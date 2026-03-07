import os
import json
from dotenv import load_dotenv
load_dotenv()
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Reels Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# gemma2-9b-it: 15,000 TPM on free tier — handles long transcripts
MODEL = "gemma2-9b-it"
# Max characters to send per request (~8k tokens worth, safe buffer)
MAX_TRANSCRIPT_CHARS = 12000


class PipelineRequest(BaseModel):
    transcript: str
    intent: str = ""
    reels_count: int = 10  # how many reels the user wants


def truncate_transcript(text: str) -> tuple[str, bool]:
    """Trim transcript to safe size. Returns (text, was_truncated)."""
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text, False
    # Cut at a newline boundary so we don't chop mid-sentence
    cut = text[:MAX_TRANSCRIPT_CHARS].rfind("\n")
    if cut == -1:
        cut = MAX_TRANSCRIPT_CHARS
    return text[:cut], True


def clean_json(raw: str) -> list:
    """Strip markdown fences and parse JSON array."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    # Find first [ ... ] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in response")
    return json.loads(match.group())


def call_groq(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=3000,
    )
    return response.choices[0].message.content


@app.post("/api/pipeline")
async def run_pipeline(req: PipelineRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required")

    transcript, was_truncated = truncate_transcript(req.transcript)

    intent_block = (
        f"\n\n--- CAMPAIGN INTENT & CONTEXT ---\n{req.intent.strip()}\n--- END INTENT ---\n"
        if req.intent.strip()
        else ""
    )
    intent_set = bool(req.intent.strip())
    n = max(5, min(req.reels_count, 20))  # clamp between 5 and 20
    # Agent 2 needs to return more items than requested so user sees full ranking
    n_shortlist = n + 5

    # ── AGENT 1: Shortlist ──
    shortlist_system = (
        "You are a social media content strategist specialising in short-form video (Reels/Shorts). "
        "You identify high-performing clip moments from real transcripts. "
        "You ALWAYS use exact timestamps found in the provided transcript — never invent timestamps. "
        + ("You strictly respect the campaign intent — filtering out irrelevant or unwanted content. " if intent_set else "")
        + "Return ONLY a valid JSON array, no markdown, no explanation."
    )
    shortlist_user = (
        f"{intent_block}"
        f"Here is the YouTube transcript:\n\n{transcript}\n\n"
        f"Identify exactly {n_shortlist} best moments for Facebook/Instagram Reels (15–60 second clips). "
        f"You MUST return {n_shortlist} items — do not return fewer. "
        f"Use EXACT timestamps from the transcript.\n"
        + (
            "IMPORTANT: Use the campaign intent above to filter moments. Only pick moments that align with what we want. Actively EXCLUDE topics listed as things to avoid.\n"
            if intent_set
            else "Focus on: strong hooks, surprising facts, emotional moments, actionable tips, relatable insights.\n"
        )
        + '\nReturn ONLY a valid JSON array:\n'
          '[{"timestamp":"exact timestamp","title":"Short punchy title","description":"Why this works as a reel","hook_type":"Hook/Insight/Story/Tip/Emotion","intent_match":true}]'
    )

    try:
        shortlist_raw = call_groq(shortlist_system, shortlist_user)
        shortlisted = clean_json(shortlist_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 1 failed: {str(e)}")

    # ── AGENT 2: Prioritize ──
    prioritize_system = (
        "You are a content prioritisation agent. You score and rank short-form video clips. "
        + (
            "Intent alignment is a PRIMARY factor — clips matching stated goals rank higher, off-topic clips rank lower. "
            if intent_set
            else "Score by virality potential. "
        )
        + "Return ONLY a valid JSON array sorted by score descending, no markdown."
    )
    prioritize_user = (
        f"{intent_block}"
        f"Here are the shortlisted reel moments:\n{json.dumps(shortlisted, indent=2)}\n\n"
        f"Score and rank each from 1–10. Return ALL {len(shortlisted)} items ranked, the user wants at least {n} reels. "
        + (
            "Your scoring MUST factor in the campaign intent — moments that align closely score higher, off-topic moments score lower. "
            if intent_set
            else ""
        )
        + "Also consider: watch-through likelihood, shareability, emotional resonance, comment/reaction potential.\n\n"
          'Return ONLY a valid JSON array sorted by score descending:\n'
          '[{"timestamp":"...","title":"...","description":"...","hook_type":"...","intent_match":true,"score":9,"why":"One sentence reason"}]'
    )

    try:
        prioritize_raw = call_groq(prioritize_system, prioritize_user)
        prioritized = clean_json(prioritize_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 2 failed: {str(e)}")

    return {
        "shortlisted": shortlisted,
        "prioritized": prioritized,
        "intent_used": intent_set,
        "truncated": was_truncated,
    }


class IntentRequest(BaseModel):
    raw: str  # anything the user pastes — caption, title, description, rough notes


@app.post("/api/build-intent")
async def build_intent(req: IntentRequest):
    if not req.raw.strip():
        raise HTTPException(status_code=400, detail="Please provide some context")

    system = (
        "You are a campaign intent writer for a social media content team. "
        "Given rough notes, a video caption, a title, or any short description, "
        "you write a structured campaign intent document that will be used to guide AI agents "
        "in selecting and scoring short-form video clips (Reels/Shorts). "
        "Always write in plain text with clear sections. Never use JSON. Never use markdown headers with #."
    )

    user = (
        f"Here is what the user gave me:\n\n{req.raw.strip()}\n\n"
        "Based on this, write a structured campaign intent with these exact sections:\n\n"
        "Who you are:\n"
        "(Describe the company or team posting this content)\n\n"
        "About this video:\n"
        "(Describe the video, who is speaking, their role, and what they are discussing)\n\n"
        "What we want:\n"
        "(List 3-5 bullet points of the types of moments, themes, or clips to look for)\n\n"
        "What to avoid:\n"
        "(List 2-4 bullet points of topics, tones, or content types to exclude)\n\n"
        "Write only the intent document. No preamble, no explanation."
    )

    try:
        result = call_groq(system, user)
        return {"intent": result.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent builder failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}
