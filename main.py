import os
import json
import re
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Reels Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 7000


class PipelineRequest(BaseModel):
    transcript: str
    intent: str = ""
    reels_count: int = 10


def chunk_transcript(text: str) -> list[str]:
    chunks = []
    while len(text) > CHUNK_SIZE:
        cut = text[:CHUNK_SIZE].rfind("\n")
        if cut == -1:
            cut = CHUNK_SIZE
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def clean_json(raw: str) -> list:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in response")
    return json.loads(match.group())


def call_groq(system: str, user: str, max_tokens: int = 2000) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


@app.post("/api/pipeline")
async def run_pipeline(req: PipelineRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required")

    chunks = chunk_transcript(req.transcript)
    intent_block = (
        f"\n\n--- CAMPAIGN INTENT ---\n{req.intent.strip()}\n--- END INTENT ---\n"
        if req.intent.strip() else ""
    )
    intent_set = bool(req.intent.strip())
    n = max(5, min(req.reels_count, 20))
    per_chunk = max(3, (n + 5) // len(chunks) + 2)

    shortlist_system = (
        "You are a social media content strategist specialising in short-form video (Reels/Shorts). "
        "You identify high-performing clip moments from real transcripts. "
        "You ALWAYS use exact timestamps found in the provided transcript — never invent them. "
        "For each moment, identify a start_time and end_time from the transcript. "
        "The clip must be maximum 90 seconds long (1 minute 30 seconds). "
        "Pick the end_time as the natural conclusion of the moment — where the point lands. "
        + ("You strictly respect the campaign intent. " if intent_set else "")
        + "Return ONLY a valid JSON array, no markdown, no explanation."
    )

    # ── AGENT 1: Process each chunk ──
    all_shortlisted = []
    for i, chunk in enumerate(chunks):
        chunk_user = (
            f"{intent_block}"
            f"This is part {i+1} of {len(chunks)} of the transcript:\n\n{chunk}\n\n"
            f"Find the {per_chunk} best moments for Reels from this section. "
            f"Use EXACT timestamps from this transcript section. Max clip length: 90 seconds.\n"
            + (
                "Filter by campaign intent — only pick moments that align with what we want.\n"
                if intent_set
                else "Focus on: strong hooks, surprising facts, emotional moments, actionable tips.\n"
            )
            + '\nReturn ONLY a valid JSON array:\n'
              '[{"timestamp":"exact start timestamp","start_time":"exact start timestamp","end_time":"exact end timestamp","title":"Short punchy title","description":"Why this works as a reel","hook_type":"Hook/Insight/Story/Tip/Emotion","intent_match":true}]'
        )
        try:
            if i > 0:
                time.sleep(10)
            raw = call_groq(shortlist_system, chunk_user, max_tokens=1500)
            chunk_results = clean_json(raw)
            all_shortlisted.extend(chunk_results)
        except Exception as e:
            continue

    if not all_shortlisted:
        raise HTTPException(status_code=500, detail="Agent 1 failed: No moments found")

    # ── AGENT 2: Rank all collected moments ──
    prioritize_system = (
        "You are a content prioritisation agent. You score and rank short-form video clips. "
        + (
            "Intent alignment is a PRIMARY factor — clips matching stated goals rank higher. "
            if intent_set
            else "Score by virality potential. "
        )
        + "Each clip has a start_time and end_time — preserve these exactly. "
        + "Return ONLY a valid JSON array sorted by score descending, no markdown."
    )

    candidates = all_shortlisted[:30]

    prioritize_user = (
        f"{intent_block}"
        f"Here are the shortlisted reel moments:\n{json.dumps(candidates, indent=2)}\n\n"
        f"Score and rank each from 1-10. Return the top 15 items ranked by score. "
        + (
            "Scoring MUST factor in campaign intent — matching moments score higher. "
            if intent_set else ""
        )
        + "Also consider: watch-through likelihood, shareability, emotional resonance.\n\n"
          'Return ONLY a valid JSON array sorted by score descending:\n'
          '[{"timestamp":"...","start_time":"...","end_time":"...","title":"...","description":"...","hook_type":"...","intent_match":true,"score":9,"why":"One sentence reason"}]'
    )

    try:
        time.sleep(10)
        prioritize_raw = call_groq(prioritize_system, prioritize_user, max_tokens=2500)
        prioritized = clean_json(prioritize_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 2 failed: {str(e)}")

    return {
        "shortlisted": all_shortlisted,
        "prioritized": prioritized,
        "intent_used": intent_set,
        "chunks_processed": len(chunks),
    }


class IntentRequest(BaseModel):
    raw: str


@app.post("/api/build-intent")
async def build_intent(req: IntentRequest):
    if not req.raw.strip():
        raise HTTPException(status_code=400, detail="Please provide some context")

    system = (
        "You are a campaign intent writer for a social media content team. "
        "Given rough notes, a video caption, a title, or any short description, "
        "you write a structured campaign intent document. "
        "Always write in plain text with clear sections. Never use JSON."
    )

    user = (
        f"Here is the context:\n\n{req.raw.strip()}\n\n"
        "Write a campaign intent document with these sections:\n\n"
        "Who you are:\n"
        "About this video:\n"
        "What we want:\n"
        "What to avoid:\n\n"
        "Write only the intent document, no explanation."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        result = response.choices[0].message.content
        return {"intent": result.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent builder failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}
