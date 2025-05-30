"""
Light‑weight local text‑enhancement helper.

1. Adds punctuation (comma / period / ellipsis) using a tiny offline model.
2. Optionally prepends a bracketed instruction tag the TTS engine can
   interpret for tone or style hints (e.g. "[sad]" or "[excited]").

The function is deliberately simple so it runs quickly on‑device.
Caching avoids re‑processing identical input + instruction pairs.
"""

from llama_cpp import Llama
from diskcache import Cache
import hashlib
import pathlib
import os
import re

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLM = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=35,
    n_ctx=2048,
    n_threads=os.cpu_count(),
)
cache = Cache("./enhancer_cache")

ALLOWED_BARK_TOKENS = {
    "laughter", "laughs", "sighs", "music", "gasps", "clears throat", "whispers",
    "giggles", "snickers", "coughs", "groans", "yells", "whimpers", "sobs", "murmurs",
    "chuckles", "hums", "sneezes", "grunts", "shrieks", "hiccups", "stammers", "stutters",
    "grumbles", "snorts", "howls", "moans", "guffaws", "sighs deeply", "laughs nervously",
    "cries", "sniffs", "smacks lips", "claps", "yawns", "mumbles", "shushes", "exhales sharply",
    "snaps", "whistles", "crunches", "slurps", "clicks", "clinks", "clatters", "sizzles",
    "rustles", "splashes", "taps", "thumps", "rumbles", "drums", "jingles", "jangles",
    "pops", "bangs", "hisses", "scratches", "squeaks", "screeches", "buzzes", "swooshes",
    "swoops", "clangs", "whirrs", "chirps", "beeps", "tick-tocks", "thuds", "swishes",
    "crackles", "fizzes", "humming"
}

ALLOWED_BARK_TOKEN_MAP = {tok.lower(): tok for tok in ALLOWED_BARK_TOKENS}

BASE_RULES = f"""Add expressive vocal tokens to the text below to make it sound more human and emotional.

Use tokens like: {', '.join(f'[{token}]' for token in ALLOWED_BARK_TOKENS)}.

Only insert tokens between words. Do not delete, change, or rearrange the original text.

Include multiple tokens spread throughout the piece. Think like a voice actor—where would you breathe, laugh, whisper, or sigh?

Avoid clumping tokens at the start or end. Space them out naturally.

Do not add any explanation or commentary—return only the annotated text.
"""

EXAMPLE_USAGE = """
Example 1:
Input: She whispered the secret and then laughed quietly.
Output: She[whispering] whispered the secret and then[laughter] laughed quietly.

Example 2:
Input: He ran down the hallway, gasping and shouting her name.
Output: He[gasping] ran down the hallway, [shouting] shouting her name.

Example 3:
Input: Make it sound like a quiet, emotional conversation.
Output: "I didn’t mean to..."[sighs] she said[whispers]. "Please understand."

Example 4:
Input: Make it comedic in tone, with punchy breaks and laughter.
Output: So I walk into the room...[chuckles] and guess what I see? [snorts] A duck—yes, a duck—[laughs] wearing sunglasses.

Example 5:
Input: Make this feel very natural, human, and full of character. Add emotional sounds or expressions where they fit.
Output: "You came back for me?"[gasping] she said. "After all this time..."[sighs] His eyes narrowed[chuckles]. "Well, I guess miracles happen." He stepped closer, [clears throat] "But you owe me an explanation."
"""

PROMPT_TEMPLATE = """SYSTEM:
You are a voice-enhancement assistant. Enrich the user’s text with expressive vocal markers for use in a text-to-speech engine.

{rules}

=== USER TEXT START ===
{text}
=== USER TEXT END ===

If the user includes tone or style instructions, interpret them creatively using only the allowed tokens or natural punctuation.

Return only the annotated version of the text. No extra comments or explanation.

Include multiple expressive tokens and distribute them throughout the full text to simulate realistic, emotive delivery.
"""

def enhance_text(text: str, instruction: str = "", model_id: str = "bark") -> str:
    key = hashlib.sha256((text + instruction).encode()).hexdigest()
    if key in cache:
        return cache[key]

    # Trim example usage for long inputs
    trimmed_examples = EXAMPLE_USAGE
    if len(text.split()) > 300:
        trimmed_examples = "\n".join(EXAMPLE_USAGE.split("\n")[:10])

    rules = BASE_RULES + "\n\n" + trimmed_examples
    rules += "\n\nAdditional tone/style instruction: "
    rules += instruction.strip() or "Add expressive vocal cues like [gasping], [laughs], [sighs], etc., in ways that fit the text naturally, as if performed by a dramatic voice actor. Capture emotion and variation in tone using the allowed token set."

    full_prompt = PROMPT_TEMPLATE.format(rules=rules, text=text)
    approx_prompt_tokens = len(full_prompt.split())
    max_tokens_estimate = max(64, min(2048 - approx_prompt_tokens, 512))

    try:
        result = LLM(
            full_prompt,
            max_tokens=max_tokens_estimate,
            temperature=0.4,
            top_p=0.95,
            stop=["USER:", "SYSTEM:"],
        )
        cleaned = result["choices"][0]["text"].strip()
        if "[" not in cleaned:
            cleaned = re.sub(r"(\.|\!|\?)", r"[sighs]\1", cleaned, count=1)
    except Exception as e:
        print("LLM generation error:", e)
        return text

    # Sanitize based on model
    if model_id.startswith("tts_models/multilingual/multi-dataset/xtts"):
        cleaned = re.sub(r"\[(.*?)\]", "", cleaned)
    elif model_id.startswith("bark"):
        def _filter(m):
            token_raw = m.group(1).strip()
            canon = ALLOWED_BARK_TOKEN_MAP.get(token_raw.lower())
            return f"[{canon}]" if canon else ""
        cleaned = re.sub(r"\[(.*?)\]", _filter, cleaned)

    instruction_lc = instruction.lower()

    if "less tags" in instruction_lc:
        tokens = re.findall(r"\[(.*?)\]", cleaned)
        if len(tokens) > 2:
            allowed = tokens[:2]
            cleaned = re.sub(r"\[(.*?)\]", "", cleaned)
            for token in allowed:
                cleaned = re.sub(r"(\b\w+\b)", r"\1[{}]".format(token), cleaned, count=1)

    if "longer gaps" in instruction_lc:
        cleaned = re.sub(r"([.!?])(\s*)", r"\1 ...\2", cleaned)

    if "after each sentence add a laugh" in instruction_lc:
        cleaned = re.sub(r"([.!?])(\s*)", r"\1[laughter]\2", cleaned)

    # Basic validation
    def _tok(s): return re.findall(r"\b[\w']+\b", s.lower())
    if _tok(text) != _tok(re.sub(r"\[(.*?)\]", "", cleaned)):
        cleaned = text

    cache[key] = cleaned
    return cleaned