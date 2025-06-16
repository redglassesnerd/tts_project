import requests
import json
import re
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
DEFAULT_TEMP = 0.4
MAX_TOKENS = 1024

# Load local JSON config for tag definitions and mappings
def load_prompt_profile():
    path = os.path.join(os.path.dirname(__file__), "prompts", "bark_emotion_tag_mapping.json")
    with open(path, "r") as f:
        profile = json.load(f)
        return profile, profile  # legacy support

# Call Ollama model
def LLM(prompt, temperature=DEFAULT_TEMP, max_tokens=MAX_TOKENS):
    try:
        chat_url = "http://localhost:11434/api/chat"
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        headers = {"Content-Type": "application/json"}
        print(f"\n\n[DEBUG] --- LLM Prompt ---\n{prompt}\n--- End Prompt ---\n")
        response = requests.post(chat_url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM ERROR] Ollama failed: {e}")
        return ""

# Build the prompt for enhancement
def build_enhancement_prompt(paragraph, tone_summary, tag_profile, emotion_map, creativity=DEFAULT_TEMP):
    tag_guidance = []
    for category, tags in tag_profile.get("tag_definitions", {}).items():
        for tag, meaning in tags.items():
            tag_guidance.append(f"[{tag}] — {meaning}")
    emotion_mappings = []
    for emotion, tags in emotion_map.get("emotion_tag_mapping", {}).items():
        emotion_mappings.append(f"{emotion}: {', '.join(tags)}")

    # Adjust instructions based on creativity
    if creativity <= 0.4:
        creativity_note = "Be subtle, minimal, and sparse with tag use. Only add tags where truly justified, and avoid over-embellishing the text."
    elif creativity >= 0.8:
        creativity_note = "Be expressive, emotional, and generous with tag placement. Emphasize drama and nuance; add tags for emotional effect, even if not strictly necessary."
    else:
        creativity_note = "Balance clarity and expressiveness. Use tags where they add value, but do not overuse."

    return f"""
You are a vocal director for an AI narrator.

Task:
Add expressive Bark tags (e.g., [sigh], [pause_long], [shout]) to this paragraph based on tone and emotion.

Instructions:
- DO NOT rephrase or explain. Only insert tags inline.
- Use only Bark-supported tags below.
- Place tags *before* the line or sentence they describe, ideally within quotation marks if dialogue.
- Avoid placing tags on blank lines or before formatting characters (e.g., em dashes, quotation marks on separate lines).
- Use no more than 2 tags of the same type per paragraph unless dramatically justified.
- Consider the speaker: Sarah’s dialogue may trend toward [angry], [shout], [growl]; Kimmy’s internal reflections lean toward [sigh], [pause_long], [soft].
- {creativity_note}

Formatting Notes:
- Use standard line breaks `\\n` and avoid excessive indentation.
- Return a single enhanced paragraph with tag placements integrated inline.
- Do not include extra commentary or markup beyond Bark tags.

Tone summary: {tone_summary}

Supported Tags:
{chr(10).join(tag_guidance)}

Emotion → Tag Hints:
{chr(10).join(emotion_mappings)}

Paragraph:
{paragraph}
""".strip()

# Main enhancer
def enhance_text(text, instruction="", creativity=DEFAULT_TEMP):
    tag_profile, emotion_map = load_prompt_profile()
    tone_prompt = f"""
    You are analyzing a dramatic monologue for performance.

    Task:
    Summarize the overall emotional tone, pacing, and dramatic intent of the text in 1–2 sentences, as if advising a voice actor.

    Avoid factual corrections. Focus only on mood, tension, emotional arc, and style.

    Text:
    {text}

    Tone Summary:""".strip()
    tone_summary = LLM(tone_prompt, temperature=creativity)
    tone_summary = tone_summary or "Reflective, bittersweet, emotionally rich monologue with themes of loss and empathy."

    # Preserve paragraph breaks and avoid stripping blank lines or multiple spaces
    # Split on double newlines to preserve paragraphs
    paragraph_blocks = re.split(r'(\n\s*\n)', text)
    enhanced = []
    for block in paragraph_blocks:
        # If block is just whitespace or a blank line, preserve as is
        if not block.strip():
            enhanced.append(block)
            continue
        # Remove only leading/trailing newlines, preserve inner formatting
        para = block.strip("\r\n")
        if not para:
            enhanced.append(block)
            continue
        prompt = build_enhancement_prompt(
            para, f"{tone_summary}. {instruction}", tag_profile, emotion_map, creativity=creativity
        )
        print(f"[DEBUG] Final prompt being sent for paragraph:\n{prompt}")
        result = LLM(prompt, temperature=creativity)

        # Strip LLM commentary if present
        lines = [line.rstrip() for line in result.splitlines()]
        content = []
        for line in lines:
            if re.search(r"^\s*(note|explanation|output|result|tone|label)[:\s]", line.lower()):
                continue
            if line.lower().startswith("the word") and "appears most frequently" in line.lower():
                continue
            content.append(line)
        enhanced_para = "\n".join(content).strip()

        # Clean out any residual "×" characters from the enhanced paragraph
        enhanced_para = enhanced_para.replace("×", "")

        # --- Strip unnecessary quotes around Bark tags ---
        # Replace occurrences like "[ 'sigh' ]", '"crack"', or "'sigh'" with [sigh]
        # This will match tags in single/double quotes, possibly with brackets
        # e.g., "[\"sigh\"]" or "'crack'" or '"sigh"'
        def strip_quotes_around_bark_tags(text):
            # Replace '"[tag]"' or "'[tag]'" with [tag]
            text = re.sub(r'["\']\s*(\[[a-zA-Z0-9_\- ]+\])\s*["\']', r'\1', text)
            # Replace ["tag"] or ['tag'] (no brackets) with [tag]
            text = re.sub(r'\[\s*["\']([a-zA-Z0-9_\- ]+)["\']\s*\]', r'[\1]', text)
            # Replace just "tag" or 'tag' (not inside []) with [tag] if tag is a Bark token
            # get all Bark tokens
            bark_tokens = set()
            for category, tags in tag_profile.get("tag_definitions", {}).items():
                bark_tokens.update(tags.keys())
            # Replace "tag" or 'tag' with [tag] if tag is a Bark token
            def replace_quoted_tag(m):
                tag = m.group(2)
                if tag in bark_tokens:
                    return f"[{tag}]"
                return m.group(0)
            # Match "tag" or 'tag' as a word, not inside [] or within a word
            text = re.sub(r'(^|[\s.,;:!?])([\'"]([a-zA-Z0-9_\- ]+)[\'"])(?=[\s.,;:!?]|$)', replace_quoted_tag, text)
            return text

        enhanced_para = strip_quotes_around_bark_tags(enhanced_para)

        # Heuristic fallback
        if "[" not in enhanced_para:
            print("[LLM WARNING] No tags found; using original paragraph.")
            enhanced.append(block)
        else:
            enhanced.append(enhanced_para)

    return "".join(enhanced).strip()