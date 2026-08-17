import requests
import json
import re
import os

DEFAULT_TEMP = 0.4
MAX_TOKENS = 1024

def get_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "mistral",
            "device": "auto",
            "output_folder": "output",
            "setup_completed": False
        }

# Load local JSON config for tag definitions and mappings
def load_prompt_profile():
    path = os.path.join(os.path.dirname(__file__), "prompts", "bark_emotion_tag_mapping.json")
    with open(path, "r") as f:
        profile = json.load(f)
        return profile, profile  # legacy support

# Call Ollama model
def LLM(prompt, temperature=DEFAULT_TEMP, max_tokens=MAX_TOKENS, response_format=None):
    config = get_config()
    ollama_url = config.get("ollama_url", "http://localhost:11434").rstrip("/")
    chat_url = f"{ollama_url}/api/chat"
    model = config.get("ollama_model", "mistral")
    try:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        if response_format == "json":
            payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        print(f"\n\n[DEBUG] --- LLM Prompt ---\n{prompt}\n--- End Prompt ---\n")
        response = requests.post(chat_url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM ERROR] Ollama failed: {e}")
        return ""

# Build the prompt for enhancement
def build_enhancement_prompt(paragraph, tone_summary, tag_profile, emotion_map, creativity=DEFAULT_TEMP, allowed_tokens=None):
    if allowed_tokens is not None:
        # Normalize allowed tokens to lowercase and strip brackets
        normalized_allowed = {t.replace("[", "").replace("]", "").strip().lower() for t in allowed_tokens}
    else:
        normalized_allowed = None

    tag_guidance = []
    for category, tags in tag_profile.get("tag_definitions", {}).items():
        for tag, meaning in tags.items():
            if normalized_allowed is not None and tag.lower() not in normalized_allowed:
                continue
            tag_guidance.append(f"[{tag}] — {meaning}")

    # Fallback to direct listing if allowed_tokens is custom and doesn't match standard bark tag definitions
    if normalized_allowed is not None and not tag_guidance:
        for t in allowed_tokens:
            clean_t = t.replace("[", "").replace("]", "").strip()
            tag_guidance.append(f"[{clean_t}] — model expressive emotion tag")

    emotion_mappings = []
    for emotion, tags in emotion_map.get("emotion_tag_mapping", {}).items():
        filtered_tags = []
        for tag in tags:
            clean_tag = tag.replace("[", "").replace("]", "").strip().lower()
            if normalized_allowed is None or clean_tag in normalized_allowed:
                filtered_tags.append(tag)
        if filtered_tags:
            emotion_mappings.append(f"{emotion}: {', '.join(filtered_tags)}")

    # Adjust instructions based on creativity
    if creativity <= 0.4:
        creativity_note = "Be subtle, minimal, and sparse with tag use. Only add tags where truly justified, and avoid over-embellishing the text."
    elif creativity >= 0.8:
        creativity_note = "Be expressive, emotional, and generous with tag placement. Emphasize drama and nuance; add tags for emotional effect, even if not strictly necessary."
    else:
        creativity_note = "Balance clarity and expressiveness. Use tags where they add value, but do not overuse."

    allowed_list_str = ", ".join(f"[{t.replace('[', '').replace(']', '')}]" for t in (allowed_tokens if allowed_tokens else ["laughter", "sighs", "music"]))

    example_tags = list(normalized_allowed)[:3] if normalized_allowed else ["laughter", "sighs", "music"]
    example_str = ", ".join(f"[{e}]" for e in example_tags)

    return f"""
You are a vocal director for an AI narrator.

Task:
Add expressive vocal emotion tags (e.g., {allowed_list_str}) to this paragraph based on tone and emotion.

Instructions:
- DO NOT rephrase or explain. Only insert tags inline.
- Use ONLY the supported tags listed below. Do NOT use any tags not listed in Supported Tags.
- Place tags *before* the line or sentence they describe, ideally within quotation marks if dialogue.
- Avoid placing tags on blank lines or before formatting characters.
- Use no more than 2 tags of the same type per paragraph unless dramatically justified.
- Consider the speaker: use tags (such as {example_str}) to reflect the speaker's emotional state or intent.
- {creativity_note}

Formatting Notes:
- Use standard line breaks `\\n` and avoid excessive indentation.
- Return a single enhanced paragraph with tag placements integrated inline.
- Do not include extra commentary or markup beyond supported tags.

Tone summary: {tone_summary}

Supported Tags:
{chr(10).join(tag_guidance)}

Emotion → Tag Hints:
{chr(10).join(emotion_mappings)}

Paragraph:
{paragraph}
""".strip()

# Main enhancer
def enhance_text(text, instruction="", creativity=DEFAULT_TEMP, allowed_tokens=None):
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
            para, f"{tone_summary}. {instruction}", tag_profile, emotion_map, creativity=creativity, allowed_tokens=allowed_tokens
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
            # get all Bark tokens and merge with allowed_tokens
            bark_tokens = set()
            for category, tags in tag_profile.get("tag_definitions", {}).items():
                bark_tokens.update(tags.keys())
            if allowed_tokens is not None:
                bark_tokens.update(t.replace("[", "").replace("]", "").strip().lower() for t in allowed_tokens)
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

        # Sanitization: Strip any tags that are not explicitly in the allowed_tokens list
        if allowed_tokens is not None:
            allowed_brackets = {f"[{t.replace('[', '').replace(']', '').strip().lower()}]" for t in allowed_tokens}
            def tag_sanitizer(m):
                tag = m.group(0).strip().lower()
                return tag if tag in allowed_brackets else ""
            enhanced_para = re.sub(r"\[[^\[\]]+\]", tag_sanitizer, enhanced_para)
            enhanced_para = re.sub(r"\s+", " ", enhanced_para).strip()

        # Heuristic fallback
        if "[" not in enhanced_para:
            print("[LLM WARNING] No tags found; using original paragraph.")
            enhanced.append(block)
        else:
            enhanced.append(enhanced_para)

    return "".join(enhanced).strip()

def analyze_script_soundscape(script, mood_prompt):
    prompt = f"""
You are an audio soundscape designer and film score coordinator.

Task:
Analyze the following script dialogue and the requested overall mood: "{mood_prompt}".
Identify relevant background sounds, music moods, or sound effect cues (SFX) that align with this style.
Suggest estimated timecode placements or cue conditions (such as 'Intro', 'Midway transition', or 'Ending climax') for these sounds.

Instructions:
- Provide a clean, structured JSON list of suggestions.
- Do NOT write conversational filler. Only return JSON.
- Each suggestion must have:
  1. "type": "music" or "sfx"
  2. "cue": name of music style/sound effect (e.g., "Rain on window", "Slow sax solo")
  3. "timecode": estimated timeline location (e.g., "0:00", "0:15", "0:45")
  4. "reason": briefly explain why this fits the script moment and mood.

Script:
{script}

Output JSON:
"""
    result = LLM(prompt)
    clean = re.sub(r"^```json\s*|```$", "", result, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except Exception:
        # Simple fallback parser if LLM returns imperfect JSON format
        import parse
        match = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return [{
            "type": "music",
            "cue": f"{mood_prompt} backing track",
            "timecode": "0:00",
            "reason": "Overall backing score matching mood request."
        }]

def suggest_text_edit(text, correction_note):
    prompt = f"""
You are a professional editor.

Task:
Revise the following text block to address this correction issue: "{correction_note}".

Instructions:
- You MUST maintain the original sentence structure, tone, grammar, syntax, and formatting as closely as possible.
- Maintain any inline emotional/expressive tags (like [sigh] or [laughter]) if they are present.
- ONLY return the rewritten text block. Do not explain your edit or include conversational prefix/suffix.

Original Text:
{text}

Revised Text:
"""
    result = LLM(prompt, temperature=0.2)
    return result.strip()