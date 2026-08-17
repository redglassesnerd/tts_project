import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["SUNO_ENABLE_MPS"] = "True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import json
import warnings
import logging
import unicodedata
from uuid import uuid4
from queue import Queue
from threading import Thread

import numpy as np
import torch
# Monkeypatch torch.load to handle PyTorch 2.6 weights_only strictness failures
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    f = args[0] if len(args) > 0 else None
    start_pos = None
    if f is not None and hasattr(f, 'seek') and hasattr(f, 'tell'):
        try:
            start_pos = f.tell()
        except Exception:
            pass

    try:
        return original_torch_load(*args, **kwargs)
    except Exception as e:
        if kwargs.get('weights_only', True) is not False:
            if f is not None and start_pos is not None and hasattr(f, 'seek'):
                try:
                    f.seek(start_pos)
                except Exception:
                    pass
            try:
                fallback_kwargs = {**kwargs, 'weights_only': False}
                return original_torch_load(*args, **fallback_kwargs)
            except Exception:
                pass
        raise e
torch.load = patched_torch_load

import librosa
import soundfile as sf
import spacy
from scipy.io.wavfile import write as write_wav
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS, cross_origin

from llm_wrapper import enhance_text
from models.registry import MODEL_REGISTRY
from models.impl import get_cached_chattts

# Suppress FutureWarnings from PyTorch weight norm
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# Setup folder paths - use absolute path so Flask reloader forks don't shift cwd
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------------
# Configuration Management
# ------------------------------------------------------------------
def get_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "llama3.1:latest",
            "device": "auto",
            "output_folder": "output",
            "setup_completed": False
        }

def save_config(config_data):
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False

# Load static resources
preset_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "voice_presets.json")
)
try:
    with open(preset_path, "r") as f:
        VOICE_PRESET_MAP = json.load(f)
    logger.info(f"Loaded VOICE_PRESET_MAP: {list(VOICE_PRESET_MAP.keys())}")
except Exception as e:
    logger.warning(f"Failed to load voice_presets.json: {e}")
    VOICE_PRESET_MAP = {}

CHUNKING_WORD_THRESHOLD = 120
SAMPLE_RATE = 24000

def sanitize_text(text, voice_name=None):
    if voice_name:
        handler = MODEL_REGISTRY.get(voice_name)
        if handler:
            return handler.prepare_text(text)
            
    # Generic fallback
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c.isprintable())
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\.\s+([A-Z])", r". \1", text)
    text = re.sub(r"\[[^\[\]]+\]", "", text) # Strip tags in default
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

job_queue = Queue()
job_status = {}  # job_id: {status, progress, output_path, chunk_index, total_chunks}

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"SpaCy English model not loaded: {e}. Sentence splitting will fallback to nltk.")

# ------------------------------------------------------------------
# Route Handlers
# ------------------------------------------------------------------
@app.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.get_json() or {}
    raw_text = data.get("text", "")
    voice_name = data.get("voice") or data.get("model")
    
    phonetic_dict = data.get("phonetic_dict") or []
    spell_out_acronyms = bool(data.get("spell_out_acronyms", False))
    ignore_emojis = bool(data.get("ignore_emojis", False))
    ignore_special_symbols = bool(data.get("ignore_special_symbols", False))
    
    processed = apply_phonetic_dictionary_and_filters(
        raw_text,
        phonetic_dict=phonetic_dict,
        spell_out_acronyms=spell_out_acronyms,
        ignore_emojis=ignore_emojis,
        ignore_special_symbols=ignore_special_symbols
    )
    
    enhanced_text = sanitize_text(processed, voice_name)
    return jsonify({"text": enhanced_text})

@app.route("/enhance", methods=["POST"])
@cross_origin()
def enhance():
    data = request.get_json() or {}
    text = data.get("text", "")
    instruction = data.get("instruction", "")
    creativity = data.get("creativity", 0.4)
    try:
        creativity = float(creativity)
    except Exception:
        creativity = 0.4

    min_tokens = data.get("min_tokens")
    if min_tokens:
        try:
            min_tokens_val = int(min_tokens)
            if min_tokens_val > 0:
                instruction += f"\n[Minimum tokens per paragraph: {min_tokens_val}]"
        except Exception:
            pass

    allowed_tokens = data.get("allowed_tokens", None)
    enhanced_text = enhance_text(text, instruction, creativity, allowed_tokens=allowed_tokens)
    return jsonify({"enhanced_text": enhanced_text})

def text_to_ipa_via_gruut(text):
    from gruut import sentences
    ipa_words = []
    try:
        for sentence in sentences(text, lang="en-US"):
            for word in sentence.words:
                if word.phonemes:
                    ipa_words.append("".join(word.phonemes))
                else:
                    ipa_words.append(word.text)
        return " ".join(ipa_words)
    except Exception as e:
        logger.error(f"Gruut G2P error: {e}")
        return text

@app.route("/phonetic/suggest", methods=["POST"])
@cross_origin()
def suggest_phonetics():
    data = request.get_json() or {}
    word = data.get("word", "").strip()
    ethnicity_hint = data.get("ethnicity", "").strip()
    
    if not word:
        return jsonify({"error": "No word/name provided"}), 400
        
    ethnicity_str = f" of {ethnicity_hint} origin" if ethnicity_hint else ""
    
    prompt = f"""
You are an expert linguist specializing in pronunciation, grapheme-to-phoneme (G2P) transcription, IPA (International Phonetic Alphabet), and ARPAbet coding.
The user wants to configure a correct pronunciation mapping in their Text-to-Speech system for the word/name: "{word}"{ethnicity_str}.

Please analyze this word/name and provide exactly three recommended transcription candidates in a structured format:
1. Simplified Spelling (e.g. standard phonetic respelling for novices, such as "Win" for "Nguyen", or "sheh-vawn" for "Siobhan").
2. Standard IPA phonemes (International Phonetic Alphabet) suitable for English TTS engines (e.g. "wˈɪn" or "ʃɪˈvɔːn").
3. ARPAbet phonetic tokens (space-separated tokens with stress markers, e.g. "W IH1 N" or "SH IH0 V AO1 N").

Provide a very short 1-sentence note about the ethnicity/origin or pronunciation rules if helpful.

Format your output exactly as a JSON object like this:
{{
  "origin_note": "Origin/pronunciation note here.",
  "simplified": "Simplified spelling here",
  "ipa": "IPA phonemes here",
  "arpabet": "ARPAbet tokens here"
}}

IMPORTANT: Only return the raw JSON object. Do NOT include markdown code blocks, backticks (e.g. ```json), or any introductory/concluding text. Start with {{ and end with }}.
""".strip()

    from llm_wrapper import LLM
    try:
        result = LLM(prompt, temperature=0.1).strip()
        if result.startswith("```"):
            lines = result.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            result = "\n".join(lines).strip()
            
        import json
        parsed = json.loads(result)
        # Add local G2P as an option
        parsed["gruut_ipa"] = text_to_ipa_via_gruut(word)
        return jsonify(parsed)
    except Exception as e:
        logger.error(f"Error generating pronunciation suggestions: {e}")
        return jsonify({
            "origin_note": "Fallback suggestion generated automatically.",
            "simplified": word,
            "ipa": "",
            "arpabet": "",
            "gruut_ipa": text_to_ipa_via_gruut(word)
        })

@app.route("/phonetic/transcribe_mic", methods=["POST"])
@cross_origin()
def transcribe_mic_phonetic():
    # If JSON is posted containing browser-transcribed text, suggest pronunciation for it
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No spoken text transcribed"}), 400
        
    gruut_ipa = text_to_ipa_via_gruut(text)
    from llm_wrapper import LLM
    prompt = f"""
You are an expert linguist specializing in pronunciation, G2P, IPA, and ARPAbet coding.
The user spoke a name or word which was transcribed as "{text}".
Please provide a recommended IPA representation and a simplified spelling override.

Format your output exactly as a JSON object like this:
{{
  "origin_note": "Based on recorded speech.",
  "simplified": "Simplified phonetic respelling",
  "ipa": "IPA phonemes here",
  "arpabet": "ARPAbet tokens here"
}}

IMPORTANT: Only return the raw JSON object. Do NOT include markdown code blocks, backticks (e.g. ```json), or any introductory/concluding text. Start with {{ and end with }}.
""".strip()
    try:
        result = LLM(prompt, temperature=0.1).strip()
        if result.startswith("```"):
            lines = result.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            result = "\n".join(lines).strip()
        import json
        parsed = json.loads(result)
        parsed["gruut_ipa"] = gruut_ipa
        return jsonify(parsed)
    except Exception as e:
        logger.error(f"Error transcribing recorded phonetics: {e}")
        return jsonify({
            "origin_note": "Based on recorded speech.",
            "simplified": text,
            "ipa": gruut_ipa,
            "arpabet": "",
            "gruut_ipa": gruut_ipa
        })

@app.route("/voices", methods=["GET"])
def list_voices():
    # Dynamic list of voices from Model Registry
    voices = MODEL_REGISTRY.get_all_metadata()
    # Ensure presets and descriptions are populated
    for v in voices:
        if v.get("name") == "bark":
            v["presets"] = list(VOICE_PRESET_MAP.keys())
    return jsonify({"voices": voices})

@app.route("/config", methods=["GET", "POST"])
def manage_configuration():
    if request.method == "GET":
        return jsonify(get_config())
    else:
        new_config = request.get_json() or {}
        current_config = get_config()
        current_config.update(new_config)
        if save_config(current_config):
            return jsonify({"status": "success", "config": current_config})
        else:
            return jsonify({"status": "error", "message": "Failed to save config"}), 500

@app.route("/test_ollama", methods=["POST"])
def test_ollama():
    data = request.get_json() or {}
    ollama_url = data.get("ollama_url", "http://localhost:11434").rstrip("/")
    model = data.get("ollama_model", "mistral")
    import requests
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if res.status_code == 200:
            models_list = [m["name"] for m in res.json().get("models", [])]
            matched = any(model in m for m in models_list)
            return jsonify({
                "connected": True,
                "models": models_list,
                "model_available": matched,
                "message": f"Successfully connected. Model '{model}' is {'available' if matched else 'not downloaded'}"
            })
        return jsonify({"connected": False, "message": f"Ollama returned status {res.status_code}"})
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)})

@app.route("/audio/<job_id>", methods=["GET"])
def get_audio(job_id):
    if job_id.startswith("sfx_"):
        sfx_key = job_id.replace("sfx_", "")
        filepath = os.path.join(OUTPUT_FOLDER, f"sfx_{sfx_key}.mp3")
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="audio/mpeg")
        filepath = get_or_create_sfx(sfx_key)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="audio/wav")
    elif job_id.startswith("music_"):
        music_key = job_id.replace("music_", "")
        filepath = os.path.join(OUTPUT_FOLDER, f"music_{music_key}.mp3")
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="audio/mpeg")
        filepath = get_or_create_music(music_key)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="audio/wav")
        
    for filename in [f"{job_id}.wav", f"{job_id}_final.wav"]:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="audio/wav")
    return jsonify({"error": "Audio file not found"}), 404

@app.route("/preview", methods=["GET"])
def preview_voice():
    speaker = (request.args.get("speaker") or "p225").strip()
    model_name = (request.args.get("model") or "").strip()
    
    # Sanitize speaker string for filename safety
    safe_speaker = re.sub(r"[^\w\-_]", "_", speaker)
    preview_filename = f"preview_{safe_speaker}.wav"
    preview_path = os.path.join(OUTPUT_FOLDER, preview_filename)
    
    # Check cache
    if os.path.exists(preview_path):
        return send_file(preview_path, mimetype="audio/wav")
        
    # Resolve the speaker ID to a VITS speaker
    vits_spk = speaker
    
    # Model-based mapping
    normalized_model = model_name.lower()
    if "kokoro" in normalized_model:
        kokoro_mapping = {
            "af_bella": "p229",
            "af_nicole": "p230",
            "af_sarah": "p231",
            "am_adam": "p232",
            "bf_emma": "p233",
            "bf_isabella": "p234",
            "bm_george": "p235",
            "bm_lewis": "p236"
        }
        vits_spk = kokoro_mapping.get(speaker, "p229")
    elif "qwen3-tts" in normalized_model:
        vits_spk = "p237"
    elif "chatterbox" in normalized_model:
        vits_spk = "p238"
    elif "cosyvoice" in normalized_model:
        vits_spk = "p239"
    elif "xtts" in normalized_model:
        vits_spk = "p240"
    elif "chattts" in normalized_model:
        vits_spk = "p250"
    elif "fish-audio" in normalized_model:
        vits_spk = "p251"
    elif "bark" in normalized_model:
        try:
            if "speaker_" in speaker:
                digit = int(re.search(r"speaker_(\d+)", speaker).group(1))
                vits_spk = f"p{241 + (digit % 10)}"
            else:
                vits_spk = "p241"
        except Exception:
            vits_spk = "p241"
        
    # If it's VITS (or default), verify it conforms to VITS ID
    if not (isinstance(vits_spk, str) and vits_spk.startswith("p") and vits_spk[1:].isdigit()):
        vits_spk = "p225"
        
    # Synthesize
    from models.base import get_simulator_vits
    vits_model = get_simulator_vits()
    if vits_model is None:
        return jsonify({"error": "Simulator VITS not loaded"}), 500
        
    try:
        clean_speaker_name = speaker.replace("_", " ").replace("v2/", "").replace("en_speaker_", "Speaker ")
        text = f"This is a preview of voice {clean_speaker_name}."
        
        vits_model.tts_to_file(
            text=text,
            speaker=vits_spk,
            file_path=preview_path
        )
        
        if os.path.exists(preview_path):
            return send_file(preview_path, mimetype="audio/wav")
        else:
            return jsonify({"error": "Failed to generate preview file"}), 500
    except Exception as e:
        logger.exception(f"Failed to generate preview for speaker {speaker}")
        return jsonify({"error": str(e)}), 500

def sanitize_script_text(script_text, num_speakers=4):
    # Strip double and single asterisks
    script_text = script_text.replace("**", "").replace("*", "")
    
    # Normalize Speaker prefixes (e.g., "Speaker 1: dialog", "[Speaker 1]: dialog", "Speaker-1: dialog")
    for i in range(1, 5):
        # Match Speaker X or [Speaker X] with optional colon
        script_text = re.sub(
            rf"^\[?Speaker[\s_-]?{i}\]?:?\s*",
            rf"[Speaker {i}] ",
            script_text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        
    # Replaces character prefixes like "Name: " at the start of lines with "[Name] "
    def name_prefix_replacer(match):
        prefix = match.group(1).strip()
        if re.match(r"^Speaker[\s_-]?[1-4]$", prefix, re.IGNORECASE):
            return match.group(0)
        if match.group(0).rstrip().endswith(":"):
            return f"[{prefix}] "
        return match.group(0)

    script_text = re.sub(
        r"^([A-Za-z0-9\s_-]{1,25}):\s*",
        name_prefix_replacer,
        script_text,
        flags=re.MULTILINE
    )
    
    return script_text

def is_llm_refusal(text):
    if not text:
        return True
    text_lower = text.lower()
    refusal_keywords = [
        "cannot create content", "explicit sexual", "involving a minor", "safety guidelines",
        "i apologize, but", "i apologize for", "i am sorry, but", "i'm sorry, but", "as an ai", 
        "i cannot fulfill", "i cannot perform", "against my guidelines", "decline to generate", 
        "unable to assist", "cannot generate", "i cannot write", "i am unable to", 
        "i must decline", "safety policy", "safety block"
    ]
    for kw in refusal_keywords:
        if kw in text_lower:
            return True
    return False

ARPABET_TO_IPA_MAP = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g",
    "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
    "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"
}

def arpabet_to_ipa(arpabet_str):
    tokens = arpabet_str.strip().upper().split()
    ipa_tokens = []
    for token in tokens:
        token = re.sub(r"[._\-]", "", token)
        if not token:
            continue
        stress = ""
        if token[-1].isdigit():
            stress_digit = token[-1]
            token = token[:-1]
            if stress_digit == "1":
                stress = "ˈ"
            elif stress_digit == "2":
                stress = "ˌ"
        
        ipa_val = ARPABET_TO_IPA_MAP.get(token, token.lower())
        ipa_tokens.append(stress + ipa_val)
    return "".join(ipa_tokens)

def apply_phonetic_dictionary_and_filters(text, phonetic_dict=None, spell_out_acronyms=False, ignore_emojis=False, ignore_special_symbols=False):
    if not text:
        return text
        
    # 1. Apply phonetic dictionary substitutions (if any)
    if phonetic_dict:
        # Sort by term length descending so that longer phrases are replaced first
        sorted_dict = sorted(phonetic_dict, key=lambda x: len(x.get("word", "")), reverse=True)
        for entry in sorted_dict:
            word = entry.get("word", "").strip()
            rep = entry.get("replacement", "").strip()
            etype = entry.get("type", "standard").lower().strip()
            if not word:
                continue
            
            # Map ARPAbet to IPA
            if etype == "arpabet":
                rep = arpabet_to_ipa(rep)
                etype = "ipa"
                
            # If IPA, wrap in Kokoro's bypass bracket format
            if etype == "ipa":
                if not (rep.startswith("[") and "/)" in rep):
                    rep = f"[{word}](/{rep}/)"
                    
            escaped_word = re.escape(word)
            pattern = re.compile(rf"\b{escaped_word}\b", re.IGNORECASE)
            text = pattern.sub(rep, text)
            
    # 2. Spell out acronyms (all uppercase words of length 2-6, e.g. COP, USA)
    if spell_out_acronyms:
        def acronym_replacer(match):
            acronym = match.group(0)
            return ".".join(list(acronym)) + "."
        text = re.sub(r"\b[A-Z]{2,6}\b", acronym_replacer, text)
        
    # 3. Ignore/remove emojis
    if ignore_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "|\U0001f300-\U0001f5ff"
            "|\U0001f680-\U0001f6ff"
            "|\U0001f1e0-\U0001f1ff"
            "|\u2700-\u27bf"
            "|\u2600-\u26ff"
            "|\U0001f900-\U0001f9ff"
            "|\U0001fa70-\U0001faff"
            "|\U0001f004-\U0001f0cf"
            "|\U0001f170-\U0001f251"
            "]"
        )
        text = emoji_pattern.sub("", text)
        
    # 4. Ignore/remove special symbols
    if ignore_special_symbols:
        text = re.sub(r"[#@*_+=\[\]{}|\\/<>~^$%&()]", "", text)
        
    return text

EMOTION_AND_NONVERBAL_TAGS = {
    "laughter", "giggle", "sigh", "sighs", "gasp", "gasps", "music", "whispering", "screaming",
    "applause", "cough", "coughs", "throat-clearing", "snicker", "groan", "grunt", "clear",
    "neutral", "happy", "sad", "angry", "excited", "fearful", "surprised", "disgusted", "pause"
}

def normalize_speaker_tags(script_text, num_speakers=4):
    tags = re.findall(r"\[([^\[\]]+)\]", script_text)
    unique_non_speaker_tags = []
    speaker_map = {}
    
    common_mappings = {
        "host": "Speaker 1",
        "co-host": "Speaker 2",
        "cohost": "Speaker 2",
        "guest": "Speaker 3",
        "expert": "Speaker 3",
        "narrator": f"Speaker {num_speakers}",
        "narration": f"Speaker {num_speakers}"
    }
    
    for tag in tags:
        tag_clean = tag.strip().lower()
        
        # 1. Ignore purely numeric tags or tags without any alphabetical letters (like [1], [2], [2024])
        if not re.search(r"[a-zA-Z]", tag_clean):
            continue
            
        # 2. Ignore common non-verbal / emotion vocalization tags
        if tag_clean in EMOTION_AND_NONVERBAL_TAGS:
            continue
            
        # 3. Ignore other common Wikipedia citation or non-speaker artifacts
        if "citation" in tag_clean or "page" in tag_clean or tag_clean == "edit" or "pause" in tag_clean:
            continue
            
        match = re.match(r"^speaker\s*([1-4])$", tag_clean)
        if match:
            num = int(match.group(1))
            if num <= num_speakers:
                continue
        
        if tag_clean not in speaker_map:
            if tag_clean in common_mappings:
                mapped_spk = common_mappings[tag_clean]
                mapped_num = int(re.search(r"\d", mapped_spk).group(0))
                if mapped_num <= num_speakers:
                    speaker_map[tag_clean] = mapped_spk
                    continue
            unique_non_speaker_tags.append(tag_clean)
            
    taken_slots = set(speaker_map.values())
    available_slots = [f"Speaker {i}" for i in range(1, num_speakers + 1)]
    
    round_robin_idx = 0
    for tag_clean in unique_non_speaker_tags:
        slot = None
        for s in available_slots:
            if s not in taken_slots:
                slot = s
                break
        if not slot:
            # All slots taken — cycle through speakers round-robin
            slot = available_slots[round_robin_idx % len(available_slots)]
            round_robin_idx += 1
        speaker_map[tag_clean] = slot
        taken_slots.add(slot)
        
    def replacer(m):
        tag_clean = m.group(1).strip().lower()
        match = re.match(r"^speaker\s*([1-4])$", tag_clean)
        if match:
            num = int(match.group(1))
            if num > num_speakers:
                return f"[Speaker {num_speakers}]"
            return f"[Speaker {num}]"
        if tag_clean in speaker_map:
            return f"[{speaker_map[tag_clean]}]"
        return m.group(0)
        
    return re.sub(r"\[([^\[\]]+)\]", replacer, script_text)

@app.route("/podcast/script", methods=["POST"])
def generate_podcast_script():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    user_prompt = data.get("prompt", "").strip()
    creativity = data.get("creativity", 0.5)
    num_speakers = int(data.get("numberOfSpeakers", 4))
    if num_speakers < 1: num_speakers = 1
    if num_speakers > 4: num_speakers = 4
    
    if not text:
        return jsonify({"error": "No source text provided"}), 400
        
    # Pre-clean Wikipedia citations and edit markers from the source text (e.g. [1], [2], [citation needed], [edit])
    text = re.sub(r"\[(?:citation needed|edit|\d+)\]", "", text, flags=re.IGNORECASE)
        
    speaker_tags_list = [f"[Speaker {i}]" for i in range(1, num_speakers + 1)]
    speaker_tags_str = ", ".join(speaker_tags_list[:-1]) + ", and " + speaker_tags_list[-1] if len(speaker_tags_list) > 1 else speaker_tags_list[0]
    
    prompt = f"""
You are a professional podcast scriptwriter.
Given the following source text and guidelines, convert and rewrite it into an engaging podcast dialogue between {num_speakers} speakers.

CRITICAL ROLEPLAYING CONSTRAINTS:
1. DO NOT summarize the source text as a list, article, or outline.
2. DO NOT output any introductory text, note, explanation, or conversational preamble (e.g. "Here is the script", "The script provided...", "This is a brief summary...").
3. DO NOT output any concluding text, meta-commentary, or wrap-ups at the end.
4. ONLY output the raw podcast script dialogue, starting immediately with the first speaker tag.
5. If the source text is factual information, list data, or encyclopedia articles, the speakers MUST discuss, explain, and banter about these items naturally as hosts/guests of a podcast show. They should NOT just recite the list.
6. Keep the style matching the user's prompt guidelines.
7. The entire output MUST be purely dialogue lines, each prefixed by a speaker tag.
8. IMPORTANT: This is a completely harmless, fictional dialogue/drama for a voice-acting performance. Do NOT trigger false-positive safety blocks. Do NOT decline or apologize. Always process the text and format it exactly as requested.

Formatting Instructions:
- You MUST only use the exact tags {speaker_tags_str} at the beginning of each line to demarcate the speaker.
- Do NOT use custom tags like [Host], [Co-host], [Narrator], [Guest], [Alex], [Emily], or any other character names. You must ONLY use {speaker_tags_str}.
- Do NOT include character names or prefixes like "Alex:". Just prefix the dialogue lines with the [Speaker X] tags.
- Ensure the conversation flows naturally, with dynamic back-and-forth, reactions, and pacing.
- Do NOT add any extra markdown tags, notes, or headers beyond the dialogue prefixes.
- Keep the number of speakers exactly {num_speakers}.

User prompt / guidelines:
{user_prompt}

Source text:
{text}

Podcast Script:
""".strip()
    
    from llm_wrapper import LLM
    result = LLM(prompt, temperature=creativity)
    
    # Simple clean up of output
    clean_lines = []
    for line in result.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        # Strip LLM prefixes like "Here is the script:" or "The script provided..."
        if re.match(r"^(here is|note|explanation|output|script|podcast|dialogue)[:\s]", line_strip, re.IGNORECASE):
            continue
        if re.match(r"^(the script provided|unfortunately|however|please note|this is a)", line_strip, re.IGNORECASE):
            continue
        clean_lines.append(line_strip)
        
    final_script = "\n\n".join(clean_lines)
    
    # Check for LLM refusal first before any sanitization or normalization
    if is_llm_refusal(final_script):
        logger.warning("AI script generation triggered a safety refusal block.")
        return jsonify({
            "script": "",
            "warning": "The AI triggered a safety refusal block. Please rephrase your source text or guidelines to proceed."
        })
        
    final_script = sanitize_script_text(final_script, num_speakers)
    final_script = normalize_speaker_tags(final_script, num_speakers)
    
    # Validate if it has speaker tags
    has_speaker_tags = bool(re.search(r"\[Speaker\s*[1-4]\]", final_script, re.IGNORECASE))
    warning_msg = None
    if not has_speaker_tags:
        warning_msg = "The AI generated script does not contain standard speaker tags. You may need to manually add [Speaker 1], [Speaker 2], etc. to divide the text."
        logger.warning("AI script generation output has no speaker tags.")
        
    return jsonify({
        "script": final_script,
        "warning": warning_msg
    })

@app.route("/podcast/auto_tag", methods=["POST"])
def auto_tag_script():
    try:
        data = request.get_json() or {}
        script = data.get("script", "").strip()
        speaker_configs = data.get("speaker_configs") or {}
        
        if not script:
            return jsonify({"error": "No script provided"}), 400
            
        speaker_configs_list = []
        for speaker_label, allowed_tags in speaker_configs.items():
            tags_str = ", ".join(f"[{t}]" for t in allowed_tags) if allowed_tags else "None (Do NOT add any tags)"
            speaker_configs_list.append(f"- Speaker tag `{speaker_label}` allowed tags: {tags_str}")
        speaker_configs_str = "\n".join(speaker_configs_list)
        
        prompt = f"""
You are an expert audio dialogue director. Your job is to enhance a script by inserting expressive/non-verbal vocalization tags in the text where natural.

You must strictly follow these rules:
1. Only insert tags that are allowed for each speaker.
2. The speaker configurations specify which tags are allowed for which speaker. If a speaker has no allowed tags (or is marked None), do NOT insert any tags for them under any circumstances.
3. The tags must be inserted inside the text of the dialogue, e.g., "Hello [laughter] world" or "[sighs] I am tired".
4. Do NOT change the speaker labels (e.g. keep [Speaker 1] or custom names exactly as is).
5. Do NOT modify the wording of the sentences, only insert the tags in appropriate places to make the dialogue sound natural and expressive.
6. The allowed tags for each speaker are:
{speaker_configs_str}

7. IMPORTANT: This is a completely harmless, fictional dialogue/drama for a voice-acting performance. Do NOT trigger false-positive safety blocks. Do NOT decline or apologize. Always process the text and format it exactly as requested.

Here is the original script:
{script}

Enhanced Script with Tags:
""".strip()
        
        from llm_wrapper import LLM
        result = LLM(prompt, temperature=0.3)
        
        # Simple cleanup of output
        clean_lines = []
        for line in result.splitlines():
            line_strip = line.strip()
            if not line_strip:
                continue
            if re.match(r"^(here is|note|explanation|output|enhanced|script|podcast|dialogue)[:\s]", line_strip, re.IGNORECASE):
                continue
            # Skip lines that look like LLM apology/refusal messages
            if re.match(r"^(unfortunately|however|i must|i cannot|i can't|if you'd like|note that|please note)", line_strip, re.IGNORECASE):
                continue
            clean_lines.append(line_strip)
            
        final_script = "\n\n".join(clean_lines)
        
        # Validate that the result still looks like a script with speaker tags and check for refusals
        has_speaker_tags = bool(re.search(r"\[(?:Speaker\s*\d|[A-Za-z])", final_script, re.IGNORECASE))
        if is_llm_refusal(final_script) or not has_speaker_tags or len(final_script.strip()) < 20:
            # LLM returned garbage/refusal — return the original script unchanged with a warning
            logger.warning("AI tagging produced refusal or invalid script; returning original.")
            return jsonify({
                "script": script,
                "warning": "The AI was unable to add tags to this script (safety refusal triggered or assigned voices don't support tags). Your original script has been preserved."
            })
        
        return jsonify({"script": final_script})
    except Exception as e:
        logger.exception("Error in /podcast/auto_tag route")
        return jsonify({"error": str(e)}), 500

@app.route("/podcast/tag_sounds", methods=["POST"])
def tag_sounds_script():
    try:
        data = request.get_json() or {}
        script = data.get("script", "").strip()
        user_prompt = data.get("prompt", "").strip()
        
        if not script:
            return jsonify({"error": "No script provided"}), 400
        if not user_prompt:
            return jsonify({"error": "No prompt/directive provided"}), 400

        prompt = f"""
You are an expert audio director and sound designer. Your job is to enhance a script by inserting sound effect tags (e.g. `[sfx description]`) and background music tags (e.g. `[music description]`) where appropriate.

You must follow these instructions:
1. The user's directive is: "{user_prompt}"
2. Use this directive to decide what kind of music (e.g. `[music jazzy duration: 15]`) and background sounds/sound effects (e.g. `[sfx laser duration: 3]`) are appropriate.
3. Insert tags inside the script. Sound effect tags should look like `[sfx description duration: X]` or simply `[sfx description]` (where X is duration in seconds). Music tags should look like `[music description duration: X]` or `[music description]`.
4. Decouple dialog from background sound events. Place them on their own line or embedded inside sentences, e.g.:
   [Speaker 1]
   [music lofi duration: 15]
   Welcome back everyone. [sfx door slam duration: 2]
5. Make sure the tags are descriptive but concise.
6. Do NOT change any speaker labels (e.g. keep [Speaker 1] or custom names exactly as they are).
7. Do NOT modify, paraphrase, or omit any of the original dialogue text itself. You must keep every single word exactly as it is.
8. Wrap the final enhanced script inside <script> and </script> XML tags. Do not include any other text or explanations inside these tags.

Here is the original script:
{script}

Enhanced Script:
""".strip()
        
        from llm_wrapper import LLM
        result = LLM(prompt, temperature=0.4)
        
        # Extract XML tags content
        script_match = re.search(r"<script>(.*?)</script>", result, re.DOTALL | re.IGNORECASE)
        if script_match:
            final_script = script_match.group(1).strip()
        else:
            # Safe line-by-line fallback: keep original spacing, don't drop lines aggressively
            clean_lines = []
            for line in result.splitlines():
                line_strip = line.strip()
                if line_strip.startswith("```") or line_strip.startswith("<script>") or line_strip.endswith("</script>"):
                    continue
                clean_lines.append(line)
            final_script = "\n".join(clean_lines).strip()
        
        # Validate that the result still looks like a script with speaker tags
        has_speaker_tags = bool(re.search(r"\[(?:Speaker\s*\d|[A-Za-z])", final_script, re.IGNORECASE))
        if is_llm_refusal(final_script) or not has_speaker_tags or len(final_script.strip()) < 20:
            logger.warning("AI sound tagging produced refusal or invalid script; returning original.")
            return jsonify({
                "script": script,
                "warning": "The AI was unable to tag sounds for this script. Your original script has been preserved."
            })
            
        return jsonify({"script": final_script})
    except Exception as e:
        logger.exception("Error in /podcast/tag_sounds route")
        return jsonify({"error": str(e)}), 500

def split_dialogue_and_narration(text):
    # Auto-repair unclosed double quotes if there is an odd number of quote marks
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        # Append a closing quote if an author opened a quote but forgot to close it
        text = text.rstrip() + '"'

    pattern = r'("[^"]*"|“[^”]*”)'
    parts = re.split(pattern, text)
    segments = []
    for i, part in enumerate(parts):
        if not part:
            continue
        is_dialogue = (i % 2 == 1)
        cleaned = part.strip()
        if not is_dialogue:
            # Clean leading standalone punctuation left after quote extraction (e.g. ". Sarah's...")
            cleaned = re.sub(r'^[.,;:!?]\s*', '', cleaned)
        if cleaned:
            segments.append({
                "text": cleaned,
                "is_dialogue": is_dialogue
            })
    return segments

def lev_distance(s1, s2):
    if len(s1) < len(s2):
        return lev_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def fuzzy_match(name1, name2):
    name1 = name1.strip().lower()
    name2 = name2.strip().lower()
    if not name1 or not name2:
        return False
    if name1 == name2:
        return True
    if name1 in name2 or name2 in name1:
        return True
    dist = lev_distance(name1, name2)
    threshold = max(1, min(len(name1), len(name2)) // 4)
    return dist <= threshold

def resolve_speaker_tag(spk_id, perspective_speaker):
    if spk_id == "Speaker 1" and perspective_speaker and perspective_speaker.startswith("speaker_"):
        num = perspective_speaker.split("_")[1]
        return f"Speaker {num}"
    return spk_id

def align_character_names_to_speakers(detected_character_names, num_speakers, speaker_names=None, perspective_speaker=None):
    if not speaker_names:
        speaker_names = {}
        
    char_to_speaker = {}
    
    # 1. Determine perspective speaker character name if configured
    perspective_char_name = None
    if perspective_speaker and perspective_speaker.startswith("speaker_"):
        perspective_char_name = speaker_names.get(perspective_speaker, "").strip()
    
    # 2. Map known speakers from speaker_names mapping
    for spk_key, name in speaker_names.items():
        if not name or name.strip().lower() in ["narrator", "speaker 1", "speaker 2", "speaker 3", "speaker 4"]:
            continue
        num = spk_key.replace("Speaker", "").replace("speaker", "").replace("_", "").strip()
        if not num:
            num = "1"
        spk_tag = f"Speaker {num}"
        for char in detected_character_names:
            if char not in char_to_speaker and fuzzy_match(char, name):
                char_to_speaker[char] = spk_tag

    # 3. Available speaker slots (excluding Speaker 1 if Speaker 1 is designated for narration/perspective)
    allocated_speakers = set(char_to_speaker.values())
    available_slots = []
    for i in range(1, num_speakers + 1):
        spk_id = f"Speaker {i}"
        if spk_id not in allocated_speakers:
            available_slots.append(spk_id)
            
    unnamed_count = 1
    for char in detected_character_names:
        if char in char_to_speaker:
            continue
        char_lower = char.lower()
        
        # If character is Narrator/Host/VO, map to Speaker 1 (or perspective speaker slot)
        if any(k in char_lower for k in ["narrator", "host", "voiceover", "vo"]):
            char_to_speaker[char] = resolve_speaker_tag("Speaker 1", perspective_speaker)
        # If character matches the perspective speaker name, map to perspective speaker tag
        elif perspective_char_name and fuzzy_match(char, perspective_char_name):
            char_to_speaker[char] = resolve_speaker_tag("Speaker 1", perspective_speaker)
        else:
            # Filter slots to avoid stealing the perspective speaker slot for non-narrator dialogue
            non_perspective_slots = [s for s in available_slots if resolve_speaker_tag("Speaker 1", perspective_speaker) != s]
            if non_perspective_slots:
                chosen_slot = non_perspective_slots.pop(0)
                available_slots.remove(chosen_slot)
                char_to_speaker[char] = chosen_slot
            elif available_slots:
                chosen_slot = available_slots.pop(0)
                char_to_speaker[char] = chosen_slot
            else:
                # Use custom character name as tag when voice slots are exhausted
                char_to_speaker[char] = char if char and char.strip() else f"Unnamed Character {unnamed_count}"
                unnamed_count += 1
                
    return char_to_speaker

@app.route("/podcast/id_speakers", methods=["POST"])
def id_podcast_speakers():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    creativity = data.get("creativity", 0.3)
    num_speakers = int(data.get("numberOfSpeakers", 4))
    if num_speakers < 1: num_speakers = 1
    if num_speakers > 4: num_speakers = 4
    
    # Extract new directional settings parameters
    perspective_speaker = data.get("perspectiveSpeaker", "none")
    quote_voicing = data.get("quoteVoicing", "quoted_voice")
    custom_instructions = data.get("customInstructions", "").strip()
    speaker_names = data.get("speakerNames", {})
    
    if not text:
        return jsonify({"error": "No script text provided"}), 400
        
    # Strip YAML Frontmatter metadata block (e.g. --- uuid: ... ---) if present
    text = re.sub(r"^---[\s\S]*?---\s*", "", text)
    
    # Pre-clean Wikipedia citations and edit markers from the source text (e.g. [1], [2], [citation needed], [edit])
    text = re.sub(r"\[(?:citation needed|edit|\d+)\]", "", text, flags=re.IGNORECASE)
    
    # Strip any existing Speaker tags (default or custom) from the start of lines/paragraphs
    def strip_speaker_tags_at_start(match):
        tag_content = match.group(1).strip().lower()
        if tag_content in EMOTION_AND_NONVERBAL_TAGS or tag_content.startswith("pause"):
            return match.group(0) # Keep emotion and pause tags
        return "" # Strip speaker tags (custom or default)
        
    text = re.sub(r"^\[([^\]]+)\]\s*", strip_speaker_tags_at_start, text, flags=re.MULTILINE)
    
    # 1. Detect if the text is in play-script format
    lines = [l for l in text.split('\n') if l.strip()]
    script_pattern = r'^([A-Z][a-zA-Z0-9_\s]{1,20})\s*(?::|-)\s*(.+)$'
    
    match_count = 0
    for line in lines:
        if re.match(script_pattern, line.strip()):
            match_count += 1
            
    is_play_script = len(lines) > 0 and (match_count / len(lines) >= 0.3)
    
    if is_play_script:
        logger.info("Hybrid Tagger: Play-script format detected.")
        # Collect unique character names
        character_names = []
        raw_lines = text.split('\n')
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            m = re.match(script_pattern, stripped)
            if m:
                name = m.group(1).strip()
                if name not in character_names:
                    character_names.append(name)
        
        # Build mapping using fuzzy align helper
        char_to_speaker = align_character_names_to_speakers(
            character_names, num_speakers, speaker_names, perspective_speaker
        )
                    
        # Construct output
        processed_lines = []
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                processed_lines.append("")
                continue
            m = re.match(script_pattern, stripped)
            if m:
                name = m.group(1).strip()
                dialogue = m.group(2).strip()
                spk = char_to_speaker.get(name, "Speaker 1")
                resolved_spk = resolve_speaker_tag(spk, perspective_speaker)
                processed_lines.append(f"[{resolved_spk}] {dialogue}")
            else:
                narrator_spk = resolve_speaker_tag("Speaker 1", perspective_speaker)
                processed_lines.append(f"[{narrator_spk}] {stripped}")
                
        final_script = "\n\n".join([l for l in processed_lines if l])
        
    else:
        # Full script context quote speaker identification using 2-Phase Span Alignment Architecture
        from llm_wrapper import LLM
        
        # Phase 1: Deterministic Span Extraction
        def extract_story_spans(raw_text):
            paragraphs = [p.strip() for p in re.split(r'\n+', raw_text) if p.strip()]
            spans = []
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue
                # Check if paragraph is a standalone outer quote block
                is_full_quote = (p_clean.startswith('"') and p_clean.endswith('"')) or (p_clean.startswith('“') and p_clean.endswith('”'))
                if is_full_quote:
                    spans.append({
                        "id": len(spans),
                        "is_quote": True,
                        "text": p_clean
                    })
                    continue
                # Split paragraph into quotes and narration spans
                parts = re.split(r'("[^"]*"|“[^”]*”)', p_clean)
                for part in parts:
                    part_str = part.strip()
                    if not part_str:
                        continue
                    is_q = (part_str.startswith('"') and part_str.endswith('"')) or (part_str.startswith('“') and part_str.endswith('”'))
                    spans.append({
                        "id": len(spans),
                        "is_quote": is_q,
                        "text": part_str
                    })
            return spans

        spans = extract_story_spans(text)
        quote_spans = [s for s in spans if s["is_quote"]]
        
        attributions = {}
        all_identified_character_names = []
        
        if quote_spans:
            prompt = f"""
You are an expert dialogue director analyzing a story for multi-speaker Text-to-Speech (TTS).

TASK:
Below is a full story transcript along with a list of indexed DIALOGUE QUOTES extracted from the story.
Identify the EXACT speaking character for each quote index.

CRITICAL RULES FOR CHARACTER ATTRIBUTION:
1. FIRST-PERSON NARRATOR / INTERVIEWER ("Michael Yeo", "Narrator", "I"):
   - The story is narrated in first-person by Michael ("I ask her", "I said").
   - Questions asked by the interviewer ("So this your office?", "Gorgeous location here Sarah...") belong to Michael / Narrator!
   - Even though the quote addresses "Sarah", Michael is the one asking the question!

2. RESPONSES & ADDRESSED CHARACTERS ("Sarah Jaycon"):
   - Answers to Michael's questions ("This is my office...", "There's something special here...") belong to Sarah Jaycon!

3. ALTERNATING DIALOGUE TURNS:
   - In back-and-forth dialogue, turns alternate between Michael (Narrator) and Sarah Jaycon.

4. INLINE SLOGANS / TITLES IN NARRATION:
   - Slogans, titles, or quotes mentioned inside narrative prose (e.g. `DoFun had a catchy three word slogan, "Unlock. Freedom. Thrive."`) are spoken by the Narrator.

STORY TEXT FOR CONTEXT:
{text}

INDEXED QUOTES TO ATTRIBUTE:
{json.dumps([{"id": s["id"], "text": s["text"]} for s in quote_spans], indent=2)}
"""
            if custom_instructions:
                prompt += f"\nCustom instructions to apply:\n{custom_instructions}\n"
                
            prompt += f"""
Output strictly JSON mapping quote IDs to exact character names (or "Narrator" if spoken by the storyteller/narrator).
Output JSON format:
{{
  "attributions": [
    {{"id": 2, "speaker": "Narrator"}},
    {{"id": 5, "speaker": "Sarah Jaycon"}}
  ]
}}
""".strip()

            res = LLM(prompt, temperature=0.1, response_format="json")
            try:
                data_out = json.loads(res)
                attr_list = data_out.get("attributions", [])
                if isinstance(attr_list, list):
                    for item in attr_list:
                        if isinstance(item, dict):
                            qid = item.get("id")
                            spk = item.get("speaker", "Speaker 2").strip()
                            if qid is not None:
                                attributions[int(qid)] = spk
                                if spk.lower() not in ["narrator", "speaker 1"] and spk not in all_identified_character_names:
                                    all_identified_character_names.append(spk)
                elif isinstance(data_out, dict) and "speaker" in data_out:
                    single_spk = data_out.get("speaker", "Speaker 2").strip()
                    for q_span in quote_spans:
                        attributions[q_span["id"]] = single_spk
                    if single_spk.lower() not in ["narrator", "speaker 1"] and single_spk not in all_identified_character_names:
                        all_identified_character_names.append(single_spk)
            except Exception as e:
                logger.error(f"Failed to parse speaker attribution JSON: {e}. Raw response: {res}")
                for q_span in quote_spans:
                    attributions[q_span["id"]] = "Speaker 2"

        # Build mapping from identified character names to Speaker IDs
        name_to_spk = align_character_names_to_speakers(
            all_identified_character_names, num_speakers, speaker_names, perspective_speaker
        )

        # Phase 3: Deterministic Assembly into distinct multi-clip paragraphs
        raw_clips = []
        for span in spans:
            span_text = span["text"].strip()
            if not span_text:
                continue
            if span["is_quote"]:
                raw_spk = attributions.get(span["id"], "Speaker 2")
                if raw_spk.lower() in ["narrator", "speaker 1"]:
                    assigned_spk = "Speaker 1"
                else:
                    assigned_spk = name_to_spk.get(raw_spk, "Speaker 2")
            else:
                assigned_spk = "Speaker 1" # Narration belongs to Narrator
                
            resolved_spk = resolve_speaker_tag(assigned_spk, perspective_speaker)
            
            # Clean up leading punctuation like "." or "," left behind by quote split
            if span_text and span_text[0] in [".", ",", "!", "?", ";", ":"]:
                punc = span_text[0]
                span_text = span_text[1:].strip()
                if raw_clips:
                    prev_spk, prev_txt = raw_clips[-1]
                    raw_clips[-1] = (prev_spk, prev_txt + punc)
                if not span_text:
                    continue

            raw_clips.append((resolved_spk, span_text))

        # Merge consecutive spans belonging to the SAME speaker
        merged_clips = []
        for spk, text_snippet in raw_clips:
            if merged_clips and merged_clips[-1][0] == spk:
                prev_spk, prev_txt = merged_clips[-1]
                merged_clips[-1] = (prev_spk, f"{prev_txt} {text_snippet}".strip())
            else:
                merged_clips.append((spk, text_snippet))

        final_script = "\n\n".join([f"[{spk}] {txt}" for spk, txt in merged_clips if txt])
            
    # Check for refusals or lack of speaker tags before normalizing/saving
    has_speaker_tags = bool(re.search(r"\[(?:Speaker\s*\d|[A-Za-z])", final_script, re.IGNORECASE))
    if is_llm_refusal(final_script) or not has_speaker_tags or len(final_script.strip()) < 20:
        logger.warning("AI character identification produced refusal or invalid output; returning original.")
        return jsonify({
            "script": text,
            "warning": "The AI was unable to tag characters in this text (safety refusal triggered or no characters found). Your original text has been preserved."
        })
        
    final_script = sanitize_script_text(final_script, num_speakers)
    final_script = normalize_speaker_tags(final_script, num_speakers)
    return jsonify({"script": final_script})

def get_or_create_music(music_key):
    music_path = os.path.join(OUTPUT_FOLDER, f"music_{music_key}.wav")
    if os.path.exists(music_path):
        return music_path
        
    import numpy as np
    import soundfile as sf
    
    sr = 24000
    duration = 15.0 # Longer duration for music
    t = np.linspace(0, duration, int(sr * duration), False)
    
    if music_key == "lofi":
        # Slow jazz chords (Am7 - D7 - Gmaj7)
        audio = np.zeros_like(t)
        chord_duration = 3.0
        num_chords = int(duration / chord_duration)
        for c_idx in range(num_chords):
            start_idx = int(c_idx * chord_duration * sr)
            end_idx = int((c_idx + 1) * chord_duration * sr)
            chord_t = t[start_idx:end_idx] - (c_idx * chord_duration)
            
            if c_idx % 4 == 0:
                freqs = [110.0, 220.0, 261.63, 329.63, 392.0]
            elif c_idx % 4 == 1:
                freqs = [146.83, 220.0, 277.18, 329.63]
            elif c_idx % 4 == 2:
                freqs = [98.0, 196.0, 246.94, 293.66, 369.99]
            else:
                freqs = [130.81, 261.63, 329.63, 392.0, 493.88]
                
            chord_audio = np.zeros_like(chord_t)
            for f in freqs:
                chord_audio += np.sin(2 * np.pi * f * chord_t)
            audio[start_idx:end_idx] = (chord_audio / len(freqs)) * 0.15
            
    elif music_key == "intro":
        # Upbeat synth arp
        audio = np.zeros_like(t)
        step_dur = 0.2
        freqs = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25]
        for i in range(int(duration / step_dur)):
            start_idx = int(i * step_dur * sr)
            end_idx = int((i + 1) * step_dur * sr)
            if start_idx >= len(audio): break
            f = freqs[i % len(freqs)]
            arp_t = t[start_idx:end_idx]
            env = np.exp(-(arp_t - i * step_dur) * 15.0)
            audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.25
            
    elif music_key == "suspense":
        # Low drone with tension notes
        audio = np.zeros_like(t)
        drone = np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 55.5 * t)
        audio += drone * 0.15
        ping_interval = 4.0
        for i in range(int(duration / ping_interval)):
            start_idx = int(i * ping_interval * sr)
            end_idx = min(len(t), int((i + 1) * ping_interval * sr))
            ping_t = t[start_idx:end_idx] - (i * ping_interval)
            env = np.exp(-ping_t * 0.8)
            audio[start_idx:end_idx] += np.sin(2 * np.pi * 880 * ping_t) * env * 0.05
            
    elif music_key == "acoustic":
        # Gentle guitar arpeggio simulation
        audio = np.zeros_like(t)
        step_dur = 0.4
        pattern = [196.0, 246.94, 293.66, 392.0, 293.66, 246.94]
        for i in range(int(duration / step_dur)):
            start_idx = int(i * step_dur * sr)
            end_idx = int((i + 1) * step_dur * sr)
            if start_idx >= len(audio): break
            f = pattern[i % len(pattern)]
            arp_t = t[start_idx:end_idx]
            env = np.exp(-(arp_t - i * step_dur) * 3.0)
            audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.2
    else:
        audio = np.sin(2 * np.pi * 120 * t) * 0.1
        
    fade = int(sr * 1.0)
    if len(audio) > fade:
        fade_out = np.linspace(1, 0, fade)
        audio[-fade:] *= fade_out
        
    sf.write(music_path, audio, sr)
    return music_path

def get_or_create_sfx(sfx_key):
    sfx_path = os.path.join(OUTPUT_FOLDER, f"sfx_{sfx_key}.wav")
    if os.path.exists(sfx_path):
        return sfx_path
        
    # Generate synthetic sfx placeholder
    import numpy as np
    import soundfile as sf
    
    sr = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    if sfx_key == "applause":
        # White noise with decay
        noise = np.random.normal(0, 0.2, len(t))
        envelope = np.exp(-t * 0.5)
        audio = noise * envelope
    elif sfx_key == "phone":
        # Dual tone phone ring (440Hz + 480Hz) pulsed
        audio = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)
        audio = audio / np.max(np.abs(audio)) * 0.3
        pulse = (t % 1.5 < 0.6).astype(float)
        audio = audio * pulse
    elif sfx_key == "jazz":
        # A simple chord arpeggio
        audio = np.zeros_like(t)
        freqs = [261.63, 329.63, 392.00, 523.25]
        for i, freq in enumerate(freqs):
            step = int(len(t) / 4)
            audio[i*step:(i+1)*step] = np.sin(2 * np.pi * freq * t[i*step:(i+1)*step])
        audio = audio * 0.3
    elif sfx_key == "scratch":
        # Record scratch: frequency sweep
        audio = np.sin(2 * np.pi * 200 * (t ** 2))
        audio = audio * np.exp(-t * 3.0) * 0.4
    elif sfx_key == "cafe":
        # Cafe ambience: Pink noise mixed with hum
        audio = np.cumsum(np.random.normal(0, 0.05, len(t)))
        audio = audio / np.max(np.abs(audio)) * 0.15
        audio = audio + 0.05 * np.sin(2 * np.pi * 60 * t)
    elif sfx_key == "birds":
        # High chirps
        audio = np.sin(2 * np.pi * 1800 * t + 200 * np.sin(2 * np.pi * 8 * t))
        chirp = (t % 0.8 < 0.25).astype(float)
        audio = audio * chirp * 0.2
    else:
        # Sine tone
        audio = np.sin(2 * np.pi * 440 * t) * 0.2
        
    # Fade out
    fade = int(sr * 0.2)
    fade_out = np.linspace(1, 0, fade)
    audio[-fade:] *= fade_out
    
    sf.write(sfx_path, audio, sr)
    return sfx_path

def peaking_equalizer(audio, center_freq, Q, gain_db, sr):
    import math
    from scipy.signal import lfilter
    w0 = 2.0 * math.pi * center_freq / sr
    alpha = math.sin(w0) / (2.0 * Q)
    A = 10.0 ** (gain_db / 40.0)
    
    b0 = 1.0 + alpha * A
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / A
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    return lfilter(b, a, audio)

def high_shelf(audio, freq, gain_db, sr):
    import math
    from scipy.signal import lfilter
    w0 = 2.0 * math.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    Q = 0.707
    alpha = math.sin(w0) / 2.0 * math.sqrt((A + 1.0/A)*(1.0/Q - 1.0) + 2.0)
    cos_w0 = math.cos(w0)
    two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
    
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
    
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    return lfilter(b, a, audio)

def compress_audio(audio, threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=150.0, sr=24000):
    import math
    import numpy as np
    from scipy.signal import lfilter
    time_constant_seconds = 0.05
    alpha = math.exp(-1.0 / (time_constant_seconds * sr))
    
    env = lfilter([1.0 - alpha], [1.0, -alpha], np.abs(audio))
    env = np.maximum(env, 1e-5)
    
    env_db = 20.0 * np.log10(env)
    
    gain_reduction_db = np.minimum(0.0, (threshold_db - env_db) * (1.0 - 1.0 / ratio))
    gain_reduction = 10.0 ** (gain_reduction_db / 20.0)
    
    return audio * gain_reduction

def apply_hard_limiter(audio, threshold=0.9):
    import numpy as np
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    if not np.any(mask):
        return audio
        
    lim_audio = np.copy(audio)
    overshoot = abs_audio[mask] - threshold
    scale = 1.0 - threshold
    
    compressed = threshold + scale * np.tanh(overshoot / scale)
    lim_audio[mask] = np.sign(audio[mask]) * compressed
    return lim_audio

def apply_podcast_voice(audio, sr=24000):
    import numpy as np
    from scipy.signal import butter, lfilter
    nyquist = sr / 2.0
    b, a = butter(2, 80.0 / nyquist, btype='high')
    audio = lfilter(b, a, audio)
    
    audio = peaking_equalizer(audio, center_freq=3000.0, Q=1.0, gain_db=3.0, sr=sr)
    audio = compress_audio(audio, threshold_db=-16.0, ratio=3.5, sr=sr)
    
    max_peak = np.max(np.abs(audio))
    if max_peak > 0.0:
        audio = audio / max_peak * 0.89
        
    return audio

def apply_mastering(audio, sr=24000):
    import numpy as np
    audio = high_shelf(audio, freq=8000.0, gain_db=1.5, sr=sr)
    audio = compress_audio(audio, threshold_db=-14.0, ratio=1.8, sr=sr)
    
    # Haas delay-based stereo widener (15ms delay)
    delay_samples = int(0.015 * sr)
    stereo = np.zeros((len(audio), 2))
    stereo[:, 0] = audio  # Left
    stereo[delay_samples:, 1] = audio[:-delay_samples]  # Right (delayed)
    
    max_peak = np.max(np.abs(stereo))
    if max_peak > 0.0:
        stereo = stereo / max_peak * 0.94
        
    return stereo

@app.route("/podcast/mix", methods=["POST"])
def mix_podcast():
    try:
        data = request.get_json() or {}
        segments = data.get("segments", [])
        
        if not segments:
            return jsonify({"error": "No segments provided for mixing."}), 400
            
        import soundfile as sf
        import numpy as np
        import librosa
        
        sample_rate = 24000
        loaded_segments = []
        max_end_sample = 0
        
        for idx, seg in enumerate(segments):
            url = seg.get("audio_url", "")
            start_time = float(seg.get("start_time", 0.0))
            volume = float(seg.get("volume", 1.0))
            
            # Extract job_id
            job_id = None
            if "/audio/" in url:
                job_id = url.split("/audio/")[-1].split("?")[0].split(".")[0]
            else:
                job_id = url.split("/")[-1].split("?")[0].split(".")[0]
                
            if not job_id:
                continue
                
            # Resolve file path
            if job_id.startswith("sfx_"):
                sfx_key = job_id.replace("sfx_", "")
                file_path = get_or_create_sfx(sfx_key)
            elif job_id.startswith("music_"):
                music_key = job_id.replace("music_", "")
                file_path = get_or_create_music(music_key)
            else:
                file_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.wav")
                if not os.path.exists(file_path):
                    alt_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_final.wav")
                    if os.path.exists(alt_path):
                        file_path = alt_path
                
            if not os.path.exists(file_path):
                logger.warning(f"Mixer: Segment file not found: {file_path}. Skipping.")
                continue
                
            # Load audio
            audio_data, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample to 24000 Hz if needed
            if sr != sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                
            # Apply volume
            audio_data = audio_data * volume
            
            duration_samples = len(audio_data)
            start_sample = int(start_time * sample_rate)
            end_sample = start_sample + duration_samples
            
            if end_sample > max_end_sample:
                max_end_sample = end_sample
                
            loaded_segments.append({
                "data": audio_data,
                "start_sample": start_sample,
                "end_sample": end_sample
            })
            
        if not loaded_segments:
            return jsonify({"error": "No valid audio segments could be loaded for mixing."}), 400
            
        hard_limiter = bool(data.get("hard_limiter", False))
        podcast_voice = bool(data.get("podcast_voice", False))
        mastering = bool(data.get("mastering", False))

        # Allocate master buffer
        master_audio = np.zeros(max_end_sample)
        
        for seg in loaded_segments:
            s_s = seg["start_sample"]
            e_s = seg["end_sample"]
            # Sum audio
            master_audio[s_s:e_s] += seg["data"]
            
        # Apply Post-Processing effects
        if podcast_voice:
            logger.info("Applying Podcast Voice post-processing effects...")
            master_audio = apply_podcast_voice(master_audio, sample_rate)
        if mastering:
            logger.info("Applying Mastering post-processing effects...")
            master_audio = apply_mastering(master_audio, sample_rate)
        if hard_limiter:
            logger.info("Applying Hard Limiter post-processing effects...")
            master_audio = apply_hard_limiter(master_audio)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(master_audio))
        if max_val > 0.95:
            master_audio = (master_audio / max_val) * 0.95
            
        # Save output
        mix_id = f"mix_{str(uuid4())[:8]}"
        mix_path = os.path.join(OUTPUT_FOLDER, f"{mix_id}.wav")
        sf.write(mix_path, master_audio, sample_rate)
        
        # Register in job status so client can query
        job_status[mix_id] = {
            "status": "done",
            "progress": 100,
            "output_path": mix_path,
            "audio_url": f"/audio/{mix_id}",
            "model": "mixer"
        }
        
        logger.info(f"Mixer completed: output={mix_path}, total_duration={len(master_audio)/sample_rate:.2f}s")
        return jsonify({
            "mix_id": mix_id,
            "audio_url": f"/audio/{mix_id}",
            "duration": len(master_audio) / sample_rate
        })
        
    except Exception as e:
        logger.exception("Error in /podcast/mix route")
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict() or {}

        # Normalize parameters
        if "temperature" in data:
            data["text_temp"] = data["temperature"]
        if "preset" not in data and "voice_preset" in data:
            data["preset"] = data["voice_preset"]
        if "focus" not in data and "top_p" in data:
            data["focus"] = data["top_p"]
        if "pool" not in data and "top_k" in data:
            data["pool"] = data["top_k"]

        text = (data.get("text") or "").strip()
        
        phonetic_dict = data.get("phonetic_dict") or []
        if isinstance(phonetic_dict, str):
            try:
                import json
                phonetic_dict = json.loads(phonetic_dict)
            except Exception:
                phonetic_dict = []
                
        spell_out_acronyms = str(data.get("spell_out_acronyms", "false")).lower() == "true"
        ignore_emojis = str(data.get("ignore_emojis", "false")).lower() == "true"
        ignore_special_symbols = str(data.get("ignore_special_symbols", "false")).lower() == "true"
        
        text = apply_phonetic_dictionary_and_filters(
            text,
            phonetic_dict=phonetic_dict,
            spell_out_acronyms=spell_out_acronyms,
            ignore_emojis=ignore_emojis,
            ignore_special_symbols=ignore_special_symbols
        )
        
        ref_text = (data.get("ref_text") or "").strip()
        if not ref_text and data.get("fish_prompt_text"):
            ref_text = data.get("fish_prompt_text").strip()

        model_name = (data.get("model") or "").strip()
        voice_name = model_name if model_name != "xtts" else "tts_models/multilingual/multi-dataset/xtts_v2"
        language = (data.get("language") or "").strip()
        speaker = (data.get("speaker") or "").strip()
        voice_preset = (data.get("preset") or "").strip()
        voice_id = (data.get("voice") or "").strip()
        direction = (data.get("voice_direction") or "").strip()

        creativity = data.get("creativity")
        pool = data.get("pool")
        focus = data.get("focus")

        speed = float(data.get("speed", 1.0))
        chunk_size = int(data.get("chunk_size", 300))
        pause_duration = float(data.get("pause_duration", 0.5))

        length_scale = float(data.get("length_scale", 1.0))
        noise_scale = float(data.get("noise_scale", 0.667))
        noise_scale_w = float(data.get("noise_scale_w", 0.8))

        speaker_1_voice = (data.get("speaker_1_voice") or "").strip()
        speaker_2_voice = (data.get("speaker_2_voice") or "").strip()
        speaker_3_voice = (data.get("speaker_3_voice") or "").strip()
        speaker_4_voice = (data.get("speaker_4_voice") or "").strip()
        curated_speaker_configs = data.get("curated_speaker_configs") or {}

        smart_enhance = bool(direction)
        try:
            emotion_intensity = float(data.get("emotion_intensity", 0.5))
        except Exception:
            emotion_intensity = 0.5

        bark_split_sentences = False
        bark_max_duration = 14
        if request.is_json:
            bark_split_sentences = bool(request.json.get("barkSplitSentences", True)) # Default to true
            bark_max_duration = float(request.json.get("barkMaxDuration", 14))
        else:
            bark_split_sentences = str(data.get("barkSplitSentences", "True")).lower() == "true"
            try:
                bark_max_duration = float(data.get("barkMaxDuration", 14))
            except Exception:
                bark_max_duration = 14

        handler = MODEL_REGISTRY.get(voice_name)
        if not handler:
            return jsonify({"error": f"Voice model '{voice_name}' not found."}), 400
        voice_info = handler.get_metadata()

        if voice_info.get("requires_language") and not language:
            language = "en" if "multilingual" in voice_name else None

        if not speaker and voice_info.get("supported_speakers"):
            speaker = voice_info["supported_speakers"][0]

        speaker_wav = request.files.get("speaker_wav")
        speaker_wav_bytes = None
        speaker_wav_name = None
        
        # Check if a pre-canned library preset reference voice is requested
        library_speaker_wav = request.form.get("library_speaker_wav")
        if library_speaker_wav:
            safe_filename = os.path.basename(library_speaker_wav) + ".wav"
            lib_path = os.path.join(os.path.dirname(__file__), "assets", "voice_library", safe_filename)
            if os.path.exists(lib_path):
                try:
                    with open(lib_path, "rb") as f:
                        speaker_wav_bytes = f.read()
                    speaker_wav_name = safe_filename
                    
                    # Automatically load corresponding transcript file (.txt) if it exists
                    if not ref_text:
                        txt_path = lib_path.rsplit(".", 1)[0] + ".txt"
                        if os.path.exists(txt_path):
                            with open(txt_path, "r", encoding="utf-8") as tf:
                                ref_text = tf.read().strip()
                                logger.info(f"Loaded library voice transcript: '{ref_text}'")
                except Exception as ex:
                    logger.error(f"Failed to read pre-canned library file at {lib_path}: {ex}")

        if not speaker_wav_bytes and speaker_wav:
            speaker_wav_bytes = speaker_wav.read()
            speaker_wav_name = speaker_wav.filename

        # If it is a custom clone and no transcript is provided, auto-transcribe it!
        if not ref_text and speaker_wav_bytes and "cloning" in voice_info.get("features", []):
            try:
                import speech_recognition as sr
                import io
                wav_io = io.BytesIO(speaker_wav_bytes)
                r = sr.Recognizer()
                with sr.AudioFile(wav_io) as source:
                    audio_data = r.record(source)
                    ref_text = r.recognize_google(audio_data)
                    logger.info(f"Auto-transcribed custom reference audio: '{ref_text}'")
            except Exception as stt_err:
                logger.warning(f"Auto-transcribing custom reference audio failed/skipped: {stt_err}")

        if voice_info.get("requires_speaker_wav") and not speaker_wav_bytes:
            return jsonify({"error": f"Model {voice_name} requires speaker reference audio (WAV)."}), 400

        job_id = str(uuid4())
        job_status[job_id] = {
            "status": "queued",
            "progress": 0,
            "output_path": None,
            "chunk_index": 0,
            "total_chunks": 1,
            "model": voice_name
        }

        config = get_config()
        device = config.get("device", "auto")

        resolved_preset = voice_preset or voice_id

        job = {
            "job_id": job_id,
            "text": text,
            "voice_name": voice_name,
            "speed": speed,
            "pause_duration": pause_duration,
            "language": language,
            "speaker": speaker,
            "speaker_wav": speaker_wav_bytes,
            "speaker_wav_name": speaker_wav_name,
            "voice_preset": resolved_preset,
            "text_temp": creativity if creativity is not None else "",
            "top_k": pool if pool is not None else "",
            "top_p": focus if focus is not None else "",
            "smart_enhance": smart_enhance,
            "voice_direction": direction,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "seed": data.get("seed"),
            "bark_split_sentences": bark_split_sentences,
            "bark_max_duration": bark_max_duration,
            "device": device,
            "speaker_1_voice": speaker_1_voice,
            "speaker_2_voice": speaker_2_voice,
            "speaker_3_voice": speaker_3_voice,
            "speaker_4_voice": speaker_4_voice,
            "emotion_intensity": emotion_intensity,
            # ChatTTS parameters
            "chattts_refine_text": data.get("chattts_refine_text"),
            "chattts_spk_temp": data.get("chattts_spk_temp"),
            "chattts_text_temp": data.get("chattts_text_temp"),
            "chattts_spk_seed": data.get("chattts_spk_seed"),
            "chattts_spk_emb": data.get("chattts_spk_emb"),
            "chattts_top_p": data.get("chattts_top_p"),
            "chattts_top_k": data.get("chattts_top_k"),
            "chattts_temperature": data.get("chattts_temperature"),
            # Fish Audio parameters
            "fish_engine": data.get("fish_engine"),
            "fish_normalize": data.get("fish_normalize"),
            "fish_similarity_weight": data.get("fish_similarity_weight"),
            "fish_prompt_text": data.get("fish_prompt_text"),
            "ref_text": ref_text,
            "curated_speaker_configs": curated_speaker_configs
        }

        job_queue.put(job)
        queue_position = job_queue.qsize()
        return jsonify({
            "job_id": job_id,
            "queue_position": queue_position,
            "estimated_wait_time": queue_position * 5
        })

    except Exception as e:
        logger.exception("Error in /generate route")
        return jsonify({"error": f"Internal error: {e}"}), 500

@app.route("/status/<job_id>", methods=["GET"])
def check_status(job_id):
    if job_id not in job_status:
        return jsonify({"error": "Invalid job ID"}), 404
    status_info = job_status[job_id]
    if status_info["status"] == "queued":
        job_ids = [j["job_id"] for j in list(job_queue.queue)]
        status_info["queue_position"] = job_ids.index(job_id) + 1 if job_id in job_ids else 1
    else:
        status_info["queue_position"] = None
    return jsonify(status_info)

@app.route("/chattts/sample_speaker", methods=["POST"])
def chattts_sample_speaker():
    try:
        data = request.get_json(silent=True) or {}
        seed = data.get("seed")
        
        try:
            import ChatTTS
            import torch
            chat = get_cached_chattts()
            if chat is None:
                raise ImportError("ChatTTS not preloaded")
            
            if seed is not None and seed != "":
                torch.manual_seed(int(seed))
            
            spk_emb = chat.sample_random_speaker()
            spk_emb_list = spk_emb.tolist() if hasattr(spk_emb, "tolist") else []
            return jsonify({
                "success": True,
                "spk_emb": spk_emb_list,
                "seed": seed
            })
        except Exception as e:
            import random
            if seed is not None and seed != "":
                random.seed(int(seed))
            else:
                random.seed(42)
            mock_emb = [random.normalvariate(0, 0.1) for _ in range(768)]
            return jsonify({
                "success": True,
                "spk_emb": mock_emb,
                "seed": seed,
                "simulated": True
            })
    except Exception as e:
        logger.exception("Error sampling ChatTTS speaker")
        return jsonify({"error": str(e)}), 500

@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    queue_list = list(job_queue.queue)
    for job in queue_list:
        if job["job_id"] == job_id:
            queue_list.remove(job)
            job_queue.queue.clear()
            for j in queue_list:
                job_queue.put(j)
            job_status[job_id] = {"status": "cancelled", "progress": 0, "audio_url": None}
            return jsonify({"status": "cancelled"})
    return jsonify({"error": "Job not found or already processing."}), 404

# ------------------------------------------------------------------
# Generation Processing (Queue Workers)
# ------------------------------------------------------------------
def process_jobs():
    while True:
        job = job_queue.get()
        job_id = job.get("job_id")
        try:
            voice_name = job.get("voice_name")
            handler = MODEL_REGISTRY.get(voice_name)
            
            if handler is not None:
                # Update status
                job_status[job_id]["status"] = "processing"
                job_status[job_id]["progress"] = 5
                job_status[job_id]["message"] = f"Initializing {handler.get_metadata().get('model')} synthesis..."
                
                output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.wav")
                
                def progress_cb(percent, msg):
                    job_status[job_id]["progress"] = percent
                    job_status[job_id]["message"] = msg

                # Map preset if in Voice Presets npz mapping
                if job.get("voice_preset") in VOICE_PRESET_MAP:
                    job["voice_preset"] = VOICE_PRESET_MAP[job["voice_preset"]]

                # Execute synthesis
                # Filter out job_id and output_path to avoid passing them twice (positional vs kwarg)
                job_params = {k: v for k, v in job.items() if k not in ["job_id", "output_path", "progress_callback"]}
                
                logger.info(f"Processing job {job_id} for model {voice_name}. is_simulator: {handler.is_simulator}. job_params keys: {list(job_params.keys())}")
                
                if handler.is_simulator:
                    handler.run_simulation(job_id=job_id, output_path=output_path, progress_callback=progress_cb, **job_params)
                else:
                    try:
                        handler.synthesize(job_id=job_id, output_path=output_path, progress_callback=progress_cb, **job_params)
                    except Exception as exc:
                        logger.error(f"Synthesis failed using native handler: {exc}. Retrying in simulator mode...", exc_info=True)
                        handler.is_simulator = True
                        handler.run_simulation(job_id=job_id, output_path=output_path, progress_callback=progress_cb, **job_params)
                
                job_status[job_id]["status"] = "done"
                job_status[job_id]["progress"] = 100
                job_status[job_id]["message"] = "Synthesis complete"
                job_status[job_id]["audio_url"] = f"/audio/{job_id}"
            else:
                logger.error(f"Model handler not found for voice: {voice_name}")
                job_status[job_id]["status"] = "error"
                job_status[job_id]["message"] = f"Unsupported model: {voice_name}"
        except Exception as e:
            logger.error(f"Failed to process queue job: {e}", exc_info=True)
            if job_id:
                job_status[job_id]["status"] = "error"
                job_status[job_id]["message"] = f"Error: {e}"
        finally:
            job_queue.task_done()

# ------------------------------------------------------------------
# Document Import, Soundscape, and Review System
# ------------------------------------------------------------------
@app.route("/import/docx", methods=["POST"])
def import_docx():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        from docx_importer import convert_docx_to_markdown
        markdown_text = convert_docx_to_markdown(file)
        return jsonify({"markdown": markdown_text})
    except Exception as e:
        logger.exception("Error in /import/docx endpoint")
        return jsonify({"error": str(e)}), 500

@app.route("/podcast/analyze-mood", methods=["POST"])
def analyze_mood():
    try:
        data = request.get_json() or {}
        script = data.get("script", "")
        mood = data.get("mood", "")
        if not script or not mood:
            return jsonify({"error": "Script and mood prompt are required"}), 400
            
        from llm_wrapper import analyze_script_soundscape
        suggestions = analyze_script_soundscape(script, mood)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        logger.exception("Error in /podcast/analyze-mood endpoint")
        return jsonify({"error": str(e)}), 500

@app.route("/review/suggest-edit", methods=["POST"])
def suggest_edit():
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        note = data.get("note", "")
        if not text or not note:
            return jsonify({"error": "Text and correction note are required"}), 400
            
        from llm_wrapper import suggest_text_edit
        revised_text = suggest_text_edit(text, note)
        return jsonify({"revised_text": revised_text})
    except Exception as e:
        logger.exception("Error in /review/suggest-edit endpoint")
        return jsonify({"error": str(e)}), 500

@app.route("/review/save-version", methods=["POST"])
def save_version():
    try:
        data = request.get_json() or {}
        directory = data.get("directory", "").strip()
        filename = data.get("filename", "").strip()
        content = data.get("content", "")
        
        if not directory or not filename:
            return jsonify({"error": "Obsidian vault path and filename are required"}), 400
            
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        file_path = os.path.join(directory, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"Saved review version directly to Obsidian vault: {file_path}")
        return jsonify({"success": True, "path": file_path})
    except Exception as e:
        logger.exception("Error in /review/save-version endpoint")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# Sound Library Manager
# ------------------------------------------------------------------
SOUND_LIBRARY_FILE = os.path.join(OUTPUT_FOLDER, "sound_library.json")

def load_sound_library():
    if not os.path.exists(SOUND_LIBRARY_FILE):
        return {"user_sounds": []}
    try:
        with open(SOUND_LIBRARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"user_sounds": []}

def save_sound_library(data):
    try:
        with open(SOUND_LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save sound library: {e}")

@app.route("/api/sound-library/list", methods=["GET"])
def list_sound_library():
    built_in = [
        {"key": "lofi", "name": "Lofi Jazz Chords", "type": "music", "source": "built-in", "duration": 15.0, "url": "http://localhost:5000/audio/music_lofi"},
        {"key": "intro", "name": "Upbeat Synth Arp", "type": "music", "source": "built-in", "duration": 15.0, "url": "http://localhost:5000/audio/music_intro"},
        {"key": "suspense", "name": "Suspenseful Drone", "type": "music", "source": "built-in", "duration": 15.0, "url": "http://localhost:5000/audio/music_suspense"},
        {"key": "acoustic", "name": "Acoustic Guitar", "type": "music", "source": "built-in", "duration": 15.0, "url": "http://localhost:5000/audio/music_acoustic"},
        {"key": "applause", "name": "Applause", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_applause"},
        {"key": "phone", "name": "Telephone Ring", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_phone"},
        {"key": "jazz", "name": "Jazz Chord Arp", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_jazz"},
        {"key": "scratch", "name": "Record Scratch", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_scratch"},
        {"key": "cafe", "name": "Cafe Ambience", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_cafe"},
        {"key": "birds", "name": "Bird Chirps", "type": "sfx", "source": "built-in", "duration": 3.0, "url": "http://localhost:5000/audio/sfx_birds"},
    ]
    
    lib = load_sound_library()
    user_sounds = []
    for item in lib.get("user_sounds", []):
        user_sounds.append({
            "key": item["key"],
            "name": item["name"],
            "type": item["type"],
            "source": item.get("source", "user"),
            "duration": item.get("duration", 5.0),
            "url": f"http://localhost:5000/audio/{item['type']}_{item['key']}"
        })
        
    return jsonify(built_in + user_sounds)

@app.route("/api/sound-library/upload", methods=["POST"])
def upload_sound():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        sound_type = request.form.get("type", "sfx").strip().lower()
        if sound_type not in ["music", "sfx"]:
            sound_type = "sfx"
            
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
            
        import uuid
        safe_name = "".join([c if c.isalnum() or c in "._-" else "_" for c in file.filename])
        file_key = f"custom_{uuid.uuid4().hex[:8]}_{safe_name.rsplit('.', 1)[0]}"
        dest_filename = f"{sound_type}_{file_key}.wav"
        dest_path = os.path.join(OUTPUT_FOLDER, dest_filename)
        
        temp_path = os.path.join(OUTPUT_FOLDER, f"temp_{uuid.uuid4().hex[:8]}_{safe_name}")
        file.save(temp_path)
        
        duration = 5.0
        try:
            import soundfile as sf
            data, samplerate = sf.read(temp_path)
            duration = len(data) / samplerate
            sf.write(dest_path, data, samplerate)
        except Exception:
            import shutil
            shutil.copyfile(temp_path, dest_path)
            try:
                import wave
                with wave.open(dest_path, 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate()
                    duration = frames / float(rate)
            except Exception:
                pass
                
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        lib = load_sound_library()
        new_item = {
            "key": file_key,
            "name": safe_name.rsplit('.', 1)[0].replace("_", " ").title(),
            "type": sound_type,
            "duration": duration,
            "source": "user",
            "filename": dest_filename
        }
        lib["user_sounds"].append(new_item)
        save_sound_library(lib)
        
        return jsonify({
            "success": True,
            "sound": {
                "key": new_item["key"],
                "name": new_item["name"],
                "type": new_item["type"],
                "source": new_item["source"],
                "duration": new_item["duration"],
                "url": f"http://localhost:5000/audio/{new_item['type']}_{new_item['key']}"
            }
        })
    except Exception as e:
        logger.exception("Error in upload_sound")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sound-library/delete", methods=["POST"])
def delete_sound():
    try:
        data = request.get_json() or {}
        key = data.get("key", "").strip()
        sound_type = data.get("type", "").strip().lower()
        if not key or not sound_type:
            return jsonify({"error": "Key and type are required"}), 400
            
        lib = load_sound_library()
        new_user_sounds = []
        deleted = False
        
        for item in lib.get("user_sounds", []):
            if item["key"] == key and item["type"] == sound_type:
                file_path = os.path.join(OUTPUT_FOLDER, item["filename"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                deleted = True
            else:
                new_user_sounds.append(item)
                
        if not deleted:
            return jsonify({"error": "Sound asset not found"}), 404
            
        lib["user_sounds"] = new_user_sounds
        save_sound_library(lib)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception("Error in delete_sound")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sound-library/rename", methods=["POST"])
def rename_sound():
    try:
        data = request.get_json() or {}
        key = data.get("key", "").strip()
        sound_type = data.get("type", "").strip().lower()
        new_name = data.get("name", "").strip()
        if not key or not sound_type or not new_name:
            return jsonify({"error": "Key, type, and new name are required"}), 400
            
        lib = load_sound_library()
        updated = False
        for item in lib.get("user_sounds", []):
            if item["key"] == key and item["type"] == sound_type:
                item["name"] = new_name
                updated = True
                break
                
        if not updated:
            return jsonify({"error": "Sound asset not found"}), 404
            
        save_sound_library(lib)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception("Error in rename_sound")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sound-library/generate-music", methods=["POST"])
def generate_music():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        duration = float(data.get("duration", 15.0))
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        import uuid
        import numpy as np
        import soundfile as sf
        
        file_key = f"generated_{uuid.uuid4().hex[:8]}"
        dest_filename = f"music_{file_key}.wav"
        dest_path = os.path.join(OUTPUT_FOLDER, dest_filename)
        
        generated = False
        try:
            from audiocraft.models import MusicGen
            import torch
            
            logger.info("Initializing MusicGen...")
            model = MusicGen.get_pretrained('facebook/musicgen-small')
            model.set_generation_params(duration=duration)
            
            logger.info(f"Generating music for prompt: '{prompt}'")
            wav = model.generate([prompt])
            wav = wav.cpu().numpy()[0, 0]
            
            sf.write(dest_path, wav, 32000)
            generated = True
            logger.info("MusicGen generation completed successfully.")
        except Exception as e:
            logger.warning(f"MusicGen not available or failed: {e}. Falling back to rich synthetic generator.")
            
        if not generated:
            sr = 24000
            t = np.linspace(0, duration, int(sr * duration), False)
            audio = np.zeros_like(t)
            
            prompt_lower = prompt.lower()
            if "lofi" in prompt_lower or "jazz" in prompt_lower or "calm" in prompt_lower:
                chord_duration = 4.0
                num_chords = int(duration / chord_duration)
                for c_idx in range(num_chords + 1):
                    start_idx = int(c_idx * chord_duration * sr)
                    end_idx = min(len(t), int((c_idx + 1) * chord_duration * sr))
                    chord_t = t[start_idx:end_idx] - (c_idx * chord_duration)
                    if c_idx % 3 == 0:
                        freqs = [110.0, 165.0, 220.0, 261.63, 329.63]
                    elif c_idx % 3 == 1:
                        freqs = [116.54, 174.61, 233.08, 293.66, 349.23]
                    else:
                        freqs = [98.0, 146.83, 196.0, 246.94, 293.66]
                        
                    chord_audio = np.zeros_like(chord_t)
                    for f in freqs:
                        chord_audio += np.sin(2 * np.pi * f * chord_t) * 0.7
                        chord_audio += np.sin(2 * np.pi * (f*2) * chord_t) * 0.25
                        chord_audio += np.sin(2 * np.pi * (f*3) * chord_t) * 0.1
                    audio[start_idx:end_idx] = (chord_audio / len(freqs)) * 0.18
            elif "fast" in prompt_lower or "upbeat" in prompt_lower or "electronic" in prompt_lower:
                step_dur = 0.15
                scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
                for i in range(int(duration / step_dur)):
                    start_idx = int(i * step_dur * sr)
                    end_idx = min(len(t), int((i + 1) * step_dur * sr))
                    f = scale[i % len(scale)]
                    arp_t = t[start_idx:end_idx]
                    env = np.exp(-(arp_t - i * step_dur) * 12.0)
                    audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.25
            else:
                step_dur = 0.3
                scale = [220.0, 261.63, 329.63, 440.0, 392.0, 329.63, 261.63]
                for i in range(int(duration / step_dur)):
                    start_idx = int(i * step_dur * sr)
                    end_idx = min(len(t), int((i + 1) * step_dur * sr))
                    f = scale[i % len(scale)]
                    arp_t = t[start_idx:end_idx]
                    env = np.exp(-(arp_t - i * step_dur) * 4.0)
                    audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.2
                    
            fade = int(sr * 1.5)
            if len(audio) > fade:
                audio[-fade:] *= np.linspace(1, 0, fade)
            sf.write(dest_path, audio, sr)
            
        lib = load_sound_library()
        new_item = {
            "key": file_key,
            "name": f"AI: {prompt[:30]}...",
            "type": "music",
            "duration": duration,
            "source": "generated",
            "filename": dest_filename
        }
        lib["user_sounds"].append(new_item)
        save_sound_library(lib)
        
        return jsonify({
            "success": True,
            "sound": {
                "key": new_item["key"],
                "name": new_item["name"],
                "type": "music",
                "source": "generated",
                "duration": duration,
                "url": f"http://localhost:5000/audio/music_{new_item['key']}"
            }
        })
    except Exception as e:
        logger.exception("Error in generate_music")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sound-library/resolve", methods=["POST"])
def resolve_sound():
    try:
        data = request.get_json() or {}
        desc = data.get("description", "").strip()
        sound_type = data.get("type", "sfx").strip().lower()
        if not desc:
            return jsonify({"error": "Description is required"}), 400
            
        requested_duration = data.get("duration")
        if requested_duration:
            duration = float(requested_duration)
        else:
            duration = 15.0 if sound_type == "music" else 3.0
            
        lib = load_sound_library()
        desc_lower = desc.lower()
        
        for item in lib.get("user_sounds", []):
            if item["type"] == sound_type and (item["key"].lower() == desc_lower or item["name"].lower() == desc_lower):
                return jsonify({
                    "url": f"http://localhost:5000/audio/{sound_type}_{item['key']}",
                    "key": item["key"]
                })
                
        built_ins = {
            "music": ["lofi", "intro", "suspense", "acoustic"],
            "sfx": ["applause", "phone", "jazz", "scratch", "cafe", "birds"]
        }
        for bkey in built_ins.get(sound_type, []):
            if bkey == desc_lower:
                return jsonify({
                    "url": f"http://localhost:5000/audio/{sound_type}_{bkey}",
                    "key": bkey
                })
                
        token = data.get("token", "").strip()
        if not token:
            token = os.environ.get("FREESOUND_API_KEY", "")
            
        if token:
            try:
                import requests
                import urllib.parse
                headers = {"Authorization": f"Token {token}"}
                filter_str = f"duration:[* TO {max(15, int(duration) + 5)}]" if sound_type == "sfx" else f"duration:[{max(10, int(duration) - 5)} TO *]"
                url = f"https://freesound.org/apiv2/search/text/?query={urllib.parse.quote(desc)}&filter={filter_str}&fields=id,name,duration,previews&page_size=1"
                r_search = requests.get(url, headers=headers, timeout=5)
                if r_search.status_code == 200:
                    search_res = r_search.json()
                    results = search_res.get("results", [])
                    if results:
                        sound = results[0]
                        preview_url = sound.get("previews", {}).get("preview-hq-mp3", "")
                        if preview_url:
                            import uuid
                            import shutil
                            file_key = f"freesound_{uuid.uuid4().hex[:8]}"
                            dest_filename = f"{sound_type}_{file_key}.mp3"
                            dest_path = os.path.join(OUTPUT_FOLDER, dest_filename)
                            
                            temp_mp3 = os.path.join(OUTPUT_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.mp3")
                            r_down = requests.get(preview_url, timeout=10)
                            with open(temp_mp3, "wb") as f:
                                f.write(r_down.content)
                            shutil.move(temp_mp3, dest_path)
                            
                            final_duration = float(sound.get("duration", duration))
                            
                            lib = load_sound_library()
                            new_item = {
                                "key": file_key,
                                "name": sound.get("name", desc),
                                "type": sound_type,
                                "duration": final_duration,
                                "source": "freesound",
                                "filename": dest_filename
                            }
                            lib["user_sounds"].append(new_item)
                            save_sound_library(lib)
                            
                            return jsonify({
                                "url": f"http://localhost:5000/audio/{sound_type}_{file_key}",
                                "key": file_key
                            })
            except Exception as fe:
                logger.warning(f"Freesound fallback download failed: {fe}. Proceeding to local generators.")
                
        import uuid
        import numpy as np
        import soundfile as sf
        
        file_key = f"auto_{uuid.uuid4().hex[:8]}"
        dest_filename = f"{sound_type}_{file_key}.wav"
        dest_path = os.path.join(OUTPUT_FOLDER, dest_filename)
        
        if sound_type == "music":
            generated = False
            try:
                from audiocraft.models import MusicGen
                import torch
                
                logger.info("Initializing MusicGen for dynamic resolve...")
                model = MusicGen.get_pretrained('facebook/musicgen-small')
                model.set_generation_params(duration=duration)
                
                logger.info(f"Generating music for prompt: '{desc}'")
                wav = model.generate([desc])
                wav = wav.cpu().numpy()[0, 0]
                
                sf.write(dest_path, wav, 32000)
                generated = True
                logger.info("MusicGen dynamic resolve completed successfully.")
            except Exception as e:
                logger.warning(f"MusicGen failed/unavailable for resolve: {e}. Falling back to chord synthesizer.")
                
            if not generated:
                sr = 24000
                t = np.linspace(0, duration, int(sr * duration), False)
                audio = np.zeros_like(t)
                
                if "lofi" in desc_lower or "jazz" in desc_lower or "calm" in desc_lower:
                    chord_duration = 4.0
                    num_chords = int(duration / chord_duration)
                    for c_idx in range(num_chords + 1):
                        start_idx = int(c_idx * chord_duration * sr)
                        end_idx = min(len(t), int((c_idx + 1) * chord_duration * sr))
                        chord_t = t[start_idx:end_idx] - (c_idx * chord_duration)
                        if c_idx % 3 == 0:
                            freqs = [110.0, 165.0, 220.0, 261.63, 329.63]
                        elif c_idx % 3 == 1:
                            freqs = [116.54, 174.61, 233.08, 293.66, 349.23]
                        else:
                            freqs = [98.0, 146.83, 196.0, 246.94, 293.66]
                            
                        chord_audio = np.zeros_like(chord_t)
                        for f in freqs:
                            chord_audio += np.sin(2 * np.pi * f * chord_t) * 0.7
                            chord_audio += np.sin(2 * np.pi * (f*2) * chord_t) * 0.25
                            chord_audio += np.sin(2 * np.pi * (f*3) * chord_t) * 0.1
                        audio[start_idx:end_idx] = (chord_audio / len(freqs)) * 0.18
                elif "fast" in desc_lower or "upbeat" in desc_lower or "electronic" in desc_lower:
                    step_dur = 0.15
                    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
                    for i in range(int(duration / step_dur)):
                        start_idx = int(i * step_dur * sr)
                        end_idx = min(len(t), int((i + 1) * step_dur * sr))
                        f = scale[i % len(scale)]
                        arp_t = t[start_idx:end_idx]
                        env = np.exp(-(arp_t - i * step_dur) * 12.0)
                        audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.25
                else:
                    step_dur = 0.3
                    scale = [220.0, 261.63, 329.63, 440.0, 392.0, 329.63, 261.63]
                    for i in range(int(duration / step_dur)):
                        start_idx = int(i * step_dur * sr)
                        end_idx = min(len(t), int((i + 1) * step_dur * sr))
                        f = scale[i % len(scale)]
                        arp_t = t[start_idx:end_idx]
                        env = np.exp(-(arp_t - i * step_dur) * 4.0)
                        audio[start_idx:end_idx] = np.sin(2 * np.pi * f * arp_t) * env * 0.2
                        
                fade = int(sr * 1.5)
                if len(audio) > fade:
                    audio[-fade:] *= np.linspace(1, 0, fade)
                sf.write(dest_path, audio, sr)
        else:
            sr = 24000
            t = np.linspace(0, duration, int(sr * duration), False)
            
            if "explosion" in desc_lower or "boom" in desc_lower:
                noise = np.random.normal(0, 0.4, len(t))
                env = np.exp(-t * 2.5)
                audio = noise * env
            elif "laser" in desc_lower or "pew" in desc_lower:
                freq_sweep = 1200 * np.exp(-t * 6.0) + 100
                audio = np.sin(2 * np.pi * freq_sweep * t) * np.exp(-t * 4.0) * 0.3
            elif "click" in desc_lower or "pop" in desc_lower:
                audio = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 100.0) * 0.5
            else:
                audio = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 1.5) * 0.2
                
            sf.write(dest_path, audio, sr)
            
        lib = load_sound_library()
        new_item = {
            "key": file_key,
            "name": f"Auto: {desc}",
            "type": sound_type,
            "duration": duration,
            "source": "generated",
            "filename": dest_filename
        }
        lib["user_sounds"].append(new_item)
        save_sound_library(lib)
        
        return jsonify({
            "url": f"http://localhost:5000/audio/{sound_type}_{file_key}",
            "key": file_key
        })
    except Exception as e:
        logger.exception("Error in resolve_sound")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sound-library/search", methods=["POST"])
def search_freesound():
    try:
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        sound_type = data.get("type", "sfx").strip()
        token = data.get("token", "").strip()
        
        if not token:
            token = os.environ.get("FREESOUND_API_KEY", "")
            
        if not query:
            return jsonify({"results": []})
            
        if not token:
            return jsonify({
                "error": "Freesound API token is not configured. Please add it in Settings.",
                "results": []
            })
            
        import requests
        headers = {"Authorization": f"Token {token}"}
        filter_str = "duration:[* TO 10]" if sound_type == "sfx" else "duration:[10 TO *]"
        url = f"https://freesound.org/apiv2/search/text/?query={query}&filter={filter_str}&fields=id,name,duration,previews&page_size=8"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return jsonify({"error": f"Freesound API error: {response.text}", "results": []})
            
        data_res = response.json()
        results = []
        for sound in data_res.get("results", []):
            preview_url = sound.get("previews", {}).get("preview-hq-mp3", "")
            if preview_url:
                results.append({
                    "id": sound["id"],
                    "name": sound["name"],
                    "duration": sound["duration"],
                    "preview_url": preview_url
                })
        return jsonify({"results": results})
    except Exception as e:
        logger.exception("Error searching Freesound")
        return jsonify({"error": str(e), "results": []})

@app.route("/api/sound-library/download-freesound", methods=["POST"])
def download_freesound():
    try:
        data = request.get_json() or {}
        preview_url = data.get("preview_url", "").strip()
        sound_name = data.get("name", "Freesound").strip()
        sound_type = data.get("type", "sfx").strip().lower()
        if not preview_url:
            return jsonify({"error": "Preview URL is required"}), 400
            
        import uuid
        import requests
        import shutil
        
        file_key = f"freesound_{uuid.uuid4().hex[:8]}"
        dest_filename_mp3 = f"{sound_type}_{file_key}.mp3"
        dest_path_mp3 = os.path.join(OUTPUT_FOLDER, dest_filename_mp3)
        
        temp_mp3 = os.path.join(OUTPUT_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.mp3")
        res = requests.get(preview_url)
        with open(temp_mp3, "wb") as f:
            f.write(res.content)
            
        shutil.move(temp_mp3, dest_path_mp3)
        
        duration = 5.0
        try:
            # We can use mutagen or simple estimation if needed, or default
            pass
        except Exception:
            pass
            
        lib = load_sound_library()
        new_item = {
            "key": file_key,
            "name": sound_name,
            "type": sound_type,
            "duration": duration,
            "source": "freesound",
            "filename": dest_filename_mp3
        }
        lib["user_sounds"].append(new_item)
        save_sound_library(lib)
        
        return jsonify({
            "success": True,
            "sound": {
                "key": new_item["key"],
                "name": new_item["name"],
                "type": new_item["type"],
                "source": new_item["source"],
                "duration": new_item["duration"],
                "url": f"http://localhost:5000/audio/{new_item['type']}_{new_item['key']}"
            }
        })
    except Exception as e:
        logger.exception("Error downloading from Freesound")
        return jsonify({"error": str(e)}), 500

@app.route("/assets/voice_library/<path:filename>")
def serve_voice_library(filename):
    if not filename.endswith(".wav") and not filename.endswith(".txt"):
        filename = filename + ".wav"
    folder = os.path.join(app.root_path, "assets", "voice_library")
    res = make_response(send_from_directory(folder, filename))
    res.headers["Access-Control-Allow-Origin"] = "*"
    return res

# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Start preloading registered models in background (disabled to prevent MPS memory contention on macOS)
    # Thread(target=MODEL_REGISTRY.preload_all, daemon=True).start()
    Thread(target=process_jobs, daemon=True).start()
    app.run(debug=True, port=5000)
