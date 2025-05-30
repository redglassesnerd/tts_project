import os
saved_voices = {}
from pydub import AudioSegment
import numpy as np
import logging
import json
preset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "voice_presets.json"))
with open(preset_path, "r") as f:
    VOICE_PRESET_MAP = json.load(f)
print(f"✅ Loaded VOICE_PRESET_MAP from {preset_path}: {list(VOICE_PRESET_MAP.keys())}")
CHUNKING_WORD_THRESHOLD = 120
MAX_ATTEMPTS = 3
from nltk.tokenize import sent_tokenize

def chunk_text_by_sentence(text, word_threshold=CHUNKING_WORD_THRESHOLD):
    """
    Splits input text into chunks by sentences, keeping each chunk under the word_threshold.
    This respects sentence boundaries and preserves bracketed token syntax.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        words = sentence.split()
        current_length = sum(len(s.split()) for s in current_chunk)
        if current_length + len(words) <= word_threshold:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def crossfade_audio(a, b, duration=0.2, sr=24000):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.size == 0 or b.size == 0:
        logging.warning("[WARNING] Skipping crossfade: one of the audio chunks is empty or not a valid NumPy array.")
        return a if isinstance(a, np.ndarray) and a.size > 0 else b
    duration = min(duration, len(a) / sr, len(b) / sr)
    crossfade_samples = int(duration * sr)
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    a[-crossfade_samples:] *= fade_out
    b[:crossfade_samples] *= fade_in
    return np.concatenate([a[:-crossfade_samples], a[-crossfade_samples:] + b[:crossfade_samples], b[crossfade_samples:]])
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import soundfile as sf
import numpy as np
import re
import spacy
import unicodedata
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from uuid import uuid4
from threading import Thread
from queue import Queue
from llm_wrapper import enhance_text

 # ────────────────────────────────────────────────────────────────
# Optional lightweight, fully‑offline LLM for auto punctuation /
# pause insertion.  The model file (e.g. `ggml‑tiny.en.bin`) can be
# placed in `backend/llama.cpp/models/` or a custom path supplied
# via the `LLAMA_MODEL_PATH` env var.
try:
    from llama_cpp import Llama
    LLAMA_MODEL_PATH = os.getenv(
        "LLAMA_MODEL_PATH",
        os.path.join(os.path.dirname(__file__),
                     "llama.cpp", "models", "ggml‑tiny.en.bin")
    )
    llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048) if os.path.exists(LLAMA_MODEL_PATH) else None
except Exception:
    llm = None  # Llama‑cpp not installed or model file missing
# ────────────────────────────────────────────────────────────────
try:
    from bark import generate_audio, SAMPLE_RATE
    from bark.generation import preload_models
    from scipy.io.wavfile import write as write_wav
except ImportError:
    pass  # Bark support will be conditionally enabled

job_queue = Queue()
job_status = {}  # job_id: {status, progress, output_path}

nlp = spacy.load("en_core_web_sm")

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

 

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
#
# Lightweight endpoint for LLM-based text preprocessing (“Review” step)
# This endpoint enhances or sanitizes text for TTS input.
@app.route("/preprocess", methods=["POST"])
def preprocess():
    data          = request.get_json() or {}
    raw_text      = data.get("text", "")
    voice_dir     = data.get("voice_direction", "")
    enhanced_text = enhance_text_with_llm(raw_text) if voice_dir.strip() else sanitize_text(raw_text)
    return jsonify({"text": enhanced_text})

# ------------------------------------------------------------------
# Human‑readable blurbs to help users pick a voice in the UI
VOICE_DESCRIPTIONS = {
    "tts_models/multilingual/multi-dataset/xtts_v2":
        "✧ XTTS‑v2  ·  Multilingual, cross‑speaker cloning model.  Best when you provide a short WAV of your own voice, but also works with its default timbre.",
    "tts_models/en/vctk/vits":
        "✧ VITS / VCTK  ·  Fast English model trained on the VCTK corpus.  Neutral Midlands accent, very CPU‑friendly.",
    "bark":
        "✧ Bark  ·  Large generative model with several pre‑baked English narrators (presets 0‑7).  Slowest but most expressive.",
}
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Optional Bark generation parameters that power‑users can tweak via
# the UI.  We will only pass those that the *installed* Bark version
# actually supports (see inspect.signature below).
BARK_OPTIONAL_PARAMS = ["text_temp", "waveform_temp", "top_k", "top_p", "seed"]
# ------------------------------------------------------------------

def get_available_voices():
    manager = ModelManager()
    models_dict = manager.models_dict
    available_voices = []

    for lang, datasets in models_dict.get("tts_models", {}).items():
        for dataset, models in datasets.items():
            for model_name, model_data in models.items():
                model_key = f"tts_models/{lang}/{dataset}/{model_name}"

                supported_languages = model_data.get("supported_languages") or ["en"]
                supported_speakers = model_data.get("speakers") or []
                # ------------------------------------------------------------------
                # VCTK models often ship without the speaker list in the metadata
                # returned by ModelManager.  Hard‑code the canonical list if empty.
                if model_key == "tts_models/en/vctk/vits" and not supported_speakers:
                    supported_speakers = [
                        # Common VCTK speaker IDs
                        "p225","p226","p227","p228","p229","p230","p231","p232",
                        "p233","p234","p236","p237","p238","p239","p240","p241",
                        "p243","p244","p245","p246","p247","p248","p249","p250",
                        "p251","p252","p253","p254","p255","p256","p257","p258",
                        "p259","p260","p261","p262","p263","p264","p265","p266",
                        "p267","p268","p269","p270","p271","p272","p273","p274",
                        "p275","p276","p277","p278","p279","p280"
                    ]
                # ------------------------------------------------------------------

                requires_language = len(supported_languages) > 1
                requires_speaker_wav = any(key in model_key for key in ["xtts", "your_tts"])

                available_voices.append({
                    "name": model_key,
                    "model": model_name,
                    "requires_language": requires_language,
                    "requires_speaker_wav": requires_speaker_wav,
                    "supported_languages": supported_languages,
                    "supported_speakers": supported_speakers,
                    "description": VOICE_DESCRIPTIONS.get(model_key, "")
                })

    # Include Bark voices manually, presets generated from VOICE_PRESET_MAP keys
    bark_voices = [
        {
            "name": "bark",
            "model": "bark",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": [],
            # expose Bark “presets” so the frontend can render a 2nd dropdown
            "presets": list(VOICE_PRESET_MAP.keys()),
            "description": VOICE_DESCRIPTIONS["bark"],
            "tokens": [
                "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]", "[clears throat]",
                "[whispers]", "[giggles]", "[snickers]", "[coughs]", "[groans]", "[yells]",
                "[gasps]", "[whimpers]", "[sobs]", "[murmurs]", "[chuckles]", "[hums]",
                "[sneezes]", "[grunts]", "[shrieks]", "[hiccups]", "[stammers]", "[stutters]",
                "[grumbles]", "[snorts]", "[howls]", "[moans]", "[guffaws]", "[sighs deeply]",
                "[laughs nervously]", "[cries]", "[sniffs]", "[smacks lips]", "[claps]",
                "[yawns]", "[mumbles]", "[shushes]", "[exhales sharply]", "[snaps]",
                "[whistles]", "[crunches]", "[slurps]", "[clicks]", "[clinks]", "[clatters]",
                "[sizzles]", "[rustles]", "[splashes]", "[taps]", "[thumps]", "[rumbles]",
                "[drums]", "[jingles]", "[jangles]", "[pops]", "[bangs]", "[hisses]",
                "[scratches]", "[squeaks]", "[screeches]", "[buzzes]", "[swooshes]",
                "[swoops]", "[clangs]", "[whirrs]", "[chirps]", "[beeps]", "[tick-tocks]",
                "[thuds]", "[swishes]", "[crackles]", "[fizzes]", "[humming]"
            ],
        }
    ]
    available_voices.extend(bark_voices)

    # ------------------------------------------------------------------
    # Curate a *short‑list* of high‑fidelity, CPU‑friendly voices suited
    # for audiobook drafting.  Only these will be exposed to the front‑end.
    PREFERRED_MODELS = {
        "tts_models/multilingual/multi-dataset/xtts_v2",  # cloning / multilingual
        "tts_models/en/vctk/vits",                       # neutral, clear English
        "bark",                                          # high‑quality, slow CPU synthesis
    }

    available_voices = [v for v in available_voices if v["name"] in PREFERRED_MODELS]
    for v in available_voices:
        if "presets" not in v:
            v["presets"] = []
        if "description" not in v:
            v["description"] = ""
    # ------------------------------------------------------------------

    return available_voices

AVAILABLE_VOICES = get_available_voices()
print(f"✅ Loaded {len(AVAILABLE_VOICES)} voice models.")

DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def get_tts_instance(model_name):
    """
    Returns a TTS instance for the given model name, or None for Bark.
    """
    if model_name == "bark":
        return None  # Skip Bark, handled separately
    try:
        tts_instance = TTS(model_name, gpu=False)
        logging.info(f"✅ Loaded TTS model: {model_name}")
        return tts_instance
    except Exception as e:
        logging.exception(f"❌ Error loading TTS model {model_name}: {e}")
        return None

 # ------------------------------------------------------------------
# Use the offline Llama model to insert punctuation & SSML breaks.
def enhance_text_with_llm(raw_text: str) -> str:
    """
    Very lightweight prompt that asks the local LLM to
    1) add standard English punctuation,
    2) insert <break time="0.6s"/> between major thoughts.
    Falls back to the original text if Llama is unavailable.
    """
    if not llm:
        return raw_text.strip()

    prompt = (
        "You are a helpful assistant that fixes punctuation in a paragraph "
        "and inserts the tag <break time=\"0.6s\"/> wherever the narrator "
        "should pause for breath.  Return ONLY the corrected text.\n\n"
        f"### INPUT:\n{raw_text.strip()}\n\n### OUTPUT:\n"
    )
    try:
        completion = llm(prompt, max_tokens=512, stop=["###"])
        improved = completion["choices"][0]["text"].strip()
        return improved if improved else raw_text.strip()
    except Exception as e:
        logging.exception("LLM text enhancement failed")
        return raw_text.strip()
# ------------------------------------------------------------------

def sanitize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = ''.join(c for c in text if c.isprintable())
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."}
    # Drop combining marks (category "Mn") such as the '͡' tie‑bar
    text = ''.join(ch for ch in text if unicodedata.category(ch) != "Mn")
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Fix abrupt period followed by capital letter with space
    text = re.sub(r'\.\s+([A-Z])', r'. \1', text)
    # Retain Bark tokens: [laughter], [music], etc., but preserve all user tokens matching Bark's allowed tokens (case-sensitive and as written).
    # Use tokens from AVAILABLE_VOICES for Bark
    bark_voice = next((v for v in AVAILABLE_VOICES if v.get("name") == "bark"), None)
    allowed_tokens = set()
    if bark_voice and "tokens" in bark_voice:
        allowed_tokens = set(bark_voice["tokens"])
    # Remove [bracketed] tokens EXCEPT those in allowed_tokens (case-sensitive, preserve case)
    def token_replacer(m):
        token = m.group(0)
        return token if token in allowed_tokens else ""
    # Only replace bracketed tokens that do not match the allowed list exactly
    text = re.sub(r'\[[^\[\]]+\]', token_replacer, text)
    # Remove extra whitespace
    return re.sub(r'\s+', ' ', text).strip()


# ------------------------------------------------------------------
# Bark Consistency Table
# Attempt #11b - Strategy B1 Local Prompt Loader
# Description: Locally loads `.npz` preset as embedding dict instead of importing from Bark
def load_history_prompt_npz(file_path):
    """
    Loads a Bark-compatible history prompt from an .npz file and returns a dictionary
    with keys: 'semantic_prompt', 'coarse_prompt', 'fine_prompt'.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            "semantic_prompt": data["semantic_prompt"],
            "coarse_prompt": data["coarse_prompt"],
            "fine_prompt": data["fine_prompt"],
        }
    except Exception as e:
        logging.exception(f"Failed to load history prompt from {file_path}: {e}")
        return None

def concatenate_audio_files(audio_files, output_path, pause_duration=0.5):
    """
    Concatenates multiple audio files with silence pauses in between.
    """
    combined_audio = []
    sr = None

    for file in audio_files:
        y, sr = librosa.load(file, sr=None)
        combined_audio.append(y)
        silence = np.zeros(int(sr * pause_duration))
        combined_audio.append(silence)

    if combined_audio:
        merged_audio = np.concatenate(combined_audio)
        sf.write(output_path, merged_audio, sr)

#
# Main TTS generation endpoint.
# Accepts text and synthesis parameters, queues a job, and returns job status.
@app.route("/generate", methods=["POST"])
def generate():
    try:
        # The front‑end *usually* sends JSON, but older calls (or file uploads)
        # may arrive as multipart/form‑data.  Handle both seamlessly.
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict() or {}
        # ------------------------------------------------------------------
        # (Voice saving logic moved to dedicated endpoint)
        # ------------------------------------------------------------------
        # Core fields sent by the new UI
        text             = (data.get("text") or "").strip()
        model_name       = (data.get("model") or "").strip()      # e.g. "xtts", "bark"
        voice_name       = model_name if model_name != "xtts" else "tts_models/multilingual/multi-dataset/xtts_v2"
        language         = (data.get("language") or "").strip()
        speaker          = (data.get("speaker") or "").strip()    # VITS speaker id
        voice_preset     = (data.get("preset")  or "").strip()    # Bark preset id
        voice_id         = (data.get("voice")   or "").strip()    # Bark custom speaker
        direction        = (data.get("voice_direction") or "").strip()

        # Bark fine‑tune sliders
        creativity = data.get("creativity")   # --> text_temp  (float 0‑1)
        pool       = data.get("pool")         # --> top_k      (int)
        focus      = data.get("focus")        # --> top_p      (float 0‑1)

        # Queue‑engine misc
        speed           = float(data.get("speed", 1.0))
        chunk_size      = int(data.get("chunk_size", 300))
        pause_duration  = float(data.get("pause_duration", 0.5))

        # XTTS config fields
        length_scale    = float(data.get("length_scale", 1.0))
        noise_scale     = float(data.get("noise_scale", 0.667))
        noise_scale_w   = float(data.get("noise_scale_w", 0.8))

        # Turn on the punctuation/SSML enhancer iff the user supplied any direction text
        smart_enhance = bool(direction)
        # ------------------------------------------------------------------

        voice_info = next((v for v in AVAILABLE_VOICES if v["name"] == voice_name), None)
        # --- New logic for language, speaker, and speaker_wav handling ---
        if voice_info:
            if voice_info.get("requires_language") and not language:
                # Apply default for multilingual models
                if "multilingual" in voice_name:
                    language = "en"
                else:
                    return jsonify({"error": f"Language is required for model {voice_name}."}), 400
            elif not voice_info.get("requires_language"):
                language = None

        if not speaker and voice_info and voice_info.get("supported_speakers"):
            speaker = voice_info["supported_speakers"][0]

        speaker_wav = request.files.get("speaker_wav")
        if voice_info and voice_info.get("requires_speaker_wav"):
            if not speaker_wav or not speaker_wav.filename:
                return jsonify({"error": f"Model {voice_name} requires a speaker reference audio file (WAV)."}), 400
        # ---------------------------------------------------------------

        if "bark" in voice_name:
            job_id = str(uuid4())
            job_status[job_id] = {"status": "queued", "progress": 0, "output_path": None}

            job_queue.put({
                "job_id": job_id,
                "text": text,
                "voice_name": voice_name,
                "speed": speed,
                "pause_duration": pause_duration,
                "language": language,
                "speaker": speaker,
                "speaker_wav": speaker_wav.read() if speaker_wav else None,
                "speaker_wav_name": speaker_wav.filename if speaker_wav else None,
                "use_bark": True,
                # Updated/inserted Bark job keys:
                "voice_preset": voice_preset or voice_id,
                # sliders mapped to Bark arg names
                "text_temp":   creativity if creativity is not None else "",
                "top_k":       pool       if pool       is not None else "",
                "top_p":       focus      if focus      is not None else "",
                # (Retain other Bark options if needed)
                "smart_enhance": smart_enhance,
                # Add XTTS config fields for consistency (not used by Bark)
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_scale_w": noise_scale_w,
                # Pass through Bark seed for voice consistency
                "seed": data.get("seed"),
            })

            queue_position = job_queue.qsize()
            estimated_wait_time = queue_position * 5
            return jsonify({
                "job_id": job_id,
                "queue_position": queue_position,
                "estimated_wait_time": estimated_wait_time
            })

        if not text:
            return jsonify({"error": "Text input is required."}), 400
        if not voice_info:
            return jsonify({"error": f"Voice model '{voice_name}' not found."}), 400
        if speaker and voice_info["supported_speakers"] and speaker not in voice_info["supported_speakers"]:
            return jsonify({"error": f"Invalid speaker '{speaker}' for model {voice_name}."}), 400

        job_id = str(uuid4())
        job_status[job_id] = {"status": "queued", "progress": 0, "output_path": None}

        job_queue.put({
            "job_id": job_id,
            "text": text,
            "voice_name": voice_name,
            "speed": speed,
            "pause_duration": pause_duration,
            "language": language,
            "speaker": speaker,
            "speaker_wav": speaker_wav.read() if speaker_wav else None,
            "speaker_wav_name": speaker_wav.filename if speaker_wav else None,
            "voice_preset": voice_preset,
            "smart_enhance": smart_enhance,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
        })

        queue_position = job_queue.qsize()
        estimated_wait_time = queue_position * 5  # rough estimate: 5 seconds per job
        return jsonify({
            "job_id": job_id,
            "queue_position": queue_position,
            "estimated_wait_time": estimated_wait_time
        })
    except Exception as e:
        logging.exception("Error in generate endpoint")
        return jsonify({"error": f"Internal error: {e}"}), 500

#
# Job status endpoint.
# Returns the status, progress, and audio URL (if done) for a given job.
@app.route("/status/<job_id>", methods=["GET"])
def check_status(job_id):
    if job_id not in job_status:
        return jsonify({"error": "Invalid job ID"}), 404
    status_info = job_status[job_id]
    # Include chunk progress if available (for frontend progress bar)
    chunk_index = status_info.get("chunk_index")
    total_chunks = status_info.get("total_chunks")
    # Add chunk info to status for frontend progress estimation
    if chunk_index is not None and total_chunks:
        status_info["chunk_index"] = chunk_index
        status_info["total_chunks"] = total_chunks
    if status_info["status"] == "queued":
        job_ids = [j["job_id"] for j in job_queue.queue]
        queue_position = job_ids.index(job_id) + 1 if job_id in job_ids else None
        status_info["queue_position"] = queue_position
    if status_info["status"] == "done":
        status_info["audio_url"] = f"/audio/{job_id}"
        return jsonify({
            "status": status_info.get("status"),
            "progress": status_info.get("progress", 0),
            "chunk_index": status_info.get("chunk_index"),
            "total_chunks": status_info.get("total_chunks"),
            "audio_url": status_info.get("audio_url"),
            "queue_position": status_info.get("queue_position")
        })
    return jsonify({
        "status": status_info.get("status"),
        "progress": status_info.get("progress", 0),
        "chunk_index": status_info.get("chunk_index"),
        "total_chunks": status_info.get("total_chunks"),
        "audio_url": status_info.get("audio_url"),
        "queue_position": status_info.get("queue_position")
    })

#
# Cancel a queued job by job_id.
@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    queue_list = list(job_queue.queue)
    for job in queue_list:
        if job["job_id"] == job_id:
            queue_list.remove(job)
            job_queue.queue.clear()
            for j in queue_list:
                job_queue.put(j)
            job_status[job_id] = {"status": "cancelled", "progress": 0}
            job_status[job_id]["audio_url"] = None
            return jsonify({"status": "cancelled"})
    return jsonify({"error": "Job not found or already processing."}), 404

def synthesize_audio_with_vits(text, speaker_id=None, noise_scale=0.667, duration_scale=1.0, use_phonemes=False):
    """
    Stub for VITS synthesis with support for additional parameters.
    Currently logs the parameters and returns None.
    """
    logging.info(f"[VITS] synthesize_audio_with_vits called with: text='{text[:30]}...', speaker_id={speaker_id}, noise_scale={noise_scale}, duration_scale={duration_scale}, use_phonemes={use_phonemes}")
    # TODO: Actual VITS synthesis implementation goes here
    # For now, just return None or raise NotImplementedError
    return None

def safe_float(value, default):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def synthesize_audio_with_bark(job, job_id, raw_input_text, output_path):
    """
    Handles Bark TTS synthesis with chunking and voice prompt preservation.
    Uses semantic token generation and waveform synthesis for improved reliability.
    Applies crossfading to improve audio transitions between chunks.
    Reports chunk progress for frontend progress bar.
    Refactored to ensure consistent voice by reusing history_prompt.
    """
    # 📘 Table: Bark Consistency Attempts
    # Attempt #11b - Strategy B1 Local Prompt Loader
    # Description: Locally loads `.npz` preset as embedding dict instead of importing from Bark
    try:
        import inspect
        from bark import SAMPLE_RATE, preload_models, generate_audio
        from scipy.io.wavfile import write as write_wav
        # Set Bark generation parameters using global API
        from bark import generation
        logger = logging.getLogger("bark")
        preload_models()
        processed_text = sanitize_text(raw_input_text)

        # Insert debug logging for Bark input
        print(f"[DEBUG] Bark input (raw) length {len(raw_input_text)} characters: {raw_input_text}")
        print(f"[DEBUG] Bark input (sanitized) length {len(processed_text)} characters: {processed_text}")

        # Check for missing punctuation at end
        if not processed_text.strip().endswith(".") and not processed_text.strip().endswith("!") and not processed_text.strip().endswith("?"):
            print("[WARNING] Input text does not end with punctuation, may cause incomplete generation.")

        # Semantic chunking using chunk_text_by_sentence for sentence-aware splitting
        semantic_chunks = chunk_text_by_sentence(processed_text)
        total_chunks = len(semantic_chunks)
        job_status[job_id]["total_chunks"] = total_chunks

        # New logic: load Bark preset path from mapping, using installed Bark package location
        voice_name = job.get("voice_preset", "Default (Speaker 0)")
        filename = VOICE_PRESET_MAP.get(voice_name, VOICE_PRESET_MAP["Default (Speaker 0)"])
        import bark
        bark_root = os.path.dirname(bark.__file__)
        voice_preset_path = os.path.join(bark_root, "assets", "prompts", filename)
        # Insert file existence check and debug logging before np.load
        if not os.path.isfile(voice_preset_path):
            raise FileNotFoundError(f"Voice preset file not found: {voice_preset_path}")
        logging.debug(f"Attempting to load voice preset file: {voice_preset_path}")

        history_prompt_data = load_history_prompt_npz(voice_preset_path)

        # Extract Bark generation parameters from job
        seed = job.get("seed")
        text_temp = safe_float(job.get("text_temp"), 0.7)
        top_k = int(job.get("top_k") or 50)
        top_p = safe_float(job.get("top_p"), 0.95)

        # Set Bark generation parameters globally before the loop
        import torch
        torch.manual_seed(seed or 42)
        generation.TEXT_TEMP = text_temp
        generation.TOP_K = top_k
        generation.TOP_P = top_p

        # Bark generation loop with consistent voice
        audio_array = []
        for i, semantic_chunk in enumerate(semantic_chunks):
            print(f"[DEBUG] Generating chunk {i+1}/{len(semantic_chunks)}")
            job_status[job_id]["chunk_index"] = i + 1
            job_status[job_id]["progress"] = min(int(((i + 1) / total_chunks) * 100), 99)
            try:
                audio = generate_audio(
                    semantic_chunk,
                    history_prompt=voice_preset_path  # consistently use the original preset
                )
                # Apply crossfade if not the first chunk
                if audio_array:
                    audio = crossfade_audio(audio_array[-1], audio)
                    audio_array[-1] = audio  # Replace last with crossfaded
                else:
                    audio_array.append(audio)
            except Exception as e:
                logger.warning(f"[WARNING] Bark chunk {i+1} failed: {e}")
                continue

        if audio_array and any(isinstance(a, np.ndarray) and a.size > 0 for a in audio_array):
            final_audio = np.concatenate(audio_array, axis=-1)
            # Save as 16-bit PCM WAV
            pcm_audio = np.clip(final_audio, -1.0, 1.0)
            pcm_audio = (pcm_audio * 32767).astype(np.int16)
            write_wav(output_path, SAMPLE_RATE, pcm_audio)
            print(f"[DEBUG] Saved final audio to {output_path}")
        else:
            logger.error(f"[BARK ERROR] Job {job_id} failed: No valid audio was generated.")
            job_status[job_id] = {
                "status": "error",
                "progress": 0,
                "message": "No valid audio was generated."
            }
            raise ValueError("No valid audio was generated.")
    except Exception as bark_error:
        logging.exception(f"[BARK ERROR] Job {job_id} failed.")
        job_status[job_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Bark synthesis failed: {bark_error}"
        }
        raise

def process_jobs():
    while True:
        job = job_queue.get()
        if job_status.get(job["job_id"], {}).get("status") == "cancelled":
            continue
        job_id = job["job_id"]
        # If the user toggled the auto‑punctuation option, run the
        # paragraph through the local LLM before sanitising / splitting.
        raw_input_text = job["text"]
        if job.get("smart_enhance"):
            raw_input_text = enhance_text_with_llm(raw_input_text)
        job_status[job_id]["status"] = "processing"
        try:
            output_path = f"{OUTPUT_FOLDER}/speech_{job_id}.wav"
            kwargs = {
                "text": sanitize_text(raw_input_text),
                "file_path": output_path,
                "speed": job["speed"],
                # XTTS config fields
                "length_scale": job.get("length_scale", 1.0),
                "noise_scale": job.get("noise_scale", 0.667),
                "noise_scale_w": job.get("noise_scale_w", 0.8),
            }
            # Safety‑net: ensure a language code is passed for multilingual models
            if "multilingual" in job["voice_name"] and not kwargs.get("language"):
                kwargs["language"] = "en"
            if job["speaker"]:
                kwargs["speaker"] = job["speaker"]

            # Handle XTTS speaker_wav upload
            model_id = job.get("voice_name") or ""
            speaker_wav_bytes = job.get("speaker_wav")
            speaker_wav_name = job.get("speaker_wav_name")
            if speaker_wav_bytes and speaker_wav_name:
                speaker_path = f"{OUTPUT_FOLDER}/speaker_{job_id}.wav"
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                with open(speaker_path, "wb") as f:
                    f.write(speaker_wav_bytes)
                # Validate file existence and size before passing to the model
                if os.path.exists(speaker_path):
                    if os.path.getsize(speaker_path) == 0:
                        raise ValueError(f"Speaker file at {speaker_path} is empty.")
                else:
                    raise FileNotFoundError(f"Speaker reference file not found at {speaker_path}")
                # Validate and re-encode to ensure compatibility with torchaudio.load
                try:
                    audio = AudioSegment.from_file(speaker_path)
                    audio = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)
                    audio.export(speaker_path, format="wav")
                except Exception as e:
                    raise ValueError(f"Failed to auto-convert speaker reference to WAV format: {e}")
                # Pass as speaker_wav for XTTS compatibility
                kwargs["speaker_wav"] = speaker_path
                print(f"[DEBUG] Saved speaker reference to {speaker_path}")
            elif model_id == "tts_models/multilingual/multi-dataset/xtts_v2":
                raise ValueError("XTTS requires a speaker_wav file, but none was provided.")

            tts_instance = get_tts_instance(job["voice_name"])
            if not tts_instance and not job.get("use_bark"):
                raise RuntimeError("TTS model failed to load")
            if job.get("use_bark"):
                try:
                    synthesize_audio_with_bark(job, job_id, raw_input_text, output_path)
                except Exception:
                    continue
            else:
                # Remove unsupported XTTS kwargs
                model_path = job.get("voice_name", "")
                settings = kwargs
                # Sanitize settings for xtts_v2 to avoid passing unsupported model_kwargs
                model_name = model_path.split("/")[-1] if "/" in model_path else model_path
                if model_name == "xtts_v2":
                    for key in ["length_scale", "noise_scale", "noise_scale_w"]:
                        settings.pop(key, None)
                tts_instance.tts_to_file(**settings)
            job_status[job_id]["audio_url"] = f"/audio/{job_id}"
            job_status[job_id]["status"] = "done"
            # Always set progress to 100% on job completion for frontend
            job_status[job_id]["progress"] = 100
            job_status[job_id]["download_url"] = f"/audio/{job_id}"
            job_status[job_id]["audio_url"] = f"/audio/{job_id}"
        except Exception as e:
            logging.exception(f"Job {job_id} failed")
            job_status[job_id] = {"status": "error", "progress": 0, "message": str(e)}

Thread(target=process_jobs, daemon=True).start()

#
# Serves the generated audio file for a completed job.
@app.route("/audio/<job_id>", methods=["GET"])
def get_audio(job_id):
    info = job_status.get(job_id)
    if not info or info.get("status") != "done":
        return jsonify({"error": "Audio not available."}), 404
    return send_file(f"{OUTPUT_FOLDER}/speech_{job_id}.wav", as_attachment=True)

#
# Returns the list of available voices and their metadata.
@app.route("/voices", methods=["GET"])
def list_voices():
    print("📢 Listing available voices")
    # Include supported_speakers in the response for frontend secondary dropdowns
    voices_with_speakers = []
    for voice in AVAILABLE_VOICES:
        voices_with_speakers.append({
            "name": voice["name"],
            "model": voice["model"],
            "requires_language": voice["requires_language"],
            "requires_speaker_wav": voice["requires_speaker_wav"],
            "supported_languages": voice["supported_languages"],
            "supported_speakers": voice["supported_speakers"],
            "presets": voice.get("presets", []),
            "description": voice.get("description", ""),
            "tokens": voice.get("tokens", [])
        })
    return jsonify({"voices": voices_with_speakers})

#
# Root endpoint: returns a status message.
@app.route("/")
def home():
    return "Coqui TTS backend running. Use /generate for speech synthesis."

#
# Text enhancement endpoint: applies LLM-based enhancements.
@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data["text"]
    instruction = data.get("instruction", "")
    try:
        enhanced = enhance_text(text, instruction)
    except Exception as e:
        logging.exception("LLM enhancement error")
        return jsonify({"error": "Enhancement failed"}), 500
    return jsonify({"enhanced_text": enhanced})

# ------------------------------------------------------------------
# Entry point: confirm app is running when executed directly.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✅ Flask app starting...")
    app.run(host="0.0.0.0", port=5000)



# ------------------------------------------------------------------
# Dedicated endpoints for CRUD operations on saved voice profiles

@app.route("/save_voice", methods=["POST"])
def save_voice():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing voice name"}), 400
    saved_voices[name] = {
        "name": name,
        "seed": data.get("seed"),
        "text_temp": data.get("text_temp"),
        "top_k": data.get("top_k"),
        "top_p": data.get("top_p"),
        "pinned": data.get("pinned", False)
    }
    return jsonify({"message": f"Voice '{name}' saved."})

@app.route("/saved_voices", methods=["GET"])
def get_saved_voices():
    # Return as a list of dicts for frontend
    return jsonify(list(saved_voices.values()))

@app.route("/saved_voices/<name>", methods=["PUT"])
def update_voice(name):
    if name not in saved_voices:
        return jsonify({"error": "Voice not found"}), 404
    data = request.get_json()
    saved_voices[name].update(data)
    return jsonify({"message": f"Voice '{name}' updated."})

@app.route("/saved_voices/<name>", methods=["DELETE"])
def delete_voice(name):
    if name in saved_voices:
        del saved_voices[name]
        return jsonify({"message": f"Voice '{name}' deleted."})
    return jsonify({"error": "Voice not found"}), 404

#
# Example VITS synthesis endpoint (stub)
@app.route("/synthesize_vits", methods=["POST"])
def synthesize_vits():
    """
    Endpoint to synthesize speech using VITS with extra parameters.
    """
    try:
        text = request.form.get("text", "").strip()
        # Extract additional parameters from the form
        speaker_id = request.form.get("speaker_id", None)
        noise_scale = float(request.form.get("noise_scale", 0.667))
        duration_scale = float(request.form.get("duration_scale", 1.0))
        use_phonemes = request.form.get("use_phonemes", "false").lower() == "true"
        # Call the VITS audio generator with all parameters
        audio = synthesize_audio_with_vits(
            text=text,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            duration_scale=duration_scale,
            use_phonemes=use_phonemes
        )
        # For now, since the function is stubbed, just return a message
        return jsonify({
            "status": "ok",
            "message": "VITS synthesis stub called.",
            "params": {
                "text": text,
                "speaker_id": speaker_id,
                "noise_scale": noise_scale,
                "duration_scale": duration_scale,
                "use_phonemes": use_phonemes
            }
        })
    except Exception as e:
        logging.exception("Error in synthesize_vits endpoint")
        return jsonify({"error": f"Internal error: {e}"}), 500