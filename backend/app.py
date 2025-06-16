import json
import os
import bark

def process_jobs():
    while True:
        job = job_queue.get()
        try:
            if job.get("use_bark"):
                # Synthesize audio using run_bark_synthesis with all Bark parameters
                audio_array = run_bark_synthesis(
                    job.get("job_id"),
                    job.get("text"),
                    job.get("speaker"),
                    job.get("voice"),
                    job.get("preset"),
                    job.get("text_temp"),
                    job.get("top_k"),
                    job.get("top_p"),
                    job.get("seed"),
                    job.get("speed"),
                    job.get("pause_duration"),
                    job.get("barkSplitSentences"),
                    job.get("barkMaxDuration"),
                    job.get("smart_enhance"),
                    job.get("language"),
                    job.get("voice_preset"),
                )
                # Optionally: Handle audio_array, save, or update job status as needed
            else:
                print(f"[WARNING] Unsupported job type or model: {job}")
        except Exception as e:
            print(f"[ERROR] Failed to process job: {e}")
        finally:
            job_queue.task_done()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")


# --- Device selection utility ---
import torch

def get_torch_device(use_mps: bool = False):
    if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[DEBUG] Apple MPS is available and selected.")
        return torch.device("mps")
    print(f"[DEBUG] Apple MPS not used or not available. use_mps={use_mps}, torch.backends.mps.is_available()={torch.backends.mps.is_available()}, torch.backends.mps.is_built()={torch.backends.mps.is_built()}")
    return torch.device("cpu")

# ────────────── Bark low-level imports for advanced synthesis ──────────────
from bark.generation import (
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    codec_decode,
)
import numpy as np
import os
from scipy.io.wavfile import write as write_wav

SAMPLE_RATE = 24000
saved_voices = {}
from pydub import AudioSegment
import numpy as np
import logging

preset_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "voice_presets.json")
)
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
    if (
        not isinstance(a, np.ndarray)
        or not isinstance(b, np.ndarray)
        or a.size == 0
        or b.size == 0
    ):
        logging.warning(
            "[WARNING] Skipping crossfade: one of the audio chunks is empty or not a valid NumPy array."
        )
        return a if isinstance(a, np.ndarray) and a.size > 0 else b
    duration = min(duration, len(a) / sr, len(b) / sr)
    crossfade_samples = int(duration * sr)
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    a[-crossfade_samples:] *= fade_out
    b[:crossfade_samples] *= fade_in
    return np.concatenate(
        [
            a[:-crossfade_samples],
            a[-crossfade_samples:] + b[:crossfade_samples],
            b[crossfade_samples:],
        ]
    )


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

from bark.generation import generate_text_semantic, generate_coarse, generate_fine, codec_decode
from bark.generation import preload_models
from scipy.io.wavfile import write as write_wav

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
    data = request.get_json() or {}
    raw_text = data.get("text", "")
    enhanced_text = sanitize_text(raw_text)
    return jsonify({"text": enhanced_text})


# ------------------------------------------------------------------
# /enhance endpoint for Bark AI Enhance feature (frontend uses this)
from flask_cors import cross_origin

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
    enhanced_text = enhance_text(text, instruction, creativity)
    return jsonify({"enhanced_text": enhanced_text})


# ------------------------------------------------------------------
# Human‑readable blurbs to help users pick a voice in the UI
VOICE_DESCRIPTIONS = {
    "tts_models/multilingual/multi-dataset/xtts_v2": "✧ XTTS‑v2  ·  Multilingual, cross‑speaker cloning model.  Best when you provide a short WAV of your own voice, but also works with its default timbre.",
    "tts_models/en/vctk/vits": "✧ VITS / VCTK  ·  Fast English model trained on the VCTK corpus.  Neutral Midlands accent, very CPU‑friendly.",
    "bark": "✧ Bark  ·  Large generative model with several pre‑baked English narrators (presets 0‑7).  Slowest but most expressive.",
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
                        "p225",
                        "p226",
                        "p227",
                        "p228",
                        "p229",
                        "p230",
                        "p231",
                        "p232",
                        "p233",
                        "p234",
                        "p236",
                        "p237",
                        "p238",
                        "p239",
                        "p240",
                        "p241",
                        "p243",
                        "p244",
                        "p245",
                        "p246",
                        "p247",
                        "p248",
                        "p249",
                        "p250",
                        "p251",
                        "p252",
                        "p253",
                        "p254",
                        "p255",
                        "p256",
                        "p257",
                        "p258",
                        "p259",
                        "p260",
                        "p261",
                        "p262",
                        "p263",
                        "p264",
                        "p265",
                        "p266",
                        "p267",
                        "p268",
                        "p269",
                        "p270",
                        "p271",
                        "p272",
                        "p273",
                        "p274",
                        "p275",
                        "p276",
                        "p277",
                        "p278",
                        "p279",
                        "p280",
                    ]
                # ------------------------------------------------------------------

                requires_language = len(supported_languages) > 1
                requires_speaker_wav = any(
                    key in model_key for key in ["xtts", "your_tts"]
                )

                available_voices.append(
                    {
                        "name": model_key,
                        "model": model_name,
                        "requires_language": requires_language,
                        "requires_speaker_wav": requires_speaker_wav,
                        "supported_languages": supported_languages,
                        "supported_speakers": supported_speakers,
                        "description": VOICE_DESCRIPTIONS.get(model_key, ""),
                    }
                )

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
                "[laughter]",
                "[laughs]",
                "[sighs]",
                "[music]",
                "[gasps]",
                "[clears throat]",
                "[whispers]",
                "[giggles]",
                "[snickers]",
                "[coughs]",
                "[groans]",
                "[yells]",
                "[gasps]",
                "[whimpers]",
                "[sobs]",
                "[murmurs]",
                "[chuckles]",
                "[hums]",
                "[sneezes]",
                "[grunts]",
                "[shrieks]",
                "[hiccups]",
                "[stammers]",
                "[stutters]",
                "[grumbles]",
                "[snorts]",
                "[howls]",
                "[moans]",
                "[guffaws]",
                "[sighs deeply]",
                "[laughs nervously]",
                "[cries]",
                "[sniffs]",
                "[smacks lips]",
                "[claps]",
                "[yawns]",
                "[mumbles]",
                "[shushes]",
                "[exhales sharply]",
                "[snaps]",
                "[whistles]",
                "[crunches]",
                "[slurps]",
                "[clicks]",
                "[clinks]",
                "[clatters]",
                "[sizzles]",
                "[rustles]",
                "[splashes]",
                "[taps]",
                "[thumps]",
                "[rumbles]",
                "[drums]",
                "[jingles]",
                "[jangles]",
                "[pops]",
                "[bangs]",
                "[hisses]",
                "[scratches]",
                "[squeaks]",
                "[screeches]",
                "[buzzes]",
                "[swooshes]",
                "[swoops]",
                "[clangs]",
                "[whirrs]",
                "[chirps]",
                "[beeps]",
                "[tick-tocks]",
                "[thuds]",
                "[swishes]",
                "[crackles]",
                "[fizzes]",
                "[humming]",
            ],
        }
    ]
    available_voices.extend(bark_voices)

    # ------------------------------------------------------------------
    # Curate a *short‑list* of high‑fidelity, CPU‑friendly voices suited
    # for audiobook drafting.  Only these will be exposed to the front‑end.
    PREFERRED_MODELS = {
        "tts_models/multilingual/multi-dataset/xtts_v2",  # cloning / multilingual
        "tts_models/en/vctk/vits",  # neutral, clear English
        "bark",  # high‑quality, slow CPU synthesis
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


def sanitize_text(text):
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
    # Drop combining marks (category "Mn") such as the '͡' tie‑bar
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Fix abrupt period followed by capital letter with space
    text = re.sub(r"\.\s+([A-Z])", r". \1", text)
    # Retain Bark tokens: [laughter], [music], etc., but preserve all user tokens matching Bark's allowed tokens (case-sensitive and as written).
    # Use tokens from AVAILABLE_VOICES for Bark
    bark_voice = next((v for v in AVAILABLE_VOICES if v.get("name") == "bark"), None)
    allowed_tokens = set()
    if bark_voice and "tokens" in bark_voice:
        allowed_tokens = set(bark_voice["tokens"])

    # Remove [bracketed] tokens EXCEPT those in allowed_tokens (case-sensitive, preserve case)
    def token_replacer(m):
        token = m.group(0).strip()
        if token not in allowed_tokens:
            print(f"[DEBUG] Stripped unknown token: {token}")
        return token if token in allowed_tokens else ""

    # Only replace bracketed tokens that do not match the allowed list exactly
    text = re.sub(r"\[[^\[\]]+\]", token_replacer, text)
    # Remove extra whitespace
    return re.sub(r"\s+", " ", text).strip()


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
        # ✅ Update job status in status_store
        status_store[job_id]["audio_url"] = f"/output/{job_id}.wav"
        status_store[job_id]["status"] = "done"
        status_store[job_id]["progress"] = 100


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
        # Normalize Bark tuning keys for backend compatibility
        if "temperature" in data:
            data["text_temp"] = data["temperature"]
        if "preset" not in data and "voice_preset" in data:
            data["preset"] = data["voice_preset"]
        if "focus" not in data and "top_p" in data:
            data["focus"] = data["top_p"]
        if "pool" not in data and "top_k" in data:
            data["pool"] = data["top_k"]
        # ------------------------------------------------------------------
        # (Voice saving logic moved to dedicated endpoint)
        # ------------------------------------------------------------------
        # Core fields sent by the new UI
        text = (data.get("text") or "").strip()
        model_name = (data.get("model") or "").strip()  # e.g. "xtts", "bark"
        # --- DEBUG LOGGING ---
        print("[DEBUG] /generate hit with model:", model_name)
        # ---------------------
        voice_name = (
            model_name
            if model_name != "xtts"
            else "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        language = (data.get("language") or "").strip()
        speaker = (data.get("speaker") or "").strip()  # VITS speaker id
        voice_preset = (data.get("preset") or "").strip()  # Bark preset id
        voice_id = (data.get("voice") or "").strip()  # Bark custom speaker
        direction = (data.get("voice_direction") or "").strip()

        # Bark fine‑tune sliders
        creativity = data.get("creativity")  # --> text_temp  (float 0‑1)
        pool = data.get("pool")  # --> top_k      (int)
        focus = data.get("focus")  # --> top_p      (float 0‑1)

        # Queue‑engine misc
        speed = float(data.get("speed", 1.0))
        chunk_size = int(data.get("chunk_size", 300))
        pause_duration = float(data.get("pause_duration", 0.5))

        # XTTS config fields
        length_scale = float(data.get("length_scale", 1.0))
        noise_scale = float(data.get("noise_scale", 0.667))
        noise_scale_w = float(data.get("noise_scale_w", 0.8))

        # Turn on the punctuation/SSML enhancer iff the user supplied any direction text
        smart_enhance = bool(direction)
        # ------------------------------------------------------------------

        # --- Bark-specific chunking options ---
        # Accept both JSON and form-data, so get from both places
        # Try to parse as bool and float, fallback to defaults if not present
        bark_split_sentences = False
        bark_max_duration = 14
        # Try to get from JSON (if present)
        if request.is_json:
            bark_split_sentences = bool(request.json.get("barkSplitSentences", False))
            bark_max_duration = float(request.json.get("barkMaxDuration", 14))
        else:
            # For form-data, values are strings
            bark_split_sentences = (
                str(data.get("barkSplitSentences", "False")).lower() == "true"
            )
            try:
                bark_max_duration = float(data.get("barkMaxDuration", 14))
            except Exception:
                bark_max_duration = 14
        # ------------------------------------------------------------------
        # Debug print for Bark chunking options
        print(
            f"[DEBUG] barkSplitSentences: {bark_split_sentences}, barkMaxDuration: {bark_max_duration}"
        )

        voice_info = next(
            (v for v in AVAILABLE_VOICES if v["name"] == voice_name), None
        )
        # --- New logic for language, speaker, and speaker_wav handling ---
        if voice_info:
            if voice_info.get("requires_language") and not language:
                # Apply default for multilingual models
                if "multilingual" in voice_name:
                    language = "en"
                else:
                    return (
                        jsonify(
                            {"error": f"Language is required for model {voice_name}."}
                        ),
                        400,
                    )
            elif not voice_info.get("requires_language"):
                language = None

        if not speaker and voice_info and voice_info.get("supported_speakers"):
            speaker = voice_info["supported_speakers"][0]

        speaker_wav = request.files.get("speaker_wav")
        if voice_info and voice_info.get("requires_speaker_wav"):
            if not speaker_wav or not speaker_wav.filename:
                return (
                    jsonify(
                        {
                            "error": f"Model {voice_name} requires a speaker reference audio file (WAV)."
                        }
                    ),
                    400,
                )
        # ---------------------------------------------------------------

        if "bark" in voice_name:
            print("[DEBUG] Starting Bark synthesis")
            import traceback

            try:
                # Ensure job_id is assigned before accessing it
                job_id = None
                if request.is_json and request.json is not None:
                    job_id = request.json.get("job_id")
                if not job_id:
                    job_id = str(uuid4())
                print(f"[DEBUG] status_store snapshot: {job_status.get(job_id, {})}")
                print(f"[DEBUG] full request body: {request.json}")
                job_status[job_id] = {
                    "status": "queued",
                    "progress": 0,
                    "output_path": None,
                }

                # Ensure advanced Bark parameters are included in the job dict
                job = {
                    "job_id": job_id,
                    "text": text,
                    "voice_name": voice_name,
                    "speed": speed,
                    "pause_duration": pause_duration,
                    "language": language,
                    "speaker": speaker,
                    "speaker_wav": speaker_wav.read() if speaker_wav else None,
                    "speaker_wav_name": (speaker_wav.filename if speaker_wav else None),
                    "use_bark": True,
                    # Updated/inserted Bark job keys:
                    "voice_preset": voice_preset or voice_id,
                    # sliders mapped to Bark arg names
                    "text_temp": creativity if creativity is not None else "",
                    "top_k": pool if pool is not None else "",
                    "top_p": focus if focus is not None else "",
                    # (Retain other Bark options if needed)
                    "smart_enhance": smart_enhance,
                    # Add XTTS config fields for consistency (not used by Bark)
                    "length_scale": length_scale,
                    "noise_scale": noise_scale,
                    "noise_scale_w": noise_scale_w,
                    # Pass through Bark seed for voice consistency
                    "seed": data.get("seed"),
                    # Pass new Bark chunking options
                    "bark_split_sentences": bark_split_sentences,
                    "bark_max_duration": bark_max_duration,
                    # Add all Bark fields for synthesize_audio_with_bark compatibility
                    "voice": voice_id or voice_name,
                    "preset": voice_preset,
                    "barkSplitSentences": bark_split_sentences,
                    "barkMaxDuration": bark_max_duration,
                }
                # Add advanced Bark parameters if present in data
                for adv_param in ["top_k", "top_p", "seed"]:
                    if adv_param in data:
                        job[adv_param] = data.get(adv_param)

                job_queue.put(job)

                queue_position = job_queue.qsize()
                estimated_wait_time = queue_position * 5
                print("[DEBUG] Completed all synthesis steps")
                print(f"[DEBUG] Finished Bark synthesis for job_id: {job_id}")
                return jsonify(
                    {
                        "job_id": job_id,
                        "queue_position": queue_position,
                        "estimated_wait_time": estimated_wait_time,
                    }
                )
            except Exception as e:
                print(f"[ERROR] Bark synthesis failed: {e}")
                traceback.print_exc()

        if not text:
            return jsonify({"error": "Text input is required."}), 400
        if not voice_info:
            return jsonify({"error": f"Voice model '{voice_name}' not found."}), 400
        if (
            speaker
            and voice_info["supported_speakers"]
            and speaker not in voice_info["supported_speakers"]
        ):
            return (
                jsonify(
                    {"error": f"Invalid speaker '{speaker}' for model {voice_name}."}
                ),
                400,
            )

        job_id = str(uuid4())
        job_status[job_id] = {"status": "queued", "progress": 0, "output_path": None}

        job_queue.put(
            {
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
            }
        )

        queue_position = job_queue.qsize()
        estimated_wait_time = queue_position * 5  # rough estimate: 5 seconds per job
        return jsonify(
            {
                "job_id": job_id,
                "queue_position": queue_position,
                "estimated_wait_time": estimated_wait_time,
            }
        )
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
        return jsonify(
            {
                "status": status_info.get("status"),
                "progress": status_info.get("progress", 0),
                "chunk_index": status_info.get("chunk_index"),
                "total_chunks": status_info.get("total_chunks"),
                "audio_url": status_info.get("audio_url"),
                "queue_position": status_info.get("queue_position"),
            }
        )
    return jsonify(
        {
            "status": status_info.get("status"),
            "progress": status_info.get("progress", 0),
            "chunk_index": status_info.get("chunk_index"),
            "total_chunks": status_info.get("total_chunks"),
            "audio_url": status_info.get("audio_url"),
            "queue_position": status_info.get("queue_position"),
        }
    )


# ------------------------------------------------------------------
# List available voices endpoint.
@app.route("/voices", methods=["GET"])
def list_voices():
    return jsonify({"voices": AVAILABLE_VOICES})


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


def synthesize_audio_with_vits(
    text, speaker_id=None, noise_scale=0.667, duration_scale=1.0, use_phonemes=False
):
    """
    Stub for VITS synthesis with support for additional parameters.
    Currently logs the parameters and returns None.
    """
    logging.info(
        f"[VITS] synthesize_audio_with_vits called with: text='{text[:30]}...', speaker_id={speaker_id}, noise_scale={noise_scale}, duration_scale={duration_scale}, use_phonemes={use_phonemes}"
    )
    # TODO: Actual VITS synthesis implementation goes here
    # For now, just return None or raise NotImplementedError
    return None


def safe_float(value, default):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


import re

# --- New semantic chunking function ---
def create_semantic_chunks(text, max_chars):
    """
    Splits the input text into chunks based on punctuation and conjunctions,
    trying to keep chunks below max_chars.
    """
    # Define break points for semantic chunking
    breakpoints = re.split(r'(?<=[\.\?!])\s+|(?<=,)\s+|(?<= and )', text)
    chunks = []
    current_chunk = ""

    for part in breakpoints:
        if len(current_chunk) + len(part) <= max_chars:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


preload_models()

def run_bark_synthesis(
    job_id,
    text,
    speaker,
    voice,
    preset,
    text_temp,
    top_k,
    top_p,
    seed,
    speed,
    pause_duration,
    barkSplitSentences,
    barkMaxDuration,
    smart_enhance,
    language,
    voice_preset,
    **kwargs
):
    """
    Handles Bark TTS synthesis with chunking and voice prompt preservation.
    Uses semantic token generation and waveform synthesis for improved reliability.
    Applies crossfading to improve audio transitions between chunks.
    Reports chunk progress for frontend progress bar.
    Accepts split_sentences and max_duration to control chunking logic.
    """
    import numpy as np
    import os
    import logging
    from scipy.io.wavfile import write as write_wav
    logger = logging.getLogger("bark")

    # --- Device selection logic based on use_mps setting ---
    # The frontend should send a setting "use_mps" to control this.
    job = locals()
    data = job
    use_mps = data.get("use_mps", False)
    if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch_device = "mps"
    else:
        torch_device = "cpu"
    print(f"[DEBUG] Apple MPS status - Requested: {use_mps}, Available: {torch.backends.mps.is_available()}, Built: {torch.backends.mps.is_built()}")
    print(f"[DEBUG] Final torch_device set to: {torch_device} (MPS requested: {use_mps})")

    raw_input_text = text
    processed_text = sanitize_text(raw_input_text) if raw_input_text is not None else sanitize_text(text)
    final_output_path = f"output/{job_id}_final.wav" if job_id else "output/final.wav"
    print(f"[DEBUG] Bark input (raw) length {len(text)} characters: {text}")
    print(f"[DEBUG] Bark input (sanitized) length {len(processed_text)} characters: {processed_text}")
    if (
        not processed_text.strip().endswith(".")
        and not processed_text.strip().endswith("!")
        and not processed_text.strip().endswith("?")
    ):
        print("[WARNING] Input text does not end with punctuation, may cause incomplete generation.")
    # --- Bark chunking logic ---
    print("[DEBUG] Creating semantic chunks")
    job = locals()
    barkSplitSentences = job.get("barkSplitSentences", True)
    barkMaxDuration = job.get("barkMaxDuration", barkMaxDuration)
    sanitized_text = processed_text
    semantic_chunks = []
    words_per_second = 2.5
    char_threshold = 300
    if barkSplitSentences:
        print("[DEBUG] Performing chunking before synthesis")
        import re
        sentence_delimiters = re.compile(r'(?<=[\.\!\?])\s+')
        sentence_candidates = sentence_delimiters.split(sanitized_text)
        temp_chunk = ""
        for sentence in sentence_candidates:
            candidate_chunk = (temp_chunk + " " + sentence).strip() if temp_chunk else sentence.strip()
            est_duration = len(candidate_chunk.split()) / words_per_second
            logger.debug(f"[DEBUG] Estimated duration for chunk: {est_duration:.2f}s")
            if len(candidate_chunk) > char_threshold or est_duration > barkMaxDuration:
                if temp_chunk.strip():
                    semantic_chunks.append(temp_chunk.strip())
                temp_chunk = sentence.strip()
            else:
                if temp_chunk:
                    temp_chunk = (temp_chunk + " " + sentence).strip()
                else:
                    temp_chunk = sentence.strip()
        if temp_chunk.strip():
            semantic_chunks.append(temp_chunk.strip())
        final_chunks = []
        for chunk in semantic_chunks:
            est_duration = len(chunk.split()) / words_per_second
            if len(chunk) > char_threshold or est_duration > barkMaxDuration:
                subchunks = re.split(r'(?<=,)\s+|(?<= and )', chunk)
                temp_sub = ""
                for sub in subchunks:
                    candidate_sub = (temp_sub + " " + sub).strip() if temp_sub else sub.strip()
                    est_sub_duration = len(candidate_sub.split()) / words_per_second
                    logger.debug(f"[DEBUG] Estimated duration for subchunk: {est_sub_duration:.2f}s")
                    if len(candidate_sub) > char_threshold or est_sub_duration > barkMaxDuration:
                        if temp_sub.strip():
                            final_chunks.append(temp_sub.strip())
                        temp_sub = sub.strip()
                    else:
                        if temp_sub:
                            temp_sub = (temp_sub + " " + sub).strip()
                        else:
                            temp_sub = sub.strip()
                if temp_sub.strip():
                    final_chunks.append(temp_sub.strip())
            else:
                final_chunks.append(chunk)
        semantic_chunks = [c for c in final_chunks if c.strip()]
        print(f"[DEBUG] Generated {len(semantic_chunks)} semantic chunks")
    else:
        semantic_chunks = [sanitized_text]
    for idx, chunk in enumerate(semantic_chunks):
        estimated_duration = len(chunk.split()) / 2.5
        print(f"[DEBUG] Chunk {idx} estimated duration: {estimated_duration:.2f}s")
    if len(semantic_chunks) == 1:
        estimated_duration = len(semantic_chunks[0].split()) / 2.5
        if estimated_duration > barkMaxDuration:
            print("[WARN] Only 1 chunk created and it's too long. Rechunking based on max duration.")
            words = semantic_chunks[0].split()
            chunk_word_limit = int(barkMaxDuration * 2.5)
            semantic_chunks = [' '.join(words[i:i + chunk_word_limit]) for i in range(0, len(words), chunk_word_limit)]
            print(f"[DEBUG] Rechunked to {len(semantic_chunks)} fallback chunks due to duration constraint.")
    if semantic_chunks:
        print(f"[DEBUG] First chunk preview: {semantic_chunks[0][:200]}")
    else:
        print("[DEBUG] No semantic chunks generated.")
    print(f"[DEBUG] semantic_chunks created: {len(semantic_chunks)} chunks")
    for i, chunk in enumerate(semantic_chunks):
        print(f"[DEBUG] semantic_chunks[{i}]: {repr(chunk)}")
        est_duration = len(chunk.split()) / words_per_second
        logger.debug(f"[DEBUG] Estimated duration for chunk: {est_duration:.2f}s")
    if not semantic_chunks:
        print("[ERROR] No semantic chunks generated. Exiting early.")
        return ""
    if len(semantic_chunks) == 1 and len(text) > 300:
        logger.warning("[WARN] Only 1 chunk created despite long input. Chunking might not be working as expected.")
    total_chunks = len(semantic_chunks)
    if job_id is not None and job_status.get(job_id) is not None:
        job_status[job_id]["total_chunks"] = total_chunks

    # --- Begin retry logic for semantic chunks ---
    # For each chunk, if generation fails (empty or gibberish), retry once
    audio_arrays = []
    chunk_attempts = []
    # Prepare Bark parameters for the pipeline
    bark_history_prompt = None
    if voice_preset:
        # Map voice_preset from VOICE_PRESET_MAP to actual identifier or .npz path
        mapped_value = VOICE_PRESET_MAP.get(voice_preset)
        if mapped_value:
            # Use Bark's installed path for prompt location if mapped_value is a filename (not full path)
            if mapped_value.endswith(".npz"):
                # Try to resolve absolute path if not already absolute
                if not os.path.isabs(mapped_value):
                    bark_root = os.path.dirname(bark.__file__)
                    speaker_path = os.path.join(bark_root, "assets", "prompts", mapped_value)
                else:
                    speaker_path = mapped_value
                if os.path.exists(speaker_path):
                    bark_history_prompt = load_history_prompt_npz(speaker_path)
                else:
                    print(f"[WARN] Speaker prompt file '{speaker_path}' not found.")
            else:
                bark_history_prompt = mapped_value
        else:
            print(f"[WARN] voice_preset '{voice_preset}' not found in VOICE_PRESET_MAP. Falling back.")
    # Prepare Bark parameter values
    _text_temp = float(text_temp) if text_temp not in (None, "") else 0.7
    _top_k = int(top_k) if top_k not in (None, "") else 50
    _top_p = float(top_p) if top_p not in (None, "") else 0.95
    _seed = int(seed) if seed not in (None, "") else None
    _voice_preset = voice_preset or "v2/en_speaker_9"
    # Ensure top_k, top_p, seed are extracted from incoming request if not already
    top_k = job.get("top_k")
    top_p = job.get("top_p")
    seed = job.get("seed")
    for chunk_idx, chunk in enumerate(semantic_chunks):
        attempt = 1
        max_attempts = 2
        chunk_success = False
        chunk_audio = None
        while attempt <= max_attempts and not chunk_success:
            print(f"[DEBUG] Synthesizing chunk {chunk_idx+1}/{len(semantic_chunks)} (attempt {attempt})")
            try:
                # Use the advanced Bark synthesis pipeline supporting top_k, top_p, seed, etc.
                audio_arr = synthesize_audio_with_bark(
                    chunk,
                    history_prompt=bark_history_prompt if bark_history_prompt is not None else _voice_preset,
                    text_temp=_text_temp,
                    waveform_temp=_text_temp,
                    top_k=top_k if top_k not in (None, "") else _top_k,
                    top_p=top_p if top_p not in (None, "") else _top_p,
                    seed=seed if seed not in (None, "") else _seed,
                )
                # Check: if output is empty or gibberish, retry once
                if audio_arr is None or (isinstance(audio_arr, np.ndarray) and audio_arr.size == 0):
                    print(f"[WARN] Chunk {chunk_idx+1} synthesis failed (empty output).")
                    if attempt == 1:
                        print(f"Retrying chunk {chunk_idx+1}...")
                    attempt += 1
                    continue
                duration_sec = len(audio_arr) / SAMPLE_RATE if hasattr(audio_arr, "__len__") else 0
                if duration_sec < 0.2:
                    print(f"[WARN] Chunk {chunk_idx+1} output too short ({duration_sec:.2f}s).")
                    if attempt == 1:
                        print(f"Retrying chunk {chunk_idx+1}...")
                    attempt += 1
                    continue
                chunk_success = True
                chunk_audio = audio_arr
                print(f"[DEBUG] Chunk {chunk_idx+1} succeeded on attempt {attempt}.")
            except Exception as e:
                print(f"[ERROR] Exception during chunk {chunk_idx+1} synthesis (attempt {attempt}): {e}")
                if attempt == 1:
                    print(f"Retrying chunk {chunk_idx+1}...")
                attempt += 1
        if not chunk_success:
            print(f"[ERROR] Failed to synthesize chunk {chunk_idx+1} after {max_attempts} attempts. Inserting silence.")
            chunk_audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        audio_arrays.append(chunk_audio)
        chunk_attempts.append(attempt)
        # Progress update
        if job_id is not None and job_status.get(job_id) is not None:
            job_status[job_id]["chunk_index"] = chunk_idx + 1
            job_status[job_id]["progress"] = int(100 * (chunk_idx + 1) / total_chunks)

    # --- Merge chunks with silence buffer between ---
    print("[DEBUG] Decoding and concatenating audio chunks with smooth merging...")
    intro_silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
    merged_audio = [intro_silence]
    for idx, arr in enumerate(audio_arrays):
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            arr = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        merged_audio.append(arr)
        if idx < len(audio_arrays) - 1:
            silence = np.zeros(int(SAMPLE_RATE // 4), dtype=np.float32)
            merged_audio.append(silence)
            print(f"Inserted pause between chunk {idx+1} and {idx+2}")
    merged_audio_np = np.concatenate(merged_audio)
    write_wav(final_output_path, SAMPLE_RATE, merged_audio_np)
    if job_id is not None and job_status.get(job_id) is not None:
        job_status[job_id]["status"] = "done"
        job_status[job_id]["progress"] = 100
        job_status[job_id]["audio_url"] = f"/output/{os.path.basename(final_output_path)}"
    print(f"[DEBUG] Bark synthesis complete. Output saved to {final_output_path}")
    return final_output_path
# -------------------------
# Custom Bark synthesis function supporting advanced sampling parameters
def synthesize_audio_with_bark(
    text,
    history_prompt=None,
    text_temp=0.7,
    waveform_temp=0.7,
    top_k=None,
    top_p=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    semantic_tokens = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        top_k=top_k,
        top_p=top_p
    )

    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
    )

    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
    )

    audio_array = codec_decode(fine_tokens)

    return audio_array

## Background job processor for handling queued TTS jobs.
def process_jobs():
    while True:
        job = job_queue.get()
        try:
            if job.get("use_bark"):
                run_bark_synthesis(**job)
            else:
                # Placeholder for VITS / XTTS jobs (not implemented yet)
                print(f"[WARNING] Unsupported job type or model: {job}")
        except Exception as e:
            print(f"[ERROR] Failed to process job: {e}")
        finally:
            job_queue.task_done()

# Ensure Flask app runs when executed directly
if __name__ == "__main__":
    preload_models()
    from threading import Thread
    Thread(target=process_jobs, daemon=True).start()
    app.run(debug=True)
# ------------------------------------------------------------------
# Enhance endpoint for local LLM-based text enhancement (“AI Enhance”)
@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json() or {}
    text = data.get("text", "")
    instruction = data.get("instruction", "")
    # Read extra params for smartEnhance
    creativity = data.get("creativity", 0.4)
    min_tokens = data.get("min_tokens", 0)
    # Pass these to enhance_text if needed (extend function signature if using)
    # For now, inject into the prompt if present
    extra_instruction = ""
    if creativity is not None:
        try:
            creativity_val = float(creativity)
            extra_instruction += f"\n[Creativity: {creativity_val}]"
        except Exception:
            pass
    if min_tokens is not None:
        try:
            min_tokens_val = int(min_tokens)
            if min_tokens_val > 0:
                extra_instruction += f"\n[Minimum tokens per paragraph: {min_tokens_val}]"
        except Exception:
            pass
    effective_instruction = (instruction or "") + extra_instruction
    enhanced_text = enhance_text(text, effective_instruction)
    return jsonify({"enhanced_text": enhanced_text})
