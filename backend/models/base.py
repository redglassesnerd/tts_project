import os
import re
import unicodedata
import logging
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

from scipy.io.wavfile import write as write_wav

logger = logging.getLogger("backend.models.base")

# Simple cache for the simulator's VITS model to avoid reloading
_SIMULATOR_VITS_CACHE = None

# Standard VCTK speakers classified by gender characteristics to provide diverse simulator failovers
MALE_SPEAKERS = ["p226", "p232", "p236", "p237", "p241", "p243", "p245", "p246", "p247", "p249", "p251", "p252", "p254", "p255", "p256", "p257", "p259", "p260", "p262", "p263", "p265", "p266", "p268", "p269", "p270", "p271", "p273", "p274", "p275", "p277", "p278", "p279", "p280", "p281", "p283", "p284", "p286", "p287", "p288", "p293", "p294", "p295", "p298", "p299", "p301", "p302", "p303", "p304", "p306", "p307", "p308", "p311", "p312", "p313", "p316", "p318", "p323", "p326", "p330", "p334", "p335", "p339", "p340", "p343", "p347", "p361", "p363", "p374"]
FEMALE_SPEAKERS = ["p225", "p228", "p229", "p230", "p231", "p233", "p234", "p236", "p238", "p239", "p240", "p244", "p248", "p250", "p253", "p258", "p261", "p264", "p267", "p272", "p276", "p282", "p285", "p292", "p297", "p300", "p305", "p310", "p314", "p317", "p329", "p330", "p333", "p336", "p341", "p345", "p351", "p360", "p362", "p364", "p376"]

def is_stage_direction(text: str, allowed_tokens=None) -> bool:
    text_strip = text.strip()
    if not text_strip:
        return True

    # If the text starts with a bracketed allowed token (e.g. [laughter] or [laughter] Ha ha ha), it is NOT a stage direction!
    if allowed_tokens:
        text_lower = text_strip.lower()
        for t in allowed_tokens:
            bracketed = f"[{t.strip('[]').lower()}]"
            if text_lower.startswith(bracketed):
                return False

    # If the text is fully enclosed in parentheses, it is a parenthetical cue (stage direction)
    if text_strip.startswith("(") and text_strip.endswith(")"):
        return True

    cleaned = text_strip.lower().strip("().*[]:- ")
    if not cleaned:
        return True
    
    # If the text is a bracketed tag and it's in the allowed tokens, it is NOT a stage direction!
    if text_strip.startswith("[") and text_strip.endswith("]"):
        tag_lower = text_strip.lower()
        if allowed_tokens and tag_lower in [t.lower() for t in allowed_tokens]:
            return False
            
    # Common sound effect / music / cue patterns
    direction_keywords = [
        "intro", "outro", "music", "transition", "sfx", "sound", "scene", "act", "narrator",
        "intro music", "outro music", "theme music", "ambient music", 
        "sound effect", "music fade", "music swell", "music plays", 
        "fade in", "fade out", "applause", "cheering", "laughter", 
        "sound of", "silence", "pause", "sigh", "clears throat",
        "episode title", "podcast title", "episode", "music begins",
        "music starts", "music ends", "opening theme", "closing theme"
    ]
    
    if cleaned in direction_keywords:
        return True
        
    for kw in direction_keywords:
        if cleaned == kw or cleaned.startswith(kw + " ") or cleaned.endswith(" " + kw) or cleaned.startswith(kw + ":"):
            return True
            
    return False

def generate_simulated_tag_audio(tag: str, sample_rate: int, gender: str = "female") -> np.ndarray:
    tag_clean = tag.strip("[]<>— ").lower()
    
    # Pauses and breaks
    if tag_clean in ["uv_break", "lbreak", "pause", "silence", "—"]:
        duration = 0.8 if tag_clean in ["lbreak", "—"] else 0.4
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
        
    # Laughter
    elif tag_clean in ["laughter", "laughs", "giggles", "chuckles", "laugh"]:
        base_pitch = 220.0 if gender == "female" else 130.0
        t = np.linspace(0, 0.7, int(sample_rate * 0.7), False)
        # 5 chuckles
        env = (np.sin(2 * np.pi * 7.0 * t) ** 2) * np.exp(-t * 2.0)
        freq = base_pitch + 50 * np.sin(2 * np.pi * 7.0 * t)
        chuckle = np.sin(2 * np.pi * freq * t) * env
        noise = np.random.normal(0, 0.08, len(t)) * env
        return (chuckle * 0.5 + noise * 0.5) * 0.3
        
    # Sigh / Breath
    elif tag_clean in ["sighs", "sigh", "breath", "whispers", "whispering", "gasp", "gasps"]:
        duration = 0.5 if tag_clean in ["gasp", "gasps"] else 0.8
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        noise = np.random.normal(0, 0.12, len(t))
        env = np.sin(np.pi * t / duration) ** 2
        return noise * env * 0.25
        
    # Cough
    elif tag_clean in ["coughs", "cough"]:
        t = np.linspace(0, 0.4, int(sample_rate * 0.4), False)
        noise = np.random.normal(0, 0.15, len(t))
        env = np.zeros_like(t)
        burst1_len = int(sample_rate * 0.12)
        burst2_start = int(sample_rate * 0.18)
        burst2_end = int(sample_rate * 0.3)
        env[0:burst1_len] = np.sin(np.pi * t[0:burst1_len] / 0.12)
        env[burst2_start:burst2_end] = np.sin(np.pi * (t[burst2_start:burst2_end] - 0.18) / 0.12) * 0.6
        return noise * env * 0.25
        
    # Clears throat
    elif tag_clean in ["clears throat", "throat"]:
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
        drone = np.sin(2 * np.pi * 90 * t) + np.sin(2 * np.pi * 130 * t)
        noise = np.random.normal(0, 0.2, len(t))
        env = np.sin(np.pi * t / 0.5) ** 2
        return (drone * 0.3 + noise * 0.7) * env * 0.2
        
    # Music
    elif tag_clean in ["music"]:
        t = np.linspace(0, 1.2, int(sample_rate * 1.2), False)
        freqs = [261.63, 329.63, 392.00, 523.25]  # C major arpeggio
        audio = np.zeros_like(t)
        for i, f in enumerate(freqs):
            start_t = i * 0.15
            slice_mask = t >= start_t
            slice_t = t[slice_mask] - start_t
            env = np.exp(-slice_t * 4.0)
            audio[slice_mask] += np.sin(2 * np.pi * f * slice_t) * env * 0.15
        return audio
        
    return None

def apply_whisper_filter(audio_data: np.ndarray) -> np.ndarray:
    alpha = 0.85
    whispered = audio_data - alpha * np.roll(audio_data, 1)
    whispered[0] = audio_data[0]
    noise = np.random.normal(0, 0.005, len(audio_data))
    return (whispered * 0.5 + noise) * 0.6

def get_simulator_vits():
    global _SIMULATOR_VITS_CACHE
    if _SIMULATOR_VITS_CACHE is None:
        try:
            from TTS.api import TTS
            logger.info("Simulator: Loading fallback VITS model...")
            # Load VITS model on CPU to ensure it runs anywhere
            _SIMULATOR_VITS_CACHE = TTS("tts_models/en/vctk/vits", gpu=False)
            logger.info("Simulator: Fallback VITS model loaded successfully.")
        except Exception as e:
            logger.error(f"Simulator: Failed to load VITS model: {e}")
    return _SIMULATOR_VITS_CACHE


class BaseTTSModel:
    """
    Abstract base class defining the standard interface for all pluggable TTS models.
    """
    def __init__(self):
        self.is_simulator = False

    def preload(self) -> bool:
        """
        Preload the model weights/pipelines.
        Should return True if loaded successfully, or False if there was an error.
        If it returns False, the model registry can mark it as running in Simulator Mode.
        """
        raise NotImplementedError

    def get_metadata(self) -> dict:
        """
        Return model metadata including:
        - name (str)
        - model (str)
        - requires_language (bool)
        - requires_speaker_wav (bool)
        - supported_languages (list)
        - supported_speakers (list)
        - presets (list)
        - tokens (list)
        - features (list of str: "tags", "instructions", "cloning", "streaming", "multi_speaker")
        - description (str)
        """
        raise NotImplementedError

    def prepare_text(self, text: str) -> str:
        """
        Modular text preprocessing.
        Applies general unicode normalization and handles tag filtering dynamically:
        - If the model supports the 'tags' feature, keeps only supported tokens.
        - Otherwise, strips all bracketed expressions [...] to avoid acoustic artifacts.
        - Strips parenthetical stage directions like (upbeat and friendly) so they are
          not read aloud (they are routed separately as voice_direction / instruct params).
        """
        if not text:
            return ""

        # Strip markdown symbols that shouldn't be read out loud
        text = text.replace("*", "").replace("_", "").replace("#", "")

        # General unicode cleaning
        text = unicodedata.normalize("NFKC", text)
        text = "".join(c for c in text if c.isprintable())
        
        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
        }
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        for bad, good in replacements.items():
            text = text.replace(bad, good)
            
        text = re.sub(r"\.\s+([A-Z])", r". \1", text)
        
        # Strip leading parenthetical voice direction cues (e.g., "(upbeat and friendly welcome) Hello!")
        text = re.sub(r"^\s*\([^)]+\)\s*", "", text)
        # Strip any remaining mid-text parenthetical stage directions (fully enclosed in parens)
        text = re.sub(r"\s*\([^)]*(?:pause|sigh|whisper|laugh|cough|gasp|clears?\s*throat|breath|music|sfx|transition|softly|gently|excitedly|sarcastically|nervously|cheerfully|angrily|sadly|dramatically|warmly|enthusiastically|upbeat|friendly|serious|calm|energetic)[^)]*\)\s*", " ", text, flags=re.IGNORECASE)
        
        metadata = self.get_metadata()
        features = metadata.get("features", [])
        
        if "tags" in features:
            # Get allowed tokens for this model (normalized to lowercase)
            allowed_tokens = {t.strip().lower() for t in metadata.get("tokens", [])}
            
            def token_replacer(m):
                token = m.group(0).strip().lower()
                return token if token in allowed_tokens else ""
                
            text = re.sub(r"\[[^\[\]]+\]", token_replacer, text)
        else:
            # Strip all bracketed tags
            text = re.sub(r"\[[^\[\]]+\]", "", text)
            
        return re.sub(r"\s+", " ", text).strip()

    def synthesize(self, job_id: str, text: str, output_path: str, progress_callback, **kwargs) -> str:
        """
        Execute speech synthesis.
        Should update progress using progress_callback(percentage, message) and write output to output_path.
        Must return the path to the generated WAV file, or raise an exception on failure.
        """
        raise NotImplementedError

    def run_simulation(self, job_id: str, text: str, output_path: str, progress_callback, **kwargs) -> str:
        """
        Standard fallback simulation mode.
        - If multi-speaker is requested, parses segments and merges VITS outputs.
        - If voice cloning is requested, runs the real XTTS v2 model as a clone fallback.
        - Otherwise, runs the VITS model, validating speaker IDs to avoid KeyError.
        """
        logger.warning(f"Job {job_id}: Running in SIMULATOR mode for {self.__class__.__name__}")
        progress_callback(10, "Simulator: Initializing simulated audio generator...")
        
        # Clean text
        clean_text = self.prepare_text(text)
        if not clean_text:
            clean_text = "Simulator mode has received empty text."

        # Check if model supports multi_speaker and text has speaker tags
        features = self.get_metadata().get("features", [])
        if "multi_speaker" in features and any(tag in text.lower() for tag in ["[speaker 1]", "[speaker 2]", "[speaker 3]", "[speaker 4]"]):
            progress_callback(15, "Simulator: Multi-speaker script detected. Parsing segments...")
            
            # Map VibeVoice speakers to configured VITS speakers or default distinct ones
            speaker_mapping = {
                "speaker_1": kwargs.get("speaker_1_voice") or "p225",
                "speaker_2": kwargs.get("speaker_2_voice") or "p226",
                "speaker_3": kwargs.get("speaker_3_voice") or "p227",
                "speaker_4": kwargs.get("speaker_4_voice") or "p228"
            }
            
            segments = []
            pattern = r"\[Speaker[\s_-]?([1-4])\]"
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            default_spk = kwargs.get("speaker") or "Speaker_1"
            
            if not matches:
                segments = [(default_spk, text)]
            else:
                first_start = matches[0].start()
                if first_start > 0:
                    initial_text = text[:first_start].strip()
                    if initial_text:
                        segments.append((default_spk, initial_text))
                for i, match in enumerate(matches):
                    spk_num = match.group(1)
                    spk_name = f"Speaker_{spk_num}"
                    start = match.end()
                    end = matches[i+1].start() if i + 1 < len(matches) else len(text)
                    seg_text = text[start:end].strip()
                    if seg_text:
                        segments.append((spk_name, seg_text))
            
            if True: # Always attempt compilation if segments are found
                try:
                    import soundfile as sf
                    merged_audio = []
                    sample_rate = 24000
                    
                    for idx, (spk_name, seg_text) in enumerate(segments):
                        # 1. Resolve speaker voice mapping and allowed tokens
                        target_kwargs = {}
                        voice_val = speaker_mapping.get(spk_name.lower(), "p225")
                        curated_config = None
                        temp_wav_path = None
                        
                        if voice_val and voice_val.startswith("curated:"):
                            curated_id = voice_val.split(":", 1)[1]
                            curated_speaker_configs = kwargs.get("curated_speaker_configs") or {}
                            curated_config = curated_speaker_configs.get(curated_id)
                            if curated_config:
                                actual_model = curated_config.get("model", "vits")
                                actual_voice = curated_config.get("voice", "p225")
                                voice_val = f"{actual_model}:{actual_voice}"
                                
                                # Apply curated settings
                                settings = curated_config.get("settings", {})
                                for k, v in settings.items():
                                    target_kwargs[k] = v
                                    
                                # Decode reference audio
                                file_base64 = curated_config.get("fileBase64")
                                if file_base64:
                                    import base64
                                    try:
                                        if "," in file_base64:
                                            file_base64 = file_base64.split(",", 1)[1]
                                        wav_bytes = base64.b64decode(file_base64)
                                        temp_wav_path = os.path.join(os.path.dirname(output_path), f"temp_curated_{job_id}_{idx}.wav")
                                        with open(temp_wav_path, "wb") as f:
                                            f.write(wav_bytes)
                                        target_kwargs["speaker_wav"] = temp_wav_path
                                    except Exception as e:
                                        logger.error(f"Failed to decode curated reference audio: {e}")
                                        
                        allowed_tokens = []
                        if ":" in voice_val:
                            model_key = voice_val.split(":", 1)[0]
                            from models.registry import MODEL_REGISTRY
                            handler = MODEL_REGISTRY.get(model_key)
                            if handler:
                                allowed_tokens = handler.get_metadata().get("tokens", [])
                                
                        # 2. Filter lines using allowed_tokens
                        lines = [l.strip() for l in seg_text.split("\n") if l.strip()]
                        # If it's the initial text (before first speaker), allowed_tokens is empty, stripping all directions
                        valid_lines = [l for l in lines if not is_stage_direction(l, allowed_tokens if idx > 0 or not matches else [])]
                        if not valid_lines:
                            if temp_wav_path and os.path.exists(temp_wav_path):
                                try:
                                    os.remove(temp_wav_path)
                                except Exception:
                                    pass
                            continue
                        seg_text_clean = "\n\n".join(valid_lines).strip()
                        
                        # 3. Parse parenthetical emotional style cues
                        seg_direction = ""
                        style_match = re.match(r"^\(([^()]+)\)", seg_text_clean)
                        if style_match:
                            seg_direction = style_match.group(1).strip()
                            seg_text_clean = seg_text_clean[style_match.end():].strip()
                            
                        target_model = None
                        if seg_direction:
                            target_kwargs["voice_direction"] = seg_direction
                        # Pass down the general emotion intensity setting
                        if "emotion_intensity" not in target_kwargs:
                            target_kwargs["emotion_intensity"] = kwargs.get("emotion_intensity", 0.5)
                        
                        if ":" in voice_val:
                            parts = voice_val.split(":", 1)
                            model_key = parts[0]
                            voice_id = parts[1]
                            
                            from models.registry import MODEL_REGISTRY
                            handler = MODEL_REGISTRY.get(model_key)
                            if handler:
                                target_model = handler
                                if model_key == "bark":
                                    target_kwargs["preset"] = voice_id
                                    target_kwargs["voice_preset"] = voice_id
                                elif model_key in ["kokoro", "vits"]:
                                    target_kwargs["speaker"] = voice_id
                                else:
                                    target_kwargs["speaker"] = voice_id
                        else:
                            from models.registry import MODEL_REGISTRY
                            target_model = MODEL_REGISTRY.get("vits")
                            target_kwargs["speaker"] = voice_val
                            
                        if not target_model:
                            from models.registry import MODEL_REGISTRY
                            target_model = MODEL_REGISTRY.get("vits")
                            target_kwargs["speaker"] = "p225"

                        clean_seg_text = target_model.prepare_text(seg_text_clean)
                        if not clean_seg_text:
                            continue
                            
                        progress_callback(
                            int(30 + 60 * (idx / len(segments))),
                            f"Simulator: Synthesizing segment {idx+1}/{len(segments)} ({spk_name})..."
                        )
                            
                        temp_seg_path = os.path.join(os.path.dirname(output_path), f"temp_seg_{job_id}_{idx}.wav")
                        try:
                            # Synthesize using target model (either native or run_simulation fallback)
                            if target_model.is_simulator:
                                target_model.run_simulation(
                                    job_id=f"{job_id}_seg_{idx}",
                                    text=clean_seg_text,
                                    output_path=temp_seg_path,
                                    progress_callback=lambda p, m: None,
                                    **target_kwargs
                                )
                            else:
                                try:
                                    target_model.synthesize(
                                        job_id=f"{job_id}_seg_{idx}",
                                        text=clean_seg_text,
                                        output_path=temp_seg_path,
                                        progress_callback=lambda p, m: None,
                                        **target_kwargs
                                    )
                                except Exception:
                                    target_model.run_simulation(
                                        job_id=f"{job_id}_seg_{idx}",
                                        text=clean_seg_text,
                                        output_path=temp_seg_path,
                                        progress_callback=lambda p, m: None,
                                        **target_kwargs
                                    )
                                    
                            if os.path.exists(temp_seg_path):
                                data, sr = sf.read(temp_seg_path)
                                if sr != sample_rate:
                                    import librosa
                                    data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
                                merged_audio.append(data)
                                # Add a 0.3s pause between segments
                                pause = np.zeros(int(sample_rate * 0.3))
                                merged_audio.append(pause)
                        finally:
                            if os.path.exists(temp_seg_path):
                                try:
                                    os.remove(temp_seg_path)
                                except Exception:
                                    pass
                            if temp_wav_path and os.path.exists(temp_wav_path):
                                try:
                                    os.remove(temp_wav_path)
                                except Exception:
                                    pass
                                    
                    if merged_audio:
                        merged_audio.pop()  # remove the trailing pause
                        final_audio = np.concatenate(merged_audio)
                        sf.write(output_path, final_audio, sample_rate)
                        progress_callback(100, "Simulator Completed: Multi-speaker script synthesized successfully.")
                        return output_path
                except Exception as e:
                    logger.error(f"Simulator multi-speaker synthesis failed: {e}")

        # Check if reference audio is provided for voice cloning simulation
        speaker_wav = kwargs.get("speaker_wav")
        if speaker_wav and "cloning" in self.get_metadata().get("features", []):
            progress_callback(20, "Simulator: Zero-shot voice cloning requested. Loading XTTS v2 fallback...")
            try:
                from models.impl import get_cached_tts, resolve_torch_device
                device = kwargs.get("device", "auto")
                device_str = resolve_torch_device(device)
                
                # Get the XTTS v2 model from the cached TTS instances
                xtts_model = get_cached_tts("tts_models/multilingual/multi-dataset/xtts_v2", device_str)
                
                if xtts_model is not None:
                    progress_callback(50, "Simulator: Synthesizing voice clone via XTTS v2 fallback...")
                    # Save reference audio to a temporary file
                    temp_wav_path = os.path.join(os.path.dirname(output_path), f"temp_sim_ref_{job_id}.wav")
                    with open(temp_wav_path, "wb") as f:
                        f.write(speaker_wav)
                    
                    try:
                        xtts_model.tts_to_file(
                            text=clean_text,
                            speaker_wav=temp_wav_path,
                            language=kwargs.get("language") or "en",
                            file_path=output_path
                        )
                        progress_callback(100, f"Simulator Completed: Voice cloned successfully via XTTS v2 fallback. [Note: Install {self.get_metadata().get('model')} for native weights]")
                        return output_path
                    finally:
                        if os.path.exists(temp_wav_path):
                            try:
                                os.remove(temp_wav_path)
                            except Exception as err:
                                logger.warning(f"Failed to remove temp ref: {err}")
            except Exception as e:
                logger.error(f"Simulator voice cloning failed: {e}. Falling back to standard VITS...")

        progress_callback(35, "Simulator: Loading lightweight speech fallback...")
        vits_model = get_simulator_vits()
        
        if vits_model is not None:
            try:
                progress_callback(60, "Simulator: Synthesizing speech using VITS baseline...")
                
                # Check for voice style/direction instructions to dynamically simulate Qwen3 style
                voice_direction = (kwargs.get("voice_direction") or "").lower()
                speed_factor = float(kwargs.get("speed", 1.0))
                speaker_id = kwargs.get("speaker")
                emotion_intensity = float(kwargs.get("emotion_intensity", 0.5))
                
                # Apply speed factor adjustments based on style prompt
                if any(k in voice_direction for k in ["fast", "speedy", "rapid", "news", "anchor", "excited"]):
                    speed_factor *= 1.2
                elif any(k in voice_direction for k in ["slow", "drawl", "dramatic", "older", "old", "elderly"]):
                    speed_factor *= 0.75
                elif any(k in voice_direction for k in ["whisper", "soft", "quiet"]):
                    speed_factor *= 0.85
                
                # Dynamic speaker resolution based on voice_direction prompt
                vits_speakers = getattr(vits_model, "speakers", []) or []
                p_speakers = sorted([s for s in vits_speakers if isinstance(s, str) and s.startswith("p")])
                if not p_speakers:
                    p_speakers = sorted(list(set(MALE_SPEAKERS + FEMALE_SPEAKERS)))
                
                if voice_direction and p_speakers:
                    import hashlib
                    # Filter speaker pool based on gender keywords in the voice style instructions
                    female_kws = ["female", "woman", "girl", "lady", "mother", "grandma", "sister", "she", "her", "chinese lady"]
                    male_kws = ["male", "man", "boy", "guy", "gentleman", "father", "grandpa", "brother", "he", "his"]
                    
                    is_female = any(k in voice_direction for k in female_kws)
                    is_male = any(k in voice_direction for k in male_kws)
                    
                    if is_female:
                        chosen_pool = [s for s in FEMALE_SPEAKERS if s in p_speakers]
                    elif is_male:
                        chosen_pool = [s for s in MALE_SPEAKERS if s in p_speakers]
                    else:
                        chosen_pool = p_speakers
                        
                    if not chosen_pool:
                        chosen_pool = p_speakers
                        
                    # Deterministically index speaker based on prompt hash
                    prompt_hash = int(hashlib.md5(voice_direction.encode("utf-8")).hexdigest(), 16)
                    speaker_id = chosen_pool[prompt_hash % len(chosen_pool)]
                else:
                    # Apply VITS speaker mapping based on legacy style directions if no prompt hash needed
                    has_direction = False
                    if "excited" in voice_direction or "high pitch" in voice_direction or "happy" in voice_direction:
                        speaker_id = "p225"
                        has_direction = True
                    elif "deep" in voice_direction or "male" in voice_direction or "low pitch" in voice_direction or "authoritative" in voice_direction or "man" in voice_direction:
                        speaker_id = "p226"
                        has_direction = True
                    elif "soft" in voice_direction or "whisper" in voice_direction or "quiet" in voice_direction:
                        speaker_id = "p227"
                        has_direction = True
                    elif "news" in voice_direction or "anchor" in voice_direction or "broadcast" in voice_direction:
                        speaker_id = "p228"
                        has_direction = True
                        
                    if not has_direction:
                        model_name = self.get_metadata().get("model", "")
                        if model_name == "kokoro":
                            kokoro_mapping = {
                                "af_bella": "p229",
                                "af_nicole": "p230",
                                "af_sarah": "p231",
                                "am_adam": "p232",
                                "bf_emma": "p233",
                                "bf_isabella": "p234",
                                "bm_george": "p226",  # Map p226 as male speaker fallback
                                "bm_lewis": "p236"
                            }
                            speaker_id = kokoro_mapping.get(speaker_id, "p229")
                        elif model_name == "qwen3-tts":
                            speaker_id = "p237"
                        elif model_name == "chatterbox-turbo":
                            speaker_id = "p238"
                        elif model_name == "cosyvoice2-styletts2":
                            speaker_id = "p239"
                        elif model_name == "xtts_v2":
                            speaker_id = "p240"
                        elif model_name == "chattts":
                            speaker_id = "p250"
                            spk_seed = kwargs.get("chattts_spk_seed")
                            if spk_seed:
                                try:
                                    import hashlib
                                    seed_hash = int(hashlib.md5(str(spk_seed).encode("utf-8")).hexdigest(), 16)
                                    speaker_id = p_speakers[seed_hash % len(p_speakers)]
                                except Exception:
                                    pass
                        elif model_name == "fish-audio":
                            speaker_id = "p251"
                            fish_engine = kwargs.get("fish_engine") or "s2"
                            if fish_engine == "s2_1_pro":
                                logger.info("Simulator: Simulating premium Fish Audio S2.1 Pro Engine")
                                speed_factor *= 1.08
                
                # If speaker_id is not in the list of available VITS speakers, fallback to 'p225'
                vits_speakers = getattr(vits_model, "speakers", []) or []
                if not vits_speakers:
                    all_possible_speakers = set(MALE_SPEAKERS + FEMALE_SPEAKERS)
                    if not (isinstance(speaker_id, str) and speaker_id in all_possible_speakers):
                        speaker_id = "p225"
                else:
                    if not (isinstance(speaker_id, str) and speaker_id in vits_speakers):
                        speaker_id = "p225"

                logger.info(f"Simulator VITS resolved: speaker_id={speaker_id}, speed_factor={speed_factor} for voice_direction='{voice_direction}'")
                
                # Parse tags and synthesize segment-by-segment to simulate tags in VITS simulator
                tag_pattern = r"(\[[^\[\]]+\]|<[^<>]+>| — )"
                parts = re.split(tag_pattern, text)
                
                gender_str = "female" if speaker_id in FEMALE_SPEAKERS else "male"
                sample_rate = 24000
                audio_segments = []
                
                import soundfile as sf
                
                for idx, part in enumerate(parts):
                    if not part:
                        continue
                    
                    if re.match(tag_pattern, part):
                        tag_audio = generate_simulated_tag_audio(part, sample_rate, gender_str)
                        if tag_audio is not None:
                            audio_segments.append(tag_audio)
                    else:
                        part_clean = self.prepare_text(part)
                        if part_clean:
                            temp_wav = os.path.join(os.path.dirname(output_path), f"temp_sim_part_{job_id}_{idx}.wav")
                            try:
                                vits_model.tts_to_file(
                                    text=part_clean,
                                    speaker=speaker_id,
                                    speed=speed_factor,
                                    file_path=temp_wav
                                )
                                if os.path.exists(temp_wav):
                                    part_audio, sr = sf.read(temp_wav)
                                    if sr != sample_rate:
                                        import librosa
                                        part_audio = librosa.resample(part_audio, orig_sr=sr, target_sr=sample_rate)
                                    if len(part_audio.shape) > 1:
                                        part_audio = np.mean(part_audio, axis=1)
                                    audio_segments.append(part_audio)
                            finally:
                                if os.path.exists(temp_wav):
                                    try:
                                        os.remove(temp_wav)
                                    except Exception:
                                        pass
                
                if audio_segments:
                    merged_audio = np.concatenate(audio_segments)
                    
                    # Apply whisper filter
                    if any(k in voice_direction for k in ["whisper", "soft", "quiet"]):
                        merged_audio = apply_whisper_filter(merged_audio)
                        

                        
                    sf.write(output_path, merged_audio, sample_rate)
                else:
                    sf.write(output_path, np.zeros(sample_rate // 2), sample_rate)
                
                progress_callback(100, f"Simulator: Completed. [Note: Dependency packages for {self.get_metadata().get('model', 'model')} are not installed. Running in simulation mode]")
                return output_path
            except Exception as e:
                logger.error(f"Simulator VITS execution failed: {e}")
        
        # Absolute fallback: Generate synthetic sine wave if VITS fails/is not available
        progress_callback(80, "Simulator: Generating synthetic placeholder tone...")
        sample_rate = 24000
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate a dual tone (A440 and E660) to sound slightly robotic/audible
        tone = np.sin(2 * np.pi * 448 * t) + np.sin(2 * np.pi * 672 * t)
        # Normalize
        tone = tone / np.max(np.abs(tone))
        # Fade out
        fade_len = int(sample_rate * 0.1)
        fade_out = np.linspace(1, 0, fade_len)
        tone[-fade_len:] *= fade_out
        
        write_wav(output_path, sample_rate, (tone * 32767).astype(np.int16))
        progress_callback(100, f"Simulator: Completed absolute fallback tone. [Install packages for {self.get_metadata().get('model', 'model')} to use the real engine]")
        return output_path
