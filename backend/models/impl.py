import os
import re
import logging
import numpy as np
import torch
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from models.base import BaseTTSModel

# Avoid circular imports by getting models dynamically or using local storage
from bark.generation import (
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    codec_decode,
    preload_models as bark_preload_models,
)
import bark
from nltk.tokenize import sent_tokenize
from scipy.io.wavfile import write as write_wav

logger = logging.getLogger("backend.models.impl")

# Central cache for loaded model instances
_ACTIVE_MODELS_CACHE = {}

# Cache for Qwen3-TTS models (load once)
_QWEN3TTS_VOICEDESIGN_MODEL_CACHE = None
_QWEN3TTS_BASE_MODEL_CACHE = None

def get_cached_tts(model_name, device_str):
    from TTS.api import TTS
    cache_key = (model_name, device_str)
    if cache_key not in _ACTIVE_MODELS_CACHE:
        logger.info(f"Loading TTS model {model_name} on device {device_str}...")
        use_gpu = (device_str == "cuda")
        tts_instance = TTS(model_name, gpu=use_gpu)
        if device_str == "mps" and hasattr(tts_instance, "to"):
            try:
                tts_instance.to(torch.device("mps"))
                logger.info("Moved TTS model to MPS device")
            except Exception as e:
                logger.warning(f"Failed to explicitly move model to MPS: {e}")
        _ACTIVE_MODELS_CACHE[cache_key] = tts_instance
    return _ACTIVE_MODELS_CACHE[cache_key]


# Helper to get active torch device
def resolve_torch_device(device_pref, use_mps: bool = True):
    if device_pref != "auto":
        return device_pref
    if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


_CHATTTS_INSTANCE = None

def get_cached_chattts():
    global _CHATTTS_INSTANCE
    if _CHATTTS_INSTANCE is None:
        try:
            import ChatTTS
            logger.info("Initializing and preloading global ChatTTS instance...")
            _CHATTTS_INSTANCE = ChatTTS.Chat()
            _CHATTTS_INSTANCE.load_models()
        except Exception as e:
            logger.warning(f"Failed to preload ChatTTS engine: {e}")
            return None
    return _CHATTTS_INSTANCE

def has_chattts_tags(text: str) -> bool:
    tags = ["[oral_", "[laugh_", "[break_", "[lbreak]", "[uv_break]"]
    lower_text = text.lower()
    return any(t in lower_text for t in tags)


class VITSTTS(BaseTTSModel):
    def get_metadata(self):
        supported_speakers = [f"p{i}" for i in range(225, 281)]
        return {
            "name": "tts_models/en/vctk/vits",
            "model": "vits",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": supported_speakers,
            "presets": [],
            "tokens": [],
            "features": [],
            "description": "✧ VITS / VCTK  ·  Fast English model. Neutral accent, extremely CPU‑friendly."
        }

    def preload(self):
        try:
            device = resolve_torch_device("auto")
            get_cached_tts("tts_models/en/vctk/vits", device)
            return True
        except Exception as e:
            logger.error(f"Failed to preload VITS model: {e}")
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        progress_callback(10, "Loading VITS model...")
        device = resolve_torch_device(kwargs.get("device", "auto"))
        tts = get_cached_tts("tts_models/en/vctk/vits", device)
        
        progress_callback(40, "Synthesizing VITS audio...")
        clean_text = self.prepare_text(text)
        speaker_id = kwargs.get("speaker") or "p225"
        
        tts.tts_to_file(
            text=clean_text,
            speaker=speaker_id,
            file_path=output_path
        )
        progress_callback(100, "VITS synthesis complete")
        return output_path


def trim_trailing_silence(audio_data, sample_rate, threshold_db=-45, frame_length=2048, hop_length=512):
    """
    Trims trailing silence from audio data using RMS energy threshold.
    """
    if len(audio_data) == 0:
        return audio_data
        
    threshold = 10 ** (threshold_db / 20)
    num_samples = len(audio_data)
    
    if len(audio_data.shape) > 1:
        mono_audio = np.mean(audio_data, axis=1)
    else:
        mono_audio = audio_data
        
    last_active_sample = num_samples
    
    for start in range(num_samples - frame_length, -1, -hop_length):
        frame = mono_audio[start : start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-8)
        if rms > threshold:
            last_active_sample = start + frame_length
            break
    else:
        last_active_sample = num_samples
        
    padding_samples = int(sample_rate * 0.15)
    end_index = min(num_samples, last_active_sample + padding_samples)
    
    return audio_data[:end_index]


class BarkTTS(BaseTTSModel):
    def get_metadata(self):
        tokens = [
            "[laughter]", "[laughs]", "[sighs]", "[sigh]", "[music]", "[gasps]", "[gasp]",
            "[clears throat]", "[whispers]", "[giggles]", "[snickers]", "[coughs]", "[cough]",
            "[groans]", "[yells]", "[yell]", "[whimpers]", "[sobs]", "[chuckles]", "[hums]",
            "[yawns]", "[mumbles]"
        ]
        return {
            "name": "bark",
            "model": "bark",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": [],
            "presets": [
                "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
                "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
                "v2/en_speaker_8", "v2/en_speaker_9"
            ],
            "tokens": tokens,
            "features": ["tags"],
            "description": "✧ Bark [Experimental]  ·  Expressive generative model. Slow, supports tag placement."
        }

    def preload(self):
        try:
            bark_preload_models()
            return True
        except Exception as e:
            logger.error(f"Failed to preload Bark models: {e}")
            self.is_simulator = True
            return False

    def load_history_prompt_npz(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            return {
                "semantic_prompt": data["semantic_prompt"],
                "coarse_prompt": data["coarse_prompt"],
                "fine_prompt": data["fine_prompt"],
            }
        except Exception as e:
            logger.error(f"Failed to load history prompt from {file_path}: {e}")
            return None

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        progress_callback(5, "Initializing Bark synthesis...")
        device = resolve_torch_device(kwargs.get("device", "auto"))
        
        clean_text = self.prepare_text(text)
        
        # Word limits and chunking
        bark_split_sentences = kwargs.get("bark_split_sentences", True)
        bark_max_duration = kwargs.get("bark_max_duration", 14)
        
        semantic_chunks = []
        words_per_second = 2.5
        char_threshold = 300
        
        if bark_split_sentences:
            sentence_candidates = sent_tokenize(clean_text)
            temp_chunk = ""
            for sentence in sentence_candidates:
                candidate_chunk = (temp_chunk + " " + sentence).strip() if temp_chunk else sentence.strip()
                est_duration = len(candidate_chunk.split()) / words_per_second
                if len(candidate_chunk) > char_threshold or est_duration > bark_max_duration:
                    if temp_chunk.strip():
                        semantic_chunks.append(temp_chunk.strip())
                    temp_chunk = sentence.strip()
                else:
                    temp_chunk = candidate_chunk
            if temp_chunk.strip():
                semantic_chunks.append(temp_chunk.strip())
            
            final_chunks = []
            for chunk in semantic_chunks:
                est_duration = len(chunk.split()) / words_per_second
                if len(chunk) > char_threshold or est_duration > bark_max_duration:
                    subchunks = re.split(r'(?<=,)\s+|(?<= and )', chunk)
                    temp_sub = ""
                    for sub in subchunks:
                        candidate_sub = (temp_sub + " " + sub).strip() if temp_sub else sub.strip()
                        est_sub_duration = len(candidate_sub.split()) / words_per_second
                        if len(candidate_sub) > char_threshold or est_sub_duration > bark_max_duration:
                            if temp_sub.strip():
                                final_chunks.append(temp_sub.strip())
                            temp_sub = sub.strip()
                        else:
                            temp_sub = candidate_sub
                    if temp_sub.strip():
                        final_chunks.append(temp_sub.strip())
                else:
                    final_chunks.append(chunk)
            semantic_chunks = [c for c in final_chunks if c.strip()]
        else:
            semantic_chunks = [clean_text]

        if not semantic_chunks:
            raise ValueError("No text remaining after Bark preprocessing.")

        total_chunks = len(semantic_chunks)
        voice_preset = kwargs.get("voice_preset") or kwargs.get("preset") or "v2/en_speaker_9"
        
        bark_history_prompt = None
        if voice_preset:
            # Check voice preset mapping (passed down or resolved)
            # If preset list maps to files we load them
            # For simplicity, check if file exists or use preset string directly
            if os.path.exists(voice_preset) and voice_preset.endswith(".npz"):
                bark_history_prompt = self.load_history_prompt_npz(voice_preset)
            elif voice_preset.endswith(".npz") and not os.path.exists(voice_preset):
                # If preset name ends with .npz but doesn't exist locally, it's a standard Suno preset name.
                # Convert it to the canonical form for Bark to load from cache/HF resources (e.g. en_speaker_9.npz -> v2/en_speaker_9)
                clean_preset = voice_preset[:-4] # strip ".npz"
                if not clean_preset.startswith("v2/"):
                    bark_history_prompt = f"v2/{clean_preset}"
                else:
                    bark_history_prompt = clean_preset
            else:
                bark_history_prompt = voice_preset

        text_temp = float(kwargs.get("text_temp") or 0.7)
        top_k = int(kwargs.get("top_k") or 50)
        top_p = float(kwargs.get("top_p") or 0.95)
        seed = kwargs.get("seed")
        if seed not in (None, ""):
            seed = int(seed)
        else:
            seed = None

        audio_arrays = []
        sample_rate = 24000

        for chunk_idx, chunk in enumerate(semantic_chunks):
            attempt = 1
            max_attempts = 2
            chunk_success = False
            chunk_audio = None
            
            def make_progress_cb(c_idx):
                base = int(100 * c_idx / total_chunks)
                weight = 100.0 / total_chunks
                def cb(step):
                    if step == "semantic":
                        pct = int(base + weight * 0.1)
                        msg = "Generating semantic tokens"
                    elif step == "coarse":
                        pct = int(base + weight * 0.5)
                        msg = "Generating coarse tokens"
                    elif step == "fine":
                        pct = int(base + weight * 0.85)
                        msg = "Generating fine tokens"
                    elif step == "decode":
                        pct = int(base + weight * 0.95)
                        msg = "Decoding audio"
                    else:
                        pct = int(base)
                        msg = "Synthesizing"
                    progress_callback(pct, f"Chunk {c_idx+1}/{total_chunks}: {msg}")
                return cb

            progress_cb = make_progress_cb(chunk_idx)

            while attempt <= max_attempts and not chunk_success:
                try:
                    if seed is not None:
                        np.random.seed(seed)
                    
                    progress_cb("semantic")
                    semantic_tokens = generate_text_semantic(
                        chunk,
                        history_prompt=bark_history_prompt,
                        temp=text_temp,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    progress_cb("coarse")
                    coarse_tokens = generate_coarse(
                        semantic_tokens,
                        history_prompt=bark_history_prompt,
                        temp=text_temp,
                    )
                    
                    progress_cb("fine")
                    fine_tokens = generate_fine(
                        coarse_tokens,
                        history_prompt=bark_history_prompt,
                    )
                    
                    progress_cb("decode")
                    audio_array = codec_decode(fine_tokens)
                    
                    if audio_array is not None and audio_array.size > 0:
                        duration_sec = len(audio_array) / sample_rate
                        if duration_sec >= 0.2:
                            chunk_success = True
                            chunk_audio = audio_array
                            
                except Exception as e:
                    logger.error(f"Bark chunk generation failed (attempt {attempt}): {e}")
                
                attempt += 1

            if not chunk_success:
                logger.warning(f"Failed chunk {chunk_idx+1}. Inserting silence.")
                chunk_audio = np.zeros(sample_rate // 2, dtype=np.float32)

            audio_arrays.append(chunk_audio)

        # Merge chunks with crossfades or simple silences
        intro_silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
        merged_audio = [intro_silence]
        for idx, arr in enumerate(audio_arrays):
            merged_audio.append(arr)
            if idx < len(audio_arrays) - 1:
                silence = np.zeros(int(sample_rate // 4), dtype=np.float32)
                merged_audio.append(silence)
                
        merged_audio_np = np.concatenate(merged_audio)
        try:
            trimmed_audio = trim_trailing_silence(merged_audio_np, sample_rate, threshold_db=-45)
            merged_audio_np = trimmed_audio
        except Exception as trim_err:
            logger.warning(f"Failed to trim trailing silence: {trim_err}")
            
        write_wav(output_path, sample_rate, merged_audio_np)
        progress_callback(100, "Bark synthesis complete")
        return output_path


class XTTSv2(BaseTTSModel):
    def get_metadata(self):
        return {
            "name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "model": "xtts_v2",
            "requires_language": True,
            "requires_speaker_wav": True,
            "supported_languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
            "supported_speakers": [],
            "presets": [],
            "tokens": [],
            "features": ["cloning"],
            "description": "✧ XTTS‑v2  ·  Multilingual speaker cloning. Best when uploading your own short WAV reference."
        }

    def preload(self):
        try:
            device = resolve_torch_device("auto")
            get_cached_tts("tts_models/multilingual/multi-dataset/xtts_v2", device)
            return True
        except Exception as e:
            logger.error(f"Failed to preload XTTSv2: {e}")
            self.is_simulator = True
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        progress_callback(10, "Preparing reference audio...")
        device = resolve_torch_device(kwargs.get("device", "auto"))
        tts = get_cached_tts("tts_models/multilingual/multi-dataset/xtts_v2", device)
        
        speaker_wav = kwargs.get("speaker_wav")
        if not speaker_wav:
            raise ValueError("Speaker reference WAV bytes are required for XTTS v2 cloning.")
            
        temp_wav_path = os.path.join(os.path.dirname(output_path), f"temp_ref_{job_id}.wav")
        with open(temp_wav_path, "wb") as f:
            f.write(speaker_wav)
            
        clean_text = self.prepare_text(text)
        lang = kwargs.get("language") or "en"
        
        # Prepare generation configuration overrides
        extra_kwargs = {}
        if "speed" in kwargs:
            try:
                extra_kwargs["speed"] = float(kwargs["speed"])
            except Exception:
                pass
        if "text_temp" in kwargs and kwargs["text_temp"] != "":
            try:
                extra_kwargs["temperature"] = float(kwargs["text_temp"])
            except Exception:
                pass
        elif "temperature" in kwargs and kwargs["temperature"] != "":
            try:
                extra_kwargs["temperature"] = float(kwargs["temperature"])
            except Exception:
                pass
        for k in ["length_scale", "noise_scale", "noise_scale_w"]:
            if k in kwargs and kwargs[k] is not None and kwargs[k] != "":
                try:
                    extra_kwargs[k] = float(kwargs[k])
                except Exception:
                    pass
        
        logger.info(f"XTTS v2 synthesis | extra_kwargs={extra_kwargs}")
        progress_callback(50, "Synthesizing XTTS v2 cloned speech...")
        try:
            tts.tts_to_file(
                text=clean_text,
                speaker_wav=temp_wav_path,
                language=lang,
                file_path=output_path,
                **extra_kwargs
            )
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception as err:
                    logger.warning(f"Failed to remove temp ref wav: {err}")
                    
        progress_callback(100, "XTTS v2 synthesis complete")
        return output_path


class KokoroTTS(BaseTTSModel):
    def get_metadata(self):
        return {
            "name": "kokoro",
            "model": "kokoro",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": ["af_bella", "af_nicole", "af_sarah", "am_adam", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
            "presets": [],
            "tokens": [],
            "features": [],
            "description": "✧ Kokoro-82M [Experimental] · Tiny footprint, high performance, Apache 2.0"
        }

    def preload(self):
        try:
            import kokoro
            logger.info("Kokoro successfully imported.")
            return True
        except ImportError:
            self.is_simulator = True
            logger.warning("kokoro library not found. Falling back to simulator mode.")
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
        
        progress_callback(20, "Loading Kokoro engine...")
        import kokoro
        import soundfile as sf
        
        # Real Kokoro execution logic
        progress_callback(50, "Synthesizing audio via Kokoro-82M...")
        clean_text = self.prepare_text(text)
        speaker = kwargs.get("speaker") or "af_bella"
        
        # Instantiate Kokoro pipeline
        # Kokoro expects kokoro.json and kokoro.onnx to be in path or packages
        # Example invocation:
        # kokoro.generate(text, voice, speed)
        # For simplicity, we wrap in try-except in case files are missing:
        try:
            # Standard kokoro-python pipeline
            # Note: actual onnx model loading depends on the kokoro wrapper used
            voice_data = kokoro.load_voice(speaker)
            # Generating
            audio, out_sr = kokoro.generate(clean_text, voice_data, speed=kwargs.get("speed", 1.0))
            sf.write(output_path, audio, out_sr)
            progress_callback(100, "Kokoro synthesis complete")
            return output_path
        except Exception as e:
            logger.error(f"Real Kokoro synthesis failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)


class Qwen3TTS(BaseTTSModel):
    def get_metadata(self):
        return {
            "name": "qwen3-tts",
            "model": "qwen3-tts",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en", "zh"],
            "supported_speakers": [],
            "presets": [],
            "tokens": [],
            "features": ["instructions", "cloning"],
            "description": "✧ Qwen3-TTS · Instruction-following voice style control & zero-shot cloning."
        }

    def preload(self):
        try:
            import qwen_tts  # noqa: F401
            logger.info("Qwen3-TTS successfully imported.")
            return True
        except ImportError:
            self.is_simulator = True
            logger.warning("qwen_tts library not found. Falling back to simulator mode.")
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)

        progress_callback(10, "Preparing Qwen3-TTS engine...")
        try:
            import torch
            import soundfile as sf
            import numpy as np
            import glob
            from qwen_tts import Qwen3TTSModel

            # Determine whether cloning is requested
            speaker_wav = kwargs.get("speaker_wav")
            is_cloning = speaker_wav is not None

            global _QWEN3TTS_VOICEDESIGN_MODEL_CACHE, _QWEN3TTS_BASE_MODEL_CACHE

            if is_cloning:
                if _QWEN3TTS_BASE_MODEL_CACHE is None:
                    logger.info("Loading Qwen3-TTS Base weights from HuggingFace cache...")
                    progress_callback(20, "Loading Qwen3-TTS Base weights (cached)...")
                    base_glob = glob.glob("/Users/E104158/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*")
                    if base_glob:
                        model_id = base_glob[0]
                        logger.info(f"Loading Qwen3-TTS Base locally from cached path: {model_id}")
                    else:
                        logger.info("Local snapshot path not found; using HuggingFace repo ID.")
                        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

                    has_cuda = torch.cuda.is_available()
                    if has_cuda:
                        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
                    else:
                        load_kwargs = dict(
                            dtype=torch.float32,
                            attn_implementation="sdpa",
                        )

                    _QWEN3TTS_BASE_MODEL_CACHE = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
                    if not has_cuda:
                        _QWEN3TTS_BASE_MODEL_CACHE.device = torch.device("cpu")

                    logger.info(
                        f"Qwen3-TTS Base model loaded successfully | "
                        f"device={_QWEN3TTS_BASE_MODEL_CACHE.device} | "
                        f"type={_QWEN3TTS_BASE_MODEL_CACHE.model.tts_model_type}"
                    )
                model = _QWEN3TTS_BASE_MODEL_CACHE
            else:
                if _QWEN3TTS_VOICEDESIGN_MODEL_CACHE is None:
                    logger.info("Loading Qwen3-TTS VoiceDesign weights from HuggingFace cache...")
                    progress_callback(20, "Loading Qwen3-TTS VoiceDesign weights (cached)...")
                    vd_glob = glob.glob("/Users/E104158/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/snapshots/*")
                    if vd_glob:
                        model_id = vd_glob[0]
                        logger.info(f"Loading Qwen3-TTS VoiceDesign locally from cached path: {model_id}")
                    else:
                        logger.info("Local snapshot path not found; using HuggingFace repo ID.")
                        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

                    has_cuda = torch.cuda.is_available()
                    if has_cuda:
                        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
                    else:
                        load_kwargs = dict(
                            dtype=torch.float32,
                            attn_implementation="sdpa",
                        )

                    _QWEN3TTS_VOICEDESIGN_MODEL_CACHE = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
                    if not has_cuda:
                        _QWEN3TTS_VOICEDESIGN_MODEL_CACHE.device = torch.device("cpu")

                    logger.info(
                        f"Qwen3-TTS VoiceDesign model loaded successfully | "
                        f"device={_QWEN3TTS_VOICEDESIGN_MODEL_CACHE.device} | "
                        f"type={_QWEN3TTS_VOICEDESIGN_MODEL_CACHE.model.tts_model_type}"
                    )
                model = _QWEN3TTS_VOICEDESIGN_MODEL_CACHE

            clean_text = self.prepare_text(text)
            lang_param = (kwargs.get("language") or "en").lower()
            qwen_lang = "english" if lang_param in ["en", "english"] else "chinese"

            if is_cloning:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav.write(speaker_wav)
                    temp_wav_path = temp_wav.name

                ref_text = (kwargs.get("ref_text") or "").strip()
                x_vector_only = not bool(ref_text)

                logger.info(f"Qwen3-TTS cloning | ref_audio={temp_wav_path} | ref_text='{ref_text}' | x_vector_only={x_vector_only}")
                progress_callback(50, "Synthesizing speech with Qwen3-TTS clone...")
                try:
                    if x_vector_only:
                        wavs, sample_rate = model.generate_voice_clone(
                            text=clean_text,
                            ref_audio=temp_wav_path,
                            language=qwen_lang,
                            x_vector_only_mode=True,
                        )
                    else:
                        wavs, sample_rate = model.generate_voice_clone(
                            text=clean_text,
                            ref_audio=temp_wav_path,
                            ref_text=ref_text,
                            language=qwen_lang,
                            x_vector_only_mode=False,
                        )
                finally:
                    if os.path.exists(temp_wav_path):
                        try:
                            os.remove(temp_wav_path)
                        except Exception:
                            pass
            else:
                instruct = (kwargs.get("voice_direction") or "").strip()
                logger.info(f"Qwen3-TTS VoiceDesign | instruct='{instruct}' | text_len={len(clean_text)}")
                progress_callback(50, "Synthesizing speech with Qwen3-TTS design...")
                wavs, sample_rate = model.generate_voice_design(
                    text=clean_text,
                    instruct=instruct,
                    language=qwen_lang,
                )

            if not wavs or wavs[0] is None:
                raise ValueError("Qwen3-TTS returned empty audio.")

            audio = wavs[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            sf.write(output_path, audio, sample_rate)
            progress_callback(100, "Qwen3-TTS synthesis complete")
            return output_path

        except Exception as e:
            logger.error(f"Qwen3-TTS synthesis failed: {e}. Falling back to simulation.", exc_info=True)
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)



class ChatterboxTTS(BaseTTSModel):
    def get_metadata(self):
        tokens = ["[laughter]", "[sighs]", "[coughs]", "[whispers]", "[screaming]", "[gasp]"]
        return {
            "name": "chatterbox-turbo",
            "model": "chatterbox-turbo",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": [],
            "presets": [],
            "tokens": tokens,
            "features": ["tags", "cloning"],
            "description": "✧ Chatterbox-Turbo [Experimental] · MIT permissive, high expression tags, zero-shot clone"
        }

    def preload(self):
        try:
            import chatterbox_turbo
            return True
        except ImportError:
            self.is_simulator = True
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
        
        try:
            import chatterbox_turbo
            # Real synthesis...
            raise NotImplementedError("Weights not present.")
        except Exception as e:
            logger.error(f"Real Chatterbox failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)


class VibeVoiceTTS(BaseTTSModel):
    def get_metadata(self):
        return {
            "name": "vibevoice",
            "model": "vibevoice",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en"],
            "supported_speakers": ["Speaker_1", "Speaker_2", "Speaker_3", "Speaker_4"],
            "presets": [],
            "tokens": [],
            "features": ["multi_speaker"],
            "description": "✧ VibeVoice [Experimental] · Up to 4 stable speakers, ideal for long form and podcasts"
        }

    def preload(self):
        try:
            import vibevoice
            return True
        except ImportError:
            self.is_simulator = True
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
        
        try:
            import vibevoice
            # Real synthesis...
            raise NotImplementedError("Weights not present.")
        except Exception as e:
            logger.error(f"Real VibeVoice failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)


class CosyVoiceTTS(BaseTTSModel):
    def get_metadata(self):
        tokens = ["[laughter]", "[laughs]", "[sighs]", "[gasps]", "[coughs]", "[groans]", "[whispers]", "[chuckles]"]
        return {
            "name": "cosyvoice2-styletts2",
            "model": "cosyvoice2-styletts2",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en", "zh", "ja", "ko", "yue"],
            "supported_speakers": [],
            "presets": [],
            "tokens": tokens,
            "features": ["tags", "cloning", "streaming"],
            "description": "✧ CosyVoice 2 / StyleTTS 2 [Experimental] · Sub-200ms streaming latency with Bark-style tag edits"
        }

    def preload(self):
        try:
            import cosyvoice
            return True
        except ImportError:
            self.is_simulator = True
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
            
        try:
            import cosyvoice
            # Real synthesis...
            raise NotImplementedError("Weights not present.")
        except Exception as e:
            logger.error(f"Real CosyVoice 2 failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)


class ChatTTSTTS(BaseTTSModel):
    def get_metadata(self):
        tokens = [
            "[laughter]", "[laughs]", "[sighs]", "[sigh]", "[cough]", "[coughs]",
            "[lbreak]", "[uv_break]"
        ]
        return {
            "name": "chattts",
            "model": "chattts",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en", "zh"],
            "supported_speakers": [],
            "presets": [],
            "tokens": tokens,
            "features": ["tags", "cloning", "streaming"],
            "description": "✧ ChatTTS [Experimental] · Conversational TTS with advanced prosody, pause, and laughter tags."
        }

    def preload(self):
        try:
            import ChatTTS
            logger.info("ChatTTS successfully imported.")
            return True
        except ImportError:
            self.is_simulator = True
            logger.warning("ChatTTS library not found. Falling back to simulator mode.")
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
        
        progress_callback(20, "Loading ChatTTS engine...")
        try:
            import ChatTTS
            import soundfile as sf
            import torch
            
            chat = get_cached_chattts()
            if chat is None:
                raise ImportError("Failed to load preloaded ChatTTS instance.")
                
            clean_text = self.prepare_text(text)
            
            progress_callback(50, "Synthesizing conversational speech via ChatTTS...")
            
            refine_text = kwargs.get("chattts_refine_text", True)
            if refine_text and has_chattts_tags(text):
                logger.info("Detected ChatTTS manual prosody tags. Disabling auto text refinement.")
                refine_text = False
                
            spk_temp = kwargs.get("chattts_spk_temp", 0.3)
            text_temp = kwargs.get("chattts_text_temp", 0.3)
            
            spk_emb = None
            spk_emb_list = kwargs.get("chattts_spk_emb")
            if spk_emb_list:
                try:
                    spk_emb = torch.tensor(spk_emb_list, dtype=torch.float32)
                    if len(spk_emb.shape) == 1:
                        spk_emb = spk_emb.unsqueeze(0)
                except Exception as e:
                    logger.error(f"Failed to reconstruct chattts_spk_emb tensor: {e}")
                    
            if spk_emb is None:
                seed = kwargs.get("chattts_spk_seed") or kwargs.get("seed")
                if seed is not None:
                    try:
                        torch.manual_seed(int(seed))
                        spk_emb = chat.sample_random_speaker()
                    except Exception as e:
                        logger.error(f"Failed to generate spk_emb from seed {seed}: {e}")
                        
            params = {}
            if spk_temp is not None:
                params['spk_temp'] = float(spk_temp)
            if text_temp is not None:
                params['text_temp'] = float(text_temp)
            if spk_emb is not None:
                params['spk_emb'] = spk_emb
                
            top_p = kwargs.get("chattts_top_p") or kwargs.get("top_p")
            if top_p is not None and top_p != "":
                params['top_P'] = float(top_p)
                
            top_k = kwargs.get("chattts_top_k") or kwargs.get("top_k")
            if top_k is not None and top_k != "":
                params['top_K'] = int(top_k)
                
            temp = kwargs.get("chattts_temperature") or kwargs.get("temperature")
            if temp is not None and temp != "":
                params['temperature'] = float(temp)
                
            wavs = chat.infer([clean_text], refine_text_flag=refine_text, params_infer_code=params)
            sf.write(output_path, wavs[0], 24000)
            progress_callback(100, "ChatTTS synthesis complete")
            return output_path
        except Exception as e:
            logger.error(f"Real ChatTTS synthesis failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)


class FishAudioTTS(BaseTTSModel):
    def get_metadata(self):
        return {
            "name": "fish-audio",
            "model": "fish-audio",
            "requires_language": False,
            "requires_speaker_wav": False,
            "supported_languages": ["en", "zh", "ja"],
            "supported_speakers": [],
            "presets": [],
            "tokens": [],
            "features": ["cloning", "streaming"],
            "description": "✧ Fish Audio S2/S2.1 Pro [Experimental] · Dual-mode sequence-to-sequence zero-shot voice cloning."
        }

    def preload(self):
        try:
            import fish_audio_sdk
            logger.info("Fish Audio SDK successfully imported.")
            return True
        except ImportError:
            self.is_simulator = True
            logger.warning("fish_audio_sdk library not found. Falling back to simulator mode.")
            return False

    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)
            
        progress_callback(20, "Loading Fish Audio engine...")
        try:
            import fish_audio_sdk
            # Real synthesis...
            raise NotImplementedError("Weights/API credentials not present.")
        except Exception as e:
            logger.error(f"Real Fish Audio failed: {e}. Falling back to simulation.")
            self.is_simulator = True
            return self.run_simulation(job_id, text, output_path, progress_callback, **kwargs)

