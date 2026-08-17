import sys
sys.path.append("/Users/E104158/Documents/localDev/tts_project-main/backend")

from models.registry import MODEL_REGISTRY
import app

# Mock job dictionary
job = {
    "job_id": "test_job_id",
    "text": "Hello",
    "voice_name": "kokoro",
    "speed": 1.0,
    "pause_duration": 0.5,
    "language": "en",
    "speaker": "af_bella",
    "speaker_wav": None,
    "speaker_wav_name": None,
    "voice_preset": "",
    "text_temp": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "smart_enhance": False,
    "length_scale": 1.0,
    "noise_scale": 0.667,
    "noise_scale_w": 0.8,
    "seed": None,
    "bark_split_sentences": True,
    "bark_max_duration": 14,
    "device": "cpu"
}

job_params = {k: v for k, v in job.items() if k not in ["job_id", "output_path", "progress_callback"]}
print("job_params keys:", list(job_params.keys()))

handler = MODEL_REGISTRY.get("kokoro")
MODEL_REGISTRY.preload_all()
print("handler is_simulator after preload:", handler.is_simulator)

# Let's try calling synthesize directly like in app.py
def progress_cb(pct, msg):
    print(f"pct={pct}, msg={msg}")

try:
    print("Calling run_simulation directly...")
    handler.run_simulation(job_id=job["job_id"], output_path="test_out.wav", progress_callback=progress_cb, **job_params)
    print("Direct run_simulation completed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()


