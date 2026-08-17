import sys
import os
sys.path.append("/Users/E104158/Documents/localDev/tts_project-main/backend")

from models.registry import MODEL_REGISTRY

def test_cloning():
    handler = MODEL_REGISTRY.get("cosyvoice2-styletts2")
    MODEL_REGISTRY.preload_all()
    
    # Load mock speaker wav bytes
    ref_path = "/Users/E104158/.gemini/antigravity/brain/1aba119a-4e4f-4e08-ab33-e9336cb9d904/scratch/mock_speaker.wav"
    with open(ref_path, "rb") as f:
        speaker_wav_bytes = f.read()
        
    def progress_cb(pct, msg):
        print(f"[{pct}%] {msg}")
        
    output_path = "test_cloning_out.wav"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print("Calling run_simulation for CosyVoice with speaker_wav...")
    handler.run_simulation(
        job_id="test_clone_job",
        text="Hello, this is a cloned simulation voice.",
        output_path=output_path,
        progress_callback=progress_cb,
        speaker_wav=speaker_wav_bytes,
        device="cpu"
    )
    
    if os.path.exists(output_path):
        print("Success! Output wav created at:", output_path)
        os.remove(output_path)
    else:
        print("Failure: Output wav not created.")

if __name__ == "__main__":
    test_cloning()
