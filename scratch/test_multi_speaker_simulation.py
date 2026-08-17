import sys
import os
sys.path.append("/Users/E104158/Documents/localDev/tts_project-main/backend")

from models.registry import MODEL_REGISTRY

def test_multi_speaker():
    handler = MODEL_REGISTRY.get("vibevoice")
    MODEL_REGISTRY.preload_all()
    
    script = "[Speaker 1] Hello and welcome to the future of speech synthesis. [Speaker 2] It sounds incredibly realistic, doesn't it? [Speaker 3] Yes, even when running in simulation fallback!"
    
    def progress_cb(pct, msg):
        print(f"[{pct}%] {msg}")
        
    output_path = "test_multi_spk_out.wav"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print("Synthesizing multi-speaker script in simulation mode...")
    handler.run_simulation(
        job_id="test_multi_spk_job",
        text=script,
        output_path=output_path,
        progress_callback=progress_cb,
        device="cpu"
    )
    
    if os.path.exists(output_path):
        import soundfile as sf
        data, sr = sf.read(output_path)
        duration = len(data) / sr
        print(f"Success! Output wav created at: {output_path} (Duration: {duration:.2f}s, SR: {sr}Hz)")
        os.remove(output_path)
    else:
        print("Failure: Output wav not created.")

if __name__ == "__main__":
    test_multi_speaker()
