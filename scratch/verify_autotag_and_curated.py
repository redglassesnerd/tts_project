import requests
import base64
import os
import time
import sys

def test_autotag_endpoint():
    print("Testing /podcast/auto_tag endpoint...")
    
    # We define speaker configs mapping speaker labels to their allowed tags
    payload = {
        "script": "[Speaker 1] Hello! That was funny. [Speaker 2] Yes it was, but let's be quiet.",
        "speaker_configs": {
            "Speaker 1": ["laughter", "laughs"],
            "Speaker 2": []  # empty allowed tags means no tags should be inserted for Speaker 2
        }
    }
    
    res = requests.post("http://localhost:5000/podcast/auto_tag", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    
    data = res.json()
    script = data.get("script", "")
    print(f"Original script: {payload['script']}")
    print(f"Auto-tagged script: {script}")
    
    assert "[Speaker 1]" in script, "Speaker 1 tag was lost"
    assert "[Speaker 2]" in script, "Speaker 2 tag was lost"
    
    # Check that Speaker 2 has no tags added
    speaker2_part = script.split("[Speaker 2]")[-1]
    assert "[" not in speaker2_part, "Speaker 2 should not have any emotional tags inserted"
    print("✓ Auto-tag endpoint verification: PASSED")

def test_curated_voice_generation():
    print("\nTesting curated voice decoding and multi-speaker generation...")
    
    # Path to the mock speaker wav artifact
    mock_wav_path = "/Users/E104158/.gemini/antigravity/brain/1aba119a-4e4f-4e08-ab33-e9336cb9d904/scratch/mock_speaker.wav"
    if not os.path.exists(mock_wav_path):
        print(f"Mock wav file not found at {mock_wav_path}. Creating a dummy wav bytes content...")
        # Simple dummy 44-byte WAV header + mock content
        wav_bytes = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00"
    else:
        with open(mock_wav_path, "rb") as f:
            wav_bytes = f.read()
            
    encoded_bytes = base64.b64encode(wav_bytes).decode("utf-8")
    
    payload = {
        "model": "vibevoice",
        "text": "[Speaker 1] This is the first speaker using a curated voice profile. [Speaker 2] And this is Speaker 2.",
        "speaker_1_voice": "curated:test_curated_voice_1",
        "speaker_2_voice": "tts_models/en/vctk/vits:p225",
        "curated_speaker_configs": {
            "test_curated_voice_1": {
                "model": "chattts",
                "voice": "default",
                "settings": {
                    "chattts_spk_seed": 42,
                    "chattts_refine_text": True
                },
                "fileBase64": encoded_bytes
            }
        }
    }
    
    res = requests.post("http://localhost:5000/generate", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    
    job_info = res.json()
    job_id = job_info.get("job_id")
    print(f"Queued multi-speaker job with curated voice: {job_id}")
    
    completed = False
    for i in range(20):
        status_res = requests.get(f"http://localhost:5000/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        status = status_data.get("status")
        progress = status_data.get("progress")
        msg = status_data.get("message")
        print(f"Status: {status} | Progress: {progress}% | Message: {msg}")
        
        if status == "done":
            completed = True
            break
        elif status == "error":
            print(f"Job failed: {msg}")
            break
        time.sleep(1)
        
    assert completed, "Multi-speaker synthesis with curated voice failed to complete successfully!"
    print("✓ Curated voice synthesis verification: PASSED")
    
    # Check audio file downloads successfully
    audio_res = requests.get(f"http://localhost:5000/audio/{job_id}")
    assert audio_res.status_code == 200
    print(f"✓ Curated audio generated successfully. Size: {len(audio_res.content)} bytes.")

if __name__ == "__main__":
    try:
        test_autotag_endpoint()
        test_curated_voice_generation()
        print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
