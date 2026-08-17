import requests
import time
import sys

def verify_generate_endpoint():
    print("Testing /generate endpoint for CosyVoice 2 (Simulator mode)...")
    payload = {
        "model": "cosyvoice2-styletts2",
        "text": "Hello from the dynamic registry verification! [laughter] Let's see if this works.",
        "speed": 1.0,
        "barkSplitSentences": True,
        "barkMaxDuration": 14
    }
    
    res = requests.post("http://localhost:5000/generate", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    
    job_info = res.json()
    job_id = job_info.get("job_id")
    print(f"Queued generation job: {job_id}")
    
    # Poll status until done
    completed = False
    for i in range(15): # max 15 seconds
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
            print(f"Job failed with error message: {msg}")
            break
            
        time.sleep(1)
        
    assert completed, "Job did not complete successfully in time!"
    print("✓ Job finished with 'done'.")
    
    # Check that audio file is served
    audio_res = requests.get(f"http://localhost:5000/audio/{job_id}")
    assert audio_res.status_code == 200, f"Expected 200 for audio download, got {audio_res.status_code}"
    print(f"✓ Audio file served successfully. Size: {len(audio_res.content)} bytes.")

    print("\nTesting /generate endpoint for VITS (Native mode)...")
    payload_vits = {
        "model": "vits",
        "text": "This is a native VITS synthesis test to verify registry routing.",
        "speaker": "p225"
    }
    
    res_vits = requests.post("http://localhost:5000/generate", json=payload_vits)
    assert res_vits.status_code == 200
    job_id_vits = res_vits.json().get("job_id")
    print(f"Queued VITS job: {job_id_vits}")
    
    completed_vits = False
    for i in range(15):
        status_res = requests.get(f"http://localhost:5000/status/{job_id_vits}")
        status_data = status_res.json()
        status = status_data.get("status")
        progress = status_data.get("progress")
        msg = status_data.get("message")
        print(f"Status: {status} | Progress: {progress}% | Message: {msg}")
        
        if status == "done":
            completed_vits = True
            break
        elif status == "error":
            print(f"VITS Job failed: {msg}")
            break
        time.sleep(1)
        
    assert completed_vits, "VITS Job failed to complete successfully!"
    print("✓ VITS Job finished with 'done'.")
    
    audio_res_vits = requests.get(f"http://localhost:5000/audio/{job_id_vits}")
    assert audio_res_vits.status_code == 200
    print(f"✓ VITS Audio file served successfully. Size: {len(audio_res_vits.content)} bytes.")

if __name__ == "__main__":
    try:
        verify_generate_endpoint()
        print("\nAll Backend Generate Integration Tests Verified successfully!")
    except Exception as e:
        print(f"Failed verification: {e}")
        sys.exit(1)
