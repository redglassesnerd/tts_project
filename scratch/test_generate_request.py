import requests
import time
import json

def test_generate():
    url = "http://localhost:5000/generate"
    payload = {
        "text": "Hello, this is a test of Kokoro simulation.",
        "model": "kokoro",
        "speed": 1.0,
        "language": "en",
        "speaker": "af_bella"
    }
    
    print("Sending POST /generate...")
    res = requests.post(url, json=payload)
    print("Response status:", res.status_code)
    print("Response JSON:", res.json())
    
    job_id = res.json().get("job_id")
    if not job_id:
        print("No job_id returned.")
        return
        
    status_url = f"http://localhost:5000/status/{job_id}"
    while True:
        res_status = requests.get(status_url)
        status_data = res_status.json()
        print("Job Status:", status_data)
        if status_data.get("status") in ["done", "error", "cancelled"]:
            break
        time.sleep(1)

if __name__ == "__main__":
    test_generate()
