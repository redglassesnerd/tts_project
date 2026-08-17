import os
import urllib.request
import wave
import struct
import math

def generate_mock_wav(path, frequency=220, duration=5.0, sample_rate=22050):
    """Generate a clean sine wave mock reference audio file."""
    num_samples = int(duration * sample_rate)
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(16383.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

def download_file(url, path):
    try:
        print(f"Downloading testing asset from {url} to {path}...")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Failed to download asset: {e}")
        return False

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(backend_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Public domain Elvis Presley 1956 Interview ogg
    elvis_url = "https://upload.wikimedia.org/wikipedia/commons/e/e1/Elvis_Presley_interview_1956_Portland.ogg"
    elvis_path = os.path.join(output_dir, "elvis_reference.ogg")
    
    # John Lennon interview clip mp3
    lennon_url = "https://archive.org/download/JohnLennonInterview1971/JohnLennonInterview1971_56kb.mp3"
    lennon_path = os.path.join(output_dir, "lennon_reference.mp3")
    
    elvis_wav_path = os.path.join(output_dir, "elvis_reference.wav")
    lennon_wav_path = os.path.join(output_dir, "lennon_reference.wav")
    
    # Generate fallbacks first to ensure the user always has a working asset
    generate_mock_wav(elvis_wav_path, frequency=160, duration=5.0)  # Deeper baritone
    generate_mock_wav(lennon_wav_path, frequency=240, duration=5.0) # Mid-range tenor
    print(f"Generated mock voice reference files at: \n  {elvis_wav_path}\n  {lennon_wav_path}")
    
    # Attempt downloads
    download_file(elvis_url, elvis_path)
    download_file(lennon_url, lennon_path)

if __name__ == "__main__":
    main()
