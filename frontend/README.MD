# Coqui TTS Project

A simple Flask + React application for text-to-speech using [Coqui TTS](https://github.com/coqui-ai/TTS).

## Requirements
- llama‑cpp‑python (only if you want fully‑offline punctuation/LLM preprocessing)
- Python 3.9+ (ideally 3.10)
- Node.js 16+ / npm or Yarn

## Backend Setup
1. `cd backend`
2. `python -m venv venv`
3. `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. `pip install -r requirements.txt`
5.  pip install llama-cpp-python diskcache   # enables the small local LLM used for smart punctuation & prompts
6. `python app.py`
   - Runs on [http://localhost:5000](http://localhost:5000)

### Local LLM (optional but recommended)
1. Download a GGML/GGUF model — the 23 MB English‑only tiny model works great:

   `curl -L -o backend/llama.cpp/models/ggml-tiny.en.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`

   (create the `backend/llama.cpp/models` folder if it doesn’t exist).  
   You can use any other GGML / GGUF model; just adjust the filename.

2. **OR** put the model anywhere else and set  
   `export LLAMA_MODEL_PATH=/full/path/to/your_model.bin`

Nothing else to run – `python app.py` will automatically spin up the LLM in‑process when you make a request that needs it.

## Frontend Setup
1. `cd frontend`
2. `npm install`
3. `npm run dev`
   - Runs on [http://localhost:5173](http://localhost:5173)

Adjust `.env` or config if you want to change the port. 

## Quick start

```bash
# one‑time
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python==0.2.58 diskcache
# You may need to ensure compatible Python (3.10 recommended) and install other packages manually if missing.
# If running into model loading errors due to torch.load(), make sure Bark models are trusted or adjust torch.load settings manually.
curl -L -o llama.cpp/models/ggml-tiny.en.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
python app.py  # starts Flask + LLM

# new terminal
cd frontend && npm install && npm run dev  # opens http://localhost:5173
```

## Troubleshooting

- **`ModuleNotFoundError`**: If you see errors like `ModuleNotFoundError: No module named 'bark'`, make sure all required packages are installed and you’re in the right Python environment.
- **`ImportError: cannot import name 'url_quote'`**: Downgrade `werkzeug` to a compatible version:  
  `pip install werkzeug==2.3.7`
- **Torch `UnpicklingError`**: If using PyTorch 2.6+, model loading may fail due to stricter defaults. Update model loading to use `torch.load(..., weights_only=False)` if you're confident the checkpoint is safe.
- **Slow or failing Bark inference**: If you’re not using a GPU, Bark inference may be very slow or hit memory errors. Reduce chunk size or upgrade to a system with GPU support.
- **Python version issues**: Many packages (like TTS) are pinned to specific Python versions. Python 3.10 is recommended. Using `pyenv` can help manage versions cleanly.