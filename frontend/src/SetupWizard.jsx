import React, { useState } from "react";
import axios from "axios";

export default function SetupWizard({ onComplete }) {
  const [step, setStep] = useState(1);
  const [device, setDevice] = useState("auto");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState("llama3.1:latest");
  const [outputFolder, setOutputFolder] = useState("output");
  
  // Ollama test connection state
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null); // { connected: bool, message: string, model_available: bool }

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await axios.post("http://localhost:5000/test_ollama", {
        ollama_url: ollamaUrl,
        ollama_model: ollamaModel,
      });
      setTestResult(res.data);
    } catch (err) {
      setTestResult({
        connected: false,
        message: "Failed to communicate with Flask backend setup diagnostic endpoint."
      });
    } finally {
      setTesting(false);
    }
  };

  const handleFinish = () => {
    const config = {
      device,
      ollama_url: ollamaUrl,
      ollama_model: ollamaModel,
      output_folder: outputFolder,
      setup_completed: true,
    };
    onComplete(config);
  };

  const steps = [
    { id: 1, name: "Welcome" },
    { id: 2, name: "Hardware" },
    { id: 3, name: "AI Enhance" },
    { id: 4, name: "Ready" }
  ];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-gray-900 bg-opacity-75 backdrop-blur-sm p-4 font-sans">
      <div className="bg-white dark:bg-slate-900 w-[600px] max-w-full rounded-2xl shadow-2xl overflow-hidden border border-gray-100 dark:border-slate-800 flex flex-col min-h-[480px] text-gray-900 dark:text-slate-100">
        
        {/* Wizard Header with Progress Steps */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-6 text-white text-center">
          <h2 className="text-xl font-bold tracking-tight">Voication Configuration Setup</h2>
          <p className="text-blue-100 text-xs mt-1">Configure your Text-to-Speech environment for optimal results</p>
          
          <div className="flex justify-center items-center mt-6 gap-2">
            {steps.map((s, idx) => (
              <React.Fragment key={s.id}>
                <div className="flex items-center">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold border transition-all duration-300 ${
                    step >= s.id 
                      ? "bg-white text-blue-700 border-white shadow" 
                      : "bg-blue-800 text-blue-300 border-blue-600"
                  }`}>
                    {s.id}
                  </div>
                  <span className={`text-xs ml-2 hidden sm:inline ${step === s.id ? "font-bold text-white" : "text-blue-200"}`}>
                    {s.name}
                  </span>
                </div>
                {idx < steps.length - 1 && (
                  <div className={`h-[2px] w-8 sm:w-12 transition-colors duration-300 ${
                    step > s.id ? "bg-white" : "bg-blue-800"
                  }`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Step Contents */}
        <div className="flex-1 p-8 overflow-y-auto">
          {step === 1 && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white">Welcome to Voication!</h3>
              <p className="text-gray-600 dark:text-slate-350 text-sm leading-relaxed">
                Voication is a premium studio environment combining generative voice synthesis (Bark) and high-fidelity text-to-speech pipelines (VITS, XTTS v2).
              </p>
              <div className="bg-blue-50 dark:bg-blue-950/20 border-l-4 border-blue-500 p-4 rounded-r-lg">
                <h4 className="text-xs font-bold text-blue-800 dark:text-blue-300 uppercase tracking-wider mb-1">Supported Technologies</h4>
                <ul className="text-xs text-blue-700 dark:text-blue-200 space-y-1 list-disc list-inside">
                  <li><strong>VITS (VCTK)</strong>: Lightweight, ultra-fast synthesis ideal for CPUs.</li>
                  <li><strong>XTTS v2</strong>: High-fidelity voice cloning using a 3-second audio sample.</li>
                  <li><strong>Bark [Experimental]</strong>: Artistic generation including laughter, sighs, and custom tags.</li>
                </ul>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Let's take a moment to configure your hardware and dependencies so you get clean, stutter-free output.
              </p>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white">Device & Hardware Acceleration</h3>
              <p className="text-gray-600 dark:text-slate-350 text-sm">
                Voice models require significant computation. Selecting the right backend accelerator ensures best performance.
              </p>

              <div className="space-y-3">
                <label className={`flex items-start border p-4 rounded-xl cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800/40 transition ${
                  device === "mps" ? "border-blue-500 bg-blue-50/10 dark:bg-blue-950/20" : "border-gray-200 dark:border-slate-800"
                }`}>
                  <input
                    type="radio"
                    name="device"
                    value="mps"
                    checked={device === "mps"}
                    onChange={() => setDevice("mps")}
                    className="mt-1 mr-3 h-4 w-4 text-blue-600"
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-850 dark:text-slate-100 flex items-center">
                      Apple Silicon (MPS)
                      <span className="ml-2 bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-400 dark:border dark:border-green-900/50 text-[10px] font-bold px-2 py-0.5 rounded-full">Recommended for Mac</span>
                    </h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Uses the unified GPU on M1, M2, or M3 chips. Speeds up XTTS v2 and Bark significantly.</p>
                  </div>
                </label>

                <label className={`flex items-start border p-4 rounded-xl cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800/40 transition ${
                  device === "cuda" ? "border-blue-500 bg-blue-50/10 dark:bg-blue-950/20" : "border-gray-200 dark:border-slate-800"
                }`}>
                  <input
                    type="radio"
                    name="device"
                    value="cuda"
                    checked={device === "cuda"}
                    onChange={() => setDevice("cuda")}
                    className="mt-1 mr-3 h-4 w-4 text-blue-600"
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-850 dark:text-slate-100">NVIDIA CUDA GPU</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Recommended if you run Windows or Linux with a dedicated NVIDIA graphic card.</p>
                  </div>
                </label>

                <label className={`flex items-start border p-4 rounded-xl cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800/40 transition ${
                  device === "cpu" ? "border-blue-500 bg-blue-50/10 dark:bg-blue-950/20" : "border-gray-200 dark:border-slate-800"
                }`}>
                  <input
                    type="radio"
                    name="device"
                    value="cpu"
                    checked={device === "cpu"}
                    onChange={() => setDevice("cpu")}
                    className="mt-1 mr-3 h-4 w-4 text-blue-600"
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-850 dark:text-slate-100">CPU Only (Safe Mode)</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Runs everything on the CPU. Safest compatibility, but slower for generative Bark models.</p>
                  </div>
                </label>

                <label className={`flex items-start border p-4 rounded-xl cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800/40 transition ${
                  device === "auto" ? "border-blue-500 bg-blue-50/10 dark:bg-blue-950/20" : "border-gray-200 dark:border-slate-800"
                }`}>
                  <input
                    type="radio"
                    name="device"
                    value="auto"
                    checked={device === "auto"}
                    onChange={() => setDevice("auto")}
                    className="mt-1 mr-3 h-4 w-4 text-blue-600"
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-850 dark:text-slate-100">Auto-Detect</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Automatically checks for available hardware acceleration at startup (recommended fallback).</p>
                  </div>
                </label>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white flex items-center">
                AI Text Preprocessing Settings
                <span className="ml-2 bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-400 dark:border dark:border-amber-900/50 text-[10px] font-bold px-2 py-0.5 rounded-full">Experimental</span>
              </h3>
              <p className="text-gray-600 dark:text-slate-350 text-sm">
                Voication uses a local LLM server (via Ollama) to automatically analyze narrative text and insert emotional tags (e.g. whispers, laughter) for Bark.
              </p>

              <div className="space-y-3 pt-2">
                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">Ollama API URL</label>
                  <input
                    type="text"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    className="w-full p-2 border border-gray-250 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                    placeholder="e.g. http://localhost:11434"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">Ollama Model</label>
                  <input
                    type="text"
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    className="w-full p-2 border border-gray-250 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                    placeholder="e.g. mistral, llama3, tinyllama"
                  />
                </div>

                <div className="pt-2 flex items-center gap-4">
                  <button
                    type="button"
                    onClick={handleTestConnection}
                    disabled={testing}
                    className="px-4 py-2 border border-blue-600 text-blue-600 dark:text-blue-450 hover:bg-blue-50 dark:hover:bg-blue-950/20 text-xs font-bold rounded transition disabled:opacity-50"
                  >
                    {testing ? "Testing Connection..." : "Test Connection"}
                  </button>

                  {testResult && (
                    <div className="flex-1 text-xs">
                      {testResult.connected ? (
                        testResult.model_available ? (
                          <span className="text-green-600 dark:text-green-400 font-medium">✓ Connection successful! Model ready.</span>
                        ) : (
                          <span className="text-amber-600 dark:text-amber-400 font-medium">⚠ Connection successful, but model '{ollamaModel}' not found. Download it in terminal using `ollama run {ollamaModel}`.</span>
                        )
                      ) : (
                        <span className="text-red-500 dark:text-red-400 font-medium">✗ Failed: {testResult.message || "Is Ollama running?"}</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white">Ready to Launch!</h3>
              <p className="text-gray-600 dark:text-slate-350 text-sm">
                Your configurations are complete. We've verified your settings and everything is ready.
              </p>

              <div className="bg-gray-50 dark:bg-slate-950 border border-gray-150 dark:border-slate-800 rounded-xl p-4 space-y-2 text-xs">
                <div className="flex justify-between border-b border-gray-100 dark:border-slate-800 pb-2">
                  <span className="font-semibold text-gray-600 dark:text-slate-400">Hardware Accelerator:</span>
                  <span className="text-gray-800 dark:text-slate-200 capitalize font-medium">{device}</span>
                </div>
                <div className="flex justify-between border-b border-gray-100 dark:border-slate-800 pb-2">
                  <span className="font-semibold text-gray-600 dark:text-slate-400">Ollama API URL:</span>
                  <span className="text-gray-800 dark:text-slate-200 font-medium">{ollamaUrl}</span>
                </div>
                <div className="flex justify-between border-b border-gray-100 dark:border-slate-800 pb-2">
                  <span className="font-semibold text-gray-600 dark:text-slate-400">Ollama model:</span>
                  <span className="text-gray-800 dark:text-slate-200 font-medium">{ollamaModel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-semibold text-gray-600 dark:text-slate-400">Audio output directory:</span>
                  <span className="text-gray-800 dark:text-slate-200 font-medium">backend/{outputFolder}</span>
                </div>
              </div>

              <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-normal">
                Note: At first generation, Coqui TTS or Bark will automatically download their model checkpoints (approx. 1-2GB) to your device. This will happen silently in the background and may take a few minutes depending on your internet connection.
              </p>
            </div>
          )}
        </div>

        {/* Wizard Footer Navigation */}
        <div className="bg-gray-50 dark:bg-slate-950 px-8 py-4 border-t border-gray-100 dark:border-slate-800 flex justify-between items-center">
          <button
            type="button"
            disabled={step === 1}
            onClick={() => setStep((s) => s - 1)}
            className="px-4 py-2 border border-gray-250 dark:border-slate-800 rounded-lg text-sm text-gray-650 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-850 transition disabled:opacity-50"
          >
            Back
          </button>

          {step < 4 ? (
            <button
              type="button"
              onClick={() => setStep((s) => s + 1)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 transition"
            >
              Continue
            </button>
          ) : (
            <button
              type="button"
              onClick={handleFinish}
              className="px-6 py-2 bg-green-600 text-white rounded-lg text-sm font-bold hover:bg-green-700 transition shadow-md"
            >
              Launch Studio
            </button>
          )}
        </div>

      </div>
    </div>
  );
}
