import React, { useEffect, useState, useRef } from "react";
import TagEditor from "./TagEditor";
import axios from "axios";
// VITS-specific fields are now moved into App function.
// --- Static Bark token list used for in‑editor autocomplete ---
// (Only used when the selected voice model name contains "bark")
const BARK_TOKENS = [
  "laughter",
  "whisper",
  "shout",
  "sigh",
  "gasp",
  "cry",
  "laughs",
  "breath",
  "cough",
  "music",
  "pause_short",
  "pause_long",
  "whispering",
  "excited",
  "angry",
  "sad",
  "surprised",
  "singing",
  "soft",
  "serious"
];

// --- MAIN COMPONENT ---
function AppInner({
  barkPresets,
  setBarkPresets,
}) {
  // VITS-specific fields (must be inside component)
  // Bark seed for reproducible variation
  const [seed, setSeed] = useState(424242);
  const [vitsSpeaker, setVitsSpeaker] = useState("");
  const [vitsNoiseScale, setVitsNoiseScale] = useState(0.667);
  const [vitsDurationScale, setVitsDurationScale] = useState(1.0);
  const [vitsUsePhonemes, setVitsUsePhonemes] = useState(false);
  const [text, setText] = useState("");
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [language, setLanguage] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [speakerWav, setSpeakerWav] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [chunkSize, setChunkSize] = useState(300);
  const [pauseDuration, setPauseDuration] = useState(0.5);

  // Bark advanced knobs
  const [barkTemperature, setBarkTemperature] = useState(0.7); // 0‑1 precise → creative
  const [barkTopK, setBarkTopK] = useState(50);               // 0‑100 smaller → larger vocab
  const [barkTopP, setBarkTopP] = useState(0.9);              // 0‑1 safe → diverse

  // XTTS tuning knobs (not used for XTTS v2)
  const [xttsLengthScale, setXttsLengthScale] = useState(1.0);
  const [xttsNoiseScale, setXttsNoiseScale] = useState(0.667);
  const [xttsNoiseScaleW, setXttsNoiseScaleW] = useState(0.8);

  const [voicePreset, setVoicePreset] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const [smartEnhance, setSmartEnhance] = useState(false); // offline LLM punctuation
  const [enhancePrompt, setEnhancePrompt] = useState("");
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [queue, setQueue] = useState([]);
  const [showQueue, setShowQueue] = useState(false);
  const [unreadCompletions, setUnreadCompletions] = useState(0);

  const chimeRef = useRef(new Audio("/chime.wav"));

  const selectedVoiceData = voices.find((v) => v.name === selectedVoice);

  // --- XTTS Voice Sample (Speaker Reference) recording state ---
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const formDataRef = useRef(new FormData());

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new window.MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        // Save the recording as a WebM blob, then convert to a File for backend transcoding
        const blob = new Blob(chunks, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        setRecordedBlob(file);
        setSpeakerWav(file);
        formDataRef.current.set("speaker_wav", file);
        // Stop all tracks after recording
        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      alert("Unable to access microphone.");
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const downloadRecording = () => {
    if (recordedBlob) {
      const url = URL.createObjectURL(recordedBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "speaker_reference.wav";
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  /* ---------- Dynamic capability discovery ---------- */
  const speakerList =
    (selectedVoiceData?.supported_speakers?.length
      ? selectedVoiceData.supported_speakers
      : selectedVoiceData?.speakers?.length
      ? selectedVoiceData.speakers
      : selectedVoiceData?.speaker_list?.length
      ? selectedVoiceData.speaker_list
      : selectedVoiceData?.speaker_ids?.length
      ? selectedVoiceData.speaker_ids
      : []);

  const presetList = selectedVoiceData?.presets || [];

// Resolve tokens list – static list for Bark, empty for others
  const tokensList = selectedVoiceData?.model?.toLowerCase().includes("bark")
    ? BARK_TOKENS
    : [];

  /* ---------- Load voices on mount ---------- */
  useEffect(() => {
    axios.get("http://localhost:5000/voices").then((res) => {
      // Sort so Bark model is first
      setVoices(
        res.data.voices.sort((a, b) => {
          if (a.model?.toLowerCase().includes("bark")) return -1;
          if (b.model?.toLowerCase().includes("bark")) return 1;
          return 0;
        })
      );
      if (res.data.voices.length > 0) {
        setSelectedVoice(res.data.voices[0].name);
      }
    });
  }, []);

  /* ---------- React to voice change ---------- */
  useEffect(() => {
    if (selectedVoiceData?.supported_languages?.length) {
      if (!selectedVoiceData.supported_languages.includes(language)) {
        setLanguage(selectedVoiceData.supported_languages[0]);
      }
    } else {
      setLanguage("");
    }

    setSpeaker("");
    setSpeakerWav(null);
    setVoicePreset("");
    setVoiceDescription(selectedVoiceData?.description || "");
  }, [selectedVoiceData]);

  useEffect(() => {
    if (speakerList.length > 1 && !speaker && !selectedVoiceData?.requires_speaker_wav) {
      setSpeaker(speakerList[0]);
    } else if (speakerList.length <= 1) {
      setSpeaker("");
    }
  }, [selectedVoiceData]);

  /* ---------- Chime when item finishes ---------- */
  const prevQueueRef = useRef([]);
  useEffect(() => {
    queue.forEach((item) => {
      const prev = prevQueueRef.current.find((q) => q.id === item.id);
      if (item.status === "done" && (!prev || prev.status !== "done")) {
        chimeRef.current.play();
        setUnreadCompletions((prevCount) => prevCount + 1);
      }
    });
    prevQueueRef.current = queue;
  }, [queue]);

  /* ---------- Helper ---------- */
  // Run local-LLM punctuation / style pass
  const runEnhancement = () => {
    if (!smartEnhance || !text.trim()) return;
    setIsEnhancing(true);
    axios
      .post("http://localhost:5000/enhance", {
        text,
        instruction: enhancePrompt,
      })
      .then((res) => {
        setText(res.data.enhanced_text);
      })
      .catch((err) => {
        console.error(err);
        alert("Local LLM enhancement failed.");
      })
      .finally(() => setIsEnhancing(false));
  };

  const allRequiredReady = () => {
    if (!text.trim()) return false;
    if (selectedVoiceData?.requires_speaker_wav && !formDataRef.current.has("speaker_wav")) return false;
    if (!selectedVoiceData?.requires_speaker_wav && speakerList.length && !speaker) return false;
    if (selectedVoiceData?.requires_language && !language) return false;
    if (presetList.length && !voicePreset) return false;
    return true;
  };

  /* ---------- Generate ---------- */
  const generateSpeech = () => {
    // --- Bark model: ensure a voice is selected, including index 0 ---
    if (
      selectedVoiceData?.model?.toLowerCase().includes("bark") &&
      (!selectedVoice && selectedVoice !== 0)
    ) {
      alert("Please select a voice for Bark.");
      return;
    }

    if (!allRequiredReady()) {
      alert("Fill in required fields first.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue((prev) => [...prev, { id: queueId, text, status: "queued", progress: 0 }]);

    const formData = new FormData();
    formData.append("text", text);
    formData.append("model", selectedVoice);
    formData.append("speed", speed);
    formData.append("chunk_size", chunkSize);
    formData.append("pause_duration", pauseDuration);
    formData.append("smart_enhance", smartEnhance);

    if (selectedVoiceData?.model?.toLowerCase().includes("bark")) {
      formData.append("temperature", barkTemperature);
      formData.append("top_k", barkTopK);
      formData.append("top_p", barkTopP);
      formData.append("seed", seed);
    }
    if (selectedVoiceData?.model?.toLowerCase().includes("xtts")) {
      formData.append("length_scale", xttsLengthScale);
      formData.append("noise_scale", xttsNoiseScale);
      formData.append("noise_scale_w", xttsNoiseScaleW);
    }
    if (selectedVoiceData?.model?.toLowerCase().includes("vits")) {
      formData.append("speaker_id", vitsSpeaker);
      formData.append("noise_scale", vitsNoiseScale);
      formData.append("duration_scale", vitsDurationScale);
      formData.append("use_phonemes", vitsUsePhonemes.toString());
    }
    if (selectedVoiceData?.requires_speaker_wav && formDataRef.current.has("speaker_wav")) {
      const wav = formDataRef.current.get("speaker_wav");
      console.log("[DEBUG] Attaching speaker_wav to formData:", wav);
      formData.append("speaker_wav", wav);
    } else if (speakerList.length && speaker) {
      formData.append("speaker", speaker);
    }
    if (selectedVoiceData?.requires_language && language) {
      formData.append("language", language);
    }
    if (presetList.length && voicePreset) {
      formData.append("voice_preset", voicePreset);
    }

    axios.post("http://localhost:5000/generate", formData).then((res) => {
      const id = res.data.job_id;
      setJobId(id);
      setJobStatus("queued");

      setQueue((prev) =>
        prev.map((item) =>
          item.id === queueId
            ? {
                ...item,
                id,
                status: "queued",
                progress: 0,
                originalIndex:
                  item.originalIndex ?? prev.findIndex((q) => q.id === queueId),
                downloadUrl:
                  res.data.status === "done"
                    ? `http://localhost:5000/audio/${id}`
                    : undefined,
              }
            : item
        )
      );

      const poll = setInterval(() => {
        axios.get(`http://localhost:5000/status/${id}`).then((res) => {
          setJobStatus(res.data.status);
          setJobProgress(res.data.progress || 0);

          setQueue((prev) =>
            prev.map((item) =>
              item.id === id
                ? { ...item, status: res.data.status, progress: res.data.progress || 0 }
                : item
            )
          );

          if (res.data.status === "done") {
            clearInterval(poll);
            setAudioUrl(res.data.download_url);
            setQueue((prev) =>
              prev.map((item) =>
                item.id === id
                  ? {
                      ...item,
                      status: "done",
                      progress: 100,
                      downloadUrl: `http://localhost:5000/audio/${id}`,
                    }
                  : item
              )
            );
          } else if (res.data.status === "error") {
            clearInterval(poll);
            alert("Generation failed.");
          }
        });
      }, 1000);
    });
  };
     

  const cancelQueueItem = (id) => {
    axios.delete(`http://localhost:5000/cancel/${id}`).catch((err) => console.error(err));
    setQueue((prev) => prev.filter((item) => item.id !== id));
  };

  const estimatedTimePerItem = 5;
  const startTimesRef = useRef({});

  /* ---------- UI ---------- */
  return (
    <>
      <div className="fixed top-0 left-0 right-0 bg-white shadow-md z-50 px-6 py-3 flex justify-between items-center">
        <h1 className="text-lg font-bold">Voication</h1>
        <button
          className="relative text-blue-600 hover:underline"
          onClick={() => {
            setShowQueue((prev) => {
              if (!prev) setUnreadCompletions(0);
              return !prev;
            });
          }}
        >
          <span className="flex items-center gap-2">
            Queue
            {/* Spinner if any job in progress */}
            {queue.some((item) => item.status !== "done" && item.status !== "error") && (
              <svg className="ml-1 animate-spin h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
              </svg>
            )}
          </span>
          {/* Unread badge */}
          {unreadCompletions > 0 && (
            <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center shadow">
              {unreadCompletions}
            </span>
          )}
        </button>
      </div>
      <div className="h-[60px]" />
      <div className="relative p-6 max-w-2xl mx-auto bg-white shadow-xl rounded-2xl mt-8">
        <h1 className="text-2xl font-bold mb-6 text-center">Voication</h1>

      {/* Voice selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold mb-2">Choose a Voice Model</label>
        <div className="flex gap-4 overflow-x-auto pb-2 pr-4">
          {voices.map((v) => (
            <label
              key={v.name}
              className={`border p-4 rounded-xl cursor-pointer w-[180px] shrink-0 whitespace-normal break-words ${
                selectedVoice === v.name ? "border-blue-500 ring-2 ring-blue-300" : "hover:border-gray-400"
              }`}
            >
              <input
                type="radio"
                name="voice"
                value={v.name}
                checked={selectedVoice === v.name}
                onChange={() => setSelectedVoice(v.name)}
                className="hidden"
              />
              <h3 className="font-semibold whitespace-normal break-words">{v.name}</h3>
              {v.description && <p className="text-sm text-gray-600 mb-1 whitespace-normal break-words">{v.description}</p>}
              {selectedVoice === v.name && v.model?.toLowerCase().includes("bark") && (
                <ul className="text-xs text-gray-500 list-disc ml-5">
                  <li>Token-level control</li>
                </ul>
              )}
            </label>
          ))}
        </div>
      </div>

      <label className="block text-sm font-semibold mb-2">Add your text</label>
      {selectedVoiceData?.model?.toLowerCase().includes("bark") ? (
        <>
          <TagEditor
            value={text}
            onChange={setText}
            tokens={tokensList}
            placeholder="Type or paste your narration here – type “[” to see token suggestions…"
            className="w-full p-4 border rounded-xl mb-6 focus:ring-2 focus:ring-blue-500 min-h-[8rem]"
          />
        </>
      ) : (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste your narration here"
          className="w-full p-4 border rounded-xl mb-6 focus:ring-2 focus:ring-blue-500 min-h-[8rem] resize-none"
        />
      )}

      {/* AI Enhance toggle and enhancement pane (only for Bark models) */}
      {selectedVoiceData?.model?.toLowerCase().includes("bark") && (
        <>
          <label className="flex items-center gap-3 mb-6 cursor-pointer select-none">
            <span className="text-sm font-medium">Enable AI&nbsp;Enhance</span>
            <span className="relative inline-block w-10 align-middle select-none transition duration-200 ease-in">
              <input
                type="checkbox"
                checked={smartEnhance}
                onChange={(e) => setSmartEnhance(e.target.checked)}
                className="sr-only"
                id="ai-enhance-toggle"
              />
              <span
                className={
                  "block w-10 h-6 rounded-full transition-colors " +
                  (smartEnhance ? "bg-blue-600" : "bg-gray-300")
                }
              ></span>
              <span
                className={
                  "dot absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition transform " +
                  (smartEnhance ? "translate-x-4" : "")
                }
              ></span>
            </span>
          </label>
          {smartEnhance && (
            <div className="mb-4">
              <label className="block text-xs font-medium mb-1">
                Add a prompt to help give the LLM more direction (optional)
              </label>
              <textarea
                className="w-full p-3 border rounded-xl mb-2 focus:ring-2 focus:ring-purple-500 text-sm"
                rows="3"
                placeholder='Tell the assistant how the narration should feel… e.g. “dramatic and tense”'
                value={enhancePrompt}
                onChange={(e) => setEnhancePrompt(e.target.value)}
                disabled={isEnhancing}
              />
              <button
                type="button"
                onClick={runEnhancement}
                disabled={isEnhancing || !text.trim()}
                className={`px-4 py-2 rounded-lg text-white ${
                  isEnhancing
                    ? "bg-gray-400 cursor-wait"
                    : "bg-purple-600 hover:bg-purple-700"
                }`}
              >
                {isEnhancing ? "Enhancing…" : "AI Enhance"}
              </button>
            </div>
          )}
        </>
      )}

      {/* Speaker dropdown */}
      {speakerList.length > 0 && !selectedVoiceData?.requires_speaker_wav && (
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">👤 Speaker:</label>
          <select
            className="w-full p-3 border rounded-xl focus:ring-2 focus:ring-blue-500"
            value={speaker}
            onChange={(e) => setSpeaker(e.target.value)}
          >
            {speakerList.map((sp) => (
              <option key={sp} value={sp}>
                {sp}
              </option>
            ))}
          </select>
        </div>
      )}


      {/* Preset dropdown - moved above Bark tuning and relabelled */}
      {presetList.length > 0 && (
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">Select a Bark Voice:</label>
          <select
            className="w-full p-3 border rounded-xl focus:ring-2 focus:ring-blue-500"
            value={voicePreset}
            onChange={(e) => setVoicePreset(e.target.value)}
          >
            {presetList.map((p) => (
              <option key={p} value={p}>
                {barkPresets[p] || barkPresets[p?.split?.("/")?.pop?.()] || p.replace("en_speaker_", "Speaker ")}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Bark advanced section */}
      {selectedVoiceData?.model?.toLowerCase().includes("bark") && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-3">Bark Voice Tuning</h3>

          <div className="flex gap-2 mb-4">
            <button
              type="button"
              onClick={() => {
                setBarkTemperature(0.4);
                setBarkTopK(30);
                setBarkTopP(0.85);
              }}
              className="px-3 py-1 text-xs rounded-lg bg-gray-200 hover:bg-gray-300"
              title="Steady and easy to follow, ideal for step-by-step instructions"
            >
              Calm &amp; Clear
            </button>
            <button
              type="button"
              onClick={() => {
                setBarkTemperature(0.7);
                setBarkTopK(50);
                setBarkTopP(0.9);
              }}
              className="px-3 py-1 text-xs rounded-lg bg-gray-200 hover:bg-gray-300"
              title="A good balance between clarity and expression"
            >
              Balanced
            </button>
            <button
              type="button"
              onClick={() => {
                setBarkTemperature(0.9);
                setBarkTopK(80);
                setBarkTopP(0.95);
              }}
              className="px-3 py-1 text-xs rounded-lg bg-gray-200 hover:bg-gray-300"
              title="Adds more variation and emotion, perfect for stories"
            >
              Expressive &amp; Varied
            </button>
          </div>

          {/* Seed input for Bark */}
          <label className="block text-xs font-medium mb-1">
            Seed – Voice Variation (integer): {seed}
          </label>
          <input
            type="number"
            min="0"
            max="999999"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10))}
            className="w-full mb-4 border rounded-xl p-2"
          />

          <label className="block text-xs font-medium mb-1">
            Creativity – Temperature (temperature, 0-1 → creative): {barkTemperature.toFixed(2)}
            <div className="relative inline-block group ml-1">
              <span className="text-gray-500 cursor-pointer">?</span>
              <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                0 = straightforward narration; 1 = highly imaginative and playful
              </div>
            </div>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={barkTemperature}
            onChange={(e) => setBarkTemperature(parseFloat(e.target.value))}
            className="w-full mb-4 accent-blue-600"
          />

          <label className="block text-xs font-medium mb-1">
            Variation – Top-K (top_k, 0-100 → larger pool): {barkTopK}
            <div className="relative inline-block group ml-1">
              <span className="text-gray-500 cursor-pointer">?</span>
              <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                How many of the most likely options to consider; low = focused, high = varied
              </div>
            </div>
          </label>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={barkTopK}
            onChange={(e) => setBarkTopK(parseInt(e.target.value, 10))}
            className="w-full mb-4 accent-blue-600"
          />

          <label className="block text-xs font-medium mb-1">
            Diversity – Top-P (top_p, 0-1 → nucleus sampling): {barkTopP.toFixed(2)}
            <div className="relative inline-block group ml-1">
              <span className="text-gray-500 cursor-pointer">?</span>
              <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                Include tokens until their cumulative probability reaches this value; low = precise, high = broad
              </div>
            </div>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={barkTopP}
            onChange={(e) => setBarkTopP(parseFloat(e.target.value))}
            className="w-full accent-blue-600"
          />
        </div>
      )}

      {/* XTTS v2: Only show voice reference UI, no tuning sliders */}
      {selectedVoiceData?.model?.toLowerCase().includes("xtts") && (
        <>
          <div className="mb-6">
            <h3 className="text-sm font-semibold mb-2">Voice Sample (Speaker Reference)</h3>
            <p className="text-xs mb-2 text-gray-600">
              Upload or record a voice sample that the system will use to match the voice style. You only need to do one.
            </p>
            <div className="text-xs italic bg-gray-100 p-2 rounded mb-3">
              “The sun sets behind the hills, and the sky turns orange. I really enjoy storytelling and character voices.”
            </div>
            {/* Upload Speaker Reference (WAV) input */}
            <div className="mb-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Upload Speaker Reference (WAV):
              </label>
              <input
                type="file"
                accept=".wav"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    const file = e.target.files[0];
                    formDataRef.current.set("speaker_wav", file);
                    setRecordedBlob(file);
                  }
                }}
                className="block w-full text-sm text-gray-600 file:mr-4 file:py-1 file:px-2 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            {/* Voice recording interface */}
            <div className="flex gap-2 items-center mb-3">
              <button
                type="button"
                className="px-3 py-1 bg-blue-600 text-white text-sm rounded"
                onClick={startRecording}
                disabled={isRecording}
              >
                {isRecording ? "Recording..." : "Record"}
              </button>
              <button
                type="button"
                className="px-3 py-1 bg-gray-700 text-white text-sm rounded"
                onClick={stopRecording}
                disabled={!isRecording}
              >
                Stop
              </button>
              {recordedBlob && (
                <button
                  type="button"
                  className="px-3 py-1 bg-green-600 text-white text-sm rounded"
                  onClick={downloadRecording}
                >
                  Download
                </button>
              )}
            </div>
            {recordedBlob && (
              <audio controls src={URL.createObjectURL(recordedBlob)} className="w-full mt-2" />
            )}
          </div>
          {/* Explanatory text for XTTS v2 */}
          {selectedVoiceData?.model?.toLowerCase() === "xtts_v2" && (
            <div className="mb-6 text-xs text-blue-700 bg-blue-50 rounded p-3 border border-blue-100">
              XTTS v2 uses your uploaded or recorded voice to generate speech. It does not support custom sliders for emotion, speed, or noise – expressiveness is inferred from the reference audio.
            </div>
          )}
        </>
      )}

      {/* Language dropdown */}
      {selectedVoiceData?.requires_language && selectedVoiceData.supported_languages && (
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">🌏 Language:</label>
          <select
            className="w-full p-3 border rounded-xl focus:ring-2 focus:ring-blue-500"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            {selectedVoiceData.supported_languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Speaker WAV upload (handled above for XTTS) */}
      {selectedVoiceData?.requires_speaker_wav &&
        !selectedVoiceData?.model?.toLowerCase().includes("xtts") && (
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">
            Upload Speaker Reference (WAV):
          </label>
          <input
            type="file"
            accept="audio/wav"
            onChange={(e) => setSpeakerWav(e.target.files[0])}
            className="w-full p-3 border rounded-xl focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Short mono WAV, 22 kHz preferred. Speak clearly and naturally.
          </p>
        </div>
      )}

      {/* Generate button */}
      {(() => {
        // Add to Queue button disables if text is empty, enhancing, or XTTS_v2 and no speaker_wav
        const speakerWavPresent =
          selectedVoiceData?.model?.toLowerCase() !== "xtts_v2" ||
          formDataRef.current.has("speaker_wav");
        return (
          <button
            onClick={generateSpeech}
            disabled={
              isEnhancing ||
              !text.trim() ||
              !speakerWavPresent
            }
            className={`w-full ${
              !text.trim() || isEnhancing || !speakerWavPresent 
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            } text-white font-bold py-3 px-6 rounded-xl transition`}
          >
            Add to queue
          </button>
        );
      })()}

   
        {showQueue && (
          <div className="fixed right-0 top-0 w-96 max-w-full h-full bg-white shadow-lg p-4 overflow-y-auto z-50 border-l border-gray-200">
            <h2 className="text-lg font-bold mb-4">Queue</h2>
            <button
              onClick={() => setShowQueue(false)}
              className="absolute top-4 right-4 text-gray-500 hover:text-black"
              title="Close"
            >
              ✕
            </button>
            {queue.map((item) => {
              if (!startTimesRef.current[item.id]) {
                startTimesRef.current[item.id] = Date.now();
              }
              let progress = item.progress ?? 0;
              if (item.status !== "done" && !item.progress) {
                const elapsed = (Date.now() - startTimesRef.current[item.id]) / 1000;
                progress = Math.min(90, Math.max(10, Math.round((elapsed / estimatedTimePerItem) * 80 + 10)));
              }
              return (
                <div key={item.id} className="mb-4 p-3 bg-gray-100 rounded-xl">
                  <p className="text-sm font-medium truncate">{item.text}</p>
                  <p className="text-xs text-gray-600">{`Status: ${item.status}`}</p>
                  {item.status !== "done" && (
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
                      <div
                        className="bg-blue-500 h-2.5 rounded-full"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  )}
                  {item.status !== "done" && (
                    <button
                      onClick={() => cancelQueueItem(item.id)}
                      className="text-xs text-red-600 hover:underline"
                    >
                      Cancel
                    </button>
                  )}
                  {item.status === "done" && (
                    <>
                      <button
                        onClick={() => setQueue((prev) => prev.filter((q) => q.id !== item.id))}
                        className="text-xs text-gray-600 hover:text-black float-right"
                        title="Clear"
                      >
                        ❌
                      </button>
                      <audio controls src={item.downloadUrl} className="w-full mt-2" />
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}

// Bark voice presets are now loaded dynamically from backend for consistency.
function AppInnerWrapper() {
  const [barkPresets, setBarkPresets] = useState({});
  useEffect(() => {
    fetch("/voices")
      .then((res) => res.json())
      .then((data) => setBarkPresets(data))
      .catch((err) => console.error("Failed to fetch voice presets", err));
  }, []);
  return <AppInner barkPresets={barkPresets} setBarkPresets={setBarkPresets} />;
}

// Top-level error boundary for debugging
function App() {
  try {
    return <AppInnerWrapper />;
  } catch (e) {
    return (
      <div style={{ color: "red", padding: 20, fontWeight: "bold" }}>
        Error: {e && e.message ? e.message : String(e)}
      </div>
    );
  }
}

export default App;