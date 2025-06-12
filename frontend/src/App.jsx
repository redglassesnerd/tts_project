import React, { useEffect, useState, useRef } from "react";
import { Cog6ToothIcon } from "@heroicons/react/24/outline";
import axios from "axios";
import TagEditor from "./TagEditor";
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
  "serious",
];

// --- MAIN COMPONENT ---
function AppInner({ barkPresets, setBarkPresets }) {
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

  // Bark sentence splitting and max duration (persistent)
  const [barkSplitSentences, setBarkSplitSentences] = useState(() => {
    try {
      return localStorage.getItem("barkSplitSentences") === "true";
    } catch {
      return false;
    }
  });

  const [barkMaxDuration, setBarkMaxDuration] = useState(() => {
    try {
      const val = localStorage.getItem("barkMaxDuration");
      return val ? Number(val) : 14;
    } catch {
      return 14;
    }
  });

  // Bark advanced knobs
  const [barkTemperature, setBarkTemperature] = useState(0.7); // 0‑1 precise → creative
  const [barkTopK, setBarkTopK] = useState(50); // 0‑100 smaller → larger vocab
  const [barkTopP, setBarkTopP] = useState(0.9); // 0‑1 safe → diverse

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
  const [showSettings, setShowSettings] = useState(false);

  // Success notification for settings save
  const [showSaveSuccess, setShowSaveSuccess] = useState(false);

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
  const speakerList = selectedVoiceData?.supported_speakers?.length
    ? selectedVoiceData.supported_speakers
    : selectedVoiceData?.speakers?.length
    ? selectedVoiceData.speakers
    : selectedVoiceData?.speaker_list?.length
    ? selectedVoiceData.speaker_list
    : selectedVoiceData?.speaker_ids?.length
    ? selectedVoiceData.speaker_ids
    : [];

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
    if (
      speakerList.length > 1 &&
      !speaker &&
      !selectedVoiceData?.requires_speaker_wav
    ) {
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
    if (
      selectedVoiceData?.requires_speaker_wav &&
      !formDataRef.current.has("speaker_wav")
    )
      return false;
    if (
      !selectedVoiceData?.requires_speaker_wav &&
      speakerList.length &&
      !speaker
    )
      return false;
    if (selectedVoiceData?.requires_language && !language) return false;
    if (presetList.length && !voicePreset) return false;
    return true;
  };

  /* ---------- Generate ---------- */
  // --- Start polling helper ---
  const startPolling = (jobId) => {
    if (!jobId) return;
    const poll = setInterval(() => {
      fetch(`http://localhost:5000/status/${jobId}`)
        .then((res) => res.json())
        .then((data) => {
          // Debug: show status and audio_url
          console.log("Job status:", {
            status: data.status,
            audio_url: data.audio_url,
          });
          setJobStatus(data.status);
          setJobProgress(data.progress || 0);
          setQueue((prev) =>
            prev.map((item) =>
              item.id === jobId
                ? { ...item, status: data.status, progress: data.progress || 0 }
                : item
            )
          );
          if (data.status === "done") {
            clearInterval(poll);
            setAudioUrl(
              data.audio_url
                ? `http://localhost:5000${data.audio_url}`
                : undefined
            );
            setQueue((prev) =>
              prev.map((item) =>
                item.id === jobId
                  ? {
                      ...item,
                      status: "done",
                      progress: 100,
                      downloadUrl: data.audio_url
                        ? `http://localhost:5000${data.audio_url}`
                        : undefined,
                    }
                  : item
              )
            );
          } else if (data.status === "error") {
            clearInterval(poll);
            alert("Generation failed.");
          }
        });
    }, 1000);
  };

  const generateSpeech = async () => {
    // --- Bark model: ensure a voice is selected, including index 0 ---
    if (
      selectedVoiceData?.model?.toLowerCase().includes("bark") &&
      !selectedVoice &&
      selectedVoice !== 0
    ) {
      alert("Please select a voice for Bark.");
      return;
    }

    if (!allRequiredReady()) {
      alert("Fill in required fields first.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue((prev) => [
      ...prev,
      { id: queueId, text, status: "queued", progress: 0 },
    ]);

    let jobId = null;
    // If Bark, send as JSON, else as FormData (for speaker_wav uploads)
    if (selectedVoiceData?.model?.toLowerCase().includes("bark")) {
      // Bark: send as JSON, include barkSplitSentences and barkMaxDuration
      const body = {
        model: selectedVoice,
        text,
        // Bark-specific
        temperature: barkTemperature,
        top_k: barkTopK,
        top_p: barkTopP,
        seed,
        barkSplitSentences,
        barkMaxDuration,
        speed,
        chunk_size: chunkSize,
        pause_duration: pauseDuration,
        smart_enhance: smartEnhance,
        // Optionals
        voice: selectedVoice, // explicitly include voice
        preset: voicePreset, // explicitly include preset
        voice_preset: voicePreset, // keep for backward compatibility
        language,
        speaker,
      };
      try {
        const response = await fetch("http://localhost:5000/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await response.json();
        jobId = data.job_id;
        setJobId(jobId);
        setJobStatus("queued");
        setQueue((prev) =>
          prev.map((item) =>
            item.id === queueId
              ? {
                  ...item,
                  id: jobId,
                  status: "queued",
                  progress: 0,
                  originalIndex:
                    item.originalIndex ??
                    prev.findIndex((q) => q.id === queueId),
                  downloadUrl:
                    data.status === "done"
                      ? `http://localhost:5000/audio/${jobId}`
                      : undefined,
                }
              : item
          )
        );
        // Start polling with backend jobId
        startPolling(jobId);
      } catch (err) {
        alert("Failed to start Bark generation.");
        return;
      }
    } else {
      // Non-Bark: use FormData for possible speaker_wav
      const formData = new FormData();
      formData.append("text", text);
      formData.append("model", selectedVoice);
      formData.append("voice", selectedVoice); // explicitly include voice field
      formData.append("speed", speed);
      formData.append("chunk_size", chunkSize);
      formData.append("pause_duration", pauseDuration);
      formData.append("smart_enhance", smartEnhance);

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
      if (
        selectedVoiceData?.requires_speaker_wav &&
        formDataRef.current.has("speaker_wav")
      ) {
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
        formData.append("preset", voicePreset); // explicitly include preset
      }

      try {
        const response = await fetch("http://localhost:5000/generate", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        jobId = data.job_id;
        setJobId(jobId);
        setJobStatus("queued");
        setQueue((prev) =>
          prev.map((item) =>
            item.id === queueId
              ? {
                  ...item,
                  id: jobId,
                  status: "queued",
                  progress: 0,
                  originalIndex:
                    item.originalIndex ??
                    prev.findIndex((q) => q.id === queueId),
                  downloadUrl:
                    data.status === "done"
                      ? `http://localhost:5000/audio/${jobId}`
                      : undefined,
                }
              : item
          )
        );
        // Start polling with backend jobId
        startPolling(jobId);
      } catch (err) {
        alert("Failed to start generation.");
        return;
      }
    }
  };

  const cancelQueueItem = (id) => {
    axios
      .delete(`http://localhost:5000/cancel/${id}`)
      .catch((err) => console.error(err));
    setQueue((prev) => prev.filter((item) => item.id !== id));
  };

  const estimatedTimePerItem = 5;
  const startTimesRef = useRef({});

  // ------------- Settings modal tab state -------------
  const [settingsTab, setSettingsTab] = useState("bark");

  /* ---------- UI ---------- */
  return (
    <>
      <div className="fixed top-0 left-0 right-0 bg-white shadow-md z-50 px-6 py-3 flex items-center">
        <h1 className="text-lg font-bold">Voication</h1>
        <button
          className="relative text-blue-600 hover:underline ml-6"
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
            {queue.some(
              (item) => item.status !== "done" && item.status !== "error"
            ) && (
              <svg
                className="ml-1 animate-spin h-4 w-4 text-blue-500"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
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
        <button
          onClick={() => setShowSettings(true)}
          className="ml-auto text-gray-500 hover:text-gray-700 transition"
          title="Settings"
        >
          <Cog6ToothIcon className="h-6 w-6" />
        </button>
      </div>
      <div className="h-[60px]" />
      <div className="relative p-6 max-w-2xl mx-auto bg-white shadow-xl rounded-2xl mt-8">
        <h1 className="text-2xl font-bold mb-6 text-center">Voication</h1>

        {/* Voice selector */}
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">
            Choose a Voice Model
          </label>
          <div className="flex gap-4 overflow-x-auto pb-2 pr-4">
            {voices.map((v) => (
              <label
                key={v.name}
                className={`border p-4 rounded-xl cursor-pointer w-[180px] shrink-0 whitespace-normal break-words ${
                  selectedVoice === v.name
                    ? "border-blue-500 ring-2 ring-blue-300"
                    : "hover:border-gray-400"
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
                <h3 className="font-semibold whitespace-normal break-words">
                  {v.name}
                </h3>
                {v.description && (
                  <p className="text-sm text-gray-600 mb-1 whitespace-normal break-words">
                    {v.description}
                  </p>
                )}
                {selectedVoice === v.name &&
                  v.model?.toLowerCase().includes("bark") && (
                    <ul className="text-xs text-gray-500 list-disc ml-5">
                      <li>Token-level control</li>
                    </ul>
                  )}
              </label>
            ))}
          </div>
        </div>

        {/* --- Text input and AI Enhance toggle grouped --- */}
        <div className="mb-6">
          <p className="text-base font-semibold mb-2">Add your text</p>
          <div>
            {selectedVoiceData?.model?.toLowerCase().includes("bark") ? (
              <TagEditor
                value={text}
                onChange={setText}
                tokens={tokensList}
                placeholder="Type or paste your narration here – type “[” to see token suggestions…"
                className="w-full p-4 border rounded-xl focus:ring-2 focus:ring-blue-500 min-h-[8rem]"
              />
            ) : (
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type or paste your narration here"
                className="w-full p-4 border rounded-xl focus:ring-2 focus:ring-blue-500 min-h-[8rem] resize-none"
              />
            )}
          </div>
          {/* AI Enhance toggle and enhancement pane (only for Bark models) */}
          {selectedVoiceData?.model?.toLowerCase().includes("bark") && (
            <>
              <label className="flex items-center gap-3 mb-6 cursor-pointer select-none">
                <span className="text-sm font-medium">Enable AI Enhance</span>
                <span className="relative inline-block w-10 align-middle select-none transition duration-200 ease-in">
                  <input
                    type="checkbox"
                    id="ai-enhance-toggle"
                    className="sr-only"
                    checked={smartEnhance}
                    onChange={(e) => setSmartEnhance(e.target.checked)}
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
                    rows={3}
                    placeholder="Tell the assistant how the narration should feel… e.g. “dramatic and tense”"
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
        </div>

        {/* --- Horizontal rule after input field --- */}
        <hr className="my-4" />

        {/* --- Voice profile load/create section (Bark only) --- */}
        {selectedVoiceData?.model?.toLowerCase().includes("bark") && (
          <VoiceProfilePanel
            presetList={presetList}
            onApplyProfile={(profile) => {
              setSeed(profile.seed);
              setBarkTemperature(profile.text_temp);
              setBarkTopK(profile.top_k);
              setBarkTopP(profile.top_p);
              setVoicePreset(profile.voice_preset);
            }}
          />
        )}

        {/* --- VITS-specific tuning UI --- */}
        {selectedVoiceData?.model?.toLowerCase().includes("vits") && (
          <div className="mb-6 space-y-3">
            <label className="block text-sm font-semibold">Speaker ID</label>
            <select
              className="w-full px-3 py-2 text-sm border rounded"
              value={vitsSpeaker}
              onChange={(e) => setVitsSpeaker(e.target.value)}
            >
              <option value="">-- Select speaker --</option>
              {speakerList.map((spk) => (
                <option key={spk} value={spk}>
                  {spk}
                </option>
              ))}
            </select>
            <label className="block text-sm font-semibold">Noise Scale</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={vitsNoiseScale}
              onChange={(e) => setVitsNoiseScale(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm border rounded"
            />
            <label className="block text-sm font-semibold">
              Duration Scale
            </label>
            <input
              type="number"
              step="0.01"
              min="0.5"
              max="2"
              value={vitsDurationScale}
              onChange={(e) => setVitsDurationScale(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm border rounded"
            />
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={vitsUsePhonemes}
                onChange={(e) => setVitsUsePhonemes(e.target.checked)}
              />
              <span className="text-sm">Use Phonemes</span>
            </label>
          </div>
        )}

        {/* XTTS v2: Only show voice reference UI, no tuning sliders */}
        {selectedVoiceData?.model?.toLowerCase().includes("xtts") && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-semibold mb-2">
                Voice Sample (Speaker Reference)
              </h3>
              <p className="text-xs mb-2 text-gray-600">
                Upload or record a voice sample that the system will use to
                match the voice style. You only need to do one.
              </p>
              <div className="text-xs italic bg-gray-100 p-2 rounded mb-3">
                “The sun sets behind the hills, and the sky turns orange. I
                really enjoy storytelling and character voices.”
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
                <audio
                  controls
                  src={URL.createObjectURL(recordedBlob)}
                  className="w-full mt-2"
                />
              )}
            </div>
            {/* Explanatory text for XTTS v2 */}
            {selectedVoiceData?.model?.toLowerCase() === "xtts_v2" && (
              <div className="mb-6 text-xs text-blue-700 bg-blue-50 rounded p-3 border border-blue-100">
                XTTS v2 uses your uploaded or recorded voice to generate speech.
                It does not support custom sliders for emotion, speed, or noise
                – expressiveness is inferred from the reference audio.
              </div>
            )}
          </>
        )}

        {/* Language dropdown */}
        {selectedVoiceData?.requires_language &&
          selectedVoiceData.supported_languages && (
            <div className="mb-6">
              <label className="block text-sm font-semibold mb-2">
                🌏 Language:
              </label>
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
          // Unified canGenerate logic for Bark, XTTS_v2, VITS, etc.
          const isBark = selectedVoiceData?.model
            ?.toLowerCase()
            .includes("bark");
          const isXTTS = selectedVoiceData?.model?.toLowerCase() === "xtts_v2";
          const requiresSpeakerWav = selectedVoiceData?.requires_speaker_wav;
          const speakerWavPresent =
            !requiresSpeakerWav || formDataRef.current.has("speaker_wav");

          const canGenerate =
            text.trim() &&
            !isEnhancing &&
            speakerWavPresent &&
            (!isBark || voicePreset);

          return (
            <button
              onClick={generateSpeech}
              disabled={!canGenerate}
              className={`w-full ${
                !canGenerate
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
                const elapsed =
                  (Date.now() - startTimesRef.current[item.id]) / 1000;
                progress = Math.min(
                  90,
                  Math.max(
                    10,
                    Math.round((elapsed / estimatedTimePerItem) * 80 + 10)
                  )
                );
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
                  {item.status === "done" && item.downloadUrl && (
                    <>
                      <button
                        onClick={() =>
                          setQueue((prev) =>
                            prev.filter((q) => q.id !== item.id)
                          )
                        }
                        className="text-xs text-gray-600 hover:text-black float-right"
                        title="Clear"
                      >
                        ❌
                      </button>
                      <audio
                        controls
                        src={item.downloadUrl}
                        className="w-full mt-2"
                      />
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {/* Settings Modal (Sidebar Layout) */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white w-[700px] max-w-full rounded shadow-lg flex relative">
            <button
              onClick={() => setShowSettings(false)}
              className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
              title="Close"
            >
              ✕
            </button>

            {/* Sidebar */}
            <div className="w-1/4 bg-gray-100 p-4 border-r rounded-l">
              <h3 className="text-xs font-semibold text-gray-600 uppercase mb-4">
                Voice Models
              </h3>
              <ul className="space-y-2">
                <li>
                  <button
                    onClick={() => setSettingsTab("bark")}
                    className={`w-full text-left px-3 py-2 rounded ${
                      settingsTab === "bark"
                        ? "bg-white shadow font-semibold"
                        : "hover:bg-gray-200"
                    }`}
                  >
                    Bark
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => setSettingsTab("vits")}
                    className={`w-full text-left px-3 py-2 rounded ${
                      settingsTab === "vits"
                        ? "bg-white shadow font-semibold"
                        : "hover:bg-gray-200"
                    }`}
                  >
                    VITS
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => setSettingsTab("xttsv2")}
                    className={`w-full text-left px-3 py-2 rounded ${
                      settingsTab === "xttsv2"
                        ? "bg-white shadow font-semibold"
                        : "hover:bg-gray-200"
                    }`}
                  >
                    XTTS V2
                  </button>
                </li>
              </ul>
            </div>

            {/* Content Area */}
            <div className="w-3/4 p-6">
              {settingsTab === "bark" && (
                <>
                  <h2 className="text-xl font-semibold mb-4">Bark Settings</h2>
                  <div className="space-y-4">
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        className="toggle"
                        checked={barkSplitSentences}
                        onChange={(e) => {
                          setBarkSplitSentences(e.target.checked);
                          localStorage.setItem(
                            "barkSplitSentences",
                            e.target.checked.toString()
                          );
                        }}
                      />
                      <span>Split long sentences by max duration</span>
                    </label>

                    <label className="block">
                      <span className="text-sm text-gray-700">
                        Max sentence duration (seconds)
                      </span>
                      <input
                        type="number"
                        min={3}
                        max={20}
                        value={barkMaxDuration}
                        onChange={(e) => {
                          const value = Number(e.target.value);
                          setBarkMaxDuration(value);
                          localStorage.setItem(
                            "barkMaxDuration",
                            value.toString()
                          );
                        }}
                        className="mt-1 block w-24 border rounded px-2 py-1 text-sm"
                      />
                    </label>

                    <div className="text-xs text-gray-500">
                      Sentences longer than this limit will be split using
                      natural joiners (like commas or "and") when possible.
                    </div>
                  </div>
                  <div className="pt-4">
                    <button
                      className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700"
                      onClick={() => {
                        localStorage.setItem(
                          "barkSplitSentences",
                          barkSplitSentences.toString()
                        );
                        localStorage.setItem(
                          "barkMaxDuration",
                          barkMaxDuration.toString()
                        );
                        setShowSaveSuccess(true);
                        setTimeout(() => setShowSaveSuccess(false), 2000);
                      }}
                    >
                      Save Settings
                    </button>
                  </div>
                </>
              )}

              {settingsTab !== "bark" && (
                <>
                  <h2 className="text-xl font-semibold mb-4 capitalize">
                    {settingsTab} Settings
                  </h2>
                  <p className="text-sm text-gray-500">Coming soon...</p>
                </>
              )}
              {/* Success notification */}
              {showSaveSuccess && (
                <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg flex items-center space-x-2 z-50">
                  <svg
                    className="w-5 h-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  <span>Settings saved successfully</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
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
// ---- Voice Profile Panel ----
function VoiceProfilePanel({ presetList, onApplyProfile }) {
  // Profile editor state
  const [savedProfiles, setSavedProfiles] = useState({});
  const [selectedProfileName, setSelectedProfileName] = useState("");
  const [showProfileEditor, setShowProfileEditor] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [isPinned, setIsPinned] = useState(false);
  // Bark tuning fields
  const [seed, setSeed] = useState(424242);
  const [textTemp, setTextTemp] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  // Bark voice preset
  const [voicePreset, setVoicePreset] = useState("");
  // Save message state
  const [saveMessage, setSaveMessage] = useState("");

  // Load saved profiles from localStorage on mount, seed ExampleMailNarration if missing
  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("voiceProfiles") || "{}");
    if (!stored["ExampleMailNarration"]) {
      stored["ExampleMailNarration"] = {
        name: "ExampleMailNarration",
        pinned: true,
        seed: 424242,
        text_temp: 0.7,
        top_k: 50,
        top_p: 0.9,
        voice_preset: presetList[0] || "",
      };
      localStorage.setItem("voiceProfiles", JSON.stringify(stored));
    }
    setSavedProfiles(stored);
  }, [presetList]);

  // Rehydrate editor fields when selectedProfileName changes
  useEffect(() => {
    const val = selectedProfileName;
    if (val && savedProfiles[val]) {
      setProfileName(savedProfiles[val].name);
      setIsPinned(!!savedProfiles[val].pinned);
      setSeed(savedProfiles[val].seed ?? 424242);
      setTextTemp(savedProfiles[val].text_temp ?? 0.7);
      setTopK(savedProfiles[val].top_k ?? 50);
      setTopP(savedProfiles[val].top_p ?? 0.9);
      setVoicePreset(savedProfiles[val].voice_preset || "");
      setShowProfileEditor(true);
      if (val && savedProfiles[val] && onApplyProfile) {
        onApplyProfile(savedProfiles[val]);
      }
    } else {
      setShowProfileEditor(false);
      setProfileName("");
      setIsPinned(false);
      setSeed(424242);
      setTextTemp(0.7);
      setTopK(50);
      setTopP(0.9);
      setVoicePreset("");
    }
  }, [selectedProfileName, savedProfiles, onApplyProfile]);

  // Handler: Create new profile
  const handleCreateNewProfile = () => {
    setProfileName("");
    setIsPinned(false);
    setSeed(424242);
    setTextTemp(0.7);
    setTopK(50);
    setTopP(0.9);
    setVoicePreset("");
    setShowProfileEditor(true);
    setSelectedProfileName(""); // Deselect
  };

  // Handler: Save profile
  const handleSaveProfile = () => {
    if (!profileName.trim()) {
      alert("Profile name required.");
      return;
    }
    // Save/update profile in state and localStorage
    setSavedProfiles((prev) => {
      const updated = {
        ...prev,
        [profileName.trim()]: {
          name: profileName.trim(),
          pinned: isPinned,
          seed,
          text_temp: textTemp,
          top_k: topK,
          top_p: topP,
          voice_preset: voicePreset,
        },
      };
      localStorage.setItem("voiceProfiles", JSON.stringify(updated));
      setSaveMessage("Profile saved!");
      setTimeout(() => setSaveMessage(""), 2000);
      return updated;
    });
    setShowProfileEditor(false);
    setSelectedProfileName(profileName.trim());
  };

  // Handler: Delete profile
  const handleDeleteProfile = () => {
    if (!profileName.trim()) return;
    setSavedProfiles((prev) => {
      const updated = { ...prev };
      delete updated[profileName.trim()];
      localStorage.setItem("voiceProfiles", JSON.stringify(updated));
      return updated;
    });
    setShowProfileEditor(false);
    setSelectedProfileName("");
    setProfileName("");
  };

  // Handler: Select profile
  const handleSelectProfile = (e) => {
    const val = e.target.value;
    setSelectedProfileName(val);
    // If "Example Voice" is selected, ensure fields are hydrated
    if (val === "Example Voice") {
      setSelectedProfileName("Example Voice");
    }
  };

  // Handler: randomise seed
  const randomiseSeed = () => {
    setSeed(Math.floor(Math.random() * 1000000));
  };

  return (
    <div className="mb-6">
      {/* Pinned Profiles panel */}
      <div className="mb-4">
        <p className="text-sm font-semibold mb-2">Pinned Profiles</p>
        <div className="flex flex-wrap gap-2">
          {Object.values(savedProfiles)
            .filter((p) => p.pinned)
            .map((p) => (
              <button
                key={p.name}
                onClick={() =>
                  setSelectedProfileName(p.name) || setShowProfileEditor(true)
                }
                className="px-3 py-1 text-xs rounded-lg bg-gray-200 hover:bg-gray-300"
              >
                {p.name}
              </button>
            ))}
        </div>
      </div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-semibold">Voice Profile</p>
        <button
          onClick={handleCreateNewProfile}
          className="px-2 py-1 text-xs rounded bg-blue-500 text-white hover:bg-blue-600"
        >
          + New Profile
        </button>
      </div>

      {showProfileEditor && (
        <div className="mb-4 space-y-2">
          {/* Save message feedback */}
          {saveMessage && (
            <div className="text-green-600 text-sm mb-2">{saveMessage}</div>
          )}
          <input
            type="text"
            placeholder="Profile Name"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            className="w-full px-3 py-2 text-sm border rounded"
          />
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={isPinned}
              onChange={(e) => setIsPinned(e.target.checked)}
            />
            <span>Pin profile</span>
          </label>
          {/* Bark preset dropdown */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Select Bark Voice
            </label>
            <select
              className="w-full px-3 py-1 border rounded text-sm mb-4"
              value={voicePreset}
              onChange={(e) => setVoicePreset(e.target.value)}
            >
              <option value="">-- choose preset --</option>
              {presetList.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          {/* Bark tuning fields inserted here */}
          <h4 className="text-xs font-medium text-gray-600 mt-4">
            Bark Tuning Settings
          </h4>
          {/* Seed input */}
          <div>
            <label className="block text-xs font-medium mb-1">
              Seed – Voice Variation (integer): {seed}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="0"
                max="999999"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full px-3 py-1 border rounded text-sm"
              />
              <button
                onClick={randomiseSeed}
                className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
                title="Randomize seed"
              >
                🎲
              </button>
            </div>
          </div>
          {/* Quick Presets pill button group */}
          <div className="mt-2 mb-3">
            <label className="block text-xs font-medium mb-1">
              Quick Presets
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                className="px-3 py-1 text-xs rounded-full border border-gray-300 bg-white hover:bg-gray-100"
                onClick={() => {
                  setTextTemp(0.6);
                  setTopK(40);
                  setTopP(0.85);
                }}
              >
                🧘 Calm
              </button>
              <button
                type="button"
                className="px-3 py-1 text-xs rounded-full border border-gray-300 bg-white hover:bg-gray-100"
                onClick={() => {
                  setTextTemp(0.7);
                  setTopK(50);
                  setTopP(0.9);
                }}
              >
                🗣 Neutral
              </button>
              <button
                type="button"
                className="px-3 py-1 text-xs rounded-full border border-gray-300 bg-white hover:bg-gray-100"
                onClick={() => {
                  setTextTemp(0.85);
                  setTopK(60);
                  setTopP(0.95);
                }}
              >
                🎉 Excited
              </button>
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Creativity – Temperature (temperature, 0-1 → creative):{" "}
              {textTemp.toFixed(2)}
              <div className="relative inline-block group ml-1">
                <span className="text-gray-500 cursor-pointer">?</span>
                <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                  0 = straightforward narration; 1 = highly imaginative and
                  playful
                </div>
              </div>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={textTemp}
              onChange={(e) => setTextTemp(Number(e.target.value))}
              className="w-full accent-blue-600"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Variation – Top-K (top_k, 0-100 → larger pool): {topK}
              <div className="relative inline-block group ml-1">
                <span className="text-gray-500 cursor-pointer">?</span>
                <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                  How many of the most likely options to consider; low =
                  focused, high = varied
                </div>
              </div>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-blue-600"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Diversity – Top-P (top_p, 0-1 → nucleus sampling):{" "}
              {topP.toFixed(2)}
              <div className="relative inline-block group ml-1">
                <span className="text-gray-500 cursor-pointer">?</span>
                <div className="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 w-48 bg-gray-800 text-white text-xs rounded p-2 hidden group-hover:block z-50">
                  Include tokens until their cumulative probability reaches this
                  value; low = precise, high = broad
                </div>
              </div>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              className="w-full accent-blue-600"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSaveProfile}
              className="text-xs px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Save
            </button>
            <button
              onClick={handleDeleteProfile}
              className="text-xs px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Delete
            </button>
            <button
              type="button"
              onClick={() => {
                setShowProfileEditor(false);
                setSelectedProfileName("");
              }}
              className="text-xs px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {!showProfileEditor && (
        <select
          onChange={handleSelectProfile}
          className="w-full px-3 py-2 text-sm border rounded"
          value={selectedProfileName}
        >
          <option value="">-- Load a voice profile --</option>
          {Object.keys(savedProfiles).map((key) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
