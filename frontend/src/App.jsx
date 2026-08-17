import React, { useEffect, useState, useRef } from "react";
import { Cog6ToothIcon } from "@heroicons/react/24/outline";
import axios from "axios";
import {
  Play,
  MessageSquare,
  Pause,
  Trash2,
  Plus,
  Download,
  Upload,
  Folder,
  Settings,
  FlaskConical,
  Mic,
  BookOpen,
  Sliders,
  Settings2,
  Check,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Info,
  Moon,
  Sun,
  RefreshCw,
  FileJson,
  FileEdit,
  Users,
  Flame,
  Activity,
  Smile,
  X,
  HelpCircle,
  Volume2,
  VolumeX,
  Library,
  Sparkles,
  Music,
  Layers,
  ListMusic,
  FolderPlus,
  Cpu,
  FileText,
  AlertTriangle,
  Save,
  Search,
  Copy,
  Edit3,
  User,
  Fingerprint,
  Square,
  Undo,
  Scissors,
  Cloud,
  CloudUpload,
  CloudDownload,
  CheckCircle2
} from "lucide-react";
import { requestDriveAccessToken, uploadProjectToDrive, fetchDriveBackups, downloadDriveProject } from "./utils/googleDriveSync";
import TagEditor from "./TagEditor";
import SetupWizard from "./SetupWizard";

// Utility to convert Base64 back to Blob for persistent storage reconstruction
const base64ToBlob = (base64Data, contentType = 'audio/wav') => {
  try {
    const parts = base64Data.split(',');
    const byteCharacters = atob(parts[1] || parts[0]);
    const byteArrays = [];
    const sliceSize = 1024;
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  } catch (err) {
    console.error("base64ToBlob conversion error:", err);
    return null;
  }
};

const DEFAULT_TIMELINE_TRACKS = [
  { id: "speaker_1", name: "Speaker 1", type: "dialogue", volume: 0.8, mute: false, solo: false },
  { id: "speaker_2", name: "Speaker 2", type: "dialogue", volume: 0.8, mute: false, solo: false },
  { id: "speaker_3", name: "Speaker 3", type: "dialogue", volume: 0.8, mute: false, solo: false },
  { id: "speaker_4", name: "Speaker 4", type: "dialogue", volume: 0.8, mute: false, solo: false },
  { id: "music", name: "Background Music", type: "music", volume: 0.4, mute: false, solo: false },
  { id: "sfx", name: "Sound Effects", type: "sfx", volume: 0.6, mute: false, solo: false }
];

// --- UTILITY: Sequential Clip Reflow ---
const reflowClips = (clipsList) => {
  const dialogueClips = clipsList.filter(c => c.trackId && c.trackId.startsWith("speaker_"));
  const otherClips = clipsList.filter(c => !c.trackId || !c.trackId.startsWith("speaker_"));
  
  // Sort dialogue clips by their current startTime
  dialogueClips.sort((a, b) => a.startTime - b.startTime);
  
  let currentTime = 0;
  const reflowedDialogue = dialogueClips.map(clip => {
    if (clip.manuallyMoved) {
      currentTime = clip.startTime + (clip.duration || 0) + 0.2;
      return clip;
    } else {
      const updatedClip = { ...clip, startTime: currentTime };
      currentTime += (clip.duration || 0) + 0.2;
      return updatedClip;
    }
  });
  
  return [...reflowedDialogue, ...otherClips];
};

// --- MAIN COMPONENT ---
function AppInner({ barkPresets, setBarkPresets, darkMode, setDarkMode, initialConfig }) {
  // ------------- Google Drive Cloud Backup State -------------
  const [googleClientId, setGoogleClientId] = useState(() => {
    try { return localStorage.getItem("voication_google_client_id") || ""; } catch { return ""; }
  });
  const [driveAccessToken, setDriveAccessToken] = useState("");
  const [driveUserEmail, setDriveUserEmail] = useState("");
  const [isDriveSyncing, setIsDriveSyncing] = useState(false);
  const [driveBackupsList, setDriveBackupsList] = useState([]);
  const [driveSyncStatusMsg, setDriveSyncStatusMsg] = useState("");
  const [projectManagerTab, setProjectManagerTab] = useState("local");

  // Config states
  const [device, setDevice] = useState(initialConfig?.device || "auto");
  const [ollamaUrl, setOllamaUrl] = useState(initialConfig?.ollama_url || "http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState(initialConfig?.ollama_model || "llama3.1:latest");
  const [outputFolder, setOutputFolder] = useState(initialConfig?.output_folder || "output");

  // ------------- Main tab navigation state -------------
  const [activeMainTab, setActiveMainTab] = useState("experiment");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scriptHistory, setScriptHistory] = useState([]);
  const [podcastText, setPodcastText] = useState(() => {
    try {
      const storedProjects = localStorage.getItem("voication_saved_projects");
      const currentId = localStorage.getItem("voication_current_project_id");
      if (storedProjects && currentId) {
        const parsedProjects = JSON.parse(storedProjects);
        const activeProj = parsedProjects.find(p => p.id === currentId);
        if (activeProj && activeProj.podcastText !== undefined) {
          return activeProj.podcastText;
        }
      }
    } catch (e) {}
    return (
      "[Speaker 1] Welcome to the Cosmic Voyager podcast. Today we are diving into some mind-blowing space facts.\n\n" +
      "[Speaker 2] Oh, I love space! What do you have for us first?\n\n" +
      "[Speaker 1] Did you know that one day on Venus is longer than one year on Venus?\n\n" +
      "[Speaker 2] Wait, seriously? How is that even possible?\n\n" +
      "[Speaker 4] Narration: Our universe is filled with wonders that defy our everyday logic, where time and motion play by different rules."
    );
  });

  const [clonedProfiles, setClonedProfiles] = useState(() => {
    const DEFAULT_REFERENCES = [
      { name: "Australian Corporate Female (Reference)", type: "reference", voiceLibraryKey: "australian_female_corporate", transcript: "Good morning and welcome to the quarterly strategy briefing. Sarah Perry here presenting our sustainable development updates across the territory." },
      { name: "Australian Casual Male (Reference)", type: "reference", voiceLibraryKey: "australian_male_casual", transcript: "G'day mate, how's it going? Sarah was taking a look down by the domed observation window earlier, catching up on the morning routine." },
      { name: "Australian Dramatic Narrator Female (Reference)", type: "reference", voiceLibraryKey: "australian_female_narrator", transcript: "Six years and six months after departure. The domed glass observation window hummed softly against the vast quiet of the sky." },
      { name: "Australian Deep Cinematic Male (Reference)", type: "reference", voiceLibraryKey: "australian_male_deep", transcript: "Kepler's tower stood tall against the Blue Mountains horizon. A fortress forged in desperate times." },
      { name: "Australian High-Energy Promo Female (Reference)", type: "reference", voiceLibraryKey: "australian_female_promo", transcript: "Get ready for the ultimate skyward adventure! Join us live as we explore the rebuilt Alps community." },
      { name: "Australian Wise Elder Male (Reference)", type: "reference", voiceLibraryKey: "australian_male_elderly", transcript: "I remember the old world before the eye of the storm expanded. We had choices back then, true proxy salvation." },
      { name: "Australian Podcast Host Female (Reference)", type: "reference", voiceLibraryKey: "australian_female_podcast", transcript: "Welcome back to Mother of the Sky! Today we're diving into radio signals, commune politics, and survival tactics." },
      { name: "Australian Tech Specialist Male (Reference)", type: "reference", voiceLibraryKey: "australian_male_tech", transcript: "The cyclical food system recycles human waste into fertilizer and soil to keep the hydroponic greens growing." },
      { name: "Australian Children Storyteller Female (Reference)", type: "reference", voiceLibraryKey: "australian_female_story", transcript: "Deep within the blue marble valley, Kimmy helped tend to the wheat grass every morning with care and diligence." },
      { name: "Australian Outback Adventurer Male (Reference)", type: "reference", voiceLibraryKey: "australian_male_outback", transcript: "Out here in the rugged mountains, you grip the radio tight and listen out for any craft crossing the horizon." },
      { name: "British RP Female (Reference)", type: "reference", voiceLibraryKey: "british_female_rp", transcript: "Sarah Perry was a veterinary nurse of exceptional distinction, residing near the Duke Street Tower in North Square." },
      { name: "British Classic Aristocrat Male (Reference)", type: "reference", voiceLibraryKey: "british_male_classic", transcript: "Indeed, the exclusive community of thinkers and creators represents the pinnacle of human ingenuity." },
      { name: "British Casual London Female (Reference)", type: "reference", voiceLibraryKey: "british_female_casual", transcript: "Cheers for tuning in! Mind the step as we head up to the main observation deck." },
      { name: "American West Coast Female (Reference)", type: "reference", voiceLibraryKey: "american_female_warm", transcript: "Hey everyone, thanks for joining. Let's take a deep breath and start our morning practice together." },
      { name: "American Deep Broadcast Male (Reference)", type: "reference", voiceLibraryKey: "american_male_broadcast", transcript: "This is Mother of the Sky Radio, broadcasting live on all emergency frequencies." },
      { name: "American Soft Southern Female (Reference)", type: "reference", voiceLibraryKey: "american_female_southern", transcript: "Well now, isn't that just a sight to behold from all the way up here in the clouds." },
      { name: "Scottish Dramatic Character Male (Reference)", type: "reference", voiceLibraryKey: "scottish_male_dramatic", transcript: "Across the highland peaks, the wind howls loud, but our resolve remains unbroken!" },
      { name: "Irish Lyrical Storyteller Female (Reference)", type: "reference", voiceLibraryKey: "irish_female_story", transcript: "Listen closely to the crackle of the radio signal, carrying tales from across the emerald waves." },
      { name: "New Zealand Kiwi Guide Male (Reference)", type: "reference", voiceLibraryKey: "kiwi_male_guide", transcript: "Kia ora! We're tracking the small crafts moving freely between the floating communes today." },
      { name: "Canadian Documentary Female (Reference)", type: "reference", voiceLibraryKey: "canadian_female_doc", transcript: "Observations over the past four hundred kilometers reveal a unique atmospheric phenomenon." }
    ];

    try {
      const stored = localStorage.getItem("voication_cloned_profiles");
      if (stored) {
        let parsed = JSON.parse(stored);
        
        // Remove templates/presets and fake celebrity library voices
        const fakes = ["luke_skywalker", "elvis_presley", "john_lennon", "jfk", "the_queen", "jane_goodall"];
        parsed = parsed.filter(p => p.type !== "preset" && !(p.type === "library" && fakes.includes(p.voiceLibraryKey)));

        // Migrate any old 'library' types for Australian voices to 'reference'
        parsed = parsed.map(p => {
          if (p.voiceLibraryKey && (p.voiceLibraryKey.includes("australian_female_corporate") || p.voiceLibraryKey.includes("australian_male_casual"))) {
            return {
              ...p,
              type: "reference",
              name: p.name.replace("(Library)", "(Reference)")
            };
          }
          
          if (p.type === "clone" && p.fileBase64) {
            const fileBlob = base64ToBlob(p.fileBase64);
            return {
              name: p.name,
              type: p.type,
              voice: p.voice,
              file: fileBlob,
              fileBase64: p.fileBase64,
              transcript: p.transcript
            };
          }
          return p;
        });

        // Ensure default reference voices are present
        DEFAULT_REFERENCES.forEach(ref => {
          const exists = parsed.some(p => p.voiceLibraryKey === ref.voiceLibraryKey);
          if (!exists) {
            parsed.push(ref);
          }
        });

        return parsed;
      }
    } catch (e) {
      console.error("Error loading cloned profiles from localStorage:", e);
    }
    return DEFAULT_REFERENCES;
  });

  const [numberOfSpeakers, setNumberOfSpeakers] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_num_speakers");
      return stored !== null ? parseInt(stored) : 4;
    } catch {
      return 4;
    }
  });

  const [customCloneName, setCustomCloneName] = useState("");
  const [customCloneTranscript, setCustomCloneTranscript] = useState("");

  const [perspectiveSpeaker, setPerspectiveSpeaker] = useState(() => localStorage.getItem("voication_perspective_speaker") || "");
  const [quoteVoicing, setQuoteVoicing] = useState(() => localStorage.getItem("voication_quote_voicing") || "quoted_voice");
  const [customInstructions, setCustomInstructions] = useState(() => localStorage.getItem("voication_custom_instructions") || "");

  useEffect(() => {
    localStorage.setItem("voication_perspective_speaker", perspectiveSpeaker);
  }, [perspectiveSpeaker]);

  useEffect(() => {
    localStorage.setItem("voication_quote_voicing", quoteVoicing);
  }, [quoteVoicing]);

  useEffect(() => {
    localStorage.setItem("voication_custom_instructions", customInstructions);
  }, [customInstructions]);

  const useMps = device === "mps";

  // VITS-specific fields (must be inside component)
  // Bark seed for reproducible variation
  const [seed, setSeed] = useState(424242);
  const [vitsNoiseScale, setVitsNoiseScale] = useState(0.667);
  const [vitsDurationScale, setVitsDurationScale] = useState(1.0);
  const [vitsUsePhonemes, setVitsUsePhonemes] = useState(false);

  // ChatTTS-specific fields
  const [chatttsRefineText, setChatttsRefineText] = useState(true);
  const [chatttsSpkTemp, setChatttsSpkTemp] = useState(0.3);
  const [chatttsTextTemp, setChatttsTextTemp] = useState(0.3);
  const [chatttsSpkSeed, setChatttsSpkSeed] = useState("");
  const [chatttsTopP, setChatttsTopP] = useState(0.7);
  const [chatttsTopK, setChatttsTopK] = useState(20);
  const [isAdvancedAccordionOpen, setIsAdvancedAccordionOpen] = useState(false);

  // Fish Audio-specific fields
  const [fishEngine, setFishEngine] = useState("s2");
  const [fishNormalize, setFishNormalize] = useState(true);
  const [fishSimilarityWeight, setFishSimilarityWeight] = useState(0.7);
  const [fishPromptText, setFishPromptText] = useState("");
  
  const [stutterFrequency, setStutterFrequency] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_stutter_frequency");
      return stored !== null ? parseFloat(stored) : 0.0;
    } catch {
      return 0.0;
    }
  });
  
  const [amusementFrequency, setAmusementFrequency] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_amusement_frequency");
      return stored !== null ? parseFloat(stored) : 0.0;
    } catch {
      return 0.0;
    }
  });

  const [text, setText] = useState("");
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [soundAssets, setSoundAssets] = useState([]);
  const [isGeneratingMusic, setIsGeneratingMusic] = useState(false);
  const [isUploadingSound, setIsUploadingSound] = useState(false);
  const [musicPrompt, setMusicPrompt] = useState("");
  const [musicDuration, setMusicDuration] = useState(15);
  const [freesoundQuery, setFreesoundQuery] = useState("");
  const [freesoundType, setFreesoundType] = useState("sfx");
  const [freesoundResults, setFreesoundResults] = useState([]);
  const [isSearchingFreesound, setIsSearchingFreesound] = useState(false);
  const [playingSoundUrl, setPlayingSoundUrl] = useState(null);
  const [freesoundToken, setFreesoundToken] = useState(() => {
    try {
      return localStorage.getItem("voication_freesound_token") || "";
    } catch {
      return "";
    }
  });

  const fetchSoundAssets = () => {
    fetch("http://localhost:5000/api/sound-library/list")
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setSoundAssets(data);
        }
      })
      .catch(err => console.error("Error listing sound library:", err));
  };

  useEffect(() => {
    fetchSoundAssets();
  }, []);
  const [voiceDescription, setVoiceDescription] = useState("");
  const [language, setLanguage] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [speakerWav, setSpeakerWav] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [emotionIntensity, setEmotionIntensity] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_emotion_intensity");
      return stored !== null ? parseFloat(stored) : 0.5;
    } catch {
      return 0.5;
    }
  });
  const [chunkSize, setChunkSize] = useState(300);
  const [pauseDuration, setPauseDuration] = useState(0.5);

  // Keep barkSettings.use_mps in-sync with the top-level toggle
  useEffect(() => {
    setBarkSettings((prev) => ({ ...prev, use_mps: useMps }));
  }, [useMps]);

  // Bark sentence splitting and max duration (persistent)
  const [barkSplitSentences, setBarkSplitSentences] = useState(() => {
    try {
      const stored = localStorage.getItem("barkSplitSentences");
      return stored !== null ? stored === "true" : true;
    } catch {
      return true;
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

  // --- Bark Use MPS setting (persistent) ---
  const [barkSettings, setBarkSettings] = useState(() => {
    try {
      return {
        use_mps: localStorage.getItem("barkUseMps") === "true" ? true : false,
        smart_enhance:
          localStorage.getItem("barkSmartEnhance") === "true" ? true : false,
        small_models:
          localStorage.getItem("barkSmallModels") === "true" ? true : false,
        skip_fine:
          localStorage.getItem("barkSkipFine") === "true" ? true : false,
      };
    } catch {
      return {
        use_mps: false,
        smart_enhance: false,
        small_models: false,
        skip_fine: false,
      };
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
  const [voiceDirection, setVoiceDirection] = useState("");
  const [streamingLatency, setStreamingLatency] = useState(false);

  const [speakerMapping, setSpeakerMapping] = useState(() => {
    try {
      const stored = localStorage.getItem("vibevoice_speaker_mapping");
      if (stored) {
        const parsed = JSON.parse(stored);
        const normalized = {};
        for (let k of ["speaker_1", "speaker_2", "speaker_3", "speaker_4"]) {
          let val = parsed[k];
          if (val && !val.includes(":")) {
            normalized[k] = `tts_models/en/vctk/vits:${val}`;
          } else {
            normalized[k] = val;
          }
        }
        return normalized;
      }
      return {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:am_adam",
        speaker_3: "bark:v2/en_speaker_9",
        speaker_4: "tts_models/en/vctk/vits:p232"
      };
    } catch {
      return {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:am_adam",
        speaker_3: "bark:v2/en_speaker_9",
        speaker_4: "tts_models/en/vctk/vits:p232"
      };
    }
  });

  const prevSpeakerMappingRef = useRef(speakerMapping);
  useEffect(() => {
    const prev = prevSpeakerMappingRef.current;
    const changedTracks = [];
    Object.keys(speakerMapping).forEach(trackId => {
      if (prev[trackId] && prev[trackId] !== speakerMapping[trackId]) {
        changedTracks.push(trackId);
      }
    });
    prevSpeakerMappingRef.current = speakerMapping;

    if (changedTracks.length > 0) {
      setPlaylistClips(clips => 
        clips.map(c => changedTracks.includes(c.trackId) ? { ...c, status: "needs-render" } : c)
      );
    }
  }, [speakerMapping]);

  const [speakerNames, setSpeakerNames] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_speaker_names");
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {}
    return {
      speaker_1: "Speaker 1",
      speaker_2: "Speaker 2",
      speaker_3: "Speaker 3",
      speaker_4: "Speaker 4"
    };
  });

  const [voicePreset, setVoicePreset] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  // Text-level AI-Enhance toggle (independent of post-processing)
  const [smartEnhance, setSmartEnhance] = useState(false);

  const [enhancePrompt, setEnhancePrompt] = useState("");
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isAutoTagging, setIsAutoTagging] = useState(false);
  const [showSoundTagger, setShowSoundTagger] = useState(false);
  const [soundTaggerPrompt, setSoundTaggerPrompt] = useState("");
  const [isAutoTaggingSound, setIsAutoTaggingSound] = useState(false);

  // --- Curated Voices, Media Formats, Active Speaker & Project Manager States ---
  const [curatedVoices, setCuratedVoices] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_curated_voices");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });
  const [loadedCuratedVoiceId, setLoadedCuratedVoiceId] = useState(() => {
    try {
      return localStorage.getItem("voication_loaded_curated_voice_id") || null;
    } catch {
      return null;
    }
  });

  const [customRecipes, setCustomRecipes] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_custom_recipes");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("voication_custom_recipes", JSON.stringify(customRecipes));
    } catch (e) {}
  }, [customRecipes]);

  useEffect(() => {
    try {
      localStorage.setItem("voication_curated_voices", JSON.stringify(curatedVoices));
    } catch (e) {}
  }, [curatedVoices]);

  const [ppHardLimiter, setPpHardLimiter] = useState(false);
  const [ppPodcastVoice, setPpPodcastVoice] = useState(false);
  const [ppMastering, setPpMastering] = useState(false);

  const [mediaFormat, setMediaFormat] = useState(() => {
    try {
      return localStorage.getItem("voication_media_format") || "podcast";
    } catch {
      return "podcast";
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("voication_media_format", mediaFormat);
    } catch (e) {}
  }, [mediaFormat]);

  const [enabledModels, setEnabledModels] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_enabled_models");
      const base = stored ? JSON.parse(stored) : ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"];
      // Auto-migrate: add new models that aren't in an old saved list
      const alwaysInclude = ["qwen3-tts"];
      const migrated = [...base];
      alwaysInclude.forEach(m => { if (!migrated.includes(m)) migrated.push(m); });
      return migrated;
    } catch {
      return ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("voication_enabled_models", JSON.stringify(enabledModels));
    } catch (e) {}
  }, [enabledModels]);

  const [showOnboardingBanner, setShowOnboardingBanner] = useState(() => {
    try {
      return localStorage.getItem("voication_show_onboarding") !== "false";
    } catch {
      return true;
    }
  });

  const getModelKey = (modelName) => {
    if (!modelName) return "";
    const name = modelName.toLowerCase();
    if (name.includes("chattts")) return "chattts";
    if (name.includes("fish-audio")) return "fish-audio";
    if (name.includes("kokoro")) return "kokoro";
    if (name.includes("bark")) return "bark";
    if (name.includes("vits")) return "vits";
    if (name.includes("qwen3-tts") || name.includes("qwen")) return "qwen3-tts";
    if (name.includes("chatterbox")) return "chatterbox-turbo";
    if (name.includes("cosy") || name.includes("cosyvoice")) return "cosyvoice2-styletts2";
    if (name.includes("xtts")) return "xtts";
    return modelName;
  };

  const isModelEnabled = (modelName) => {
    if (!currentProjectId) {
      return enabledModels.includes(getModelKey(modelName));
    }
    const activeProj = projects.find(p => p.id === currentProjectId);
    if (!activeProj) {
      return enabledModels.includes(getModelKey(modelName));
    }
    const enabled = activeProj.enabledModels || ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"];
    const key = getModelKey(modelName);
    return enabled.includes(key);
  };


  const [activeSpeakerKey, setActiveSpeakerKey] = useState("speaker_1");

  const [projects, setProjects] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_saved_projects");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  const [isSaving, setIsSaving] = useState(false);

  // --- Nested Project & Chapter States ---
  const [chapters, setChapters] = useState([
    { id: "chapter_1", name: "Chapter 1", podcastText: "[Speaker 1] Write script here.", playlistClips: [], playlistTracks: DEFAULT_TIMELINE_TRACKS, usePhoneticSettings: true }
  ]);
  const [currentChapterId, setCurrentChapterId] = useState("chapter_1");
  const [globalSummary, setGlobalSummary] = useState("A summary of this novel/project.");
  const [phoneticDict, setPhoneticDict] = useState([]);
  const [spellOutAcronyms, setSpellOutAcronyms] = useState(false);
  const [ignoreEmojis, setIgnoreEmojis] = useState(false);
  const [ignoreSpecialSymbols, setIgnoreSpecialSymbols] = useState(false);
  const [obsidianVaultPath, setObsidianVaultPath] = useState("");
  const [playgroundSubTab, setPlaygroundSubTab] = useState("voice"); // "voice" or "sound"
  const [storytellerViewMode, setStorytellerViewMode] = useState("overview"); // "overview" or "editor"
  const [chapterEditorTab, setChapterEditorTab] = useState("script"); // "script" or "multitrack"
  const [projectViewMode, setProjectViewMode] = useState("overview"); // "overview" or "multitrack"
  const [speakerColors, setSpeakerColors] = useState(() => {
    try {
      const stored = localStorage.getItem("voication_speaker_colors");
      return stored ? JSON.parse(stored) : {
        speaker_1: "#4f46e5",
        speaker_2: "#059669",
        speaker_3: "#d97706",
        speaker_4: "#e11d48",
      };
    } catch {
      return {
        speaker_1: "#4f46e5",
        speaker_2: "#059669",
        speaker_3: "#d97706",
        speaker_4: "#e11d48",
      };
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("voication_speaker_colors", JSON.stringify(speakerColors));
    } catch (e) {}
  }, [speakerColors]);

  const [reviewNotes, setReviewNotes] = useState([]);
  const [activeReviewAudioUrl, setActiveReviewAudioUrl] = useState("");
  const [rightPanelTab, setRightPanelTab] = useState("queue");
  const [selectedReviewText, setSelectedReviewText] = useState("");
  const [isSuggestingEdit, setIsSuggestingEdit] = useState(false);
  const reviewAudioRef = useRef(null);
  const [newNoteText, setNewNoteText] = useState("");
  const [newNoteTarget, setNewNoteTarget] = useState("");
  const [autoPauseOnType, setAutoPauseOnType] = useState(true);
  const [reviewAudioDuration, setReviewAudioDuration] = useState(0);
  const [reviewCurrentTime, setReviewCurrentTime] = useState(0);
  const [reviewPlaybackRate, setReviewPlaybackRate] = useState(1);

  const [currentProjectId, setCurrentProjectId] = useState(() => {
    try {
      return localStorage.getItem("voication_current_project_id") || "";
    } catch {
      return "";
    }
  });

  const [activeProjectName, setActiveProjectName] = useState("Untitled Project");
  const [currentProjectCreatedAt, setCurrentProjectCreatedAt] = useState("");
  const [showProjectManager, setShowProjectManager] = useState(false);

  // --- Phonetic Dictionary UI Helper States ---
  const [newPhoneticWord, setNewPhoneticWord] = useState("");
  const [newPhoneticReplacement, setNewPhoneticReplacement] = useState("");
  const [editingPhoneticId, setEditingPhoneticId] = useState(null);
  const [editingPhoneticWord, setEditingPhoneticWord] = useState("");
  const [editingPhoneticReplacement, setEditingPhoneticReplacement] = useState("");
  const [newPhoneticType, setNewPhoneticType] = useState("standard");
  const [editingPhoneticType, setEditingPhoneticType] = useState("standard");
  const [showPronunciationWizard, setShowPronunciationWizard] = useState(false);
  const [wizardTargetWord, setWizardTargetWord] = useState("");
  const [wizardEthnicity, setWizardEthnicity] = useState("");
  const [wizardSuggestions, setWizardSuggestions] = useState(null);
  const [wizardLoading, setWizardLoading] = useState(false);
  const [wizardRecording, setWizardRecording] = useState(false);
  const [editingProfileIdx, setEditingProfileIdx] = useState(null);
  const [editingProfileName, setEditingProfileName] = useState("");
  const [editingProfileTranscript, setEditingProfileTranscript] = useState("");
  const [wizardAudioUrl, setWizardAudioUrl] = useState(null);
  const [wizardAudioBlob, setWizardAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recognitionRef = useRef(null);
  const voiceCreatorRecognitionRef = useRef(null);

  // --- Waveform Playlist Timeline States ---
  const [playlistTracks, setPlaylistTracks] = useState(() => {
    try {
      const storedProjects = localStorage.getItem("voication_saved_projects");
      const currentId = localStorage.getItem("voication_current_project_id");
      if (storedProjects && currentId) {
        const parsedProjects = JSON.parse(storedProjects);
        const activeProj = parsedProjects.find(p => p.id === currentId);
        if (activeProj && activeProj.playlistTracks && activeProj.playlistTracks.length > 0) {
          return activeProj.playlistTracks;
        }
      }
      const stored = localStorage.getItem("vibevoice_playlist_tracks");
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && parsed.length > 0) return parsed;
      }
    } catch (e) {}
    return DEFAULT_TIMELINE_TRACKS;
  });

  const [playlistClips, setPlaylistClips] = useState(() => {
    try {
      const storedProjects = localStorage.getItem("voication_saved_projects");
      const currentId = localStorage.getItem("voication_current_project_id");
      if (storedProjects && currentId) {
        const parsedProjects = JSON.parse(storedProjects);
        const activeProj = parsedProjects.find(p => p.id === currentId);
        if (activeProj && activeProj.playlistClips) {
          return activeProj.playlistClips;
        }
      }
    } catch (e) {}
    return [];
  });

  // Sync projects list to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("voication_saved_projects", JSON.stringify(projects));
    } catch (e) {}
  }, [projects]);

  // Load curated voice back into sandbox
  const loadCuratedVoice = (curated, voicesList = voices, clonedList = clonedProfiles) => {
    if (!curated) return;
    setLoadedCuratedVoiceId(curated.id);
    try {
      localStorage.setItem("voication_loaded_curated_voice_id", curated.id);
    } catch (e) {}
    const modelData = voicesList.find(v => v.name === curated.model);
    if (!modelData) {
      alert(`Model ${curated.model} not found!`);
      return;
    }
    setSelectedVoice(curated.voice);
    
    const settings = curated.settings || {};
    if (curated.model === "vits") {
      if (settings.vitsNoiseScale !== undefined) setVitsNoiseScale(settings.vitsNoiseScale);
      if (settings.vitsDurationScale !== undefined) setVitsDurationScale(settings.vitsDurationScale);
      if (settings.vitsUsePhonemes !== undefined) setVitsUsePhonemes(settings.vitsUsePhonemes);
    } else if (curated.model === "chattts") {
      if (settings.chattts_refine_text !== undefined) setChatttsRefineText(settings.chattts_refine_text);
      if (settings.chattts_spk_temp !== undefined) setChatttsSpkTemp(settings.chattts_spk_temp);
      if (settings.chattts_text_temp !== undefined) setChatttsTextTemp(settings.chattts_text_temp);
      if (settings.chattts_spk_seed !== undefined) setChatttsSpkSeed(settings.chattts_spk_seed);
    } else if (curated.model === "fish-audio") {
      if (settings.fish_engine !== undefined) setFishEngine(settings.fish_engine);
      if (settings.fish_normalize !== undefined) setFishNormalize(settings.fish_normalize);
      if (settings.fish_similarity_weight !== undefined) setFishSimilarityWeight(settings.fish_similarity_weight);
      if (settings.fish_prompt_text !== undefined) setFishPromptText(settings.fish_prompt_text);
    } else if (curated.model === "bark") {
      if (settings.creativity !== undefined) setBarkTemperature(settings.creativity);
      if (settings.pool !== undefined) setBarkTopK(settings.pool);
      if (settings.focus !== undefined) setBarkTopP(settings.focus);
      if (settings.barkSplitSentences !== undefined) setBarkSplitSentences(settings.barkSplitSentences);
      if (settings.barkMaxDuration !== undefined) setBarkMaxDuration(settings.barkMaxDuration);
    } else if (curated.model === "xtts") {
      if (settings.length_scale !== undefined) setXttsLengthScale(settings.length_scale);
      if (settings.noise_scale !== undefined) setXttsNoiseScale(settings.noise_scale);
      if (settings.noise_scale_w !== undefined) setXttsNoiseScaleW(settings.noise_scale_w);
      if (settings.voice_direction !== undefined) setVoiceDirection(settings.voice_direction);
    } else if (curated.model === "qwen3-tts") {
      // Restore the style instruction prompt
      if (settings.voice_direction !== undefined) setVoiceDirection(settings.voice_direction);
    }

    if (settings.speed !== undefined) setSpeed(settings.speed);
    if (settings.emotion_intensity !== undefined) setEmotionIntensity(settings.emotion_intensity);
    if (settings.chunk_size !== undefined) setChunkSize(settings.chunk_size);
    if (settings.pause_duration !== undefined) setPauseDuration(settings.pause_duration);
    
    // Restore speaker and voicePreset configurations
    if (settings.speaker !== undefined) setSpeaker(settings.speaker);
    if (settings.voicePreset !== undefined) setVoicePreset(settings.voicePreset);
    
    if (settings.activeCloneProfileName) {
      const p = clonedList.find(profile => profile.name === settings.activeCloneProfileName);
      if (p) {
        setActiveCloneProfile(p);
      } else {
        setActiveCloneProfile(null);
      }
    } else {
      setActiveCloneProfile(null);
    }

    if (curated.fileBase64) {
      const fileBlob = base64ToBlob(curated.fileBase64);
      setRecordedBlob(fileBlob);
      formDataRef.current.set("speaker_wav", fileBlob);
    } else {
      setRecordedBlob(null);
      formDataRef.current.delete("speaker_wav");
    }
    alert(`Loaded curated voice "${curated.name}" parameters into the sandbox.`);
  };

  // Save curated voice configured in sandbox
  const saveAsCuratedVoice = async () => {
    const defaultName = `Curated ${selectedVoiceData?.name || "Voice"} (${new Date().toLocaleTimeString()})`;
    const name = prompt("Enter a name for this curated voice:", defaultName);
    if (!name || !name.trim()) return;

    let fileBase64 = null;
    if (recordedBlob && (selectedVoiceData?.requires_speaker_wav || selectedVoiceData?.features?.includes("cloning"))) {
      try {
        fileBase64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(recordedBlob);
        });
      } catch (e) {
        console.error("Failed to read reference WAV for curated voice:", e);
      }
    }

    const settings = {};
    if (selectedVoiceData?.name === "vits") {
      settings.vitsNoiseScale = vitsNoiseScale;
      settings.vitsDurationScale = vitsDurationScale;
      settings.vitsUsePhonemes = vitsUsePhonemes;
    } else if (selectedVoiceData?.name === "chattts") {
      settings.chattts_refine_text = chatttsRefineText;
      settings.chattts_spk_temp = chatttsSpkTemp;
      settings.chattts_text_temp = chatttsTextTemp;
      settings.chattts_spk_seed = chatttsSpkSeed;
      settings.chattts_top_p = chatttsTopP;
      settings.chattts_top_k = chatttsTopK;
    } else if (selectedVoiceData?.name === "fish-audio") {
      settings.fish_engine = fishEngine;
      settings.fish_normalize = fishNormalize;
      settings.fish_similarity_weight = fishSimilarityWeight;
      settings.fish_prompt_text = fishPromptText;
    } else if (selectedVoiceData?.name === "bark") {
      settings.creativity = barkTemperature;
      settings.pool = barkTopK;
      settings.focus = barkTopP;
      settings.barkSplitSentences = barkSplitSentences;
      settings.barkMaxDuration = barkMaxDuration;
    } else if (selectedVoiceData?.name === "xtts") {
      settings.length_scale = xttsLengthScale;
      settings.noise_scale = xttsNoiseScale;
      settings.noise_scale_w = xttsNoiseScaleW;
      settings.voice_direction = voiceDirection;
    } else if (selectedVoiceData?.name === "qwen3-tts") {
      // Save the style instruction so it can be reused from the Voice Library
      settings.voice_direction = voiceDirection;
    }

    settings.speed = speed;
    settings.emotion_intensity = emotionIntensity;
    settings.chunk_size = chunkSize;
    settings.pause_duration = pauseDuration;
    settings.speaker = speaker;
    settings.voicePreset = voicePreset;
    settings.activeCloneProfileName = activeCloneProfile ? activeCloneProfile.name : null;

    const newVoice = {
      id: `curated_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      name: name.trim(),
      model: selectedVoiceData?.name || "",
      voice: selectedVoice || "",
      settings,
      fileBase64
    };

    setCuratedVoices(prev => [...prev, newVoice]);
    alert(`Curated voice "${name}" saved!`);
  };

  // Overwrite existing loaded curated voice settings
  const updateCuratedVoice = async () => {
    if (!loadedCuratedVoiceId) return;
    const target = curatedVoices.find(v => v.id === loadedCuratedVoiceId);
    if (!target) return;

    let fileBase64 = target.fileBase64;
    if (recordedBlob && (selectedVoiceData?.requires_speaker_wav || selectedVoiceData?.features?.includes("cloning"))) {
      try {
        fileBase64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(recordedBlob);
        });
      } catch (e) {
        console.error("Failed to read reference WAV for curated voice:", e);
      }
    }

    const settings = {};
    if (selectedVoiceData?.name === "vits") {
      settings.vitsNoiseScale = vitsNoiseScale;
      settings.vitsDurationScale = vitsDurationScale;
      settings.vitsUsePhonemes = vitsUsePhonemes;
    } else if (selectedVoiceData?.name === "chattts") {
      settings.chattts_refine_text = chatttsRefineText;
      settings.chattts_spk_temp = chatttsSpkTemp;
      settings.chattts_text_temp = chatttsTextTemp;
      settings.chattts_spk_seed = chatttsSpkSeed;
      settings.chattts_top_p = chatttsTopP;
      settings.chattts_top_k = chatttsTopK;
    } else if (selectedVoiceData?.name === "fish-audio") {
      settings.fish_engine = fishEngine;
      settings.fish_normalize = fishNormalize;
      settings.fish_similarity_weight = fishSimilarityWeight;
      settings.fish_prompt_text = fishPromptText;
    } else if (selectedVoiceData?.name === "bark") {
      settings.creativity = barkTemperature;
      settings.pool = barkTopK;
      settings.focus = barkTopP;
      settings.barkSplitSentences = barkSplitSentences;
      settings.barkMaxDuration = barkMaxDuration;
    } else if (selectedVoiceData?.name === "xtts") {
      settings.length_scale = xttsLengthScale;
      settings.noise_scale = xttsNoiseScale;
      settings.noise_scale_w = xttsNoiseScaleW;
      settings.voice_direction = voiceDirection;
    } else if (selectedVoiceData?.name === "qwen3-tts") {
      settings.voice_direction = voiceDirection;
    }

    settings.speed = speed;
    settings.emotion_intensity = emotionIntensity;
    settings.chunk_size = chunkSize;
    settings.pause_duration = pauseDuration;
    settings.speaker = speaker;
    settings.voicePreset = voicePreset;
    settings.activeCloneProfileName = activeCloneProfile ? activeCloneProfile.name : null;

    setCuratedVoices(prev => prev.map(item => {
      if (item.id === loadedCuratedVoiceId) {
        return {
          ...item,
          voice: selectedVoice || "",
          settings,
          fileBase64
        };
      }
      return item;
    }));
    alert(`Curated voice "${target.name}" settings updated!`);
  };

  // ------------- Google Drive Cloud Sync Handlers -------------
  const handleConnectDrive = () => {
    setIsDriveSyncing(true);
    setDriveSyncStatusMsg("Connecting to Google Identity Services...");
    requestDriveAccessToken(
      googleClientId,
      (data) => {
        setDriveAccessToken(data.accessToken);
        setDriveUserEmail(data.userEmail);
        setIsDriveSyncing(false);
        setDriveSyncStatusMsg(`Connected as ${data.userEmail || "Google User"}`);
        handleRefreshDriveBackups(data.accessToken);
      },
      (err) => {
        setIsDriveSyncing(false);
        setDriveSyncStatusMsg(`Connection failed: ${err}`);
        alert(`Google Drive connection error: ${err}`);
      }
    );
  };

  const handleBackupProjectToDrive = async (proj) => {
    if (!driveAccessToken) {
      handleConnectDrive();
      return;
    }
    setIsDriveSyncing(true);
    setDriveSyncStatusMsg(`Backing up "${proj.name}" to Google Drive...`);
    try {
      await uploadProjectToDrive(proj, driveAccessToken);
      setDriveSyncStatusMsg(`Successfully backed up "${proj.name}" to Google Drive!`);
      handleRefreshDriveBackups(driveAccessToken);
    } catch (err) {
      console.error("Drive upload error:", err);
      setDriveSyncStatusMsg(`Backup failed: ${err.message}`);
      alert(`Failed to upload to Google Drive: ${err.message}`);
    } finally {
      setIsDriveSyncing(false);
    }
  };

  const handleBackupAllProjectsToDrive = async () => {
    if (!driveAccessToken) {
      handleConnectDrive();
      return;
    }
    if (!projects || projects.length === 0) {
      alert("No projects available to backup!");
      return;
    }
    setIsDriveSyncing(true);
    setDriveSyncStatusMsg("Backing up all projects to Google Drive...");
    try {
      for (const proj of projects) {
        await uploadProjectToDrive(proj, driveAccessToken);
      }
      setDriveSyncStatusMsg(`Successfully backed up ${projects.length} project(s) to Google Drive!`);
      handleRefreshDriveBackups(driveAccessToken);
    } catch (err) {
      console.error("Batch Drive upload error:", err);
      setDriveSyncStatusMsg(`Batch backup failed: ${err.message}`);
      alert(`Batch upload failed: ${err.message}`);
    } finally {
      setIsDriveSyncing(false);
    }
  };

  const handleRefreshDriveBackups = async (token = driveAccessToken) => {
    if (!token) return;
    setIsDriveSyncing(true);
    try {
      const backups = await fetchDriveBackups(token);
      setDriveBackupsList(backups);
    } catch (err) {
      console.error("Fetch Drive backups error:", err);
    } finally {
      setIsDriveSyncing(false);
    }
  };

  const handleRestoreFromDrive = async (file) => {
    if (!driveAccessToken) return;
    if (!confirm(`Restore "${file.name}" from Google Drive into Voication Studio?`)) return;
    setIsDriveSyncing(true);
    setDriveSyncStatusMsg(`Downloading "${file.name}"...`);
    try {
      const projectData = await downloadDriveProject(file.id, driveAccessToken);
      if (!projectData.id || !projectData.name) {
        throw new Error("Invalid project JSON structure.");
      }
      
      setProjects(prev => {
        const exists = prev.some(p => p.id === projectData.id);
        if (exists) {
          return prev.map(p => p.id === projectData.id ? projectData : p);
        } else {
          return [projectData, ...prev];
        }
      });
      
      loadProject(projectData);
      setDriveSyncStatusMsg(`Successfully restored "${projectData.name}"!`);
      alert(`Project "${projectData.name}" restored and loaded!`);
    } catch (err) {
      console.error("Drive restore error:", err);
      setDriveSyncStatusMsg(`Restore failed: ${err.message}`);
      alert(`Failed to restore project from Drive: ${err.message}`);
    } finally {
      setIsDriveSyncing(false);
    }
  };

  // Load a project config
  const loadProject = (project) => {
    if (!project) return;
    setLoadedCuratedVoiceId(null);
    
    // Automatically migrate flat project configuration
    let migrated = project;
    if (!project.chapters || project.chapters.length === 0) {
      const defaultChapter = {
        id: "chapter_1",
        name: "Chapter 1",
        podcastText: project.podcastText || "",
        playlistClips: project.playlistClips || [],
        playlistTracks: project.playlistTracks && project.playlistTracks.length > 0 ? project.playlistTracks : DEFAULT_TIMELINE_TRACKS
      };
      migrated = {
        ...project,
        globalSummary: project.globalSummary || "A summary of this novel/project.",
        obsidianVaultPath: project.obsidianVaultPath || "",
        activeChapterId: "chapter_1",
        chapters: [defaultChapter]
      };
    }

    setCurrentProjectId(migrated.id);
    setActiveProjectName(migrated.name);
    setCurrentProjectCreatedAt(migrated.createdAt || new Date().toISOString());
    setMediaFormat(migrated.mediaFormat || "podcast");
    setNumberOfSpeakers(migrated.numberOfSpeakers || 4);
    setSpeakerMapping(migrated.speakerMapping || {});
    setSpeakerNames(migrated.speakerNames || {});
    setSpeakerColors(migrated.speakerColors || {
      speaker_1: "#4f46e5",
      speaker_2: "#059669",
      speaker_3: "#d97706",
      speaker_4: "#e11d48",
    });
    setEnabledModels(migrated.enabledModels || ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"]);
    
    // Set nested states
    setGlobalSummary(migrated.globalSummary || "A summary of this novel/project.");
    setObsidianVaultPath(migrated.obsidianVaultPath || "");
    setChapters(migrated.chapters);
    
    setPerspectiveSpeaker(migrated.perspectiveSpeaker || "");
    setQuoteVoicing(migrated.quoteVoicing || "own_voice");
    setCustomInstructions(migrated.customInstructions || "");
    
    // Set active chapter
    const activeChId = migrated.activeChapterId || migrated.chapters[0].id;
    setCurrentChapterId(activeChId);
    
    const activeCh = migrated.chapters.find(c => c.id === activeChId) || migrated.chapters[0];
    setPodcastText(activeCh.podcastText || "");
    setPlaylistClips(activeCh.playlistClips || []);
    setPlaylistTracks(activeCh.playlistTracks && activeCh.playlistTracks.length > 0 ? activeCh.playlistTracks : DEFAULT_TIMELINE_TRACKS);
    
    if (migrated.curatedVoices && Array.isArray(migrated.curatedVoices)) {
      setCuratedVoices(prev => {
        const merged = [...prev];
        migrated.curatedVoices.forEach(cv => {
          if (!merged.some(item => item.id === cv.id)) {
            merged.push(cv);
          }
        });
        return merged;
      });
    }
    try {
      localStorage.setItem("voication_current_project_id", migrated.id);
    } catch (e) {}
  };

  // Create a new blank project
  const createNewProject = () => {
    const defaultId = `project_${Date.now()}`;
    const defaultProject = {
      id: defaultId,
      name: "New Studio Project",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      mediaFormat: "podcast",
      numberOfSpeakers: 4,
      speakerMapping: {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:am_adam",
        speaker_3: "bark:v2/en_speaker_9",
        speaker_4: "tts_models/en/vctk/vits:p232"
      },
      speakerNames: {
        speaker_1: "Speaker 1",
        speaker_2: "Speaker 2",
        speaker_3: "Speaker 3",
        speaker_4: "Speaker 4"
      },
      podcastText: "[Speaker 1] New script text starts here.",
      playlistClips: [],
      playlistTracks: DEFAULT_TIMELINE_TRACKS,
      curatedVoices: [],
      enabledModels: ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"]
    };
    setProjects(prev => [defaultProject, ...prev]);
    loadProject(defaultProject);
  };

  // Initial Project Load
  useEffect(() => {
    if (projects.length > 0) {
      let activeProj = projects.find(p => p.id === currentProjectId);
      if (!activeProj) {
        activeProj = projects[0];
      }
      if (activeProj) {
        setCurrentProjectId(activeProj.id);
        setActiveProjectName(activeProj.name);
        setCurrentProjectCreatedAt(activeProj.createdAt || new Date().toISOString());
        setMediaFormat(activeProj.mediaFormat || "podcast");
        setNumberOfSpeakers(activeProj.numberOfSpeakers || 4);
        setSpeakerMapping(activeProj.speakerMapping || {});
        setSpeakerNames(activeProj.speakerNames || {});
        setSpeakerColors(activeProj.speakerColors || {
          speaker_1: "#4f46e5",
          speaker_2: "#059669",
          speaker_3: "#d97706",
          speaker_4: "#e11d48",
        });
        setEnabledModels(activeProj.enabledModels || ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"]);
        setPhoneticDict(activeProj.phoneticDict || []);
        setSpellOutAcronyms(activeProj.spellOutAcronyms || false);
        setIgnoreEmojis(activeProj.ignoreEmojis || false);
        setIgnoreSpecialSymbols(activeProj.ignoreSpecialSymbols || false);
        setPpHardLimiter(activeProj.ppHardLimiter || false);
        setPpPodcastVoice(activeProj.ppPodcastVoice || false);
        setPpMastering(activeProj.ppMastering || false);

        // Load chapters if they exist; otherwise initialize from legacy flat structure
        if (activeProj.chapters && activeProj.chapters.length > 0) {
          const mappedChapters = activeProj.chapters.map(c => ({
            ...c,
            usePhoneticSettings: c.usePhoneticSettings !== undefined ? c.usePhoneticSettings : true
          }));
          setChapters(mappedChapters);
          const activeChId = activeProj.activeChapterId || activeProj.chapters[0].id;
          setCurrentChapterId(activeChId);
          const activeCh = mappedChapters.find(c => c.id === activeChId) || mappedChapters[0];
          setPodcastText(activeCh.podcastText || "");
          setPlaylistClips(activeCh.playlistClips || []);
          setPlaylistTracks(activeCh.playlistTracks && activeCh.playlistTracks.length > 0 ? activeCh.playlistTracks : DEFAULT_TIMELINE_TRACKS);
        } else {
          const legacyChapter = {
            id: "chapter_1",
            name: "Chapter 1",
            podcastText: activeProj.podcastText || "",
            playlistClips: activeProj.playlistClips || [],
            playlistTracks: activeProj.playlistTracks && activeProj.playlistTracks.length > 0 ? activeProj.playlistTracks : DEFAULT_TIMELINE_TRACKS,
            usePhoneticSettings: true
          };
          setChapters([legacyChapter]);
          setCurrentChapterId("chapter_1");
          setPodcastText(legacyChapter.podcastText);
          setPlaylistClips(legacyChapter.playlistClips);
          setPlaylistTracks(legacyChapter.playlistTracks);
        }
      }
    } else {
      const defaultId = `project_${Date.now()}`;
      const defaultProject = {
        id: defaultId,
        name: "Welcome Studio Project",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        mediaFormat: "podcast",
        numberOfSpeakers: 4,
        speakerMapping: {
          speaker_1: "kokoro:af_bella",
          speaker_2: "kokoro:am_adam",
          speaker_3: "bark:v2/en_speaker_9",
          speaker_4: "tts_models/en/vctk/vits:p232"
        },
        speakerNames: {
          speaker_1: "Speaker 1",
          speaker_2: "Speaker 2",
          speaker_3: "Speaker 3",
          speaker_4: "Speaker 4"
        },
        playlistText: "[Speaker 1] Welcome to Voication Studio. You can edit this script and voices here.",
        playlistClips: [],
        playlistTracks: DEFAULT_TIMELINE_TRACKS,
        curatedVoices: [],
        enabledModels: ["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"],
        phoneticDict: [],
        spellOutAcronyms: false,
        ignoreEmojis: false,
        ignoreSpecialSymbols: false
      };
      setProjects([defaultProject]);
      setCurrentProjectId(defaultId);
      setActiveProjectName(defaultProject.name);
      setCurrentProjectCreatedAt(defaultProject.createdAt);
      setEnabledModels(["vits", "kokoro", "bark", "chattts", "fish-audio", "qwen3-tts"]);
      try {
        localStorage.setItem("voication_current_project_id", defaultId);
      } catch (e) {}
    }
  }, []);

  const saveProjectSync = (projId = currentProjectId) => {
    if (!projId) return;
    const projName = activeProjectName || "Untitled Project";
    
    const updatedChapters = chapters.map(ch => {
      if (ch.id === currentChapterId) {
        return {
          ...ch,
          podcastText,
          playlistClips,
          playlistTracks
        };
      }
      return ch;
    });

    const updatedProject = {
      id: projId,
      name: projName,
      createdAt: currentProjectCreatedAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      mediaFormat,
      numberOfSpeakers,
      speakerMapping,
      speakerNames,
      speakerColors,
      curatedVoices,
      enabledModels,
      globalSummary,
      obsidianVaultPath,
      phoneticDict,
      spellOutAcronyms,
      ignoreEmojis,
      ignoreSpecialSymbols,
      ppHardLimiter,
      ppPodcastVoice,
      ppMastering,
      perspectiveSpeaker,
      quoteVoicing,
      customInstructions,
      activeChapterId: currentChapterId,
      chapters: updatedChapters
    };
    
    try {
      const stored = localStorage.getItem("voication_saved_projects");
      const currentProjects = stored ? JSON.parse(stored) : [];
      const filtered = currentProjects.filter(p => p.id !== projId);
      const updatedList = [updatedProject, ...filtered];
      localStorage.setItem("voication_saved_projects", JSON.stringify(updatedList));
      localStorage.setItem("voication_current_project_id", projId);
      localStorage.setItem("vibevoice_playlist_tracks", JSON.stringify(playlistTracks));
      setProjects(updatedList);
    } catch (e) {
      console.error("Failed to sync project to local storage", e);
    }
  };

  const switchChapter = (newChapterId) => {
    const updatedChapters = chapters.map(ch => {
      if (ch.id === currentChapterId) {
        return {
          ...ch,
          podcastText,
          playlistClips,
          playlistTracks
        };
      }
      return ch;
    });
    setChapters(updatedChapters);

    const target = updatedChapters.find(ch => ch.id === newChapterId);
    if (target) {
      setCurrentChapterId(newChapterId);
      setPodcastText(target.podcastText || "");
      setPlaylistClips(target.playlistClips || []);
      setPlaylistTracks(target.playlistTracks && target.playlistTracks.length > 0 ? target.playlistTracks : DEFAULT_TIMELINE_TRACKS);
      setStorytellerViewMode("editor");
      setTimeout(() => saveProjectSync(), 100);
    }
  };

  const goBackToOverview = () => {
    const updated = chapters.map(ch => {
      if (ch.id === currentChapterId) {
        return {
          ...ch,
          podcastText,
          playlistClips,
          playlistTracks
        };
      }
      return ch;
    });
    setChapters(updated);
    setStorytellerViewMode("overview");
    setTimeout(() => {
      try {
        localStorage.setItem("voication_chapters", JSON.stringify(updated));
      } catch (err) {}
    }, 100);
  };

  const createNewChapter = () => {
    const nextNum = chapters.length + 1;
    const newId = `chapter_${Date.now()}`;
    const newCh = {
      id: newId,
      name: `Chapter ${nextNum}`,
      podcastText: `[Speaker 1] Chapter ${nextNum} content here.`,
      playlistClips: [],
      playlistTracks: DEFAULT_TIMELINE_TRACKS
    };
    
    const updated = chapters.map(ch => {
      if (ch.id === currentChapterId) {
        return {
          ...ch,
          podcastText,
          playlistClips,
          playlistTracks
        };
      }
      return ch;
    });

    const newChaptersList = [...updated, newCh];
    setChapters(newChaptersList);
    setCurrentChapterId(newId);
    setPodcastText(newCh.podcastText);
    setPlaylistClips(newCh.playlistClips);
    setPlaylistTracks(newCh.playlistTracks);
    setStorytellerViewMode("editor");
    setTimeout(() => saveProjectSync(), 100);
  };

  const deleteChapter = (chapterId) => {
    if (chapters.length <= 1) {
      alert("Projects must have at least one chapter.");
      return;
    }
    if (confirm("Are you sure you want to delete this chapter? All timeline clips will be lost.")) {
      const filtered = chapters.filter(ch => ch.id !== chapterId);
      setChapters(filtered);
      
      if (currentChapterId === chapterId) {
        const firstRemaining = filtered[0];
        setCurrentChapterId(firstRemaining.id);
        setPodcastText(firstRemaining.podcastText || "");
        setPlaylistClips(firstRemaining.playlistClips || []);
        setPlaylistTracks(firstRemaining.playlistTracks && firstRemaining.playlistTracks.length > 0 ? firstRemaining.playlistTracks : DEFAULT_TIMELINE_TRACKS);
      }
      setTimeout(() => saveProjectSync(), 100);
    }
  };

  const triggerBatchSynthesis = () => {
    if (!confirm(`Are you sure you want to batch-render all clips for all ${chapters.length} chapters? This will start parallel TTS synthesis tasks.`)) {
      return;
    }
    saveProjectSync();
    
    chapters.forEach(ch => {
      const clips = ch.playlistClips || [];
      clips.forEach(clip => {
        if (clip.status !== "done") {
          generateClipAudio(clip.id, clip, speakerMapping);
        }
      });
    });
    alert("Batch rendering started! Check the Render Queue sidebar to monitor progress.");
  };

  const addReviewNote = (text, targetPara) => {
    const playhead = reviewAudioRef.current ? reviewAudioRef.current.currentTime : 0;
    const newNote = {
      id: `note_${Date.now()}`,
      timecode: Math.round(playhead * 10) / 10,
      text: text,
      targetPara: targetPara,
      suggestion: "",
      status: "pending"
    };
    setReviewNotes(prev => [...prev, newNote]);
  };

  const deleteReviewNote = (noteId) => {
    setReviewNotes(prev => prev.filter(n => n.id !== noteId));
  };

  const requestLlmRewrite = async (noteId) => {
    const note = reviewNotes.find(n => n.id === noteId);
    if (!note || !note.targetPara) return;
    
    setIsSuggestingEdit(true);
    try {
      const response = await fetch("http://localhost:5000/review/suggest-edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: note.targetPara,
          note: note.text
        })
      });
      if (!response.ok) {
        throw new Error("Rewrite request failed");
      }
      const data = await response.json();
      setReviewNotes(prev => prev.map(n => 
        n.id === noteId ? { ...n, suggestion: data.revised_text } : n
      ));
    } catch (err) {
      alert(`Ollama rewrite failed: ${err.message}`);
    } finally {
      setIsSuggestingEdit(false);
    }
  };

  const approveRewrite = (noteId) => {
    const note = reviewNotes.find(n => n.id === noteId);
    if (!note || !note.suggestion) return;
    
    const originalText = note.targetPara.trim();
    const replacementText = note.suggestion.trim();
    
    setPodcastText(prev => {
      if (prev.includes(originalText)) {
        return prev.replace(originalText, replacementText);
      }
      const paragraphs = prev.split("\n\n");
      const updated = paragraphs.map(p => {
        if (p.trim() === originalText) {
          return replacementText;
        }
        return p;
      });
      return updated.join("\n\n");
    });

    setReviewNotes(prev => prev.map(n => 
      n.id === noteId ? { ...n, status: "approved" } : n
    ));
    
    alert("Script updated successfully!");
    setTimeout(() => saveProjectSync(), 200);
  };

  const rejectRewrite = (noteId) => {
    setReviewNotes(prev => prev.map(n => 
      n.id === noteId ? { ...n, status: "rejected", suggestion: "" } : n
    ));
  };

  const publishToObsidian = async (mode = "clean") => {
    if (!obsidianVaultPath || !obsidianVaultPath.trim()) {
      alert("Please configure your Obsidian Vault absolute path in Project Settings first.");
      setActiveMainTab("project-settings");
      return;
    }
    
    saveProjectSync();
    
    const currentChapter = chapters.find(c => c.id === currentChapterId) || { name: "Chapter 1" };
    const safeName = `${activeProjectName.replace(/[^\w\-_]/g, "_")}_${currentChapter.name.replace(/[^\w\-_]/g, "_")}${mode === "commented" ? "_commented" : ""}.md`;
    
    let exportContent = podcastText;
    
    if (mode === "commented" && reviewNotes.length > 0) {
      exportContent += "\n\n---\n\n## 📝 Review Notes & Timecoded Annotations\n\n";
      reviewNotes.forEach((note) => {
        const mins = Math.floor(note.timecode / 60);
        const secs = (note.timecode % 60).toFixed(1).padStart(4, "0");
        exportContent += `%% [${mins}:${secs}] Note: ${note.text}${note.targetPara ? ` (Target: "${note.targetPara}")` : ""}${note.suggestion ? ` -> Suggested: "${note.suggestion}"` : ""} %%\n\n`;
      });
    }

    try {
      const response = await fetch("http://localhost:5000/review/save-version", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          directory: obsidianVaultPath,
          filename: safeName,
          content: exportContent
        })
      });
      
      const data = await response.json();
      if (data.success) {
        alert(`Success! Version published and saved to Obsidian: ${data.path}`);
      } else {
        alert(`Failed to save: ${data.error}`);
      }
    } catch (err) {
      console.error(err);
      alert(`Publish failed: ${err.message}`);
    }
  };

  const loadActiveChapterMix = () => {
    const mixedItem = queue.find(item => item.model === "mixer" && item.status === "done");
    if (mixedItem && mixedItem.downloadUrl) {
      setActiveReviewAudioUrl(mixedItem.downloadUrl);
      alert("Successfully loaded completed mixed master audio!");
    } else {
      alert("No mixed master audio found in the render queue. Please export a timeline mixdown first.");
    }
  };

  // Sync state to localStorage on beforeunload (refresh/close)
  useEffect(() => {
    const handleBeforeUnload = () => {
      saveProjectSync();
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [
    currentProjectId,
    activeProjectName,
    mediaFormat,
    numberOfSpeakers,
    speakerMapping,
    speakerNames,
    podcastText,
    playlistClips,
    playlistTracks,
    curatedVoices,
    enabledModels,
    chapters,
    currentChapterId,
    globalSummary,
    obsidianVaultPath
  ]);

  // Background Autosave Hook
  useEffect(() => {
    if (!currentProjectId) return;
    setIsSaving(true);
    const timer = setTimeout(() => {
      const projId = currentProjectId;
      const projName = activeProjectName || "Untitled Project";
      
      const updatedChapters = chapters.map(ch => {
        if (ch.id === currentChapterId) {
          return {
            ...ch,
            podcastText,
            playlistClips,
            playlistTracks
          };
        }
        return ch;
      });

      const updatedProject = {
        id: projId,
        name: projName,
        createdAt: currentProjectCreatedAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        mediaFormat,
        numberOfSpeakers,
        speakerMapping,
        speakerNames,
        curatedVoices,
        enabledModels,
        globalSummary,
        obsidianVaultPath,
        activeChapterId: currentChapterId,
        chapters: updatedChapters
      };
      
      setProjects(prev => {
        const filtered = prev.filter(p => p.id !== projId);
        return [updatedProject, ...filtered];
      });
      setIsSaving(false);
    }, 1200);
    return () => clearTimeout(timer);
  }, [
    currentProjectId,
    activeProjectName,
    mediaFormat,
    numberOfSpeakers,
    speakerMapping,
    speakerNames,
    podcastText,
    playlistClips,
    playlistTracks,
    curatedVoices,
    enabledModels,
    chapters,
    currentChapterId,
    globalSummary,
    obsidianVaultPath
  ]);

  // Caret detection logic for active speakers
  const detectActiveSpeaker = (textBefore) => {
    const speakers = ["speaker_1", "speaker_2", "speaker_3", "speaker_4"];
    let lastIndex = -1;
    let detectedSpeaker = "speaker_1";
    
    speakers.forEach((spkKey) => {
      const num = spkKey.split("_")[1];
      const customName = speakerNames[spkKey] || `Speaker ${num}`;
      
      const escapedCustom = customName.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      const escapedDefault = `Speaker\\s+${num}`;
      
      const regex = new RegExp(`\\[(${escapedCustom}|${escapedDefault})\\]`, "gi");
      let match;
      while ((match = regex.exec(textBefore)) !== null) {
        if (match.index > lastIndex) {
          lastIndex = match.index;
          detectedSpeaker = spkKey;
        }
      }
    });
    return detectedSpeaker;
  };

  const handleCaretChange = (textBefore) => {
    const activeSpk = detectActiveSpeaker(textBefore);
    if (activeSpk !== activeSpeakerKey) {
      setActiveSpeakerKey(activeSpk);
    }
  };

  // Post-processing (noise-cancelling) flag from Settings modal
  const postProcessEnhance = barkSettings.smart_enhance || false;

  // AI Enhance advanced sliders
  const [enhanceCreativity, setEnhanceCreativity] = useState(0.4);
  const [queue, setQueue] = useState([]);
  const [showQueue, setShowQueue] = useState(true);
  const [autoRippleOnSync, setAutoRippleOnSync] = useState(() => localStorage.getItem("voication_auto_ripple") !== "false");
  const [defaultClipSpacing, setDefaultClipSpacing] = useState(() => {
    const saved = localStorage.getItem("voication_default_clip_spacing");
    return saved !== null ? parseFloat(saved) : 0.2;
  });
  const [playingPreview, setPlayingPreview] = useState(null);
  const previewAudioRef = useRef(null);

  const [playingClipUrl, setPlayingClipUrl] = useState(null);
  const clipAudioRef = useRef(null);

  const togglePlayClipAudio = (url) => {
    if (clipAudioRef.current) {
      clipAudioRef.current.pause();
      clipAudioRef.current = null;
    }
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      setPlayingPreview(null);
    }
    if (isPlaying) {
      stopTimeline(false);
    }
    if (isProjectPlaying) {
      stopProjectTimeline(false);
    }

    if (playingClipUrl === url) {
      setPlayingClipUrl(null);
      return;
    }

    setPlayingClipUrl(url);
    let absoluteUrl = url;
    if (url && !url.startsWith("http")) {
      absoluteUrl = `http://localhost:5000${url}`;
    }
    const audio = new Audio(absoluteUrl);
    clipAudioRef.current = audio;
    audio.play().catch(err => {
      console.error("Failed to play clip preview:", err);
      setPlayingClipUrl(null);
    });
    audio.onended = () => {
      setPlayingClipUrl(null);
    };
  };

  const playVoicePreview = (spkVal, fallbackModelName) => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }
    if (playingPreview === spkVal) {
      setPlayingPreview(null);
      return;
    }

    // Client-side playback for reference/library/clone profiles
    if (spkVal && (spkVal.includes(":clone:") || spkVal.startsWith("clone:"))) {
      let cloneName = spkVal.includes(":clone:") ? spkVal.split(":clone:")[1] : spkVal.replace("clone:", "");
      const cloneProfile = clonedProfiles.find(p => p.name === cloneName && (p.type === "clone" || p.type === "library" || p.type === "reference"));
      
      if (cloneProfile) {
        setPlayingPreview(spkVal);
        let url = "";
        if (cloneProfile.voiceLibraryKey) {
          url = `http://localhost:5000/assets/voice_library/${cloneProfile.voiceLibraryKey}.wav`;
        } else if (cloneProfile.file) {
          url = URL.createObjectURL(cloneProfile.file);
        } else if (cloneProfile.fileBase64) {
          url = cloneProfile.fileBase64;
        }
        
        if (url) {
          const audio = new Audio(url);
          previewAudioRef.current = audio;
          audio.play().then(() => {
            audio.onended = () => setPlayingPreview(null);
          }).catch((err) => {
            console.error("Failed reference preview playback, falling back to TTS preview API:", err);
            const fallbackAudio = new Audio(`http://localhost:5000/preview?speaker=${encodeURIComponent(cloneName)}&model=${encodeURIComponent(fallbackModelName || "kokoro")}`);
            previewAudioRef.current = fallbackAudio;
            fallbackAudio.play().catch(() => setPlayingPreview(null));
            fallbackAudio.onended = () => setPlayingPreview(null);
          });
          return;
        }
      }
      return;
    }

    let spk = spkVal;
    let modelName = fallbackModelName;
    if (spkVal && spkVal.includes(":")) {
      const parts = spkVal.split(":", 2);
      modelName = parts[0];
      spk = parts[1];
    }

    setPlayingPreview(spkVal);
    const audio = new Audio(`http://localhost:5000/preview?speaker=${encodeURIComponent(spk)}&model=${encodeURIComponent(modelName)}`);
    previewAudioRef.current = audio;
    audio.play().catch((err) => {
      console.error("Failed to play preview:", err);
      setPlayingPreview(null);
    });
    audio.onended = () => {
      setPlayingPreview(null);
    };
  };

  const injectGenerativeTokens = (text, modelName) => {
    if (!text) return "";
    
    let processed = text;
    const lowerModel = (modelName || "").toLowerCase();
    
    // 1. Punctuation Macro Triggers: Convert ellipses (...) to breaks
    if (lowerModel.includes("bark")) {
      processed = processed.replace(/\.\.\./g, " — ");
    } else if (
      lowerModel.includes("chatterbox") || 
      lowerModel.includes("cosyvoice") || 
      lowerModel.includes("chattts") || 
      lowerModel.includes("qwen")
    ) {
      processed = processed.replace(/\.\.\./g, " [uv_break] ");
    }
    
    // 2. Word-Weight Highlighter: Convert text inside **bold** to UPPERCASE for Bark shouting
    if (lowerModel.includes("bark")) {
      processed = processed.replace(/\*\*([^*]+)\*\*/g, (_, match) => match.toUpperCase());
      processed = processed.replace(/__([^_]+)__/g, (_, match) => match.toUpperCase());
      processed = processed.replace(/<strong>([^<]+)<\/strong>/g, (_, match) => match.toUpperCase());
    }
    
    // 3. Slider-to-Tag Injection
    const words = processed.split(/\s+/);
    const resultWords = [];
    
    words.forEach((word) => {
      resultWords.push(word);
      
      // Avoid injecting after speaker tags like [Speaker 1] or emotional cues like (Whispering)
      if (word.startsWith("[") || word.endsWith("]") || word.startsWith("(") || word.endsWith(")")) return;
      
      // Stutter/Amusement injection removed — these sliders did not function
      // reliably cross-model and produced artifacts on unsupported engines.
    });
    
    return resultWords.join(" ");
  };

  const getModelAndVoiceFromMapping = (spkVal) => {
    if (!spkVal) {
      return { model: "tts_models/en/vctk/vits", voice: "p225", isClone: false };
    }
    
    if (spkVal.startsWith("curated:")) {
      const curatedId = spkVal.split(":")[1];
      const curated = curatedVoices.find(v => v.id === curatedId);
      if (curated) {
        return {
          model: curated.model,
          voice: curated.voice,
          isClone: !!curated.fileBase64,
          isCurated: true,
          curatedId
        };
      }
    }
    
    if (spkVal.includes(":clone:")) {
      const parts = spkVal.split(":clone:");
      return { model: parts[0], voice: `clone:${parts[1]}`, isClone: true, cloneName: parts[1] };
    }
    
    if (spkVal.includes(":")) {
      const parts = spkVal.split(":", 2);
      return { model: parts[0], voice: parts[1], isClone: false };
    }
    
    return { model: spkVal, voice: "", isClone: false };
  };

  const [unreadCompletions, setUnreadCompletions] = useState(0);
  const [showSettings, setShowSettings] = useState(false);

  // Podcast Script Assistant state variables
  const [podcastSource, setPodcastSource] = useState("");
  const [podcastPrompt, setPodcastPrompt] = useState("");
  const [isGeneratingPodcast, setIsGeneratingPodcast] = useState(false);
  const [isIdentifyingSpeakers, setIsIdentifyingSpeakers] = useState(false);

  const isScriptEditorBusy = isIdentifyingSpeakers || isAutoTagging || isAutoTaggingSound || isEnhancing;
  const scriptEditorProcessName = isIdentifyingSpeakers
    ? "LLM Analyzing Dialogue & Character Attribution..."
    : isAutoTagging
    ? "LLM Analyzing Script & Tagging Emotes..."
    : isAutoTaggingSound
    ? "LLM Analyzing Script & Tagging Sound Effects..."
    : isEnhancing
    ? "AI Script Polish & Enhancement in Progress..."
    : "";

  // (Waveform Playlist Timeline States moved up to prevent initialization temporal dead zone error)
  const [playheadTime, setPlayheadTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(25); // pixels per second
  const [selectedTimelineClip, setSelectedTimelineClip] = useState(null);
  const [timelineSelectedSound, setTimelineSelectedSound] = useState("");
  const [timelineGeneratePrompt, setTimelineGeneratePrompt] = useState("");
  const [timelineGenerateType, setTimelineGenerateType] = useState("music");
  const [isClipModalOpen, setIsClipModalOpen] = useState(false);

  // Web Audio Refs
  const audioContextRef = useRef(null);
  const audioBuffersCache = useRef({});
  const activePollingIntervals = useRef({});
  const getClipPeaks = (clipId, numPeaks) => {
    const buffer = audioBuffersCache.current[clipId];
    if (!buffer) return null;
    try {
      const channelData = buffer.getChannelData(0);
      const step = Math.max(1, Math.floor(channelData.length / numPeaks));
      const peaks = [];
      for (let i = 0; i < numPeaks; i++) {
        let max = 0;
        const start = i * step;
        const end = Math.min(start + step, channelData.length);
        for (let j = start; j < end; j++) {
          const val = Math.abs(channelData[j]);
          if (val > max) max = val;
        }
        peaks.push(max);
      }
      const maxPeak = Math.max(...peaks, 0.01);
      return peaks.map(p => p / maxPeak);
    } catch (e) {
      console.error("Error calculating peaks", e);
      return null;
    }
  };
  const gainNodesRef = useRef({});
  const activeSourcesRef = useRef([]);
  const playStartTimeRef = useRef(0);
  const playStartOffsetRef = useRef(0);
  const playbackIntervalRef = useRef(null);
  const isPlayingRef = useRef(false);
  const dragStartRef = useRef(null);
  const projectDragStartRef = useRef(null);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
      activeSourcesRef.current.forEach(src => {
        try { src.stop(); } catch (e) {}
      });
    };
  }, []);

  const [projectPlayheadTime, setProjectPlayheadTime] = useState(0);
  const [isProjectPlaying, setIsProjectPlaying] = useState(false);

  const projectActiveSourcesRef = useRef([]);
  const projectPlayStartTimeRef = useRef(0);
  const projectPlayStartOffsetRef = useRef(0);
  const projectPlaybackIntervalRef = useRef(null);
  const isProjectPlayingRef = useRef(false);

  useEffect(() => {
    isProjectPlayingRef.current = isProjectPlaying;
  }, [isProjectPlaying]);

  useEffect(() => {
    return () => {
      if (projectPlaybackIntervalRef.current) {
        clearInterval(projectPlaybackIntervalRef.current);
      }
      projectActiveSourcesRef.current.forEach(src => {
        try { src.stop(); } catch (e) {}
      });
    };
  }, []);

  const podcastRecipes = [
    // --- Podcast Format ---
    {
      name: "Tech Talk Panel",
      mediaType: "podcast",
      description: "Host (warm female) & Co-host (bright male) with tech expert guest and crisp voiceover.",
      mapping: {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:am_adam",
        speaker_3: "kokoro:af_nicole",
        speaker_4: "tts_models/en/vctk/vits:p232"
      }
    },
    {
      name: "Dramatic Story",
      mediaType: "podcast",
      description: "Heroic female lead, companion male, XTTS antagonist, and a soft/hushed female narrator.",
      mapping: {
        speaker_1: "kokoro:bf_emma",
        speaker_2: "kokoro:bm_george",
        speaker_3: "bark:v2/en_speaker_9",
        speaker_4: "kokoro:af_sarah"
      }
    },
    {
      name: "True Crime Mystery",
      mediaType: "podcast",
      description: "Deep male detective host, whisper female co-host, neutral witness, and broadcast announcer.",
      mapping: {
        speaker_1: "kokoro:bm_lewis",
        speaker_2: "kokoro:bf_isabella",
        speaker_3: "tts_models/en/vctk/vits:p226",
        speaker_4: "kokoro:af_nicole"
      }
    },
    {
      name: "Casual Chat Show",
      mediaType: "podcast",
      description: "High-pitch happy female host, energetic male host, guest star, and neutral male VO.",
      mapping: {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:bm_george",
        speaker_3: "bark:v2/en_speaker_6",
        speaker_4: "tts_models/en/vctk/vits:p236"
      }
    },
    // --- Audiobook Format ---
    {
      name: "Classic Novel",
      mediaType: "audiobook",
      description: "Warm female main narrator, expressive male for character dialogue, and deep male secondary voice.",
      mapping: {
        speaker_1: "kokoro:af_bella",
        speaker_2: "kokoro:am_adam",
        speaker_3: "kokoro:bm_lewis"
      }
    },
    {
      name: "Sci-Fi Drama",
      mediaType: "audiobook",
      description: "Atmospheric narrator (Bark), heroic lead male, and a crisp synthesized AI companion voice.",
      mapping: {
        speaker_1: "bark:v2/en_speaker_9",
        speaker_2: "kokoro:bm_george",
        speaker_3: "kokoro:af_nicole"
      }
    },
    {
      name: "Solo Narrator",
      mediaType: "audiobook",
      description: "Rich, highly-expressive female narrative voice suited for single-narrator books.",
      mapping: {
        speaker_1: "kokoro:af_sarah"
      }
    },
    // --- Video Format ---
    {
      name: "Documentary Film",
      mediaType: "video",
      description: "Deep cinematic male voiceover narrator with high-fidelity female quote/translation voice.",
      mapping: {
        speaker_1: "kokoro:bm_lewis",
        speaker_2: "kokoro:af_nicole"
      }
    },
    {
      name: "Commercial Promo",
      mediaType: "video",
      description: "Energetic male announcer coupled with friendly female customer testimonial.",
      mapping: {
        speaker_1: "kokoro:am_adam",
        speaker_2: "kokoro:af_bella"
      }
    },
    {
      name: "Movie Trailer",
      mediaType: "video",
      description: "Extremely deep dramatic voice for short cinematic video intros and epic teasers.",
      mapping: {
        speaker_1: "tts_models/en/vctk/vits:p232"
      }
    }
  ];

  const allRecipes = [...podcastRecipes, ...customRecipes];

  const isRecipeActive = (recipeMapping) => {
    return Object.keys(recipeMapping).every(key => {
      return speakerMapping[key] === recipeMapping[key];
    });
  };

  const applyRecipe = (recipe) => {
    const mapping = recipe.mapping || recipe;
    setSpeakerMapping(mapping);
    const numSpeakers = recipe.speakerCount !== undefined ? recipe.speakerCount : Object.keys(mapping).length;
    setNumberOfSpeakers(numSpeakers);
    try {
      localStorage.setItem("vibevoice_speaker_mapping", JSON.stringify(mapping));
      localStorage.setItem("voication_num_speakers", numSpeakers.toString());
    } catch (err) {}

    if (recipe.names) {
      setSpeakerNames(recipe.names);
      try {
        localStorage.setItem("voication_speaker_names", JSON.stringify(recipe.names));
      } catch (err) {}
    }

    if (recipe.colors) {
      setSpeakerColors(recipe.colors);
      try {
        localStorage.setItem("voication_speaker_colors", JSON.stringify(recipe.colors));
      } catch (err) {}
    }
    setTimeout(() => {
      syncScriptToTimeline();
      saveProjectSync();
    }, 150);
  };

  const applyCustomSpeakerNamesToScript = (scriptText) => {
    if (!scriptText) return "";
    let cleaned = scriptText.replace(/\*/g, ""); // sanitize asterisks
    for (let i = 1; i <= numberOfSpeakers; i++) {
      const spkKey = `speaker_${i}`;
      const customName = speakerNames[spkKey];
      if (customName && customName.trim()) {
        const regex = new RegExp(`\\[Speaker\\s+${i}\\]`, "gi");
        cleaned = cleaned.replace(regex, `[${customName.trim()}]`);
      }
    }
    return cleaned;
  };

  const clearSpeakerTags = () => {
    if (window.confirm("Are you sure you want to remove all speaker tags from the script?")) {
      const speakerNamesList = Object.values(speakerNames).map(n => n.trim().toLowerCase());
      const speakerTagsRegex = /\[([^\]]+)\]/gi;
      
      let newText = podcastText.replace(speakerTagsRegex, (match, tag) => {
        const cleanTag = tag.trim().toLowerCase();
        const isDefaultSpeaker = /^speaker\s+\d+$/i.test(cleanTag);
        const isCustomSpeaker = speakerNamesList.includes(cleanTag);
        if (isDefaultSpeaker || isCustomSpeaker) {
          return "";
        }
        return match;
      });
      
      // Clean up multiple newlines or leading/trailing spaces
      newText = newText.split("\n").map(line => line.trim()).filter(Boolean).join("\n\n");
      setScriptHistory(prev => [...prev, podcastText]);
      setPodcastText(newText);
    }
  };

  const cleanMarkdownContent = (rawText) => {
    if (!rawText) return { title: "", cleanText: "" };
    let txt = rawText.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    
    let extractedTitle = "";
    // Strip YAML/TOML frontmatter and extract title
    const linesArr = txt.split("\n");
    let startIdx = -1;
    for (let i = 0; i < linesArr.length; i++) {
      if (linesArr[i].trim() !== "") {
        if (linesArr[i].trim().startsWith("---") || linesArr[i].trim().startsWith("+++")) {
          startIdx = i;
        }
        break;
      }
    }
    if (startIdx !== -1) {
      let endIdx = -1;
      for (let j = startIdx + 1; j < Math.min(linesArr.length, 120); j++) {
        const lineTrim = linesArr[j].trim();
        if (lineTrim.startsWith("title:")) {
          extractedTitle = lineTrim.replace(/^title:\s*["']?/, "").replace(/["']?\s*$/, "");
        }
        if (lineTrim.startsWith("---") || lineTrim.startsWith("+++")) {
          endIdx = j;
          break;
        }
      }
      if (endIdx !== -1) {
        txt = linesArr.slice(endIdx + 1).join("\n");
      }
    }

    // Strip Markdown headers (# Heading -> Heading)
    txt = txt.replace(/^#{1,6}\s+/gm, "");
    // Strip Markdown bold and italics (**text** -> text)
    txt = txt.replace(/\*\*([^*]+)\*\*/g, "$1");
    txt = txt.replace(/__([^_]+)__/g, "$1");
    txt = txt.replace(/\*([^*]+)\*/g, "$1");
    txt = txt.replace(/_([^_]+)_/g, "$1");
    // Strip Markdown links
    txt = txt.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

    return { title: extractedTitle.trim(), cleanText: txt.trim() };
  };

  const handleDocumentImport = async (e, targetContext = "chapter") => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const ext = file.name.split(".").pop().toLowerCase();
    if (["txt", "md", "markdown"].includes(ext)) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        let rawText = evt.target.result || "";
        let { title, cleanText } = cleanMarkdownContent(rawText);
        
        if (targetContext === "playground") {
          setText(cleanText);
        } else {
          setScriptHistory(prev => [...prev, podcastText]);
          setPodcastText(cleanText);
          setChapters(prev => prev.map(ch => {
            if (ch.id === currentChapterId) {
              return { ...ch, podcastText: cleanText, name: title || ch.name };
            }
            return ch;
          }));
          setTimeout(() => saveProjectSync(), 100);
        }
        alert(`Successfully imported ${file.name}${title ? ` ("${title}")` : ""}!`);
        e.target.value = "";
      };
      reader.readAsText(file);
    } else if (ext === "docx") {
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("http://localhost:5000/import/docx", {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || `Server returned ${res.status}`);
        }
        const data = await res.json();
        let { title, cleanText } = cleanMarkdownContent(data.markdown || "");
        if (targetContext === "playground") {
          setText(cleanText);
        } else {
          setScriptHistory(prev => [...prev, podcastText]);
          setPodcastText(cleanText);
          setChapters(prev => prev.map(ch => {
            if (ch.id === currentChapterId) {
              return { ...ch, podcastText: cleanText, name: title || ch.name };
            }
            return ch;
          }));
          setTimeout(() => saveProjectSync(), 100);
        }
        alert(`Successfully imported and converted ${file.name} to Markdown!`);
      } catch (err) {
        console.error("DOCX import error", err);
        alert(`Failed to import DOCX: ${err.message}`);
      } finally {
        e.target.value = "";
      }
    } else {
      alert("Unsupported file type. Please upload a .docx, .txt, or .md file.");
      e.target.value = "";
    }
  };

  const autoTagScriptWithAI = async () => {
    if (!podcastText.trim()) return;
    setIsAutoTagging(true);
    
    // Construct speaker configs payload mapping speaker labels to their allowed tags
    const speakerConfigs = {};
    for (let i = 1; i <= numberOfSpeakers; i++) {
      const spkKey = `speaker_${i}`;
      const num = spkKey.split("_")[1];
      const customName = speakerNames[spkKey] || `Speaker ${num}`;
      
      const assignedVoiceId = speakerMapping[spkKey];
      let allowedTags = [];
      let modelName = "";
      
      if (assignedVoiceId) {
        if (assignedVoiceId.startsWith("curated:")) {
          const curatedId = assignedVoiceId.split(":")[1];
          const curated = curatedVoices.find(v => v.id === curatedId);
          if (curated) {
            modelName = curated.model;
          }
        } else if (assignedVoiceId.includes(":")) {
          modelName = assignedVoiceId.split(":")[0];
        }
      }
      
      if (modelName) {
        const voiceData = voices.find(v => v.name === modelName);
        if (voiceData && voiceData.features?.includes("tags")) {
          allowedTags = (voiceData.tokens || []).map(t => t.replace(/^\[|\]$/g, ""));
        }
      }
      
      speakerConfigs[`Speaker ${i}`] = allowedTags;
      speakerConfigs[customName] = allowedTags;
    }
    
    try {
      const res = await axios.post("http://localhost:5000/podcast/auto_tag", {
        script: podcastText,
        speaker_configs: speakerConfigs
      });
      if (res.data && res.data.script) {
        if (res.data.warning) {
          // LLM couldn't add tags — show warning but preserve original script
          alert(res.data.warning);
        } else {
          setScriptHistory(prev => [...prev, podcastText]);
          const processed = applyCustomSpeakerNamesToScript(res.data.script);
          setPodcastText(processed);
          alert("Script enhanced with speaker-specific vocalization tags!");
        }
      }
    } catch (e) {
      console.error("AI tagging failed:", e);
      alert("Failed to auto-tag script: " + (e.response?.data?.error || e.message));
    } finally {
      setIsAutoTagging(false);
    }
  };

  const autoTagSoundWithAI = async () => {
    if (!podcastText.trim() || !soundTaggerPrompt.trim()) return;
    setIsAutoTaggingSound(true);
    try {
      const res = await axios.post("http://localhost:5000/podcast/tag_sounds", {
        script: podcastText,
        prompt: soundTaggerPrompt
      });
      if (res.data && res.data.script) {
        if (res.data.warning) {
          alert(res.data.warning);
        } else {
          setScriptHistory(prev => [...prev, podcastText]);
          const processed = applyCustomSpeakerNamesToScript(res.data.script);
          setPodcastText(processed);
          syncScriptToTimeline(processed);
          alert("Script auto-tagged with sound effects and music tags based on your prompt!");
        }
      }
    } catch (e) {
      console.error("AI sound tagging failed:", e);
      alert("Failed to auto-tag sound/music: " + (e.response?.data?.error || e.message));
    } finally {
      setIsAutoTaggingSound(false);
    }
  };

  const addSoundClipToTimeline = (soundKey, soundType, duration, startOffset = playheadTime) => {
    const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    const newClip = {
      id: clipId,
      trackId: soundType,
      text: `${soundType === "music" ? "Music" : "SFX"}: ${soundKey}`,
      voiceDirection: "",
      startTime: startOffset,
      duration: duration,
      status: "needs-render",
      audioUrl: `http://localhost:5000/audio/${soundType}_${encodeURIComponent(soundKey)}`,
      jobId: null,
      [soundType === "music" ? "musicKey" : "sfxKey"]: soundKey
    };
    
    setPlaylistClips(prev => {
      const updated = [...prev, newClip];
      setTimeout(() => syncTimelineToScript(updated), 50);
      return updated;
    });
    
    resolveSoundClipAudio(newClip);
  };

  const generatePodcastScript = async () => {
    if (!podcastSource.trim()) {
      alert("Please provide source text for the script.");
      return;
    }
    setIsGeneratingPodcast(true);
    try {
      const response = await fetch("http://localhost:5000/podcast/script", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: podcastSource,
          prompt: podcastPrompt,
          creativity: 0.6,
          numberOfSpeakers: numberOfSpeakers
        })
      });
      const data = await response.json();
      if (data.script) {
        if (data.warning) {
          alert(data.warning);
        }
        setScriptHistory(prev => [...prev, podcastText]);
        const processed = applyCustomSpeakerNamesToScript(data.script);
        setPodcastText(processed);
      } else if (data.error) {
        alert("Error: " + data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to generate podcast script.");
    } finally {
      setIsGeneratingPodcast(false);
    }
  };

  const identifySpeakers = async () => {
    if (!podcastText.trim()) {
      alert("The editor is empty. Please enter or generate some dialogue first.");
      return;
    }
    setIsIdentifyingSpeakers(true);
    try {
      const response = await fetch("http://localhost:5000/podcast/id_speakers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: podcastText,
          creativity: 0.3,
          numberOfSpeakers: numberOfSpeakers,
          speakerNames: speakerNames,
          perspectiveSpeaker: perspectiveSpeaker,
          quoteVoicing: quoteVoicing,
          customInstructions: customInstructions
        })
      });
      const data = await response.json();
      if (data.script) {
        if (data.warning) {
          alert(data.warning);
        }
        setScriptHistory(prev => [...prev, podcastText]);
        const processed = applyCustomSpeakerNamesToScript(data.script);
        setPodcastText(processed);
      } else if (data.error) {
        alert("Error: " + data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to identify speakers.");
    } finally {
      setIsIdentifyingSpeakers(false);
    }
  };

  const undoScriptChange = () => {
    if (scriptHistory.length === 0) return;
    const previous = scriptHistory[scriptHistory.length - 1];
    setScriptHistory(prev => prev.slice(0, prev.length - 1));
    setPodcastText(previous);
  };

  const clearCompletedQueueItems = () => {
    const completedIds = queue.filter(item => item.status === "done" || item.status === "error").map(item => item.id);
    if (completedIds.length === 0) return;
    
    setQueue(prev => prev.filter(item => !completedIds.includes(item.id)));
    const saved = JSON.parse(localStorage.getItem("savedQueue")) || [];
    localStorage.setItem(
      "savedQueue",
      JSON.stringify(saved.filter(j => !completedIds.includes(j.id)))
    );
  };

  const autoChunkSegment = (seg) => {
    if (seg.isPause || !seg.speakerKey) return;
    
    const tag = `[${seg.tagName}]`;
    let chunks = [];
    const paragraphs = seg.text.split(/\n+/).map(p => p.trim()).filter(Boolean);
    if (paragraphs.length > 1) {
      chunks = paragraphs;
    } else {
      const sentenceRegex = /[^.!?]+[.!?]+(?:\s+|$)/g;
      const sentences = seg.text.match(sentenceRegex) || [seg.text];
      const targetSentenceCount = 3;
      for (let i = 0; i < sentences.length; i += targetSentenceCount) {
        chunks.push(sentences.slice(i, i + targetSentenceCount).join(" ").trim());
      }
    }
    
    const replacementText = chunks.map(chunk => `${tag}\n${chunk}`).join("\n\n") + "\n\n";
    const before = podcastText.substring(0, seg.startIndex);
    const after = podcastText.substring(seg.endIndex);
    const updatedScript = (before + replacementText + after).trim().replace(/\n{3,}/g, "\n\n");
    setScriptHistory(prev => [...prev, podcastText]);
    setPodcastText(updatedScript);
  };

  const autoChunkAllSegments = () => {
    const segments = parseScriptTextToSegments(podcastText, numberOfSpeakers, speakerNames);
    let updatedScript = podcastText;
    
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      if (seg.isPause || !seg.speakerKey) continue;
      const wordCount = seg.text.split(/\s+/).filter(Boolean).length;
      if (wordCount > 150) {
        const tag = `[${seg.tagName}]`;
        let chunks = [];
        const paragraphs = seg.text.split(/\n+/).map(p => p.trim()).filter(Boolean);
        if (paragraphs.length > 1) {
          chunks = paragraphs;
        } else {
          const sentenceRegex = /[^.!?]+[.!?]+(?:\s+|$)/g;
          const sentences = seg.text.match(sentenceRegex) || [seg.text];
          const targetSentenceCount = 3;
          for (let k = 0; k < sentences.length; k += targetSentenceCount) {
            chunks.push(sentences.slice(k, k + targetSentenceCount).join(" ").trim());
          }
        }
        const replacementText = chunks.map(chunk => `${tag}\n${chunk}`).join("\n\n") + "\n\n";
        const before = updatedScript.substring(0, seg.startIndex);
        const after = updatedScript.substring(seg.endIndex);
        updatedScript = before + replacementText + after;
      }
    }
    
    setScriptHistory(prev => [...prev, podcastText]);
    setPodcastText(updatedScript.trim().replace(/\n{3,}/g, "\n\n"));
    alert("All long dialogue segments have been successfully chunked by paragraph or sentence groups!");
  };

  // --- Waveform Playlist Audio & Interaction Logic ---
  const loadAudioBuffer = async (clipId, audioUrl) => {
    if (!audioUrl) return;
    if (audioBuffersCache.current[clipId]) {
      return audioBuffersCache.current[clipId];
    }
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioContextRef.current;
      const response = await fetch(audioUrl);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      audioBuffersCache.current[clipId] = audioBuffer;
      
      // Update clip duration and trigger reflow of sequential clips
      setPlaylistClips(prev => {
        const updated = prev.map(c => c.id === clipId ? { ...c, duration: audioBuffer.duration } : c);
        return reflowClips(updated);
      });
      return audioBuffer;
    } catch (err) {
      console.error("Failed to load audio for clip", clipId, err);
    }
  };

  useEffect(() => {
    playlistClips.forEach(clip => {
      if (clip.isPause) return;
      if (clip.audioUrl && (clip.status === "done" || clip.status === "idle") && !audioBuffersCache.current[clip.id]) {
        loadAudioBuffer(clip.id, clip.audioUrl);
      }

      // Resume polling for clips that were queued/generating before page reload or session switch
      if (["generating", "queued", "processing"].includes(clip.status)) {
        if (clip.jobId && !activePollingIntervals.current[clip.id]) {
          pollClipStatus(clip.id, clip.jobId);
        } else if (!clip.jobId) {
          setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, status: "idle" } : c));
        }
      }
    });

    // Mirror active and generating clips in the right sidebar Queue drawer so queue is never empty during rendering
    if (playlistClips.length > 0) {
      setQueue(prevQueue => {
        const clipMap = new Map(playlistClips.map(c => [c.id, c]));
        const updatedPrev = prevQueue.map(item => {
          const matchingClip = clipMap.get(item.id);
          if (matchingClip) {
            return {
              ...item,
              status: matchingClip.status || "idle",
              progress: matchingClip.progress || 0,
              jobId: matchingClip.jobId
            };
          }
          return item;
        });

        // Add any missing clips from playlistClips that are currently generating, queued, or completed
        const existingQueueIds = new Set(prevQueue.map(q => q.id));
        const missingClips = playlistClips
          .filter(c => !existingQueueIds.has(c.id) && ["generating", "queued", "processing", "done"].includes(c.status))
          .map(c => ({
            id: c.id,
            text: c.isPause ? `<Pause: ${c.duration}s>` : c.text,
            status: c.status,
            progress: c.progress || 0,
            jobId: c.jobId,
            model: "tts"
          }));

        return [...updatedPrev, ...missingClips];
      });
    }
  }, [playlistClips]);

  useEffect(() => {
    try {
      localStorage.setItem("vibevoice_playlist_tracks", JSON.stringify(playlistTracks));
    } catch (e) {}
  }, [playlistTracks]);

  useEffect(() => {
    try {
      const serializable = clonedProfiles.map(p => {
        if (p.type === "clone") {
          return {
            name: p.name,
            type: p.type,
            voice: p.voice,
            fileBase64: p.fileBase64,
            transcript: p.transcript
          };
        }
        return p;
      });
      localStorage.setItem("voication_cloned_profiles", JSON.stringify(serializable));
    } catch (e) {
      console.error("Failed to save cloned profiles to localStorage:", e);
    }
  }, [clonedProfiles]);

  const handleTrackVoiceChange = (trackId, voiceId) => {
    setSpeakerMapping(prev => {
      const updated = { ...prev, [trackId]: voiceId };
      try {
        localStorage.setItem("vibevoice_speaker_mapping", JSON.stringify(updated));
      } catch (err) {}
      return updated;
    });
    setTimeout(() => {
      syncScriptToTimeline();
      saveProjectSync();
    }, 100);
  };

  const updateTrackVolume = (trackId, vol) => {
    setPlaylistTracks(prev => {
      const updated = prev.map(t => {
        if (t.id === trackId) {
          const updatedTrack = { ...t, volume: vol };
          updateTrackGain(updatedTrack, prev);
          return updatedTrack;
        }
        return t;
      });
      return updated;
    });
    setTimeout(() => saveProjectSync(), 150);
  };

  const toggleTrackMute = (trackId) => {
    setPlaylistTracks(prev => {
      const updatedTracks = prev.map(t => t.id === trackId ? { ...t, mute: !t.mute } : t);
      updatedTracks.forEach(t => updateTrackGain(t, updatedTracks));
      return updatedTracks;
    });
    setTimeout(() => saveProjectSync(), 150);
  };

  const toggleTrackSolo = (trackId) => {
    setPlaylistTracks(prev => {
      const updatedTracks = prev.map(t => t.id === trackId ? { ...t, solo: !t.solo } : t);
      updatedTracks.forEach(t => updateTrackGain(t, updatedTracks));
      return updatedTracks;
    });
    setTimeout(() => saveProjectSync(), 150);
  };

  const getPayloadForVoice = (voiceId, text, voiceDirection = "", usePhonetics = true) => {
    let model = "kokoro";
    let speaker = "";
    let preset = "";
    
    let isCurated = false;
    let curatedObj = null;
    let curatedId = "";
    
    if (voiceId && voiceId.startsWith("curated:")) {
      curatedId = voiceId.split(":")[1];
      curatedObj = curatedVoices.find(v => v.id === curatedId);
      if (curatedObj) {
        isCurated = true;
        model = curatedObj.model;
        preset = curatedObj.voice;
        speaker = curatedObj.voice;
      }
    } else if (voiceId && voiceId.includes(":")) {
      const parts = voiceId.split(":", 2);
      model = parts[0];
      const sub = parts[1];
      if (model.includes("vits")) {
        speaker = sub;
      } else {
        preset = sub;
        speaker = sub;
      }
    } else if (voiceId) {
      model = voiceId;
    }

    const payload = {
      model: model,
      text: text,
      voice: model,
      speaker: speaker,
      preset: preset,
      voice_preset: preset,
      voice_direction: voiceDirection,
      emotion_intensity: emotionIntensity,
      speed: speed,
      chunk_size: chunkSize,
      pause_duration: pauseDuration,
      use_mps: useMps,
      barkSplitSentences: barkSplitSentences,
      barkMaxDuration: barkMaxDuration,
      small_models: barkSettings?.small_models || false,
      skip_fine: barkSettings?.skip_fine || false,
      chattts_refine_text: chatttsRefineText,
      chattts_spk_temp: chatttsSpkTemp,
      chattts_text_temp: chatttsTextTemp,
      chattts_spk_seed: chatttsSpkSeed,
      chattts_top_p: chatttsTopP,
      chattts_top_k: chatttsTopK,
      chattts_temperature: chatttsSpkTemp,
      fish_engine: fishEngine,
      fish_normalize: fishNormalize,
      fish_similarity_weight: fishSimilarityWeight,
      fish_prompt_text: fishPromptText,
      phonetic_dict: usePhonetics ? phoneticDict : [],
      spell_out_acronyms: usePhonetics ? spellOutAcronyms : false,
      ignore_emojis: usePhonetics ? ignoreEmojis : false,
      ignore_special_symbols: usePhonetics ? ignoreSpecialSymbols : false,
    };

    if (isCurated && curatedObj && curatedObj.settings) {
      Object.assign(payload, curatedObj.settings);
    }

    const curatedConfigs = {};
    curatedVoices.forEach(v => {
      curatedConfigs[v.id] = {
        model: v.model,
        voice: v.voice,
        settings: v.settings,
        fileBase64: v.fileBase64
      };
    });
    payload.curated_speaker_configs = curatedConfigs;

    return payload;
  };

  const pollClipStatus = (clipId, jobId) => {
    if (!jobId) return;
    if (activePollingIntervals.current[clipId]) {
      clearInterval(activePollingIntervals.current[clipId]);
    }

    const poll = setInterval(() => {
      fetch(`http://localhost:5000/status/${jobId}`)
        .then(res => {
          if (!res.ok) {
            throw new Error(`Job ${jobId} not found (${res.status})`);
          }
          return res.json();
        })
        .then(data => {
          if (data.status === "done") {
            if (activePollingIntervals.current[clipId]) {
              clearInterval(activePollingIntervals.current[clipId]);
              delete activePollingIntervals.current[clipId];
            }
            const audioUrl = `http://localhost:5000${data.audio_url}`;
            
            if (audioBuffersCache.current[clipId]) {
              delete audioBuffersCache.current[clipId];
            }

            loadAudioBuffer(clipId, audioUrl).then(buf => {
              const actualDur = buf ? buf.duration : 2.0;
              setPlaylistClips(prev => {
                const updated = prev.map(c => c.id === clipId ? {
                  ...c,
                  status: "done",
                  progress: 100,
                  audioUrl: audioUrl,
                  duration: actualDur
                } : c);
                return reflowClips(updated);
              });
            });
            setQueue(prev => prev.map(item => item.id === clipId ? { ...item, status: "done", progress: 100 } : item));
          } else if (data.status === "error" || data.status === "cancelled") {
            if (activePollingIntervals.current[clipId]) {
              clearInterval(activePollingIntervals.current[clipId]);
              delete activePollingIntervals.current[clipId];
            }
            setPlaylistClips(prev => prev.map(c => c.id === clipId ? { ...c, status: "idle", jobId: null, progress: 0 } : c));
            setQueue(prev => prev.map(item => item.id === clipId ? { ...item, status: "idle", progress: 0 } : item));
          } else {
            setPlaylistClips(prev => prev.map(c => c.id === clipId && c.jobId === jobId ? {
              ...c,
              status: data.status,
              progress: data.progress || 0
            } : c));
            setQueue(prev => prev.map(item => item.id === clipId ? {
              ...item,
              status: data.status,
              progress: data.progress || 0
            } : item));
          }
        })
        .catch(err => {
          console.error("Error polling clip status", err);
          if (activePollingIntervals.current[clipId]) {
            clearInterval(activePollingIntervals.current[clipId]);
            delete activePollingIntervals.current[clipId];
          }
          // Reset status to idle if server job doesn't exist or backend restarted
          setPlaylistClips(prev => prev.map(c => c.id === clipId ? { ...c, status: "idle", jobId: null, progress: 0 } : c));
          setQueue(prev => prev.map(item => item.id === clipId ? { ...item, status: "idle", progress: 0 } : item));
        });
    }, 1000);

    activePollingIntervals.current[clipId] = poll;
  };

  const generateClipAudio = async (clipId, customClipObject = null, customSpeakerMapping = null) => {
    const clip = customClipObject || playlistClips.find(c => c.id === clipId);
    if (!clip) return;
    
    setPlaylistClips(prev => {
      const exists = prev.some(c => c.id === clip.id);
      if (exists) {
        return prev.map(c => c.id === clip.id ? { ...c, status: "generating", progress: 0 } : c);
      }
      return prev;
    });

    const mapping = customSpeakerMapping || speakerMapping;
    const voiceId = mapping[clip.trackId] || "kokoro:af_bella";
    
    // Parse speaker mapping
    const { model: modelName, voice: voiceVal, isClone, cloneName, isCurated } = getModelAndVoiceFromMapping(voiceId);
    const processedText = injectGenerativeTokens(clip.text, modelName);

    setQueue(prev => {
      const filtered = prev.filter(item => item.id !== clip.id);
      return [
        ...filtered,
        {
          id: clip.id,
          text: `[Storyteller] ${clip.text}`,
          status: "generating",
          progress: 0,
          model: modelName || "kokoro"
        }
      ];
    });

    const activeCh = chapters.find(c => c.id === currentChapterId);
    const usePhonetics = activeCh ? (activeCh.usePhoneticSettings ?? true) : true;

    try {
      let response;
      if (isClone && !isCurated) {
        // Find custom clone profile Blob
        const cloneProfile = clonedProfiles.find(p => p.name === cloneName && (p.type === "clone" || p.type === "library" || p.type === "reference"));
        const formData = new FormData();
        formData.append("model", modelName);
        formData.append("voice", modelName);
        formData.append("text", processedText);
        formData.append("voice_direction", clip.voiceDirection || "");
        formData.append("emotion_intensity", emotionIntensity);
        formData.append("speed", speed);
        formData.append("chunk_size", chunkSize);
        formData.append("pause_duration", pauseDuration);
        formData.append("use_mps", useMps);
        formData.append("barkSplitSentences", barkSplitSentences);
        formData.append("barkMaxDuration", barkMaxDuration);
        formData.append("small_models", barkSettings?.small_models || false);
        formData.append("skip_fine", barkSettings?.skip_fine || false);
        formData.append("chattts_refine_text", chatttsRefineText.toString());
        formData.append("chattts_spk_temp", chatttsSpkTemp.toString());
        formData.append("chattts_text_temp", chatttsTextTemp.toString());
        formData.append("chattts_spk_seed", chatttsSpkSeed.toString());
        formData.append("chattts_top_p", chatttsTopP.toString());
        formData.append("chattts_top_k", chatttsTopK.toString());
        formData.append("chattts_temperature", chatttsSpkTemp.toString());
        formData.append("fish_engine", fishEngine.toString());
        formData.append("fish_normalize", fishNormalize.toString());
        formData.append("fish_similarity_weight", fishSimilarityWeight.toString());
        formData.append("fish_prompt_text", fishPromptText.toString());
        formData.append("phonetic_dict", JSON.stringify(usePhonetics ? phoneticDict : []));
        formData.append("spell_out_acronyms", (usePhonetics && spellOutAcronyms).toString());
        formData.append("ignore_emojis", (usePhonetics && ignoreEmojis).toString());
        formData.append("ignore_special_symbols", (usePhonetics && ignoreSpecialSymbols).toString());
        if (cloneProfile) {
          if (cloneProfile.type === "library" || cloneProfile.type === "reference") {
            formData.append("library_speaker_wav", cloneProfile.voiceLibraryKey);
          } else if (cloneProfile.file) {
            formData.append("speaker_wav", cloneProfile.file, "clone_ref.wav");
          }
          if (cloneProfile.transcript) {
            formData.append("ref_text", cloneProfile.transcript);
          }
        }
        
        response = await fetch("http://localhost:5000/generate", {
          method: "POST",
          body: formData,
        });
      } else {
        const payload = getPayloadForVoice(voiceId, processedText, clip.voiceDirection, usePhonetics);
        response = await fetch("http://localhost:5000/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data = await response.json();
      const jobId = data.job_id;
      
      setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, jobId: jobId, status: "queued" } : c));
      setQueue(prev => prev.map(item => item.id === clip.id ? { ...item, status: "queued", jobId: jobId } : item));
      pollClipStatus(clip.id, jobId);
    } catch (err) {
      console.error("Failed to generate clip", clip.id, err);
      setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, status: "error" } : c));
      setQueue(prev => prev.map(item => item.id === clip.id ? { ...item, status: "error" } : item));
    }
  };

  const renderTrack = (trackId) => {
    const trackClips = playlistClips.filter(c => c.trackId === trackId);
    if (trackClips.length === 0) {
      alert("No clips on this track to render.");
      return;
    }
    trackClips.forEach(c => {
      generateClipAudio(c.id);
    });
  };

  const parseScriptToClips = () => {
    if (!podcastText.trim()) {
      alert("Please write or generate a script first.");
      return;
    }
    
    // Match any [Tag] or Pause tag
    const regex = /(\[([^\]]+)\]|<Pause:\s*(\d+(?:\.\d+)?)\s*seconds>|&lt;Pause:\s*(\d+(?:\.\d+)?)\s*seconds&gt;)/gi;
    const matches = [];
    let match;
    while ((match = regex.exec(podcastText)) !== null) {
      const fullMatch = match[0];
      const isPause = fullMatch.toLowerCase().includes("pause");
      if (isPause) {
        const sec = parseFloat(match[3] || match[4] || "1.0");
        matches.push({
          isPause: true,
          seconds: sec,
          tagName: `Pause: ${sec}s`,
          index: match.index,
          tagLength: fullMatch.length
        });
      } else {
        const tagContent = match[2].trim().toLowerCase();
        let matchedSpeakerNum = null;
        for (let i = 1; i <= numberOfSpeakers; i++) {
          const spkKey = `speaker_${i}`;
          const customName = (speakerNames[spkKey] || `Speaker ${i}`).trim().toLowerCase();
          const defaultName = `speaker ${i}`;
          if (tagContent === customName || tagContent === defaultName) {
            matchedSpeakerNum = i;
            break;
          }
        }
        if (matchedSpeakerNum !== null) {
          matches.push({
            isPause: false,
            speakerNum: matchedSpeakerNum,
            tagName: match[2],
            index: match.index,
            tagLength: fullMatch.length
          });
        } else if (tagContent.startsWith("music:") || tagContent.startsWith("sfx:") || tagContent.startsWith("sound effect:")) {
          const type = tagContent.startsWith("music:") ? "music" : "sfx";
          const desc = match[2].substring(match[2].indexOf(":") + 1).trim();
          matches.push({
            isPause: false,
            isSound: true,
            soundType: type,
            soundDesc: desc,
            tagName: match[2],
            index: match.index,
            tagLength: fullMatch.length
          });
        }
      }
    }

    if (matches.length === 0) {
      alert("No speaker tags found! Use buttons below the editor to insert speaker tags.");
      return;
    }

    const newClips = [];
    let currentTime = 0;
    let lastSpeakerNum = 1;

    for (let i = 0; i < matches.length; i++) {
      const current = matches[i];
      const next = matches[i + 1];
      
      const textStart = current.index + current.tagLength;
      const textEnd = next ? next.index : podcastText.length;
      
      let rawDialogue = podcastText.substring(textStart, textEnd).trim();

      if (current.isPause) {
        const clipId = `clip_${Date.now()}_pause_${Math.random().toString(36).substr(2, 5)}`;
        newClips.push({
          id: clipId,
          trackId: `speaker_${lastSpeakerNum}`,
          text: `<Pause: ${current.seconds} seconds>`,
          isPause: true,
          duration: current.seconds,
          startTime: currentTime,
          status: "done",
          progress: 100,
          audioUrl: null,
          jobId: null,
          manuallyMoved: false
        });
        currentTime += current.seconds;
      } else if (current.isSound) {
        const clipId = `clip_${Date.now()}_${current.soundType}_${Math.random().toString(36).substr(2, 5)}`;
        const duration = current.soundType === "music" ? 15.0 : 3.0;
        const audioUrl = `http://localhost:5000/audio/${current.soundType}_${encodeURIComponent(current.soundDesc)}`;
        const soundClip = {
          id: clipId,
          trackId: current.soundType,
          text: `${current.soundType === "music" ? "Music" : "SFX"}: ${current.soundDesc}`,
          voiceDirection: "",
          startTime: currentTime,
          duration: duration,
          status: "needs-render",
          audioUrl: audioUrl,
          jobId: null,
          [current.soundType === "music" ? "musicKey" : "sfxKey"]: current.soundDesc
        };
        newClips.push(soundClip);
        
        if (rawDialogue) {
          const wordCount = rawDialogue.split(/\s+/).filter(Boolean).length;
          const estDuration = Math.max(2.0, wordCount * 0.4 + 0.5);
          const diagClipId = `clip_${Date.now()}_diag_${Math.random().toString(36).substr(2, 5)}`;
          newClips.push({
            id: diagClipId,
            trackId: `speaker_${lastSpeakerNum}`,
            text: rawDialogue,
            voiceDirection: "",
            startTime: currentTime,
            duration: estDuration,
            status: "generating",
            progress: 0,
            audioUrl: null,
            jobId: null,
            manuallyMoved: false
          });
          currentTime += estDuration + 0.2;
        }
      } else {
        lastSpeakerNum = current.speakerNum;
        let voiceDirection = "";
        const parenRegex = /^\(([^)]+)\)/;
        const parenMatch = rawDialogue.match(parenRegex);
        if (parenMatch) {
          voiceDirection = parenMatch[1];
          rawDialogue = rawDialogue.substring(parenMatch[0].length).trim();
        }
        
        if (!rawDialogue) continue;

        const wordCount = rawDialogue.split(/\s+/).filter(Boolean).length;
        const estDuration = Math.max(2.0, wordCount * 0.4 + 0.5);

        const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
        const trackId = `speaker_${current.speakerNum}`;

        newClips.push({
          id: clipId,
          trackId: trackId,
          text: rawDialogue,
          voiceDirection: voiceDirection,
          startTime: currentTime,
          duration: estDuration,
          status: "generating",
          progress: 0,
          audioUrl: null,
          jobId: null,
          manuallyMoved: false
        });

        currentTime += estDuration + 0.2;
      }
    }

    setPlaylistClips(newClips);
    setPlayheadTime(0);
    audioBuffersCache.current = {};
    
    setActiveMainTab("storyteller");
    setStorytellerViewMode("editor");
    
    newClips.forEach(clip => {
      if (clip.status === "generating") {
        generateClipAudio(clip.id, clip, speakerMapping);
      } else if (clip.status === "needs-render") {
        resolveSoundClipAudio(clip);
      }
    });
  };

  const syncScriptToTimeline = (customText = podcastText, customClips = playlistClips) => {
    const segments = parseScriptTextToSegments(customText, numberOfSpeakers, speakerNames);
    if (segments.length === 0) return [];

    let currentTime = 0;
    const unusedClips = [...customClips];

    const newClips = segments.map((seg, idx) => {
      let existingClipIndex = -1;
      let targetTrackId = "speaker_1";

      if (seg.isPause) {
        targetTrackId = "speaker_1";
        existingClipIndex = unusedClips.findIndex(c => c.isPause && c.duration === (seg.duration || 1.0));
      } else if (seg.isSound) {
        targetTrackId = seg.soundType;
        existingClipIndex = unusedClips.findIndex(c => !c.isPause && c.trackId === seg.soundType && (c.musicKey === seg.soundDesc || c.sfxKey === seg.soundDesc));
      } else {
        targetTrackId = seg.speakerKey || "speaker_1";
        if (parseInt(targetTrackId.split("_")[1]) > numberOfSpeakers) {
          targetTrackId = "speaker_1";
        }
        existingClipIndex = unusedClips.findIndex(c => !c.isPause && !c.musicKey && !c.sfxKey && c.text === seg.text && c.trackId === targetTrackId);
      }

      // Fallback 1: Match by type
      if (existingClipIndex === -1) {
        if (seg.isPause) {
          existingClipIndex = unusedClips.findIndex(c => c.isPause);
        } else if (seg.isSound) {
          existingClipIndex = unusedClips.findIndex(c => c.trackId === seg.soundType);
        } else {
          existingClipIndex = unusedClips.findIndex(c => !c.isPause && !c.musicKey && !c.sfxKey);
        }
      }

      // Fallback 2: Match any remaining
      if (existingClipIndex === -1) {
        existingClipIndex = unusedClips.findIndex(() => true);
      }

      let clip;
      if (existingClipIndex !== -1) {
        const existingClip = unusedClips[existingClipIndex];
        unusedClips.splice(existingClipIndex, 1);

        if (seg.isPause) {
          const duration = seg.duration || 1.0;
          clip = {
            ...existingClip,
            isPause: true,
            trackId: "speaker_1",
            text: `<Pause: ${duration} seconds>`,
            duration: duration,
            ...(!existingClip.isPause ? { audioUrl: null, status: "done" } : {})
          };
        } else if (seg.isSound) {
          const textChanged = existingClip.trackId !== seg.soundType || (existingClip.musicKey !== seg.soundDesc && existingClip.sfxKey !== seg.soundDesc);
          clip = {
            ...existingClip,
            isPause: false,
            trackId: seg.soundType,
            text: `${seg.soundType === "music" ? "Music" : "SFX"}: ${seg.soundDesc}`,
            [seg.soundType === "music" ? "musicKey" : "sfxKey"]: seg.soundDesc,
            duration: seg.soundDuration || existingClip.duration || (seg.soundType === "music" ? 15.0 : 3.0),
            ...(textChanged ? {
              audioUrl: `http://localhost:5000/audio/${seg.soundType}_${encodeURIComponent(seg.soundDesc)}`,
              status: "needs-render",
            } : {})
          };
          if (textChanged) {
            resolveSoundClipAudio(clip);
          }
        } else {
          const textChanged = existingClip.isPause || existingClip.text !== seg.text || existingClip.trackId !== targetTrackId;
          clip = {
            ...existingClip,
            isPause: false,
            trackId: targetTrackId,
            text: seg.text,
            ...(textChanged ? {
              audioUrl: null,
              status: "generating",
              progress: 0,
              jobId: null,
              duration: Math.max(2.0, seg.text.split(/\s+/).filter(Boolean).length * 0.4 + 0.5)
            } : {})
          };
          if (textChanged) {
            generateClipAudio(clip.id, clip, speakerMapping);
          }
        }
      } else {
        const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
        if (seg.isPause) {
          const duration = seg.duration || 1.0;
          clip = {
            id: clipId,
            trackId: "speaker_1",
            text: `<Pause: ${duration} seconds>`,
            voiceDirection: "",
            startTime: currentTime,
            duration: duration,
            status: "done",
            progress: 100,
            audioUrl: null,
            jobId: null,
            manuallyMoved: false,
            isPause: true,
          };
        } else if (seg.isSound) {
          clip = {
            id: clipId,
            trackId: seg.soundType,
            text: `${seg.soundType === "music" ? "Music" : "SFX"}: ${seg.soundDesc}`,
            voiceDirection: "",
            startTime: currentTime,
            duration: seg.soundDuration || (seg.soundType === "music" ? 15.0 : 3.0),
            status: "needs-render",
            audioUrl: `http://localhost:5000/audio/${seg.soundType}_${encodeURIComponent(seg.soundDesc)}`,
            jobId: null,
            [seg.soundType === "music" ? "musicKey" : "sfxKey"]: seg.soundDesc
          };
          resolveSoundClipAudio(clip);
        } else {
          const wordCount = seg.text.split(/\s+/).filter(Boolean).length;
          const duration = Math.max(2.0, wordCount * 0.4 + 0.5);
          clip = {
            id: clipId,
            trackId: targetTrackId,
            text: seg.text,
            voiceDirection: "",
            startTime: currentTime,
            duration: duration,
            status: "generating",
            progress: 0,
            audioUrl: null,
            jobId: null,
            manuallyMoved: false,
            isPause: false,
          };
          generateClipAudio(clipId, clip, speakerMapping);
        }
      }

      if (autoRippleOnSync) {
        clip.startTime = currentTime;
        currentTime += clip.duration + defaultClipSpacing;
      } else {
        if (!clip.manuallyMoved) {
          clip.startTime = currentTime;
        }
        currentTime = clip.startTime + clip.duration + defaultClipSpacing;
      }

      return clip;
    });

    setPlaylistClips(newClips);
    audioBuffersCache.current = {};
    return newClips;
  };

  const syncTimelineToScript = (clipsList = playlistClips) => {
    if (!clipsList || clipsList.length === 0) return "";
    const sorted = [...clipsList].sort((a, b) => a.startTime - b.startTime);
    const newText = sorted.map((c) => {
      if (c.isPause) {
        return `<Pause: ${c.duration} seconds>`;
      } else if (c.trackId === "music") {
        return `[music: ${c.musicKey || c.text.replace("Music: ", "")}]`;
      } else if (c.trackId === "sfx") {
        return `[sfx: ${c.sfxKey || c.text.replace("SFX: ", "")}]`;
      } else {
        const spkNum = c.trackId.split("_")[1];
        const spkName = speakerNames[c.trackId] || `Speaker ${spkNum}`;
        const dir = c.voiceDirection ? `(${c.voiceDirection}) ` : "";
        return `[${spkName}] ${dir}${c.text}`;
      }
    }).join("\n\n");
    setPodcastText(newText);
    return newText;
  };

  const resolveSoundClipAudio = (clip) => {
    const endpoint = `http://localhost:5000/api/sound-library/resolve`;
    const payload = {
      description: clip.musicKey || clip.sfxKey,
      type: clip.trackId,
      duration: clip.duration,
      token: freesoundToken
    };
    
    setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, status: "generating" } : c));
    
    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
      if (data.url) {
        loadAudioBuffer(clip.id, data.url).then(buf => {
          const duration = buf ? buf.duration : clip.duration;
          setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { 
            ...c, 
            status: "done", 
            audioUrl: data.url, 
            duration: duration 
          } : c));
        });
      } else {
        setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, status: "error" } : c));
      }
    })
    .catch(err => {
      console.error("Failed to resolve sound clip:", err);
      setPlaylistClips(prev => prev.map(c => c.id === clip.id ? { ...c, status: "error" } : c));
    });
  };


  const playTimeline = async () => {
    if (clipAudioRef.current) {
      clipAudioRef.current.pause();
      setPlayingClipUrl(null);
    }
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioContextRef.current;
    if (ctx.state === "suspended") {
      await ctx.resume();
    }

    stopTimeline(false);

    const startTime = ctx.currentTime;
    playStartTimeRef.current = startTime;
    playStartOffsetRef.current = playheadTime;

    setIsPlaying(true);

    playlistTracks.forEach(track => {
      if (!gainNodesRef.current[track.id]) {
        const gainNode = ctx.createGain();
        gainNode.connect(ctx.destination);
        gainNodesRef.current[track.id] = gainNode;
      }
      updateTrackGain(track);
    });

    const activeSources = [];
    const hasSolo = playlistTracks.some(t => t.solo);

    playlistClips.forEach(clip => {
      if (clip.isPause) return;
      if (clip.status !== "done") return;
      const track = playlistTracks.find(t => t.id === clip.trackId);
      if (!track) return;

      const isMuted = track.mute || (hasSolo && !track.solo);
      if (isMuted) return;

      const buffer = audioBuffersCache.current[clip.id];
      if (!buffer) {
        loadAudioBuffer(clip.id, clip.audioUrl).then(buf => {
          if (buf && isPlayingRef.current) {
            const currentOffset = ctx.currentTime - playStartTimeRef.current + playStartOffsetRef.current;
            if (clip.startTime + buf.duration > playStartOffsetRef.current) {
              const src = ctx.createBufferSource();
              src.buffer = buf;
              src.connect(gainNodesRef.current[clip.trackId]);

              const playTime = playStartTimeRef.current + clip.startTime - playStartOffsetRef.current;
              const offsetInClip = Math.max(0, playStartOffsetRef.current - clip.startTime);
              const durationToPlay = buf.duration - offsetInClip;

              if (playTime >= ctx.currentTime) {
                src.start(playTime, offsetInClip, durationToPlay);
              } else {
                src.start(ctx.currentTime, offsetInClip + (ctx.currentTime - playTime), durationToPlay - (ctx.currentTime - playTime));
              }
              activeSourcesRef.current.push(src);
            }
          }
        });
        return;
      }

      const clipEnd = clip.startTime + buffer.duration;
      if (clipEnd <= playStartOffsetRef.current) {
        return;
      }

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(gainNodesRef.current[clip.trackId]);

      const offsetInClip = Math.max(0, playStartOffsetRef.current - clip.startTime);
      const playTime = startTime + Math.max(0, clip.startTime - playStartOffsetRef.current);
      const durationToPlay = buffer.duration - offsetInClip;

      src.start(playTime, offsetInClip, durationToPlay);
      activeSources.push(src);
    });

    activeSourcesRef.current = activeSources;

    isPlayingRef.current = true;
    playbackIntervalRef.current = setInterval(() => {
      const elapsed = ctx.currentTime - playStartTimeRef.current;
      const currentPos = playStartOffsetRef.current + elapsed;
      setPlayheadTime(currentPos);
      
      const maxDuration = Math.max(10, ...playlistClips.map(c => c.startTime + (c.duration || 0)));
      if (currentPos >= maxDuration) {
        stopTimeline(true);
      }
    }, 50);
  };

  const updateTrackGain = (track, tracksList = playlistTracks) => {
    const gainNode = gainNodesRef.current[track.id];
    if (!gainNode) return;
    const hasSolo = tracksList.some(t => t.solo);
    const isMuted = track.mute || (hasSolo && !track.solo);
    const vol = isMuted ? 0 : track.volume;
    if (audioContextRef.current) {
      gainNode.gain.setValueAtTime(vol, audioContextRef.current.currentTime);
    }
  };

  const stopTimeline = (resetPlayhead = false) => {
    setIsPlaying(false);
    isPlayingRef.current = false;
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
    activeSourcesRef.current.forEach(src => {
      try {
        src.stop();
      } catch (e) {}
    });
    activeSourcesRef.current = [];
    if (resetPlayhead) {
      setPlayheadTime(0);
    }
  };

  const pauseTimeline = () => {
    stopTimeline(false);
  };

  const addSfxClip = (key) => {
    const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    const audioUrl = `http://localhost:5000/audio/sfx_${key}`;
    
    loadAudioBuffer(clipId, audioUrl).then(buf => {
      const dur = buf ? buf.duration : 3.0;
      setPlaylistClips(prev => [
        ...prev,
        {
          id: clipId,
          trackId: "sfx",
          text: `SFX: ${key}`,
          voiceDirection: "",
          startTime: playheadTime,
          duration: dur,
          status: "done",
          audioUrl: audioUrl,
          jobId: null,
          sfxKey: key
        }
      ]);
    });
  };

  const addMusicClip = (key) => {
    const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    const audioUrl = `http://localhost:5000/audio/music_${key}`;
    
    loadAudioBuffer(clipId, audioUrl).then(buf => {
      const dur = buf ? buf.duration : 15.0;
      setPlaylistClips(prev => [
        ...prev,
        {
          id: clipId,
          trackId: "music",
          text: `Music: ${key}`,
          voiceDirection: "",
          startTime: playheadTime,
          duration: dur,
          status: "done",
          audioUrl: audioUrl,
          jobId: null,
          musicKey: key
        }
      ]);
    });
  };

  const deleteTimelineClip = (clipId, confirmFirst = true) => {
    if (confirmFirst && !window.confirm("Are you sure you want to delete this clip from the timeline and script?")) {
      return;
    }
    setPlaylistClips(prev => {
      const filtered = prev.filter(c => c.id !== clipId);
      setTimeout(() => {
        setPlaylistClips(latest => {
          syncTimelineToScript(latest);
          return latest;
        });
      }, 0);
      return filtered;
    });
  };

  const splitTimelineClipAtPlayhead = () => {
    // Find active clip intersecting playhead on any track
    const intersecting = playlistClips.find(c => playheadTime > c.startTime && playheadTime < c.startTime + (c.duration || 0));
    if (!intersecting) {
      alert("Please position the playhead over a clip to split it.");
      return;
    }

    const ratio = (playheadTime - intersecting.startTime) / intersecting.duration;
    if (ratio <= 0.05 || ratio >= 0.95) {
      alert("Cannot split too close to the boundaries of the clip.");
      return;
    }

    let textPart1 = intersecting.text;
    let textPart2 = "";

    const isDialogue = intersecting.trackId && intersecting.trackId.startsWith("speaker_") && !intersecting.isPause;

    if (isDialogue) {
      const words = intersecting.text.split(/\s+/).filter(Boolean);
      if (words.length <= 1) {
        alert("This clip only contains one word and cannot be split by word boundary.");
        return;
      }
      const splitIndex = Math.max(1, Math.round(words.length * ratio));
      textPart1 = words.slice(0, splitIndex).join(" ");
      textPart2 = words.slice(splitIndex).join(" ");

      if (!window.confirm(`Split this clip at playhead?\n\nPart 1: "${textPart1}"\n\nPart 2: "${textPart2}"`)) {
        return;
      }
    } else if (intersecting.isPause) {
      // Split pause duration
      textPart1 = `<Pause: ${(intersecting.duration * ratio).toFixed(1)} seconds>`;
      textPart2 = `<Pause: ${(intersecting.duration * (1 - ratio)).toFixed(1)} seconds>`;
    } else {
      // Music / SFX
      textPart1 = `${intersecting.text} (Part 1)`;
      textPart2 = `${intersecting.text} (Part 2)`;
    }

    const dur1 = intersecting.duration * ratio;
    const dur2 = intersecting.duration * (1 - ratio);

    const clipIdA = `clip_${Date.now()}_A_${Math.random().toString(36).substr(2, 4)}`;
    const clipIdB = `clip_${Date.now()}_B_${Math.random().toString(36).substr(2, 4)}`;

    const clipA = {
      ...intersecting,
      id: clipIdA,
      text: textPart1,
      duration: dur1,
      status: isDialogue ? "needs-render" : intersecting.status,
      audioUrl: isDialogue ? null : intersecting.audioUrl
    };

    const clipB = {
      ...intersecting,
      id: clipIdB,
      text: textPart2 || intersecting.text, // Fallback for non-dialogue
      duration: dur2,
      startTime: intersecting.startTime + dur1 + 0.2, // include 0.2s pause buffer
      status: isDialogue ? "needs-render" : intersecting.status,
      audioUrl: isDialogue ? null : intersecting.audioUrl
    };

    setPlaylistClips(prev => {
      const filtered = prev.filter(c => c.id !== intersecting.id);
      const updated = [...filtered, clipA, clipB];
      
      // Reflow list
      const reflowed = reflowClips(updated);
      setTimeout(() => {
        setPlaylistClips(latest => {
          syncTimelineToScript(latest);
          return latest;
        });
      }, 0);
      return reflowed;
    });
  };

  const triggerDownload = async (url, filename) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error("Failed to trigger download", err);
      // Fallback: open in new tab
      window.open(url, "_blank");
    }
  };

  const exportMixedPodcast = async () => {
    if (playlistClips.length === 0) {
      alert("No clips on the timeline to mix.");
      return;
    }
    
    // Block export while clips are still rendering
    const pendingClips = playlistClips.filter(c => c.status === "generating" || c.status === "queued");
    if (pendingClips.length > 0) {
      alert(`Cannot export yet — ${pendingClips.length} clip${pendingClips.length > 1 ? "s are" : " is"} still rendering. Please wait for all clips to finish before exporting.`);
      return;
    }
    
    const hasSolo = playlistTracks.some(t => t.solo);

    const segments = playlistClips
      .filter(clip => {
        if (clip.status !== "done" || !clip.audioUrl) return false;
        const track = playlistTracks.find(t => t.id === clip.trackId);
        if (!track) return false;
        const isMuted = track.mute || (hasSolo && !track.solo);
        return !isMuted;
      })
      .map(clip => {
        const track = playlistTracks.find(t => t.id === clip.trackId);
        return {
          audio_url: clip.audioUrl,
          start_time: clip.startTime,
          volume: track ? track.volume : 1.0
        };
      });

    if (segments.length === 0) {
      alert("No audible (non-muted) clips to mix.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue(prev => [
      ...prev,
      { id: queueId, text: "Timeline Master Mix", status: "queued", progress: 0, model: "mixer" }
    ]);

    try {
      const response = await fetch("http://localhost:5000/podcast/mix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          segments,
          hard_limiter: ppHardLimiter,
          podcast_voice: ppPodcastVoice,
          mastering: ppMastering
        })
      });
      
      const data = await response.json();
      if (data.mix_id) {
        const jobId = data.mix_id;
        const downloadUrl = `http://localhost:5000${data.audio_url}`;
        setQueue(prev =>
          prev.map(item =>
            item.id === queueId
              ? {
                  ...item,
                  id: jobId,
                  status: "done",
                  progress: 100,
                  downloadUrl: downloadUrl,
                  message: "Timeline Mix Complete"
                }
              : item
          )
        );
        chimeRef.current.play();
        
        // Trigger automatic browser download
        const safeName = `${activeProjectName.replace(/[^\w\-_]/g, "_")}_mixdown.wav`;
        triggerDownload(downloadUrl, safeName);
        
        alert("Mix complete! Audio download started and master WAV added to Render Queue.");
      } else if (data.error) {
        alert("Error mixing: " + data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to export mixed master.");
    }
  };

  const stopProjectTimeline = (resetPlayhead = true) => {
    setIsProjectPlaying(false);
    isProjectPlayingRef.current = false;
    if (projectPlaybackIntervalRef.current) {
      clearInterval(projectPlaybackIntervalRef.current);
      projectPlaybackIntervalRef.current = null;
    }
    projectActiveSourcesRef.current.forEach(src => {
      try { src.stop(); } catch (e) {}
    });
    projectActiveSourcesRef.current = [];
    if (resetPlayhead) {
      setProjectPlayheadTime(0);
    }
  };

  const playProjectTimeline = async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioContextRef.current;
    if (ctx.state === "suspended") {
      await ctx.resume();
    }

    stopProjectTimeline(false);

    const startTime = ctx.currentTime;
    projectPlayStartTimeRef.current = startTime;
    projectPlayStartOffsetRef.current = projectPlayheadTime;

    setIsProjectPlaying(true);
    isProjectPlayingRef.current = true;

    // Gather all clips across all chapters with their absolute start times
    const computedChapters = computeChapterStartTimes(chapters);
    const allClips = [];
    chapters.forEach(ch => {
      const compCh = computedChapters.find(c => c.id === ch.id);
      const chStart = compCh ? compCh.startTime : 0;
      (ch.playlistClips || []).forEach(clip => {
        if (!clip.isPause && clip.status === "done" && clip.audioUrl) {
          allClips.push({
            ...clip,
            startTime: chStart + clip.startTime
          });
        }
      });
    });

    const activeSources = [];

    allClips.forEach(clip => {
      const buffer = audioBuffersCache.current[clip.id];
      if (!buffer) {
        loadAudioBuffer(clip.id, clip.audioUrl).then(buf => {
          if (buf && isProjectPlayingRef.current) {
            const currentOffset = ctx.currentTime - projectPlayStartTimeRef.current + projectPlayStartOffsetRef.current;
            if (clip.startTime + buf.duration > projectPlayStartOffsetRef.current) {
              const src = ctx.createBufferSource();
              src.buffer = buf;
              src.connect(ctx.destination);

              const playTime = projectPlayStartTimeRef.current + clip.startTime - projectPlayStartOffsetRef.current;
              const offsetInClip = Math.max(0, projectPlayStartOffsetRef.current - clip.startTime);
              const durationToPlay = buf.duration - offsetInClip;

              if (playTime >= ctx.currentTime) {
                src.start(playTime, offsetInClip, durationToPlay);
              } else {
                src.start(ctx.currentTime, offsetInClip + (ctx.currentTime - playTime), durationToPlay - (ctx.currentTime - playTime));
              }
              projectActiveSourcesRef.current.push(src);
            }
          }
        });
        return;
      }

      const clipEnd = clip.startTime + buffer.duration;
      if (clipEnd <= projectPlayStartOffsetRef.current) {
        return;
      }

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);

      const offsetInClip = Math.max(0, projectPlayStartOffsetRef.current - clip.startTime);
      const playTime = startTime + Math.max(0, clip.startTime - projectPlayStartOffsetRef.current);
      const durationToPlay = buffer.duration - offsetInClip;

      src.start(playTime, offsetInClip, durationToPlay);
      activeSources.push(src);
    });

    projectActiveSourcesRef.current = activeSources;

    const totalDuration = computedChapters.length > 0 
      ? Math.max(...computedChapters.map(ch => ch.startTime + ch.duration))
      : 30;

    projectPlaybackIntervalRef.current = setInterval(() => {
      const elapsed = ctx.currentTime - projectPlayStartTimeRef.current;
      const currentPos = projectPlayStartOffsetRef.current + elapsed;
      setProjectPlayheadTime(currentPos);
      
      if (currentPos >= totalDuration) {
        stopProjectTimeline(true);
      }
    }, 50);
  };

  const exportProjectTimeline = async () => {
    const computedChapters = computeChapterStartTimes(chapters);
    const allSegments = [];

    chapters.forEach(ch => {
      const compCh = computedChapters.find(c => c.id === ch.id);
      const chStart = compCh ? compCh.startTime : 0;
      const chTracks = ch.playlistTracks || [];
      (ch.playlistClips || []).forEach(clip => {
        if (!clip.isPause && clip.status === "done" && clip.audioUrl) {
          const track = chTracks.find(t => t.id === clip.trackId);
          allSegments.push({
            audio_url: clip.audioUrl,
            start_time: chStart + clip.startTime,
            volume: track ? (track.volume ?? 0.8) : 0.8
          });
        }
      });
    });

    if (allSegments.length === 0) {
      alert("No rendered audio clips found across any chapters to mix.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue(prev => [
      ...prev,
      { id: queueId, text: `${activeProjectName} - Full Project Mixdown`, status: "queued", progress: 0, model: "mixer" }
    ]);

    try {
      const response = await fetch("http://localhost:5000/podcast/mix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          segments: allSegments,
          hard_limiter: ppHardLimiter,
          podcast_voice: ppPodcastVoice,
          mastering: ppMastering
        })
      });
      
      const data = await response.json();
      if (data.mix_id) {
        const jobId = data.mix_id;
        const downloadUrl = `http://localhost:5000${data.audio_url}`;
        setQueue(prev =>
          prev.map(item =>
            item.id === queueId
              ? {
                  ...item,
                  id: jobId,
                  status: "done",
                  progress: 100,
                  downloadUrl: downloadUrl,
                  message: "Project Mix Complete"
                }
              : item
          )
        );
        chimeRef.current.play();
        
        const safeName = `${activeProjectName.replace(/[^\w\-_]/g, "_")}_full_mixdown.wav`;
        triggerDownload(downloadUrl, safeName);
        
        alert("Full project mix complete! Audio download started and master WAV added to Render Queue.");
      } else if (data.error) {
        alert("Error mixing project: " + data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to export full project master.");
    }
  };

  const handleProjectRulerClick = (e, rulerContainer) => {
    const rect = rulerContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickedTime = Math.max(0, clickX / zoomLevel);
    setProjectPlayheadTime(clickedTime);
    
    if (isProjectPlayingRef.current) {
      playProjectTimeline();
    }
  };

  const exportMarkdownScript = (scope = "chapter") => {
    let title = "";
    let content = "";
    
    if (scope === "chapter") {
      if (!activeChapter) return;
      const safeChName = activeChapter.name.replace(/[^\w\-_]/g, "_");
      title = `${safeChName}_script.md`;
      content = `# Chapter Script: ${activeChapter.name}\n\n${podcastText}`;
    } else {
      const safeProjName = activeProjectName.replace(/[^\w\-_]/g, "_");
      title = `${safeProjName}_full_transcript.md`;
      content = `# Project: ${activeProjectName}\n\n`;
      chapters.forEach((ch, idx) => {
        content += `## Chapter ${idx + 1}: ${ch.name}\n\n${ch.podcastText || ""}\n\n---\n\n`;
      });
    }

    const blob = new Blob([content], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", title);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getChapterDuration = (ch) => {
    if (!ch || !ch.playlistClips || ch.playlistClips.length === 0) return 15; // default 15 seconds if empty
    return Math.max(10, ...ch.playlistClips.map(c => c.startTime + (c.duration || 0)));
  };

  const computeChapterStartTimes = (chaptersList) => {
    let current = 0;
    return chaptersList.map((ch, idx) => {
      let st = ch.startTime;
      if (st === undefined || st === null) {
        st = current;
      }
      const dur = getChapterDuration(ch);
      current = st + dur;
      return { ...ch, startTime: st, duration: dur };
    });
  };

  const handleChapterClipMouseDown = (e, chapterId) => {
    e.preventDefault();
    e.stopPropagation();
    const ch = chapters.find(c => c.id === chapterId);
    if (!ch) return;
    
    // Ensure all chapters have computed startTime/durations before dragging starts
    const computed = computeChapterStartTimes(chapters);
    const target = computed.find(c => c.id === chapterId);
    
    projectDragStartRef.current = {
      chapterId: chapterId,
      startX: e.clientX,
      initialStartTime: target.startTime,
      computedChapters: computed
    };
    document.addEventListener("mousemove", handleChapterClipMouseMove);
    document.addEventListener("mouseup", handleChapterClipMouseUp);
  };

  const handleChapterClipMouseMove = (e) => {
    if (!projectDragStartRef.current) return;
    const { chapterId, startX, initialStartTime, computedChapters } = projectDragStartRef.current;
    const deltaX = e.clientX - startX;
    const deltaSecs = deltaX / zoomLevel;
    let newStartTime = Math.max(0, initialStartTime + deltaSecs);
    const deltaShift = newStartTime - initialStartTime;

    // Apply ripple edit shifting subsequent chapters
    const targetIdx = computedChapters.findIndex(c => c.id === chapterId);
    
    setChapters(prev => {
      return prev.map((ch, idx) => {
        let computedCh = computedChapters.find(c => c.id === ch.id);
        let currentStart = computedCh ? computedCh.startTime : (ch.startTime || 0);
        if (ch.id === chapterId) {
          return { ...ch, startTime: newStartTime };
        } else if (idx > targetIdx) {
          // Ripple shift subsequent chapters
          return { ...ch, startTime: Math.max(0, currentStart + deltaShift) };
        }
        return { ...ch, startTime: currentStart };
      });
    });
  };

  const handleChapterClipMouseUp = () => {
    projectDragStartRef.current = null;
    document.removeEventListener("mousemove", handleChapterClipMouseMove);
    document.removeEventListener("mouseup", handleChapterClipMouseUp);
    setTimeout(() => saveProjectSync(), 100);
  };

  const handleClipMouseDown = (e, clipId) => {
    e.preventDefault();
    e.stopPropagation();
    const clip = playlistClips.find(c => c.id === clipId);
    if (!clip) return;
    
    // Find compatible tracks for vertical movement
    const visibleTracks = (playlistTracks || DEFAULT_TIMELINE_TRACKS).filter(track => {
      if (track.type !== "dialogue") return true;
      const num = parseInt(track.id.split("_")[1]);
      return num <= numberOfSpeakers;
    });
    
    let compatibleTracks = [visibleTracks.find(t => t.id === clip.trackId)];
    if (clip.trackId && clip.trackId.startsWith("speaker_")) {
      compatibleTracks = visibleTracks.filter(t => t.id && t.id.startsWith("speaker_"));
    } else if (clip.trackId === "music") {
      compatibleTracks = visibleTracks.filter(t => t.id === "music");
    } else if (clip.trackId === "sfx") {
      compatibleTracks = visibleTracks.filter(t => t.id === "sfx");
    }
    
    dragStartRef.current = {
      clipId: clipId,
      startX: e.clientX,
      startY: e.clientY,
      initialStartTime: clip.startTime,
      initialTrackId: clip.trackId,
      compatibleTracks: compatibleTracks
    };
    document.addEventListener("mousemove", handleClipMouseMove);
    document.addEventListener("mouseup", handleClipMouseUp);
  };

  const handleClipMouseMove = (e) => {
    if (!dragStartRef.current) return;
    const { clipId, startX, startY, initialStartTime, initialTrackId, compatibleTracks } = dragStartRef.current;
    
    // Horizontal move
    const deltaX = e.clientX - startX;
    const deltaSecs = deltaX / zoomLevel;
    let newStartTime = Math.max(0, initialStartTime + deltaSecs);
    
    // Vertical move
    const deltaY = e.clientY - startY;
    const trackIndexOffset = Math.round(deltaY / 112); // Height of h-28 is 112px
    const initialIdx = compatibleTracks.findIndex(t => t.id === initialTrackId);
    let targetIdx = initialIdx + trackIndexOffset;
    targetIdx = Math.max(0, Math.min(compatibleTracks.length - 1, targetIdx));
    const targetTrackId = compatibleTracks[targetIdx].id;
    
    setPlaylistClips(prev => {
      const updated = prev.map(c => {
        if (c.id === clipId) {
          const statusVal = (c.trackId !== targetTrackId && !c.isPause) ? "needs-render" : c.status;
          return { ...c, startTime: newStartTime, trackId: targetTrackId, status: statusVal, manuallyMoved: true };
        }
        return c;
      });
      return reflowClips(updated);
    });
  };

  const handleClipMouseUp = () => {
    if (dragStartRef.current) {
      const { clipId, initialTrackId } = dragStartRef.current;
      setPlaylistClips(prev => {
        const currentClip = prev.find(c => c.id === clipId);
        if (currentClip && currentClip.trackId !== initialTrackId) {
          // Track changed! Sync immediately
          setTimeout(() => {
            setPlaylistClips(latest => {
              syncTimelineToScript(latest);
              return latest;
            });
          }, 0);
        }
        return prev;
      });
    }
    dragStartRef.current = null;
    document.removeEventListener("mousemove", handleClipMouseMove);
    document.removeEventListener("mouseup", handleClipMouseUp);
    setTimeout(() => saveProjectSync(), 100);
  };

  const handleRulerClick = (e, rulerContainer) => {
    const rect = rulerContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickedTime = Math.max(0, clickX / zoomLevel);
    setPlayheadTime(clickedTime);
    
    if (isPlayingRef.current) {
      playTimeline();
    }
  };

  const generatePodcast = async () => {
    if (!podcastText.trim()) {
      alert("Please enter a podcast script first.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue((prev) => [
      ...prev,
      { id: queueId, text: podcastText, status: "queued", progress: 0, model: "vibevoice" },
    ]);

    const curatedConfigs = {};
    curatedVoices.forEach(v => {
      curatedConfigs[v.id] = {
        model: v.model,
        voice: v.voice,
        settings: v.settings,
        fileBase64: v.fileBase64
      };
    });

    const payload = {
      model: "vibevoice",
      text: podcastText,
      temperature: barkTemperature,
      top_k: barkTopK,
      top_p: barkTopP,
      seed,
      speed,
      chunk_size: chunkSize,
      pause_duration: pauseDuration,
      smart_enhance: postProcessEnhance,
      voice: "vibevoice",
      speaker_1_voice: speakerMapping.speaker_1 || "p225",
      speaker_2_voice: speakerMapping.speaker_2 || "p226",
      speaker_3_voice: speakerMapping.speaker_3 || "p227",
      speaker_4_voice: speakerMapping.speaker_4 || "p228",
      use_mps: useMps,
      small_models: barkSettings?.small_models || false,
      skip_fine: barkSettings?.skip_fine || false,
      emotion_intensity: emotionIntensity,
      chattts_refine_text: chatttsRefineText,
      chattts_spk_temp: chatttsSpkTemp,
      chattts_text_temp: chatttsTextTemp,
      chattts_spk_seed: chatttsSpkSeed,
      chattts_top_p: chatttsTopP,
      chattts_top_k: chatttsTopK,
      chattts_temperature: chatttsSpkTemp,
      fish_engine: fishEngine,
      fish_normalize: fishNormalize,
      fish_similarity_weight: fishSimilarityWeight,
      fish_prompt_text: fishPromptText,
      curated_speaker_configs: curatedConfigs,
      phonetic_dict: (chapters.find(c => c.id === currentChapterId)?.usePhoneticSettings ?? true) ? phoneticDict : [],
      spell_out_acronyms: (chapters.find(c => c.id === currentChapterId)?.usePhoneticSettings ?? true) ? spellOutAcronyms : false,
      ignore_emojis: (chapters.find(c => c.id === currentChapterId)?.usePhoneticSettings ?? true) ? ignoreEmojis : false,
      ignore_special_symbols: (chapters.find(c => c.id === currentChapterId)?.usePhoneticSettings ?? true) ? ignoreSpecialSymbols : false,
    };

    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      const jobId = data.job_id;
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
                model: "vibevoice",
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

      const savedQueue = JSON.parse(localStorage.getItem("savedQueue")) || [];
      savedQueue.push({ id: jobId, text: podcastText, timestamp: Date.now() });
      localStorage.setItem("savedQueue", JSON.stringify(savedQueue));

      startPolling(jobId);
    } catch (err) {
      console.error(err);
      alert("Failed to start podcast generation.");
    }
  };

  // Success notification for settings save
  const [showSaveSuccess, setShowSaveSuccess] = useState(false);

  const chimeRef = useRef(new Audio("/chime.wav"));

  const selectedVoiceData = voices.find((v) => v.name === selectedVoice);

  const textareaRef = useRef(null);
  const tagEditorRef = useRef(null);

  const insertTextAtCursor = (insertedText) => {
    const isTagBased = selectedVoiceData?.features?.includes("tags") || selectedVoiceData?.features?.includes("multi_speaker");
    if (isTagBased && tagEditorRef.current) {
      const cleanToken = insertedText.replace(/^\[|\]$/g, "");
      tagEditorRef.current.insertToken(cleanToken);
      return;
    }
    const el = textareaRef.current;
    if (!el) {
      setText(prev => {
        const space = prev && !prev.endsWith(" ") && !prev.endsWith("\n") ? " " : "";
        return prev + space + insertedText;
      });
      return;
    }
    const start = el.selectionStart;
    const end = el.selectionEnd;
    const currentVal = el.value;
    const spaceBefore = start > 0 && currentVal[start - 1] !== " " && currentVal[start - 1] !== "\n" ? " " : "";
    const spaceAfter = end < currentVal.length && currentVal[end] !== " " && currentVal[end] !== "\n" ? " " : "";
    const textToInsert = spaceBefore + insertedText + spaceAfter;
    
    const newVal = currentVal.substring(0, start) + textToInsert + currentVal.substring(end);
    setText(newVal);
    
    setTimeout(() => {
      el.focus();
      el.setSelectionRange(start + textToInsert.length, start + textToInsert.length);
    }, 0);

    try {
      navigator.clipboard.writeText(insertedText);
    } catch (e) {
      console.warn("Failed to copy to clipboard:", e);
    }
  };

  const insertChatttsTag = (tag) => {
    insertTextAtCursor(tag);
    if (chatttsRefineText) {
      setChatttsRefineText(false);
    }
  };

  // --- XTTS Voice Sample (Speaker Reference) recording state ---
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [activeCloneProfile, setActiveCloneProfile] = useState(null);
  const [showVoiceCreatorModal, setShowVoiceCreatorModal] = useState(false);
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

      // Start real-time speech recognition
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        voiceCreatorRecognitionRef.current = recognition;
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";

        recognition.onresult = (event) => {
          const resultText = event.results[0][0].transcript;
          if (resultText) {
            setCustomCloneTranscript(resultText);
          }
        };

        recognition.onerror = (err) => {
          console.error("Voice Creator Speech Recognition error:", err);
        };

        recognition.start();
      }
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
    if (voiceCreatorRecognitionRef.current) {
      voiceCreatorRecognitionRef.current.stop();
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

  // Resolve tokens list – use backend-supplied tokens if the model supports the "tags" feature
  const tokensList =
    selectedVoiceData?.features?.includes("tags") &&
    Array.isArray(selectedVoiceData?.tokens)
      ? selectedVoiceData.tokens.map((t) => t.replace(/^\[|\]$/g, "")) // strip brackets
      : selectedVoiceData?.features?.includes("multi_speaker")
      ? ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]
      : [];

  const allSelectableVoices = React.useMemo(() => {
    const list = [];
    voices.forEach((v) => {
      if (v.name === "vibevoice") return;

      const modelName = v.name === "bark" 
        ? "Bark" 
        : v.name === "tts_models/en/vctk/vits" 
        ? "VITS" 
        : v.name === "tts_models/multilingual/multi-dataset/xtts_v2" 
        ? "XTTS v2" 
        : v.name === "kokoro"
        ? "Kokoro"
        : v.name === "qwen3-tts"
        ? "Qwen3"
        : v.name === "chatterbox-turbo"
        ? "Chatterbox"
        : v.name === "cosyvoice2-styletts2"
        ? "CosyVoice 2"
        : v.name === "chattts"
        ? "ChatTTS"
        : v.name === "fish-audio"
        ? "Fish Audio"
        : v.name;

      if (v.features?.includes("cloning")) {
        clonedProfiles.forEach((p) => {
          if (p.type === "clone" || p.type === "library" || p.type === "reference") {
            list.push({
              id: `${v.name}:clone:${p.name}`,
              label: (p.type === "library" || p.type === "reference") ? `📚 ${p.name}` : `🧬 ${p.name}`,
              model: modelName,
              rawId: `clone:${p.name}`,
              modelKey: v.name,
            });
          }
        });
      }

      const spks = v.supported_speakers?.length
        ? v.supported_speakers
        : v.speakers?.length
        ? v.speakers
        : v.speaker_list?.length
        ? v.speaker_list
        : v.speaker_ids?.length
        ? v.speaker_ids
        : [];

      spks.forEach((s) => {
        list.push({
          id: `${v.name}:${s}`,
          label: s,
          model: modelName,
          rawId: s,
          modelKey: v.name,
        });
      });

      const presets = v.presets || [];
      presets.forEach((p) => {
        list.push({
          id: `${v.name}:${p}`,
          label: p,
          model: modelName,
          rawId: p,
          modelKey: v.name,
    });
  });
});

    // Add curated voices to list
    curatedVoices.forEach((cv) => {
      const modelName = cv.model === "bark" 
        ? "Bark" 
        : cv.model === "tts_models/en/vctk/vits" || cv.model === "vits"
        ? "VITS" 
        : cv.model === "tts_models/multilingual/multi-dataset/xtts_v2" || cv.model === "xtts"
        ? "XTTS v2" 
        : cv.model === "kokoro"
        ? "Kokoro"
        : cv.model === "chattts"
        ? "ChatTTS"
        : cv.model === "fish-audio"
        ? "Fish Audio"
        : cv.model;
        
      list.push({
        id: `curated:${cv.id}`,
        label: `[Curated] ${cv.name} (${cv.voice || cv.model})`,
        model: `Curated (${modelName})`,
        rawId: cv.id,
        modelKey: cv.model
      });
    });

    return list;
  }, [voices, clonedProfiles, curatedVoices]);

  const speakerColorsMap = React.useMemo(() => {
    const map = {};
    const maxSpeakers = Math.max(4, numberOfSpeakers);
    for (let i = 1; i <= maxSpeakers; i++) {
      const spkKey = `speaker_${i}`;
      const color = speakerColors[spkKey] || "#4f46e5";
      const customName = speakerNames[spkKey];
      
      map[`Speaker ${i}`] = color;
      if (customName) {
        map[customName] = color;
      }
    }
    return map;
  }, [speakerColors, speakerNames, numberOfSpeakers]);

  const podcastEditorTokens = React.useMemo(() => {
    // 1. Always allow switching speakers
    const list = ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"];
    const maxSpeakers = Math.max(4, numberOfSpeakers);
    for (let i = 1; i <= maxSpeakers; i++) {
      const spkKey = `speaker_${i}`;
      const customName = speakerNames[spkKey];
      if (customName && !list.includes(customName)) {
        list.push(customName);
      }
    }
    
    // 2. Resolve active speaker's technology/model
    const assignedVoiceId = speakerMapping[activeSpeakerKey];
    let modelName = "";
    
    if (assignedVoiceId) {
      if (assignedVoiceId.startsWith("curated:")) {
        const curatedId = assignedVoiceId.split(":")[1];
        const curated = curatedVoices.find(v => v.id === curatedId);
        if (curated) {
          modelName = curated.model;
        }
      } else if (assignedVoiceId.includes(":")) {
        modelName = assignedVoiceId.split(":")[0];
      }
    }
    
    if (modelName) {
      const voiceData = voices.find(v => v.name === modelName);
      if (voiceData && voiceData.features?.includes("tags") && Array.isArray(voiceData.tokens)) {
        voiceData.tokens.forEach(t => {
          const cleanTag = t.replace(/^\[|\]$/g, "");
          if (!list.includes(cleanTag)) {
            list.push(cleanTag);
          }
        });
      }
    }
    
    return list;
  }, [speakerMapping, voices, activeSpeakerKey, speakerNames, curatedVoices, numberOfSpeakers]);

  /* ---------- Load voices on mount ---------- */
  useEffect(() => {
    axios.get("http://localhost:5000/voices").then((res) => {
      // Sort so Bark model is first
      const sorted = res.data.voices.sort((a, b) => {
        if (a.model?.toLowerCase().includes("bark")) return -1;
        if (b.model?.toLowerCase().includes("bark")) return 1;
        return 0;
      });
      setVoices(sorted);
      if (sorted.length > 0) {
        const firstNonVibe = sorted.find(v => v.name !== "vibevoice");
        if (firstNonVibe) {
          setSelectedVoice(firstNonVibe.name);
        }
      }
    });
  }, []);

  // ---- Load persisted queue from localStorage and rehydrate jobs ----
  useEffect(() => {
    const savedJobs = JSON.parse(localStorage.getItem("savedQueue")) || [];
    const now = Date.now();
    const freshJobs = savedJobs.filter(
      (job) => now - job.timestamp < 12 * 60 * 60 * 1000
    ); // 12 hours max age

    freshJobs.forEach((job) => {
      fetch(`http://localhost:5000/status/${job.id}`)
        .then((res) => {
          if (!res.ok) throw new Error("Job not found");
          return res.json();
        })
        .then((data) => {
          setQueue((prev) => {
            // Skip if this job is already in the queue
            if (prev.some((q) => q.id === job.id)) return prev;
            return [
              ...prev,
              {
                id: job.id,
                text: job.text,
                status: data.status,
                progress: data.progress || 0,
                downloadUrl: data.audio_url
                  ? `http://localhost:5000${data.audio_url}`
                  : undefined,
                chunkIndex: data.chunk_index ?? null,
                totalChunks: data.total_chunks ?? null,
                message: data.message ?? null,
                model: data.model ?? null,
              },
            ];
          });
          if (data.status !== "done") {
            startPolling(job.id);
          }
        })
        .catch(() => {
          // Cleanup stale entry
          const updated = savedJobs.filter((j) => j.id !== job.id);
          localStorage.setItem("savedQueue", JSON.stringify(updated));
        });
    });
  }, []);

  /* ---------- Safeguard Experiment selectedVoice ---------- */
  useEffect(() => {
    if (activeMainTab === "experiment" && selectedVoice === "vibevoice") {
      const firstNonVibe = voices.find((v) => v.name !== "vibevoice");
      if (firstNonVibe) {
        setSelectedVoice(firstNonVibe.name);
      }
    }
  }, [activeMainTab, selectedVoice, voices]);

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
        creativity: enhanceCreativity,
        allowed_tokens: tokensList,
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
    if (!jobId) {
      console.warn("No job ID returned – skipping status polling.");
      return;
    }
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
                ? {
                    ...item,
                    status: data.status,
                    progress: data.progress || 0,
                    chunkIndex: data.chunk_index ?? null,
                    totalChunks: data.total_chunks ?? null,
                    message: data.message ?? null,
                    model: data.model ?? item.model,
                  }
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
                      message: data.message ?? "Synthesis complete",
                      model: data.model ?? item.model,
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

    if (selectedVoiceData?.model?.toLowerCase().includes("bark") && text.trim().length > 1000) {
      const proceed = window.confirm(
        "Warning: You are using Bark (an experimental model) to synthesize text longer than 1000 characters.\n\n" +
        "Bark is extremely resource-heavy and not optimized for long-form content. Generating this may take several minutes and could result in voice truncation.\n\n" +
        "We recommend VITS or XTTS v2 for long chapters/novels.\n\n" +
        "Do you want to proceed anyway?"
      );
      if (!proceed) return;
    }

    if (!allRequiredReady()) {
      alert("Fill in required fields first.");
      return;
    }

    const queueId = Date.now().toString();
    setQueue((prev) => [
      ...prev,
      { id: queueId, text, status: "queued", progress: 0, model: selectedVoice },
    ]);

    let jobId = null;
    const hasReferenceWav = (selectedVoiceData?.features?.includes("cloning") || selectedVoiceData?.requires_speaker_wav) && formDataRef.current.has("speaker_wav");

    if (!hasReferenceWav && !selectedVoiceData?.requires_speaker_wav) {
      // Send as JSON
      const payload = {
        model: selectedVoice,
        text,
        temperature: barkTemperature,
        top_k: barkTopK,
        top_p: barkTopP,
        seed,
        barkSplitSentences,
        barkMaxDuration,
        speed,
        chunk_size: chunkSize,
        pause_duration: pauseDuration,
        smart_enhance: postProcessEnhance,
        voice: selectedVoice,
        preset: voicePreset,
        voice_preset: voicePreset,
        language,
        speaker,
        use_mps: useMps,
        small_models: barkSettings?.small_models,
        skip_fine: barkSettings?.skip_fine,
        text_temp: barkTemperature,
        focus: barkTopP,
        pool: barkTopK,
        voice_direction: voiceDirection,
        streaming_latency: streamingLatency,
        speaker_1_voice: speakerMapping.speaker_1 || "p225",
        speaker_2_voice: speakerMapping.speaker_2 || "p226",
        speaker_3_voice: speakerMapping.speaker_3 || "p227",
        speaker_4_voice: speakerMapping.speaker_4 || "p228",
        emotion_intensity: emotionIntensity,
        chattts_refine_text: chatttsRefineText,
        chattts_spk_temp: chatttsSpkTemp,
        chattts_text_temp: chatttsTextTemp,
        chattts_spk_seed: chatttsSpkSeed,
        chattts_top_p: chatttsTopP,
        chattts_top_k: chatttsTopK,
        chattts_temperature: chatttsSpkTemp,
        fish_engine: fishEngine,
        fish_normalize: fishNormalize,
        fish_similarity_weight: fishSimilarityWeight,
        fish_prompt_text: fishPromptText,
        phonetic_dict: phoneticDict,
        spell_out_acronyms: spellOutAcronyms,
        ignore_emojis: ignoreEmojis,
        ignore_special_symbols: ignoreSpecialSymbols,
      };
      // Diagnostic log for use_mps
      console.log("Sending use_mps:", payload.use_mps);
      try {
        const response = await fetch("http://localhost:5000/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
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
                  model: selectedVoice,
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
        // ---- Save to localStorage queue ----
        const savedQueue = JSON.parse(localStorage.getItem("savedQueue")) || [];
        savedQueue.push({ id: jobId, text, timestamp: Date.now() });
        localStorage.setItem("savedQueue", JSON.stringify(savedQueue));
        // Start polling with backend jobId
        startPolling(jobId);
      } catch (err) {
        alert("Failed to start voice generation.");
        return;
      }
    } else {
      // Use FormData for speaker_wav uploads and voice cloning models
      const formData = new FormData();
      formData.append("text", text);
      formData.append("model", selectedVoice);
      formData.append("voice", selectedVoice); // explicitly include voice field
      formData.append("speed", speed.toString());
      formData.append("chunk_size", chunkSize.toString());
      formData.append("pause_duration", pauseDuration.toString());
      formData.append("smart_enhance", postProcessEnhance ? "true" : "false");

      if (selectedVoiceData?.model?.toLowerCase().includes("xtts")) {
        formData.append("length_scale", xttsLengthScale.toString());
        formData.append("noise_scale", xttsNoiseScale.toString());
        formData.append("noise_scale_w", xttsNoiseScaleW.toString());
      }
      if (selectedVoiceData?.model?.toLowerCase().includes("vits")) {
        formData.append("noise_scale", vitsNoiseScale.toString());
        formData.append("duration_scale", vitsDurationScale.toString());
        formData.append("use_phonemes", vitsUsePhonemes.toString());
      }
      
      // Attach reference audio if the model requires speaker WAV or supports cloning
      const currentVoice = activeCloneProfile ? `clone:${activeCloneProfile.name}` : "";
      const cloneProfile = activeCloneProfile;
      if (currentVoice && currentVoice.startsWith("clone:") && cloneProfile) {
        if (cloneProfile.type === "library" || cloneProfile.type === "reference") {
          formData.append("library_speaker_wav", cloneProfile.voiceLibraryKey);
        } else if (cloneProfile.file) {
          formData.append("speaker_wav", cloneProfile.file, "clone_ref.wav");
        }
        if (cloneProfile.transcript) {
          formData.append("ref_text", cloneProfile.transcript);
        }
      } else if (
        (selectedVoiceData?.requires_speaker_wav || selectedVoiceData?.features?.includes("cloning")) &&
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
      
      // Pass general features
      if (voiceDirection) {
        formData.append("voice_direction", voiceDirection);
      }
      formData.append("streaming_latency", streamingLatency.toString());
      formData.append("speaker_1_voice", speakerMapping.speaker_1 || "p225");
      formData.append("speaker_2_voice", speakerMapping.speaker_2 || "p226");
      formData.append("speaker_3_voice", speakerMapping.speaker_3 || "p227");
      formData.append("speaker_4_voice", speakerMapping.speaker_4 || "p228");
      
      // Pass Apple‑Silicon toggle to backend as well
      formData.append("use_mps", useMps.toString());
      formData.append("small_models", barkSettings.small_models.toString());
      formData.append("skip_fine", barkSettings.skip_fine.toString());
      formData.append("emotion_intensity", emotionIntensity.toString());
      formData.append("chattts_refine_text", chatttsRefineText.toString());
      formData.append("chattts_spk_temp", chatttsSpkTemp.toString());
      formData.append("chattts_text_temp", chatttsTextTemp.toString());
      formData.append("chattts_spk_seed", chatttsSpkSeed.toString());
      formData.append("chattts_top_p", chatttsTopP.toString());
      formData.append("chattts_top_k", chatttsTopK.toString());
      formData.append("chattts_temperature", chatttsSpkTemp.toString());
      formData.append("fish_engine", fishEngine.toString());
      formData.append("fish_normalize", fishNormalize.toString());
      formData.append("fish_similarity_weight", fishSimilarityWeight.toString());
      formData.append("fish_prompt_text", fishPromptText.toString());
      formData.append("phonetic_dict", JSON.stringify(phoneticDict));
      formData.append("spell_out_acronyms", spellOutAcronyms.toString());
      formData.append("ignore_emojis", ignoreEmojis.toString());
      formData.append("ignore_special_symbols", ignoreSpecialSymbols.toString());

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
                  model: selectedVoice,
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
        // ---- Save to localStorage queue ----
        const savedQueue = JSON.parse(localStorage.getItem("savedQueue")) || [];
        savedQueue.push({ id: jobId, text, timestamp: Date.now() });
        localStorage.setItem("savedQueue", JSON.stringify(savedQueue));
        // Start polling with backend jobId
        startPolling(jobId);
      } catch (err) {
        alert("Failed to start generation.");
        return;
      }
    }
  };

  const cancelQueueItem = (id) => {
    const item = queue.find((q) => q.id === id);
    const jobId = item?.jobId || id;
    axios
      .post(`http://localhost:5000/cancel/${jobId}`)
      .catch((err) => console.error("Error calling cancel on backend:", err));
    setQueue((prev) => prev.filter((item) => item.id !== id));
    setPlaylistClips((prev) => prev.map((c) => c.id === id ? { ...c, status: "idle", jobId: null, progress: 0 } : c));
    // Remove from localStorage queue as well
    const saved = JSON.parse(localStorage.getItem("savedQueue")) || [];
    localStorage.setItem(
      "savedQueue",
      JSON.stringify(saved.filter((j) => j.id !== id))
    );
  };

  const cancelAllQueue = () => {
    const activeItems = queue.filter(item => item.status === "generating" || item.status === "queued" || item.status === "processing");
    activeItems.forEach(item => {
      const jobId = item.jobId || item.id;
      axios
        .post(`http://localhost:5000/cancel/${jobId}`)
        .catch((err) => console.error("Error calling cancel on backend:", err));
    });
    const activeIds = activeItems.map(item => item.id);
    setQueue(prev => prev.filter(item => !activeIds.includes(item.id)));
    setPlaylistClips(prev => prev.map(c => activeIds.includes(c.id) ? { ...c, status: "idle", jobId: null, progress: 0 } : c));
    const saved = JSON.parse(localStorage.getItem("savedQueue")) || [];
    localStorage.setItem(
      "savedQueue",
      JSON.stringify(saved.filter((j) => !activeIds.includes(j.id)))
    );
  };

  const estimatedTimePerItem = 5;
  const startTimesRef = useRef({});

  // ------------- Settings modal tab state -------------
  const [settingsTab, setSettingsTab] = useState("general");
  
  const [testingOllama, setTestingOllama] = useState(false);
  const [ollamaTestResult, setOllamaTestResult] = useState(null);

  const testOllamaSettings = async () => {
    setTestingOllama(true);
    setOllamaTestResult(null);
    try {
      const res = await axios.post("http://localhost:5000/test_ollama", {
        ollama_url: ollamaUrl,
        ollama_model: ollamaModel,
      });
      setOllamaTestResult(res.data);
    } catch (err) {
      setOllamaTestResult({
        connected: false,
        message: "Failed to communicate with Flask backend setup diagnostic endpoint."
      });
    } finally {
      setTestingOllama(false);
    }
  };

  const saveGeneralSettings = () => {
    const payload = {
      device,
      ollama_url: ollamaUrl,
      ollama_model: ollamaModel,
      output_folder: outputFolder,
      setup_completed: true
    };
    axios.post("http://localhost:5000/config", payload)
      .then((res) => {
        setShowSaveSuccess(true);
        setTimeout(() => setShowSaveSuccess(false), 2000);
      })
      .catch((err) => {
        console.error("Failed to save config:", err);
        alert("Failed to save configuration to backend.");
      });
  };

  const parseScriptTextToSegments = (text, numSpeakers, speakerNames) => {
    if (!text) return [];
    const regex = /(\[([^\]]+)\]|<Pause:\s*(\d+(?:\.\d+)?)\s*seconds>|&lt;Pause:\s*(\d+(?:\.\d+)?)\s*seconds&gt;)/gi;
    const matches = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      const fullMatch = match[0];
      const isPause = fullMatch.toLowerCase().includes("pause");
      if (isPause) {
        const sec = parseFloat(match[3] || match[4] || "1.0");
        matches.push({
          isPause: true,
          seconds: sec,
          tagName: `Pause: ${sec}s`,
          index: match.index,
          length: fullMatch.length
        });
      } else {
        const tagContent = match[2].trim().toLowerCase();
        let matchedSpeakerKey = null;
        for (let i = 1; i <= numSpeakers; i++) {
          const spkKey = `speaker_${i}`;
          const customName = (speakerNames[spkKey] || `Speaker ${i}`).trim().toLowerCase();
          const defaultName = `speaker ${i}`;
          if (tagContent === customName || tagContent === defaultName) {
            matchedSpeakerKey = spkKey;
            break;
          }
        }
        if (matchedSpeakerKey) {
          matches.push({
            isPause: false,
            speakerKey: matchedSpeakerKey,
            tagName: match[2],
            index: match.index,
            length: fullMatch.length
          });
        } else {
          const lowerTag = tagContent.toLowerCase();
          const isMusic = lowerTag.startsWith("music:") || lowerTag.startsWith("music ");
          const isSfx = lowerTag.startsWith("sfx:") || lowerTag.startsWith("sfx ") || lowerTag.startsWith("sound effect:") || lowerTag.startsWith("sound effect ");
          
          if (isMusic || isSfx) {
            const type = isMusic ? "music" : "sfx";
            let prefixLength = 0;
            if (lowerTag.startsWith("music:")) prefixLength = 6;
            else if (lowerTag.startsWith("music ")) prefixLength = 6;
            else if (lowerTag.startsWith("sfx:")) prefixLength = 4;
            else if (lowerTag.startsWith("sfx ")) prefixLength = 4;
            else if (lowerTag.startsWith("sound effect:")) prefixLength = 13;
            else if (lowerTag.startsWith("sound effect ")) prefixLength = 13;
            
            let rawDesc = match[2].substring(prefixLength).trim();
            
            // Extract duration if present
            let duration = isMusic ? 15.0 : 3.0;
            const durRegex = /(?:duration|dur)[\s:]*(\d+(?:\.\d+)?)(?:\s*(?:s|sec|second|seconds))?/i;
            const durMatch = durRegex.exec(rawDesc);
            if (durMatch) {
              duration = parseFloat(durMatch[1]);
              rawDesc = rawDesc.replace(durRegex, "").trim();
            }
            
            matches.push({
              isPause: false,
              isSound: true,
              soundType: type,
              soundDesc: rawDesc,
              soundDuration: duration,
              tagName: match[2],
              index: match.index,
              length: fullMatch.length
            });
          }
        }
      }
    }

    const segments = [];
    if (matches.length === 0) {
      segments.push({
        speakerKey: "speaker_1",
        tagName: "Speaker 1",
        text: text,
        startIndex: 0,
        endIndex: text.length
      });
      return segments;
    }

    let lastSpeakerKey = "speaker_1";
    let lastTagName = "Speaker 1";

    for (let i = 0; i < matches.length; i++) {
      const current = matches[i];
      const next = matches[i + 1];
      
      const textStart = current.index + current.length;
      const textEnd = next ? next.index : text.length;
      
      const rawDialogue = text.substring(textStart, textEnd).trim();
      
      if (current.isPause) {
        segments.push({
          speakerKey: null,
          tagName: current.tagName,
          text: "<Pause: " + current.seconds + " seconds>",
          isPause: true,
          duration: current.seconds,
          startIndex: current.index,
          endIndex: textEnd
        });
      } else if (current.isSound) {
        segments.push({
          speakerKey: null,
          tagName: current.tagName,
          text: `[${current.tagName}]`,
          isSound: true,
          soundType: current.soundType,
          soundDesc: current.soundDesc,
          soundDuration: current.soundDuration,
          startIndex: current.index,
          endIndex: textStart
        });
        if (rawDialogue) {
          segments.push({
            speakerKey: lastSpeakerKey,
            tagName: lastTagName,
            text: rawDialogue,
            startIndex: textStart,
            endIndex: textEnd
          });
        }
      } else {
        lastSpeakerKey = current.speakerKey;
        lastTagName = current.tagName;
        segments.push({
          speakerKey: current.speakerKey,
          tagName: current.tagName,
          text: rawDialogue,
          isPause: false,
          duration: null,
          startIndex: current.index,
          endIndex: textEnd
        });
      }
    }
    return segments;
  };

  const startWizardRecording = async () => {
    try {
      setWizardAudioUrl(null);
      setWizardAudioBlob(null);
      audioChunksRef.current = [];
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        setWizardAudioBlob(blob);
        setWizardAudioUrl(url);
        
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setWizardRecording(true);
      
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";
        
        recognition.onresult = (event) => {
          const resultText = event.results[0][0].transcript;
          if (resultText) {
            setWizardTargetWord(resultText);
          }
        };
        
        recognition.onerror = (err) => {
          console.error("Speech Recognition error:", err);
        };
        
        recognition.start();
      }
    } catch (err) {
      console.error("Failed to access microphone:", err);
      alert("Could not access your microphone. Please check permissions.");
    }
  };

  const stopWizardRecording = () => {
    if (mediaRecorderRef.current && wizardRecording) {
      mediaRecorderRef.current.stop();
      setWizardRecording(false);
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.error("Failed to stop recognition:", e);
      }
    }
  };

  const transcribeRecordedVoice = async () => {
    if (!wizardTargetWord.trim()) {
      alert("No speech transcription captured. Please type the target word or speak again.");
      return;
    }
    
    setWizardLoading(true);
    try {
      const response = await fetch("http://localhost:5000/phonetic/transcribe_mic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: wizardTargetWord }),
      });
      if (!response.ok) throw new Error("Failed to transcribe audio on backend");
      const data = await response.json();
      setWizardSuggestions(data);
    } catch (err) {
      console.error(err);
      alert("Error getting transcription suggestions: " + err.message);
    } finally {
      setWizardLoading(false);
    }
  };

  const fetchWizardSuggestions = async () => {
    if (!wizardTargetWord.trim()) {
      alert("Please enter a target word or name first.");
      return;
    }
    
    setWizardLoading(true);
    setWizardSuggestions(null);
    try {
      const response = await fetch("http://localhost:5000/phonetic/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          word: wizardTargetWord,
          ethnicity: wizardEthnicity
        })
      });
      if (!response.ok) throw new Error("Failed to get pronunciation suggestions.");
      const data = await response.json();
      setWizardSuggestions(data);
    } catch (err) {
      console.error(err);
      alert("Error fetching pronunciation recommendation: " + err.message);
    } finally {
      setWizardLoading(false);
    }
  };

  const addPhoneticEntry = () => {
    if (!newPhoneticWord.trim() || !newPhoneticReplacement.trim()) {
      alert("Please enter both the word/term and its pronunciation replacement.");
      return;
    }
    const isDuplicate = phoneticDict.some(e => e.word.trim().toLowerCase() === newPhoneticWord.trim().toLowerCase());
    if (isDuplicate) {
      alert("An entry for this word/term already exists in the dictionary.");
      return;
    }
    const newEntry = {
      id: `phonetic_${Date.now()}`,
      word: newPhoneticWord.trim(),
      replacement: newPhoneticReplacement.trim(),
      type: newPhoneticType
    };
    const updated = [...phoneticDict, newEntry];
    setPhoneticDict(updated);
    setNewPhoneticWord("");
    setNewPhoneticReplacement("");
    setNewPhoneticType("standard");
    setTimeout(() => saveProjectSync(currentProjectId), 100);
  };

  const deletePhoneticEntry = (id) => {
    const updated = phoneticDict.filter(e => e.id !== id);
    setPhoneticDict(updated);
    setTimeout(() => saveProjectSync(currentProjectId), 100);
  };

  const startEditingPhoneticEntry = (entry) => {
    setEditingPhoneticId(entry.id);
    setEditingPhoneticWord(entry.word);
    setEditingPhoneticReplacement(entry.replacement);
    setEditingPhoneticType(entry.type || "standard");
  };

  const saveEditingPhoneticEntry = () => {
    if (!editingPhoneticWord.trim() || !editingPhoneticReplacement.trim()) {
      alert("Word/term and pronunciation replacement cannot be empty.");
      return;
    }
    const updated = phoneticDict.map(e => {
      if (e.id === editingPhoneticId) {
        return {
          ...e,
          word: editingPhoneticWord.trim(),
          replacement: editingPhoneticReplacement.trim(),
          type: editingPhoneticType
        };
      }
      return e;
    });
    setPhoneticDict(updated);
    setEditingPhoneticId(null);
    setEditingPhoneticWord("");
    setEditingPhoneticReplacement("");
    setTimeout(() => saveProjectSync(currentProjectId), 100);
  };

  const renderPhoneticDictionaryView = () => {
    return (
      <div className="space-y-8 animate-fadeIn text-gray-900 dark:text-slate-100">
        {/* Header Section */}
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-gray-900 dark:text-white tracking-tight">
            Phonetic Dictionary & Text Dials
          </h2>
          <p className="mt-2 text-sm text-gray-500 dark:text-slate-400 font-medium">
            Fine-tune terminology pronunciations, spell out acronyms, and filter unwanted characters before TTS synthesis.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left panel: Config Dials */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md">
              <h3 className="text-sm font-bold text-gray-800 dark:text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Sliders className="h-4 w-4 text-indigo-500" />
                Text-Processing Dials
              </h3>
              
              <div className="space-y-5">
                {/* Dial 1: Spell Out Acronyms */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <label className="text-xs font-bold text-gray-800 dark:text-slate-200">
                      Spell Out Acronyms
                    </label>
                    <p className="text-[10px] text-gray-400 dark:text-slate-400 mt-1 leading-relaxed">
                      Converts all-uppercase words (2-6 chars, e.g. "COP") to dotted format ("C.O.P.") so TTS reads them letter-by-letter.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={spellOutAcronyms}
                    onChange={(e) => {
                      setSpellOutAcronyms(e.target.checked);
                      setTimeout(() => saveProjectSync(currentProjectId), 100);
                    }}
                    className="w-4 h-4 rounded text-indigo-600 border-gray-300 dark:border-slate-800 focus:ring-indigo-500"
                  />
                </div>

                {/* Dial 2: Ignore Emojis */}
                <div className="flex items-start justify-between gap-3 pt-4 border-t border-gray-100 dark:border-slate-800">
                  <div className="flex-1">
                    <label className="text-xs font-bold text-gray-800 dark:text-slate-200">
                      Ignore Emojis
                    </label>
                    <p className="text-[10px] text-gray-400 dark:text-slate-400 mt-1 leading-relaxed">
                      Removes emoji pictographs (e.g. 📁, 🎙️) from the synthesized text stream to prevent TTS model instabilities.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={ignoreEmojis}
                    onChange={(e) => {
                      setIgnoreEmojis(e.target.checked);
                      setTimeout(() => saveProjectSync(currentProjectId), 100);
                    }}
                    className="w-4 h-4 rounded text-indigo-600 border-gray-300 dark:border-slate-800 focus:ring-indigo-500"
                  />
                </div>

                {/* Dial 3: Ignore Special Symbols */}
                <div className="flex items-start justify-between gap-3 pt-4 border-t border-gray-100 dark:border-slate-800">
                  <div className="flex-1">
                    <label className="text-xs font-bold text-gray-800 dark:text-slate-200">
                      Ignore Special Symbols
                    </label>
                    <p className="text-[10px] text-gray-400 dark:text-slate-400 mt-1 leading-relaxed">
                      Strips coding characters and punctuation markers (e.g. @, #, *, ^, _) while preserving standard punctuation (punctuation markers).
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={ignoreSpecialSymbols}
                    onChange={(e) => {
                      setIgnoreSpecialSymbols(e.target.checked);
                      setTimeout(() => saveProjectSync(currentProjectId), 100);
                    }}
                    className="w-4 h-4 rounded text-indigo-600 border-gray-300 dark:border-slate-800 focus:ring-indigo-500"
                  />
                </div>
              </div>
            </div>
            
            <div className="bg-amber-50/50 dark:bg-amber-950/20 p-5 rounded-2xl border border-amber-200/40 dark:border-amber-900/30 text-amber-800 dark:text-amber-300">
              <h4 className="text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Info className="h-4 w-4" />
                How It Works
              </h4>
              <p className="text-[10px] leading-relaxed">
                The phonetic dictionary performs strict word replacement behind the scenes at rendering time. E.g. defining <strong>"COP"</strong> &rarr; <strong>"C.O.P."</strong> ensures the voice doesn't pronounce it as "cop". The original script remains clean and unmodified for editing.
              </p>
            </div>
          </div>

          {/* Right panel: Dictionary List & Add Form */}
          <div className="lg:col-span-2 space-y-6">
            {/* Add Entry Card */}
            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md">
              <h3 className="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-widest mb-5 flex items-center gap-2">
                <BookOpen className="h-3.5 w-3.5 text-indigo-500" />
                Add Phonetic Dictionary Entry
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1.5">
                    Original Word / Term
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="e.g. COP"
                      value={newPhoneticWord}
                      onChange={(e) => setNewPhoneticWord(e.target.value)}
                      className="flex-1 px-3 py-2 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-955 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 min-w-0"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setWizardTargetWord(newPhoneticWord || "");
                        setWizardSuggestions(null);
                        setWizardAudioUrl(null);
                        setWizardAudioBlob(null);
                        setWizardEthnicity("");
                        setShowPronunciationWizard(true);
                      }}
                      className="px-3 py-2 bg-gradient-to-r from-violet-500 to-indigo-500 hover:from-violet-600 hover:to-indigo-600 text-white rounded-xl text-xs font-bold transition flex items-center gap-1 shrink-0"
                      title="Pronunciation Recommendation Wizard"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      Wizard
                    </button>
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1.5">
                    Pronunciation Type
                  </label>
                  <select
                    value={newPhoneticType}
                    onChange={(e) => setNewPhoneticType(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-955 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="standard">Standard Text-to-Text</option>
                    <option value="ipa">IPA (Phonemes)</option>
                    <option value="arpabet">ARPAbet (e.g. W IH1 N)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1.5">
                    Replacement / Phonemes
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder={
                        newPhoneticType === "standard" 
                          ? "e.g. community of practice" 
                          : newPhoneticType === "ipa" 
                            ? "e.g. kˈOkəɹO" 
                            : "e.g. K OW1 K OW0 R OW0"
                      }
                      value={newPhoneticReplacement}
                      onChange={(e) => setNewPhoneticReplacement(e.target.value)}
                      className="flex-1 px-3 py-2 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-955 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 min-w-0"
                      onKeyDown={(e) => e.key === "Enter" && addPhoneticEntry()}
                    />
                    <button
                      type="button"
                      onClick={addPhoneticEntry}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shrink-0"
                    >
                      <Plus className="h-4 w-4" />
                      Add
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* List Card */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md overflow-hidden">
              <div className="p-4 border-b border-gray-150 dark:border-slate-850 flex justify-between items-center bg-gray-50/50 dark:bg-slate-950/20">
                <span className="text-xs font-bold text-gray-800 dark:text-slate-200">
                  Active Dictionary ({phoneticDict.length})
                </span>
              </div>

              {phoneticDict.length === 0 ? (
                <div className="p-8 text-center text-gray-400 dark:text-slate-500 flex flex-col items-center gap-2">
                  <BookOpen className="h-8 w-8 text-gray-300 dark:text-slate-700" />
                  <span className="text-xs font-medium">No phonetic overrides defined yet.</span>
                  <span className="text-[10px]">Add entries above to customize how specific words are pronounced by the synthesizer.</span>
                </div>
              ) : (
                <div className="divide-y divide-gray-100 dark:divide-slate-855">
                  {phoneticDict.map((entry) => (
                    <div key={entry.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-gray-50/40 dark:hover:bg-slate-955/10 transition">
                      {editingPhoneticId === entry.id ? (
                        /* Edit mode */
                        <div className="flex-1 flex flex-col md:flex-row gap-3">
                          <input
                            type="text"
                            value={editingPhoneticWord}
                            onChange={(e) => setEditingPhoneticWord(e.target.value)}
                            className="px-3 py-1.5 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-950 rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500 md:w-1/4"
                          />
                          <select
                            value={editingPhoneticType}
                            onChange={(e) => setEditingPhoneticType(e.target.value)}
                            className="px-2 py-1.5 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-950 rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500 md:w-1/4"
                          >
                            <option value="standard">Standard</option>
                            <option value="ipa">IPA (Phonemes)</option>
                            <option value="arpabet">ARPAbet</option>
                          </select>
                          <input
                            type="text"
                            value={editingPhoneticReplacement}
                            onChange={(e) => setEditingPhoneticReplacement(e.target.value)}
                            className="flex-1 px-3 py-1.5 text-xs border border-gray-250 dark:border-slate-850 bg-gray-50 dark:bg-slate-950 rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500"
                            onKeyDown={(e) => e.key === "Enter" && saveEditingPhoneticEntry()}
                          />
                          <div className="flex gap-2 shrink-0">
                            <button
                              type="button"
                              onClick={saveEditingPhoneticEntry}
                              className="px-2.5 py-1.5 bg-green-50 hover:bg-green-100 dark:bg-green-950 dark:hover:bg-green-900/60 text-green-700 dark:text-green-300 text-[10px] font-bold rounded-lg transition"
                            >
                              Save
                            </button>
                            <button
                              type="button"
                              onClick={() => setEditingPhoneticId(null)}
                              className="px-2.5 py-1.5 border border-gray-200 dark:border-slate-800 hover:bg-gray-50 dark:hover:bg-slate-800 text-gray-750 dark:text-slate-300 text-[10px] font-bold rounded-lg transition"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        /* Read Mode */
                        <>
                          <div className="flex-1 min-w-0 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
                            <div className="sm:w-1/3 flex items-center gap-2 truncate">
                              <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/40 px-2 py-1 rounded-md">
                                {entry.word}
                              </span>
                              {entry.type === "ipa" && (
                                <span className="text-[9px] font-bold text-emerald-600 bg-emerald-50 dark:text-emerald-400 dark:bg-emerald-950/40 px-1.5 py-0.5 rounded uppercase">
                                  IPA
                                </span>
                              )}
                              {entry.type === "arpabet" && (
                                <span className="text-[9px] font-bold text-sky-600 bg-sky-50 dark:text-sky-400 dark:bg-sky-950/40 px-1.5 py-0.5 rounded uppercase">
                                  ARP
                                </span>
                              )}
                              {(!entry.type || entry.type === "standard") && (
                                <span className="text-[9px] font-bold text-gray-550 bg-gray-50 dark:text-slate-400 dark:bg-slate-800/40 px-1.5 py-0.5 rounded uppercase">
                                  STD
                                </span>
                              )}
                            </div>
                            <div className="flex-1 truncate text-xs text-gray-600 dark:text-slate-350">
                              &rarr; <span className="ml-2 font-mono font-semibold bg-gray-50 dark:bg-slate-950/60 px-1.5 py-0.5 rounded">{entry.replacement}</span>
                            </div>
                          </div>

                          <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
                            <button
                              onClick={() => startEditingPhoneticEntry(entry)}
                              className="px-2.5 py-1.5 border border-gray-200 dark:border-slate-800 hover:bg-gray-50 dark:hover:bg-slate-800 text-gray-750 dark:text-slate-300 text-[10px] font-bold rounded-lg transition"
                              type="button"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => deletePhoneticEntry(entry.id)}
                              className="p-1.5 text-red-500 hover:bg-red-50 dark:hover:bg-red-955/20 rounded-lg transition"
                              title="Delete Override"
                              type="button"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

      {/* ── Pronunciation Wizard Modal ── */}
      {showPronunciationWizard && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-0 sm:p-4">
          <div className="bg-white dark:bg-slate-900 w-full sm:max-w-2xl rounded-t-3xl sm:rounded-2xl shadow-2xl flex flex-col max-h-[92vh] overflow-hidden animate-fadeIn text-gray-900 dark:text-slate-100">

            {/* Header */}
            <div className="px-5 pt-5 pb-4 border-b border-gray-100 dark:border-slate-800 flex items-start justify-between gap-3 bg-gradient-to-r from-violet-50/70 to-indigo-50/70 dark:from-slate-950 dark:to-slate-900 shrink-0">
              <div className="min-w-0">
                <h2 className="text-base font-extrabold text-violet-900 dark:text-violet-200 flex items-center gap-2 flex-wrap">
                  <Sparkles className="h-4 w-4 text-violet-500 shrink-0" />
                  Pronunciation Wizard
                </h2>
                <p className="text-[10px] text-gray-500 dark:text-slate-400 mt-0.5 leading-relaxed">
                  Get expert IPA, ARPAbet and simplified suggestions for any name or term — especially ethnic or foreign names.
                </p>
              </div>
              <button
                onClick={() => setShowPronunciationWizard(false)}
                className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition shrink-0"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Scrollable body */}
            <div className="overflow-y-auto flex-1 p-5 space-y-5">

              {/* Target word + ethnicity */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1.5">
                    Word / Name to look up
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Siobhan, Nguyen, Xiomara…"
                    value={wizardTargetWord}
                    onChange={(e) => setWizardTargetWord(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && fetchWizardSuggestions()}
                    className="w-full px-3 py-2 text-xs border border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-950 rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1.5">
                    Language / Ethnicity hint (optional)
                  </label>
                  <select
                    value={wizardEthnicity}
                    onChange={(e) => setWizardEthnicity(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-950 rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500"
                  >
                    <option value="">Unknown / General</option>
                    <option value="Vietnamese">Vietnamese</option>
                    <option value="Irish / Gaelic">Irish / Gaelic</option>
                    <option value="Arabic">Arabic</option>
                    <option value="Indian / Hindi">Indian / Hindi</option>
                    <option value="Spanish / Latin">Spanish / Latin</option>
                    <option value="Japanese">Japanese</option>
                    <option value="Korean">Korean</option>
                    <option value="Chinese / Mandarin">Chinese / Mandarin</option>
                    <option value="Polish">Polish</option>
                    <option value="Hebrew">Hebrew</option>
                    <option value="Swahili / African">Swahili / African</option>
                    <option value="French">French</option>
                    <option value="German">German</option>
                    <option value="Greek">Greek</option>
                    <option value="Russian">Russian</option>
                    <option value="Turkish">Turkish</option>
                    <option value="Welsh">Welsh</option>
                    <option value="Thai">Thai</option>
                    <option value="Portuguese">Portuguese</option>
                    <option value="Persian / Farsi">Persian / Farsi</option>
                  </select>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={fetchWizardSuggestions}
                  disabled={wizardLoading}
                  className="flex-1 sm:flex-none px-4 py-2 bg-gradient-to-r from-violet-500 to-indigo-500 hover:from-violet-600 hover:to-indigo-600 disabled:opacity-50 text-white rounded-xl text-xs font-bold transition flex items-center justify-center gap-2"
                >
                  {wizardLoading ? (
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="h-3.5 w-3.5" />
                  )}
                  {wizardLoading ? "Generating…" : "Get Pronunciation Recommendations"}
                </button>
              </div>

              {/* Microphone Recording Panel */}
              <div className="bg-gray-50 dark:bg-slate-950 rounded-2xl border border-gray-100 dark:border-slate-800 p-4 space-y-3">
                <h4 className="text-xs font-bold text-gray-700 dark:text-slate-300 flex items-center gap-2">
                  <Mic className="h-3.5 w-3.5 text-rose-500" />
                  Record Your Pronunciation
                </h4>
                <p className="text-[10px] text-gray-400 dark:text-slate-500 leading-relaxed">
                  Speak the name or word aloud — your browser will transcribe it and suggest the phonetic spelling automatically.
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  {!wizardRecording ? (
                    <button
                      type="button"
                      onClick={startWizardRecording}
                      className="px-4 py-2 bg-rose-500 hover:bg-rose-600 text-white rounded-xl text-xs font-bold transition flex items-center gap-2"
                    >
                      <Mic className="h-3.5 w-3.5" />
                      Record
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={stopWizardRecording}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-xl text-xs font-bold transition flex items-center gap-2 animate-pulse"
                    >
                      <span className="h-2 w-2 bg-white rounded-full inline-block" />
                      Stop Recording
                    </button>
                  )}

                  {wizardAudioUrl && (
                    <audio controls src={wizardAudioUrl} className="h-8 flex-1 min-w-0 max-w-xs rounded-lg" />
                  )}

                  {wizardAudioUrl && wizardTargetWord && (
                    <button
                      type="button"
                      onClick={transcribeRecordedVoice}
                      disabled={wizardLoading}
                      className="px-3 py-2 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      Suggest from Recording
                    </button>
                  )}
                </div>

                {wizardRecording && (
                  <p className="text-[10px] text-rose-500 font-semibold animate-pulse">
                    🎙 Recording… Speak clearly now.
                  </p>
                )}
                {wizardTargetWord && wizardAudioUrl && (
                  <p className="text-[10px] text-gray-500 dark:text-slate-400">
                    Heard: <span className="font-semibold text-indigo-600 dark:text-indigo-400">"{wizardTargetWord}"</span>
                  </p>
                )}
              </div>

              {/* Suggestion Cards */}
              {wizardSuggestions && (
                <div className="space-y-3">
                  {wizardSuggestions.origin_note && (
                    <div className="bg-amber-50/60 dark:bg-amber-950/20 border border-amber-200/40 dark:border-amber-900/30 rounded-xl px-4 py-3 text-[11px] text-amber-800 dark:text-amber-300 leading-relaxed flex gap-2 items-start">
                      <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                      {wizardSuggestions.origin_note}
                    </div>
                  )}

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {/* Simplified */}
                    {wizardSuggestions.simplified && (
                      <div className="bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700 rounded-2xl p-4 flex flex-col gap-3 shadow-sm">
                        <div>
                          <span className="text-[9px] font-extrabold text-gray-400 uppercase tracking-widest">Simplified Spelling</span>
                          <p className="mt-1.5 text-base font-bold text-gray-800 dark:text-slate-100 font-mono tracking-wide">
                            {wizardSuggestions.simplified}
                          </p>
                          <p className="text-[9px] text-gray-400 mt-1">Easy to read, no special characters</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            setNewPhoneticWord(wizardTargetWord);
                            setNewPhoneticReplacement(wizardSuggestions.simplified);
                            setNewPhoneticType("standard");
                            setShowPronunciationWizard(false);
                          }}
                          className="mt-auto px-3 py-1.5 bg-gray-100 hover:bg-gray-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-200 rounded-lg text-[10px] font-bold transition flex items-center gap-1 justify-center"
                        >
                          <Plus className="h-3 w-3" /> Use This
                        </button>
                      </div>
                    )}

                    {/* IPA */}
                    {(wizardSuggestions.ipa || wizardSuggestions.gruut_ipa) && (
                      <div className="bg-white dark:bg-slate-800 border border-emerald-100 dark:border-emerald-900/40 rounded-2xl p-4 flex flex-col gap-3 shadow-sm">
                        <div>
                          <span className="text-[9px] font-extrabold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest">IPA Phonemes</span>
                          <p className="mt-1.5 text-base font-bold text-gray-800 dark:text-slate-100 font-mono tracking-wide break-all">
                            {wizardSuggestions.ipa || wizardSuggestions.gruut_ipa}
                          </p>
                          <p className="text-[9px] text-gray-400 mt-1">Bypasses the G2P engine for exact pronunciation</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            setNewPhoneticWord(wizardTargetWord);
                            setNewPhoneticReplacement(wizardSuggestions.ipa || wizardSuggestions.gruut_ipa);
                            setNewPhoneticType("ipa");
                            setShowPronunciationWizard(false);
                          }}
                          className="mt-auto px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 dark:bg-emerald-950/40 dark:hover:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 rounded-lg text-[10px] font-bold transition flex items-center gap-1 justify-center"
                        >
                          <Plus className="h-3 w-3" /> Use IPA
                        </button>
                      </div>
                    )}

                    {/* ARPAbet */}
                    {wizardSuggestions.arpabet && (
                      <div className="bg-white dark:bg-slate-800 border border-sky-100 dark:border-sky-900/40 rounded-2xl p-4 flex flex-col gap-3 shadow-sm">
                        <div>
                          <span className="text-[9px] font-extrabold text-sky-600 dark:text-sky-400 uppercase tracking-widest">ARPAbet Tokens</span>
                          <p className="mt-1.5 text-base font-bold text-gray-800 dark:text-slate-100 font-mono tracking-wide break-all">
                            {wizardSuggestions.arpabet}
                          </p>
                          <p className="text-[9px] text-gray-400 mt-1">Stress-marked token format for speech engines</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            setNewPhoneticWord(wizardTargetWord);
                            setNewPhoneticReplacement(wizardSuggestions.arpabet);
                            setNewPhoneticType("arpabet");
                            setShowPronunciationWizard(false);
                          }}
                          className="mt-auto px-3 py-1.5 bg-sky-50 hover:bg-sky-100 dark:bg-sky-950/40 dark:hover:bg-sky-900/40 text-sky-700 dark:text-sky-300 rounded-lg text-[10px] font-bold transition flex items-center gap-1 justify-center"
                        >
                          <Plus className="h-3 w-3" /> Use ARPAbet
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Gruut G2P fallback note */}
                  {wizardSuggestions.gruut_ipa && wizardSuggestions.ipa && wizardSuggestions.gruut_ipa !== wizardSuggestions.ipa && (
                    <div className="flex flex-wrap items-center gap-2 px-3 py-2 bg-slate-50 dark:bg-slate-950 rounded-xl border border-gray-100 dark:border-slate-800 text-[10px] text-gray-500 dark:text-slate-400">
                      <span className="font-bold shrink-0">Local G2P (gruut):</span>
                      <span className="font-mono text-indigo-500 dark:text-indigo-300">{wizardSuggestions.gruut_ipa}</span>
                      <button
                        type="button"
                        onClick={() => {
                          setNewPhoneticWord(wizardTargetWord);
                          setNewPhoneticReplacement(wizardSuggestions.gruut_ipa);
                          setNewPhoneticType("ipa");
                          setShowPronunciationWizard(false);
                        }}
                        className="ml-auto px-2 py-1 bg-indigo-50 dark:bg-indigo-950/30 text-indigo-600 dark:text-indigo-300 rounded-lg font-bold hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition"
                      >
                        Use
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-4 border-t border-gray-100 dark:border-slate-800 flex justify-end gap-3 bg-gray-50/60 dark:bg-slate-950/30 shrink-0">
              <button
                type="button"
                onClick={() => setShowPronunciationWizard(false)}
                className="px-4 py-2 border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-800 rounded-xl text-xs font-bold text-gray-600 dark:text-slate-300 transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
    );
  };

  const renderProjectMultitrackTimeline = () => {
    const computedChapters = computeChapterStartTimes(chapters);
    const totalDuration = computedChapters.length > 0 
      ? Math.max(...computedChapters.map(ch => ch.startTime + ch.duration))
      : 30;

    const handleProjectRulerMouseDown = (mouseDownEvent) => {
      mouseDownEvent.preventDefault();
      mouseDownEvent.stopPropagation();
      const container = mouseDownEvent.currentTarget.parentElement;
      const updateProjectPlayhead = (moveEvent) => {
        const rect = container.getBoundingClientRect();
        const clickX = moveEvent.clientX - rect.left + container.scrollLeft;
        const newTime = Math.max(0, clickX / zoomLevel);
        setProjectPlayheadTime(newTime);
      };
      updateProjectPlayhead(mouseDownEvent);

      const handleMouseMove = (moveEvent) => {
        updateProjectPlayhead(moveEvent);
      };

      const handleMouseUp = () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
        if (isProjectPlayingRef.current) {
          playProjectTimeline();
        }
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    };

    return (
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-3xl shadow-2xl space-y-6 text-slate-100 font-sans mt-4">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-850">
          <div>
            <h3 className="text-sm font-bold text-indigo-400">📽️ Project Multitrack (Chapter Sequence Mixer)</h3>
            <p className="text-[10px] text-slate-400">Drag chapter blocks to adjust start times. Dragging a chapter will ripple-shift all subsequent chapters. Double click a block to open it in the editor.</p>
          </div>
          
          <div className="flex flex-wrap items-center gap-4 bg-slate-950 px-4 py-2.5 rounded-2xl border border-slate-800 shadow-md">
            {/* Playback Controls */}
            <div className="flex items-center gap-1.5 border-r border-slate-800 pr-4">
              <button
                type="button"
                onClick={isProjectPlaying ? () => stopProjectTimeline(false) : playProjectTimeline}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition active:scale-95 ${
                  isProjectPlaying
                    ? "bg-amber-600 hover:bg-amber-700 text-white"
                    : "bg-green-600 hover:bg-green-700 text-white"
                }`}
                title={isProjectPlaying ? "Pause Playback" : "Play Project"}
              >
                {isProjectPlaying ? (
                  <Pause className="h-4 w-4 fill-current" />
                ) : (
                  <Play className="h-4 w-4 fill-current ml-0.5" />
                )}
              </button>
              <button
                type="button"
                onClick={() => stopProjectTimeline(true)}
                className="w-8 h-8 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition active:scale-95"
                title="Stop & Reset Playhead"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
              <div className="text-xs font-mono text-slate-400 w-16 text-center select-none">
                {projectPlayheadTime.toFixed(1)}s
              </div>
            </div>

            {/* Export options */}
            <div className="flex items-center gap-3 border-r border-slate-800 pr-4">
              <button
                type="button"
                onClick={exportProjectTimeline}
                className="py-1.5 px-3 rounded-xl bg-indigo-655 hover:bg-indigo-700 active:scale-[0.98] text-white text-xs font-bold transition flex items-center gap-1.5 shadow-md"
                title="Mixdown and export all chapters as a single project audio file"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Export Project Mixdown</span>
              </button>
              <button
                type="button"
                onClick={() => exportMarkdownScript("project")}
                className="py-1.5 px-3 rounded-xl bg-slate-850 hover:bg-slate-800 active:scale-[0.98] text-slate-200 text-xs font-bold transition flex items-center gap-1.5 border border-slate-700 shadow-md"
                title="Export full project transcript as markdown"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Export Transcript (.md)</span>
              </button>
            </div>

            <div className="flex items-center gap-6 text-xs">
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Total Project Duration</span>
                <span className="font-mono text-gray-200 text-sm font-bold">
                  {Math.floor(totalDuration / 60)}m {Math.floor(totalDuration % 60)}s
                </span>
              </div>
              <div className="flex flex-col w-24">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Zoom</span>
                  <span className="text-[10px] text-gray-400 font-mono">{zoomLevel} px/s</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={zoomLevel > 30 ? 10 : zoomLevel}
                  onChange={(e) => setZoomLevel(parseInt(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer h-1 rounded"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto select-none pt-4">
          {chapters.length === 0 ? (
            <div className="h-48 border-2 border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center text-center p-6 text-slate-500">
              <Folder className="h-8 w-8 text-slate-500 mb-2" />
              <p className="text-xs font-semibold">No chapters found in this project.</p>
            </div>
          ) : (
            <div className="flex bg-slate-955 border border-slate-850 rounded-2xl overflow-hidden shadow-inner relative">
              {/* Left Column - Chapter Names */}
              <div className="w-40 shrink-0 bg-slate-950 border-r border-slate-850 flex flex-col pt-8 z-20">
                {computedChapters.map((ch, idx) => (
                  <div key={ch.id} className="h-20 border-b border-slate-900 p-3 flex flex-col justify-center bg-slate-955">
                    <span className="text-xs font-bold text-indigo-400 truncate">#{idx + 1} {ch.name}</span>
                    <span className="text-[9px] text-slate-500 mt-1">Duration: {ch.duration.toFixed(1)}s</span>
                  </div>
                ))}
              </div>

              {/* Right Timeline Lanes */}
              <div 
                className="flex-1 overflow-x-auto relative min-w-[600px] bg-slate-950"
                style={{ height: `${computedChapters.length * 80 + 32}px` }}
              >
                {/* Horizontal Time Grid Ruler */}
                <div 
                  onMouseDown={handleProjectRulerMouseDown}
                  className="absolute top-0 left-0 right-0 h-8 border-b border-slate-900 bg-slate-955/70 z-10 flex items-center cursor-pointer"
                >
                  {Array.from({ length: Math.ceil(totalDuration / 10) + 5 }).map((_, stepIdx) => {
                    const sec = stepIdx * 10;
                    return (
                      <div 
                        key={sec} 
                        className="absolute border-l border-slate-900/60 h-full text-[9px] font-mono text-slate-500 pl-1 pt-1 select-none"
                        style={{ left: `${sec * zoomLevel}px` }}
                      >
                        {sec}s
                      </div>
                    );
                  })}
                </div>

                {/* Lanes Grid */}
                <div className="pt-8 relative">
                  {computedChapters.map((ch, idx) => {
                    const left = ch.startTime * zoomLevel;
                    const width = ch.duration * zoomLevel;

                    return (
                      <div key={ch.id} className="h-20 border-b border-slate-900 bg-slate-900/10 relative">
                        <div
                          onMouseDown={(e) => handleChapterClipMouseDown(e, ch.id)}
                          onDoubleClick={() => {
                            switchChapter(ch.id);
                            const target = chapters.find(c => c.id === ch.id);
                            if (target) {
                              syncScriptToTimeline(target.podcastText || "", target.playlistClips || []);
                            }
                            setChapterEditorTab("multitrack");
                            setStorytellerViewMode("editor");
                          }}
                          style={{ left: `${left}px`, width: `${width}px` }}
                          className="absolute top-2 bottom-2 rounded-xl border border-indigo-500/40 p-2 flex flex-col justify-between cursor-move select-none overflow-hidden transition-all bg-gradient-to-r from-indigo-950/90 to-purple-950/90 hover:from-indigo-900/90 hover:to-purple-900/90 hover:border-indigo-400 text-indigo-100 shadow-md z-10"
                          title="Drag to change timing, double click to edit chapter timeline"
                        >
                          <div className="flex items-center justify-between w-full min-w-0">
                            <span className="text-[10px] font-bold truncate">{ch.name}</span>
                            <span className="text-[8px] bg-indigo-900/60 px-1 py-0.5 rounded font-mono text-indigo-300">
                              Start: {ch.startTime.toFixed(1)}s
                            </span>
                          </div>
                          
                          {/* Mini Waveform Visualization */}
                          <div className="flex items-center justify-around opacity-15 px-2 pointer-events-none z-0 h-6">
                            {Array.from({ length: 20 }).map((_, i) => (
                              <div
                                key={i}
                                className="w-[1.5px] bg-indigo-200 rounded-full"
                                style={{ height: `${20 + Math.sin(i * 0.5 + idx) * 30}%` }}
                              />
                            ))}
                          </div>

                          <div className="text-[8px] opacity-75 text-right font-semibold">
                            Total: {ch.duration.toFixed(1)}s
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Vertical Playhead Line */}
                <div
                  className="absolute top-0 bottom-0 w-[2px] bg-red-500 pointer-events-none z-30 shadow-[0_0_8px_rgba(239,68,68,0.9)]"
                  style={{ left: `${projectPlayheadTime * zoomLevel}px` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMultitrackTimeline = () => {
    const clips = playlistClips || [];
    const tracks = playlistTracks || DEFAULT_TIMELINE_TRACKS;
    const isMixerRendering = clips.some(c => c.status === "generating" || c.status === "queued");
    const parsedDialogueSegments = parseScriptTextToSegments(podcastText, numberOfSpeakers, speakerNames).filter(s => !s.isPause);
    const dialogueClips = clips.filter(c => c.trackId && c.trackId.startsWith("speaker_") && !c.isPause);

    const handleRulerMouseDown = (mouseDownEvent) => {
      mouseDownEvent.preventDefault();
      mouseDownEvent.stopPropagation();
      const container = mouseDownEvent.currentTarget.parentElement;
      const updatePlayhead = (moveEvent) => {
        const rect = container.getBoundingClientRect();
        const clickX = moveEvent.clientX - rect.left + container.scrollLeft;
        const newTime = Math.max(0, clickX / zoomLevel);
        setPlayheadTime(newTime);
      };
      updatePlayhead(mouseDownEvent);

      const handleMouseMove = (moveEvent) => {
        updatePlayhead(moveEvent);
      };

      const handleMouseUp = () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    };

    return (
      <div className="bg-gray-900 border border-gray-800 p-6 rounded-3xl shadow-2xl space-y-6 text-slate-100 font-sans">
        {/* Timeline Control Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-gray-850">
          
          {/* Playback Transport buttons */}
          <div className="flex items-center gap-2 bg-gray-950 p-1.5 rounded-xl border border-gray-800">
            <button
              onClick={playTimeline}
              disabled={isPlaying || clips.length === 0}
              className={`p-2.5 rounded-lg text-sm font-bold transition flex items-center justify-center ${
                isPlaying 
                  ? "bg-gray-800 text-gray-500 cursor-not-allowed" 
                  : "bg-green-600 hover:bg-green-500 text-white shadow active:scale-95"
              }`}
              title="Play Timeline"
            >
              <Play className="h-4 w-4 mr-1.5 fill-current" />
              Play
            </button>
            <button
              onClick={pauseTimeline}
              disabled={!isPlaying}
              className={`p-2.5 rounded-lg text-sm font-bold transition flex items-center justify-center ${
                !isPlaying 
                  ? "bg-gray-800 text-gray-500 cursor-not-allowed" 
                  : "bg-amber-600 hover:bg-amber-500 text-white shadow active:scale-95"
              }`}
              title="Pause Timeline"
            >
              <Pause className="h-4 w-4 mr-1.5 fill-current" />
              Pause
            </button>
            <button
              onClick={() => stopTimeline(true)}
              disabled={clips.length === 0}
              className="p-2.5 rounded-lg text-sm font-bold bg-red-600 hover:bg-red-500 text-white shadow transition active:scale-95 flex items-center justify-center"
              title="Stop and Reset Playhead"
            >
              <span className="w-3.5 h-3.5 bg-white rounded-sm mr-1.5 shrink-0" />
              Stop
            </button>
            <button
              onClick={splitTimelineClipAtPlayhead}
              disabled={clips.length === 0}
              className="p-2.5 rounded-lg text-sm font-bold bg-indigo-600 hover:bg-indigo-550 text-white shadow transition active:scale-95 flex items-center justify-center gap-1"
              title="Split clip at the current Playhead position"
              type="button"
            >
              <Scissors className="h-4 w-4" />
              Split
            </button>
          </div>

          {/* Zoom & Position readout */}
          <div className="flex items-center gap-6 bg-gray-950 px-4 py-2 rounded-xl border border-gray-800 text-xs">
            {/* Position readout */}
            <div className="flex flex-col">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Playhead</span>
              <span className="font-mono text-gray-200 text-sm font-bold">
                {Math.floor(playheadTime / 60)}:{String(Math.floor(playheadTime % 60)).padStart(2, "0")}.
                {String(Math.floor((playheadTime % 1) * 100)).padStart(2, "0")}
              </span>
            </div>

            {/* Timeline Duration */}
            <div className="flex flex-col">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Total Duration</span>
              <span className="font-mono text-gray-200 text-sm font-bold">
                {(() => {
                  const maxDur = Math.max(0, ...clips.map(c => c.startTime + (c.duration || 0)));
                  return `${Math.floor(maxDur / 60)}:${String(Math.floor(maxDur % 60)).padStart(2, "0")}s`;
                })()}
              </span>
            </div>

            {/* Zoom Knobs */}
            <div className="flex flex-col w-28">
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Zoom</span>
                <span className="text-[10px] text-gray-400 font-mono">{zoomLevel} px/s</span>
              </div>
              <input
                type="range"
                min="10"
                max="80"
                value={zoomLevel}
                onChange={(e) => setZoomLevel(parseInt(e.target.value))}
                className="w-full accent-purple-500 cursor-pointer h-1 rounded"
              />
            </div>
          </div>

          {/* Add FX / Music presets dropdowns */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 bg-gray-950 p-1.5 rounded-xl border border-gray-805">
              <select
                id="timeline-music-select-reusable"
                className="bg-gray-900 border border-gray-800 text-[11px] rounded-lg p-1.5 text-gray-300 focus:ring-1 focus:ring-purple-500"
                defaultValue="lofi"
              >
                <option value="lofi">Ambient Lo-Fi</option>
                <option value="intro">Tech Talk Intro</option>
                <option value="suspense">Dramatic Suspense</option>
                <option value="acoustic">Happy Acoustic</option>
              </select>
              <button
                onClick={() => {
                  const sel = document.getElementById("timeline-music-select-reusable");
                  if (sel) addMusicClip(sel.value);
                }}
                className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-2.5 py-1.5 rounded-lg shadow transition active:scale-95"
              >
                + Music
              </button>
            </div>

            <div className="flex items-center gap-1.5 bg-gray-950 p-1.5 rounded-xl border border-gray-805">
              <select
                id="timeline-sfx-select-reusable"
                className="bg-gray-900 border border-gray-800 text-[11px] rounded-lg p-1.5 text-gray-300 focus:ring-1 focus:ring-purple-500"
                defaultValue="laugh"
              >
                <option value="laugh">Studio Laughter</option>
                <option value="gasp">Dramatic Gasp</option>
                <option value="applause">Short Applause</option>
                <option value="whoosh">Transition Whoosh</option>
                <option value="ding">Bell Notification</option>
              </select>
              <button
                onClick={() => {
                  const sel = document.getElementById("timeline-sfx-select-reusable");
                  if (sel) addSfxClip(sel.value);
                }}
                className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-2.5 py-1.5 rounded-lg shadow transition active:scale-95"
              >
                + SFX
              </button>
            </div>
          </div>

          {/* Master Mix and Export controls */}
          <div className="flex items-center gap-2">
            {(() => {
              const pendingClips = clips.filter((clip) => {
                if (clip.isPause) return false;
                if (clip.status === "needs-render") return true;
                if (clip.trackId && clip.trackId.startsWith("speaker_")) {
                  const myIdx = dialogueClips.findIndex(c => c.id === clip.id);
                  const correspondingSegment = myIdx !== -1 ? parsedDialogueSegments[myIdx] : null;
                  if (correspondingSegment && correspondingSegment.text !== clip.text) {
                    return true;
                  }
                }
                return false;
              });

              if (pendingClips.length > 0) {
                return (
                  <button
                    onClick={() => {
                      pendingClips.forEach(clip => {
                        let textToRender = clip.text;
                        if (clip.trackId && clip.trackId.startsWith("speaker_")) {
                          const myIdx = dialogueClips.findIndex(c => c.id === clip.id);
                          const correspondingSegment = myIdx !== -1 ? parsedDialogueSegments[myIdx] : null;
                          if (correspondingSegment) {
                            textToRender = correspondingSegment.text;
                          }
                        }
                        generateClipAudio(clip.id, { ...clip, text: textToRender }, speakerMapping);
                      });
                    }}
                    className="px-4 py-2.5 rounded-xl text-xs font-bold bg-amber-600 hover:bg-amber-500 text-white border border-amber-500/30 hover:shadow-lg active:scale-95 flex items-center gap-1.5 animate-pulse"
                    title="Render all pending clips that need to be generated"
                  >
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-100" />
                    <span>Render Pending Clips ({pendingClips.length})</span>
                  </button>
                );
              }
              return null;
            })()}

            <div className="flex items-center gap-3 bg-gray-900/60 border border-gray-800/80 px-3 py-1.5 rounded-xl mr-1 text-[11px] font-medium text-gray-300">
              <span className="text-gray-405 font-bold uppercase tracking-wider text-[9px] mr-1">Effects:</span>
              <label className="flex items-center gap-1.5 cursor-pointer hover:text-white select-none">
                <input
                  type="checkbox"
                  checked={ppHardLimiter}
                  onChange={(e) => setPpHardLimiter(e.target.checked)}
                  className="rounded border-gray-700 bg-gray-950 text-purple-600 focus:ring-purple-500 focus:ring-offset-gray-900"
                />
                <span>Hard Limiter</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer hover:text-white select-none">
                <input
                  type="checkbox"
                  checked={ppPodcastVoice}
                  onChange={(e) => setPpPodcastVoice(e.target.checked)}
                  className="rounded border-gray-700 bg-gray-950 text-purple-600 focus:ring-purple-500 focus:ring-offset-gray-900"
                />
                <span>Podcast Voice</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer hover:text-white select-none">
                <input
                  type="checkbox"
                  checked={ppMastering}
                  onChange={(e) => setPpMastering(e.target.checked)}
                  className="rounded border-gray-700 bg-gray-950 text-purple-600 focus:ring-purple-500 focus:ring-offset-gray-900"
                />
                <span>Mastering</span>
              </label>
            </div>

            <button
              onClick={exportMixedPodcast}
              disabled={isMixerRendering || clips.length === 0}
              className={`px-4 py-2.5 rounded-xl text-xs font-bold text-white shadow transition-all duration-200 flex items-center gap-1.5 ${
                isMixerRendering || clips.length === 0
                  ? "bg-purple-900/50 text-slate-400 cursor-not-allowed border border-purple-950"
                  : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 border border-purple-500/30 hover:shadow-lg active:scale-95"
              }`}
            >
              {isMixerRendering ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  <span>Mixing Tracks...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-3.5 w-3.5 shrink-0" />
                  <span>Mix & Export Audio</span>
                </>
              )}
            </button>

            <button
              onClick={() => exportMarkdownScript("chapter")}
              className="px-3 py-2.5 rounded-xl text-xs font-bold text-slate-300 hover:text-white border border-gray-800 hover:border-gray-600 bg-gray-950 transition active:scale-95 flex items-center gap-1.5"
              title="Export this chapter's script as markdown"
              type="button"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Export Script (.md)</span>
            </button>
          </div>
        </div>

        {isMixerRendering && (
          <div className="bg-purple-950/20 border border-purple-900/40 p-4 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <RefreshCw className="h-4 w-4 text-purple-400 animate-spin shrink-0" />
              <div className="text-xs">
                <span className="font-bold text-purple-200">Rendering Voice Tracks...</span>
                <span className="text-gray-400 ml-2 font-mono">
                  ({clips.length - clips.filter(c => c.status === "generating" || c.status === "queued").length} / {clips.length} clips complete)
                </span>
              </div>
            </div>
            
            <div className="flex-1 max-w-md flex items-center gap-4">
              <div className="flex-1 bg-purple-950/80 rounded-full h-2 overflow-hidden border border-purple-900/60">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-indigo-500 h-full rounded-full transition-all duration-500"
                  style={{ 
                    width: `${Math.round(((clips.length - clips.filter(c => c.status === "generating" || c.status === "queued").length) / clips.length) * 100)}%` 
                  }}
                />
              </div>
              <div className="shrink-0 text-[10px] font-bold text-purple-300 font-mono flex items-center gap-3">
                <span>{Math.round(((clips.length - clips.filter(c => c.status === "generating" || c.status === "queued").length) / clips.length) * 100)}%</span>
                <span className="bg-purple-900/50 px-2 py-0.5 rounded text-purple-200 border border-purple-800/40">
                  ETA: {Math.ceil(clips.filter(c => c.status === "generating" || c.status === "queued").length * 2.5)}s
                </span>
              </div>
            </div>

            <button
              onClick={cancelAllQueue}
              className="bg-red-600 hover:bg-red-500 text-white font-bold text-[10px] uppercase tracking-wider px-3 py-1.5 rounded-lg border border-red-500/20 shadow active:scale-95 transition"
            >
              Cancel Render
            </button>
          </div>
        )}

        {/* Sound & Music Timeline Assister Toolbar */}
        <div className="bg-gray-950 p-4 rounded-2xl border border-gray-800/80 grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
          {/* Section 1: Add existing sound from Library */}
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1.5">
                Add Sound from Library
              </label>
              <select
                value={timelineSelectedSound}
                onChange={(e) => setTimelineSelectedSound(e.target.value)}
                className="w-full p-2 border border-gray-800 rounded-lg text-xs bg-gray-900 text-gray-350 focus:ring-1 focus:ring-purple-400 font-semibold"
              >
                <option value="">-- Select Sound Asset --</option>
                {soundAssets.map(s => {
                  const sType = s.type || "sfx";
                  const sName = s.name || s.key || "Unnamed";
                  const sDur = typeof s.duration === "number" ? s.duration : parseFloat(s.duration) || 0.0;
                  return (
                    <option key={s.key} value={`${sType}:${s.key}:${sDur}`}>
                      [{sType.toUpperCase()}] {sName} ({sDur.toFixed(1)}s)
                    </option>
                  );
                })}
              </select>
            </div>
            <button
              onClick={() => {
                if (!timelineSelectedSound) return;
                const [type, key, durStr] = timelineSelectedSound.split(":");
                const dur = parseFloat(durStr || "5.0");
                addSoundClipToTimeline(key, type, dur);
              }}
              disabled={!timelineSelectedSound}
              className="mt-5 py-2 px-4 rounded-lg text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-500 transition active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              type="button"
            >
              Add Clip at Playhead
            </button>
          </div>

          {/* Section 2: Quick Asset Gen / AI Sound generation */}
          <div className="flex items-end gap-3 border-t md:border-t-0 md:border-l border-gray-800 pt-3 md:pt-0 md:pl-4">
            <div className="flex-1 space-y-1.5">
              <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Quick AI Generate Sound
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={timelineGeneratePrompt}
                  onChange={(e) => setTimelineGeneratePrompt(e.target.value)}
                  placeholder="e.g. dramatic violin build-up, heavy laser blast..."
                  className="flex-1 p-2 border border-gray-800 rounded-lg text-xs bg-gray-900 text-gray-200 focus:ring-1 focus:ring-purple-400 font-semibold text-gray-100"
                />
                <select
                  value={timelineGenerateType}
                  onChange={(e) => setTimelineGenerateType(e.target.value)}
                  className="w-20 p-2 border border-gray-800 rounded-lg text-xs bg-gray-900 text-gray-300 focus:ring-1 focus:ring-purple-400 font-semibold shrink-0"
                >
                  <option value="music">Music</option>
                  <option value="sfx">SFX</option>
                </select>
              </div>
            </div>
            <button
              onClick={async () => {
                if (!timelineGeneratePrompt.trim()) return;
                const promptVal = timelineGeneratePrompt.trim();
                const typeVal = timelineGenerateType;
                
                const clipId = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
                const initialDur = typeVal === "music" ? 15.0 : 3.0;
                
                const newClip = {
                  id: clipId,
                  trackId: typeVal,
                  text: `${typeVal === "music" ? "Music" : "SFX"}: ${promptVal}`,
                  voiceDirection: "",
                  startTime: playheadTime,
                  duration: initialDur,
                  status: "generating",
                  audioUrl: null,
                  jobId: null,
                  [typeVal === "music" ? "musicKey" : "sfxKey"]: promptVal
                };
                
                setPlaylistClips(prev => {
                  const updated = [...prev, newClip];
                  setTimeout(() => syncTimelineToScript(updated), 50);
                  return updated;
                });
                
                setTimelineGeneratePrompt("");
                
                const endpoint = `http://localhost:5000/api/sound-library/resolve`;
                try {
                  const r = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      description: promptVal,
                      type: typeVal,
                      duration: initialDur,
                      token: freesoundToken
                    })
                  });
                  const data = await r.json();
                  if (data.url) {
                    const buf = await loadAudioBuffer(clipId, data.url);
                    const finalDur = buf ? buf.duration : initialDur;
                    setPlaylistClips(prev => {
                      const updated = prev.map(c => c.id === clipId ? {
                        ...c,
                        status: "done",
                        audioUrl: data.url,
                        duration: finalDur
                      } : c);
                      setTimeout(() => syncTimelineToScript(updated), 50);
                      return updated;
                    });
                    fetchSoundAssets();
                  } else {
                    setPlaylistClips(prev => prev.map(c => c.id === clipId ? { ...c, status: "error" } : c));
                  }
                } catch (err) {
                  console.error("Failed to generate timeline sound clip:", err);
                  setPlaylistClips(prev => prev.map(c => c.id === clipId ? { ...c, status: "error" } : c));
                }
              }}
              disabled={!timelineGeneratePrompt.trim()}
              className="py-2 px-4 rounded-lg text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-500 transition active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              type="button"
            >
              Generate & Add
            </button>
          </div>
        </div>

        {/* Timeline Workspace */}
        <div className="overflow-x-auto select-none pt-4">
          {clips.length === 0 ? (
            <div className="h-48 border-2 border-dashed border-gray-800 rounded-2xl flex flex-col items-center justify-center text-center p-6 text-gray-500">
              <Sliders className="h-8 w-8 text-gray-500 mb-2" />
              <p className="text-xs font-semibold">Your Multitrack Timeline is empty.</p>
              <p className="text-[10px] text-gray-650 max-w-xs mt-1">
                Type a dialogue script in the Editor and click "Load into Multitrack Timeline" to generate tracks.
              </p>
            </div>
          ) : (
            <div className="flex bg-gray-950 border border-gray-850 rounded-2xl overflow-hidden shadow-inner relative">
              
              {/* Left Header Panel: Track Info Column */}
              <div className="w-36 sm:w-56 shrink-0 bg-gray-950 border-r border-gray-850 flex flex-col pt-8 z-20">
                {tracks.filter(track => {
                  if (track.type !== "dialogue") return true;
                  const num = parseInt(track.id.split("_")[1]);
                  return num <= numberOfSpeakers;
                }).map((track) => {
                  const isDialogue = track.type === "dialogue";
                  const activeVoice = isDialogue ? (speakerMapping[track.id] || "kokoro:af_bella") : "";
                  
                  return (
                    <div key={track.id} className="h-28 border-b border-gray-900 p-2 sm:p-3 flex flex-col justify-between select-none bg-gray-950">
                      {/* Track Name */}
                      <div className="flex items-center justify-between gap-1">
                        <span className={`text-[10px] sm:text-xs font-bold flex items-center gap-1 truncate ${
                          track.id === "music" ? "text-amber-400" :
                          track.id === "sfx" ? "text-emerald-400" : "text-purple-400"
                        }`}>
                          {track.id === "music" ? (
                            <ListMusic className="h-3 w-3 sm:h-3.5 sm:w-3.5 shrink-0" />
                          ) : track.id === "sfx" ? (
                            <Volume2 className="h-3 w-3 sm:h-3.5 sm:w-3.5 shrink-0" />
                          ) : (
                            <Mic className="h-3 w-3 sm:h-3.5 sm:w-3.5 shrink-0" />
                          )}
                          <span className="truncate">{track.type === "dialogue" ? (speakerNames[track.id] || track.name) : track.name}</span>
                        </span>

                        {/* Re-render track button */}
                        {isDialogue && (
                          <button
                            onClick={() => renderTrack(track.id)}
                            disabled={isMixerRendering || clips.some(c => c.trackId === track.id && ["generating", "queued", "processing"].includes(c.status))}
                            className={`text-[9px] border px-1 py-0.5 rounded transition flex items-center gap-1 shrink-0 font-bold ${
                              isMixerRendering || clips.some(c => c.trackId === track.id && ["generating", "queued", "processing"].includes(c.status))
                                ? "bg-gray-900 text-gray-600 border-gray-800 cursor-not-allowed opacity-50"
                                : "bg-purple-950 hover:bg-purple-900 text-purple-200 border-purple-800"
                            }`}
                            title="Re-render track speech clips"
                          >
                            <RefreshCw className={`h-2.5 w-2.5 ${isMixerRendering ? "animate-spin" : "animate-spin-once"}`} />
                            <span className="hidden sm:inline">Render</span>
                          </button>
                        )}
                      </div>

                      {/* Dialogue Voice selection Dropdown */}
                      {isDialogue && (
                        <select
                          className="w-full text-[10px] bg-gray-900 border border-gray-800 text-gray-300 rounded p-1 font-semibold"
                          value={activeVoice}
                          onChange={(e) => handleTrackVoiceChange(track.id, e.target.value)}
                        >
                          {Object.entries(
                            allSelectableVoices.reduce((acc, voice) => {
                              const cat = voice.model || "Other Models";
                              if (!acc[cat]) acc[cat] = [];
                              acc[cat].push(voice);
                              return acc;
                            }, {})
                          ).map(([cat, list]) => (
                            <optgroup key={cat} label={cat} className="bg-gray-950 text-gray-400 font-semibold">
                              {list.map((v) => (
                                <option key={v.id} value={v.id} className="text-gray-200 bg-gray-900">
                                  {v.label}
                                </option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                      )}

                      {/* Track Knobs */}
                      <div className="flex items-center justify-between gap-1.5 mt-1 sm:mt-1.5 text-[9px] sm:text-[10px] font-bold text-gray-400">
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => toggleTrackMute(track.id)}
                            className={`px-1.5 py-0.5 rounded border transition ${
                              track.mute 
                                ? "bg-red-950 border-red-800 text-red-400 font-bold" 
                                : "bg-gray-900 border-gray-800 hover:bg-gray-855"
                            }`}
                          >
                            MUTE
                          </button>
                          
                          <button
                            onClick={() => toggleTrackSolo(track.id)}
                            className={`px-1.5 py-0.5 rounded border transition ${
                              track.solo 
                                ? "bg-amber-955 border-amber-600 text-amber-400 font-bold" 
                                : "bg-gray-900 border-gray-800 hover:bg-gray-855"
                            }`}
                          >
                            SOLO
                          </button>
                        </div>
                        <span className="text-[9px] text-gray-500 font-mono">
                          {Math.round((track.volume ?? 0.8) * 100)}%
                        </span>
                      </div>

                      {/* Track Volume Slider */}
                      <div className="flex items-center gap-2 mt-1 sm:mt-1.5">
                        <Sliders className="h-3 w-3 text-slate-500 shrink-0" />
                        <input
                          type="range"
                          min="0"
                          max="1.5"
                          step="0.05"
                          value={track.volume ?? 0.8}
                          onChange={(e) => updateTrackVolume(track.id, parseFloat(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 shrink-0"
                          title="Track Volume"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Right Scrollable Timeline Workspace area */}
              <div 
                className="flex-1 overflow-x-auto relative min-w-[600px] h-[calc(6*112px+32px)] bg-gray-950"
                style={{ height: `${tracks.filter(t => t.type !== "dialogue" || parseInt(t.id.split("_")[1]) <= numberOfSpeakers).length * 112 + 32}px` }}
                onClick={(e) => {
                  const isClipClick = e.target.closest('.cursor-move');
                  if (isClipClick) return;
                  const isRulerClick = e.target.closest('.cursor-ew-resize');
                  if (isRulerClick) return;

                  const rect = e.currentTarget.getBoundingClientRect();
                  const clickX = e.clientX - rect.left + e.currentTarget.scrollLeft;
                  const newTime = Math.max(0, clickX / zoomLevel);
                  setPlayheadTime(newTime);
                }}
              >
                {/* Horizontal time grids */}
                <div 
                  className="absolute top-0 left-0 right-0 h-8 border-b border-gray-900 bg-gray-950/70 z-10 flex items-center cursor-ew-resize select-none"
                  onMouseDown={handleRulerMouseDown}
                >
                  {Array.from({ length: 120 }).map((_, sec) => (
                    <div 
                      key={sec} 
                      className="absolute border-l border-gray-900/60 h-full text-[9px] font-mono text-gray-500 pl-1 pt-1 pointer-events-none"
                      style={{ left: `${sec * zoomLevel}px` }}
                    >
                      {sec}s
                    </div>
                  ))}
                </div>

                {/* Tracks Rows drawing lane clips */}
                <div className="pt-8">
                  {tracks.filter(track => {
                    if (track.type !== "dialogue") return true;
                    const num = parseInt(track.id.split("_")[1]);
                    return num <= numberOfSpeakers;
                  }).map((track) => (
                    <div key={track.id} className="h-28 border-b border-gray-900 bg-gray-900/10 relative">
                      {clips
                        .filter((c) => c.trackId === track.id)
                        .map((clip) => {
                          const left = clip.startTime * zoomLevel;
                          const width = (clip.duration || 2.0) * zoomLevel;
                          
                          let colorClass = "from-indigo-900/90 to-purple-900/90 hover:from-indigo-800 hover:to-purple-805 border-purple-700/80 text-purple-100";
                          if (clip.trackId === "music") {
                            colorClass = "from-amber-900/90 to-orange-900/90 hover:from-amber-800 hover:to-orange-800 border-orange-700/80 text-orange-100";
                          } else if (clip.trackId === "sfx") {
                            colorClass = "from-emerald-900/90 to-teal-900/90 hover:from-emerald-800 hover:to-teal-800 border-teal-700/80 text-teal-100";
                          }
                          
                          const isSpeakerTrack = clip.trackId && clip.trackId.startsWith("speaker_");
                          const spkColor = isSpeakerTrack ? (speakerColors[clip.trackId] || "#4f46e5") : null;
                          
                          // Warning validation checks
                          const myDialogueIndex = dialogueClips.findIndex(c => c.id === clip.id);
                          const correspondingSegment = myDialogueIndex !== -1 ? parsedDialogueSegments[myDialogueIndex] : null;
                          const isBusy = ["generating", "queued", "processing"].includes(clip.status);
                          const needsReRender = !clip.isPause && !isBusy && (clip.status === "needs-render" || (correspondingSegment && correspondingSegment.text !== clip.text));

                          const inlineStyle = { left: `${left}px`, width: `${width}px` };
                          if (clip.isPause) {
                            inlineStyle.background = "linear-gradient(to right, #d97706cc, #b45309cc)";
                            inlineStyle.borderColor = "#b45309";
                          } else if (spkColor) {
                            inlineStyle.background = `linear-gradient(to right, ${spkColor}dd, ${spkColor}aa)`;
                            inlineStyle.borderColor = needsReRender ? "#f59e0b" : spkColor;
                            if (needsReRender) {
                              inlineStyle.borderWidth = "2.5px";
                            }
                          }

                          return (
                            <div
                              key={clip.id}
                              onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
                              onDoubleClick={() => {
                                setSelectedTimelineClip(clip);
                                setIsClipModalOpen(true);
                              }}
                              className={`absolute top-2 bottom-2 rounded-xl border p-2 flex flex-col justify-between cursor-move select-none overflow-hidden transition-shadow shadow-md hover:shadow-lg bg-gradient-to-r ${
                                needsReRender ? "shadow-[0_0_10px_rgba(245,158,11,0.35)] ring-1 ring-amber-500/20" : ""
                              } ${colorClass}`}
                              style={inlineStyle}
                              title={clip.isPause ? "Double click to edit duration" : "Double click to edit clip text/settings. Drag horizontally to move."}
                            >
                              {/* Visual Status Overlay */}
                              {["generating", "queued", "processing"].includes(clip.status) && (
                                <div className="absolute inset-0 bg-black/75 flex flex-col items-center justify-center backdrop-blur-[0.5px] z-25 p-1">
                                  <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mb-1" />
                                  <span className="text-[8px] font-bold tracking-wider text-white uppercase opacity-95 text-center leading-none">
                                    {clip.status === "generating" ? "Sending..." : clip.status === "queued" ? "Queued" : `Rendering ${clip.progress}%`}
                                  </span>
                                </div>
                              )}
                              
                              <div className="flex items-start justify-between gap-1 z-10 w-full min-w-0">
                                <span className="text-[10px] font-bold leading-tight line-clamp-2 select-none truncate flex-1 min-w-0">
                                  {clip.isPause ? (
                                    <span className="flex items-center gap-1 text-amber-200">
                                      ⏱️ Pause: {clip.duration}s
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          const newVal = prompt("Enter pause duration in seconds:", clip.duration);
                                          if (newVal !== null) {
                                            const sec = parseFloat(newVal);
                                            if (!isNaN(sec) && sec >= 0) {
                                              setPlaylistClips(prev => {
                                                const updated = prev.map(c => c.id === clip.id ? { ...c, duration: sec, text: `<Pause: ${sec} seconds>` } : c);
                                                const newText = updated.map((c) => {
                                                  const spkNum = c.trackId.split("_")[1];
                                                  const spkName = speakerNames[c.trackId] || `Speaker ${spkNum}`;
                                                  if (c.isPause) {
                                                    return `<Pause: ${c.duration} seconds>`;
                                                  } else {
                                                    const dir = c.voiceDirection ? `(${c.voiceDirection}) ` : "";
                                                    return `[${spkName}] ${dir}${c.text}`;
                                                  }
                                                }).join("\n\n");
                                                setPodcastText(newText);
                                                return updated;
                                              });
                                            }
                                          }
                                        }}
                                        className="px-1 py-0.5 bg-black/40 hover:bg-black/60 rounded text-[8px]"
                                      >
                                        Edit
                                      </button>
                                    </span>
                                  ) : (
                                    <>
                                      {clip.voiceDirection && <span className="italic opacity-85 mr-1 font-semibold">({clip.voiceDirection})</span>}
                                      {clip.text}
                                    </>
                                  )}
                                </span>
                                
                                <div className="flex items-center gap-1 shrink-0">
                                  {!clip.isPause && clip.audioUrl && (
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        togglePlayClipAudio(clip.audioUrl);
                                      }}
                                      className="bg-black/45 hover:bg-black/60 text-white rounded-full w-5 h-5 flex items-center justify-center transition"
                                      title="Preview clip audio"
                                      type="button"
                                    >
                                      {playingClipUrl === clip.audioUrl ? (
                                        <Pause className="h-2.5 w-2.5 fill-current" />
                                      ) : (
                                        <Play className="h-2.5 w-2.5 fill-current ml-0.5" />
                                      )}
                                    </button>
                                  )}
                                  
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      deleteTimelineClip(clip.id);
                                    }}
                                    className="bg-black/45 hover:bg-red-950/60 text-white hover:text-red-400 rounded-full w-5 h-5 flex items-center justify-center transition"
                                    title="Delete clip"
                                    type="button"
                                  >
                                    <Trash2 className="h-3 w-3" />
                                  </button>
                                </div>
                              </div>

                              {needsReRender && (
                                <span className="absolute top-1 right-1 flex items-center gap-0.5 bg-amber-100 border border-amber-305 text-amber-905 px-1 py-0.5 rounded text-[8px] font-black z-30 animate-pulse">
                                  <AlertTriangle className="h-2.5 w-2.5" />
                                  <span>Needs Render</span>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      generateClipAudio(clip.id, { ...clip, text: correspondingSegment.text }, speakerMapping);
                                    }}
                                    className="px-1 py-0.5 bg-amber-500 hover:bg-amber-600 text-white rounded text-[8px] font-bold ml-1"
                                  >
                                    Render
                                  </button>
                                </span>
                              )}

                              {(() => {
                                if (clip.isPause) return null;
                                const numPeaks = Math.max(12, Math.floor(width / 6));
                                const peaks = getClipPeaks(clip.id, numPeaks);
                                if (peaks) {
                                  return (
                                    <div className="absolute inset-0 flex items-center justify-around opacity-30 px-2 pointer-events-none z-0">
                                      {peaks.map((p, i) => (
                                        <div
                                          key={i}
                                          className="w-[1.5px] bg-white rounded-full transition-all duration-300"
                                          style={{ height: `${Math.max(5, p * 85)}%` }}
                                        />
                                      ))}
                                    </div>
                                  );
                                } else {
                                  return (
                                    <div className="absolute inset-0 flex items-center justify-around opacity-15 px-2 pointer-events-none z-0">
                                      {Array.from({ length: numPeaks }).map((_, i) => (
                                        <div
                                          key={i}
                                          className="w-[1.5px] bg-white rounded-full animate-pulse"
                                          style={{ height: `${20 + Math.sin(i * 0.4) * 15}%` }}
                                        />
                                      ))}
                                    </div>
                                  );
                                }
                              })()}

                              <div className="flex justify-between items-center text-[8px] opacity-90 z-10">
                                <span className="font-mono text-white/70">{(clip.duration || 0.0).toFixed(1)}s</span>
                                {clip.status === "generating" && (
                                  <span className="flex items-center gap-0.5 bg-black/40 px-1 py-0.5 rounded text-[8px] font-bold text-purple-350">
                                    <span className="animate-spin inline-block w-2.5 h-2.5 border border-purple-400 border-t-transparent rounded-full" />
                                    Sending...
                                  </span>
                                )}
                                {clip.status === "queued" && (
                                  <span className="bg-black/40 px-1 py-0.5 rounded font-bold text-amber-300 animate-pulse">
                                    Queued
                                  </span>
                                )}
                                {clip.status === "processing" && (
                                  <span className="flex items-center gap-1 bg-black/40 px-1 py-0.5 rounded text-[8px] font-bold text-blue-350">
                                    <span className="animate-spin inline-block w-2.5 h-2.5 border border-blue-400 border-t-transparent rounded-full" />
                                    {clip.progress}%
                                  </span>
                                )}
                                {clip.status === "error" && (
                                  <span className="flex items-center gap-0.5 bg-red-950 border border-red-800 px-1 py-0.5 rounded font-bold text-red-300">
                                    <AlertTriangle className="h-2.5 w-2.5" />
                                    <span>Fail</span>
                                  </span>
                                )}
                                {clip.status === "idle" && (
                                  <span className="bg-black/30 px-1 py-0.5 rounded font-semibold text-white/50">
                                    Not rendered
                                  </span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  ))}
                </div>

                <div
                  className="absolute top-0 bottom-0 w-[2px] bg-red-505 pointer-events-none z-30 shadow-[0_0_8px_rgba(239,68,68,0.9)]"
                  style={{ left: `${playheadTime * zoomLevel}px` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const isMixerRendering = playlistClips.some(c => c.status === "generating" || c.status === "queued");
  const isQueueRendering = queue.some(item => item.status === "generating" || item.status === "queued");

  /* ---------- UI ---------- */
  return (
    <div className={`flex h-screen w-screen overflow-hidden ${darkMode ? "dark bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-900"}`}>
      {/* Sidebar mobile overlay backdrop */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden transition-opacity duration-300"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* LHS Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 text-slate-100 flex flex-col h-full border-r border-slate-800 transition-transform duration-300 transform md:relative md:translate-x-0 ${
        mobileMenuOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full"
      }`}>
        {/* Sidebar Header: Notion-Style Project Selector */}
        <div className="p-4 border-b border-slate-800">
          <div className="relative">
            <button
              onClick={() => {
                setShowProjectManager(prev => !prev);
                setMobileMenuOpen(false);
              }}
              className="w-full flex items-center justify-between gap-2 px-3 py-2 bg-slate-800/60 hover:bg-slate-800 rounded-lg text-slate-200 transition duration-150 text-xs font-bold border border-slate-700/50"
            >
              <span className="flex items-center gap-2 truncate">
                <Folder className="h-4 w-4 text-indigo-400 shrink-0" />
                <span className="truncate">{activeProjectName}</span>
              </span>
              <ChevronDown className="h-3 w-3 text-slate-400 shrink-0" />
            </button>
            <div className="mt-2 flex items-center justify-between px-1">
              {isSaving ? (
                <span className="flex items-center gap-1 text-[10px] font-semibold text-slate-400">
                  <RefreshCw className="h-2.5 w-2.5 animate-spin text-indigo-400" />
                  Saving...
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[10px] font-semibold text-slate-400">
                  <Check className="h-2.5 w-2.5 text-emerald-400" />
                  Saved locally
                </span>
              )}
              <button
                onClick={() => {
                  saveProjectSync();
                  setIsSaving(true);
                  setTimeout(() => setIsSaving(false), 500);
                }}
                className="text-[9px] font-bold text-indigo-400 hover:text-indigo-300 transition uppercase tracking-wider"
                title="Force save project now"
              >
                Save Now
              </button>
            </div>
          </div>
        </div>

        {/* Tree View Navigation */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {/* Active Project Section */}
          <div className="space-y-1">
            <div className="px-3 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
              Workspace
            </div>
            
            <div className="space-y-0.5">
              <button
                onClick={() => { setActiveMainTab("creator"); setMobileMenuOpen(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition ${
                  activeMainTab === "creator"
                    ? "bg-slate-800 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Mic className="h-4 w-4 shrink-0 text-indigo-400" />
                <span>Voice Creator</span>
              </button>

              <button
                onClick={() => { setActiveMainTab("experiment"); setMobileMenuOpen(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition ${
                  activeMainTab === "experiment"
                    ? "bg-slate-800 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <FlaskConical className="h-4 w-4 shrink-0 text-blue-400" />
                <span>The Playground</span>
              </button>

              <div className="space-y-1">
                <button
                  onClick={() => {
                    setActiveMainTab("storyteller");
                    goBackToOverview();
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition ${
                    activeMainTab === "storyteller" && storytellerViewMode === "overview"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  }`}
                  type="button"
                >
                  <BookOpen className="h-4 w-4 shrink-0 text-purple-400" />
                  <span>Script & Dialogue</span>
                </button>

                {/* Chapters child list in sidebar */}
                <div className="pl-6 pr-2 space-y-1 pb-1">
                  {chapters.map((ch) => {
                    const isChActive = activeMainTab === "storyteller" && storytellerViewMode === "editor" && currentChapterId === ch.id;
                    return (
                      <div
                        key={ch.id}
                        onClick={() => {
                          switchChapter(ch.id);
                          setStorytellerViewMode("editor");
                          setActiveMainTab("storyteller");
                          setMobileMenuOpen(false);
                        }}
                        className={`group/item w-full flex items-center justify-between px-2 py-1 rounded-md text-[11px] font-semibold transition cursor-pointer ${
                          isChActive
                            ? "bg-slate-800 text-white font-bold"
                            : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/20"
                        }`}
                      >
                        <span className="truncate max-w-[120px]">{ch.name}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteChapter(ch.id);
                          }}
                          className="opacity-0 group-hover/item:opacity-100 text-[10px] text-slate-500 hover:text-red-400 transition"
                          title="Delete Chapter"
                          type="button"
                        >
                          ✕
                        </button>
                      </div>
                    );
                  })}

                  <button
                    onClick={() => {
                      createNewChapter();
                      setStorytellerViewMode("editor");
                      setActiveMainTab("storyteller");
                    }}
                    className="w-full flex items-center gap-1 px-2 py-1 text-[10px] text-indigo-400 hover:text-indigo-300 font-bold uppercase tracking-wider transition hover:bg-slate-800/20 rounded-md"
                    type="button"
                  >
                    <Plus className="h-3 w-3" />
                    Add Chapter
                  </button>
                </div>
              </div>



              <button
                onClick={() => { setActiveMainTab("project-settings"); setMobileMenuOpen(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition ${
                  activeMainTab === "project-settings"
                    ? "bg-slate-800 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Settings2 className="h-4 w-4 shrink-0 text-amber-400" />
                <span>Project Settings</span>
              </button>
            </div>
          </div>

          {/* Curated Voices Section */}
          <div className="space-y-1">
            <div className="px-3 flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-wider">
              <span>Curated Voices</span>
              <span className="bg-slate-800 text-slate-400 text-[9px] px-1.5 py-0.5 rounded-full">
                {curatedVoices.length}
              </span>
            </div>
            
            <div className="max-h-40 overflow-y-auto space-y-0.5 px-1">
              {curatedVoices.length === 0 ? (
                <div className="text-[10px] text-slate-500 px-3 py-1.5 italic">
                  No curated voices saved
                </div>
              ) : (
                curatedVoices.map(v => (
                  <div key={v.id} className="w-full flex items-center justify-between gap-1 group px-2 py-1 rounded hover:bg-slate-800/30">
                    <button
                      onClick={() => {
                        setSelectedVoice(v.voice);
                        setActiveMainTab("experiment");
                        loadCuratedVoice(v);
                      }}
                      className="flex-1 flex items-center gap-2 text-[11px] text-slate-400 hover:text-slate-200 truncate text-left"
                    >
                      <Library className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                      <span className="truncate">{v.name}</span>
                    </button>
                    <div className="hidden group-hover:flex items-center gap-1 shrink-0">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const newName = prompt("Rename voice to:", v.name);
                          if (newName && newName.trim()) {
                            setCuratedVoices(prev => prev.map(item => item.id === v.id ? { ...item, name: newName.trim() } : item));
                          }
                        }}
                        className="text-slate-400 hover:text-amber-400 p-0.5"
                        title="Rename"
                      >
                        <Edit3 className="h-3.5 w-3.5 text-slate-400 hover:text-amber-400" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm(`Delete curated voice "${v.name}"?`)) {
                            setCuratedVoices(prev => prev.filter(item => item.id !== v.id));
                          }
                        }}
                        className="text-slate-400 hover:text-red-400 p-0.5"
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5 text-slate-400 hover:text-red-400" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Render Queue Section */}
          <div className="space-y-1">
            <button
              onClick={() => {
                setShowQueue(prev => {
                  if (!prev) setUnreadCompletions(0);
                  return !prev;
                });
              }}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold transition ${
                showQueue
                  ? "bg-slate-800 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
              }`}
            >
              <span className="flex items-center gap-2.5">
                <ListMusic className="h-4 w-4 shrink-0 text-blue-400" />
                <span>Render Queue</span>
                {isQueueRendering && (
                  <RefreshCw className="h-3 w-3 animate-spin text-blue-400 shrink-0" />
                )}
              </span>
              
              {queue.filter(item => item.status !== "done" && item.status !== "error").length > 0 ? (
                <span className="bg-blue-600 text-white text-[9px] px-1.5 py-0.5 rounded-full animate-pulse">
                  {queue.filter(item => item.status !== "done" && item.status !== "error").length}
                </span>
              ) : (
                unreadCompletions > 0 && (
                  <span className="bg-red-500 text-white text-[9px] px-1.5 py-0.5 rounded-full">
                    {unreadCompletions}
                  </span>
                )
              )}
            </button>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/60 space-y-2">
          <button
            onClick={() => {
              setShowProjectManager(true);
              setMobileMenuOpen(false);
            }}
            className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            <FolderPlus className="h-4 w-4 text-indigo-400" />
            <span>Manage Projects</span>
          </button>
          
          <div className="flex items-center justify-between gap-2">
            <button
              onClick={() => {
                setShowSettings(true);
                setMobileMenuOpen(false);
              }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              title="Application Settings"
            >
              <Settings className="h-4 w-4 text-slate-400" />
              <span>App Settings</span>
            </button>

            <button
              onClick={() => {
                const updated = !darkMode;
                setDarkMode(updated);
                localStorage.setItem(
                  "voicationSetting",
                  JSON.stringify({ darkMode: updated })
                );
              }}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Main Workspace Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Mobile Header Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-800 shrink-0">
          <button
            onClick={() => setMobileMenuOpen(true)}
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-655 dark:text-slate-350 transition"
            title="Open Menu"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          
          <div className="flex items-center gap-1.5 truncate max-w-[180px]">
            <span className="text-xs font-extrabold tracking-wide uppercase text-indigo-600 dark:text-indigo-400 truncate">
              {activeProjectName}
            </span>
            {isSaving ? (
              <RefreshCw className="h-3 w-3 animate-spin text-indigo-500 shrink-0" />
            ) : (
              <Check className="h-3 w-3 text-emerald-500 shrink-0" />
            )}
          </div>
          
          <button
            onClick={() => setShowQueue(prev => !prev)}
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-655 dark:text-slate-350 transition relative"
            title="Render Queue"
          >
            <ListMusic className="h-5 w-5" />
            {queue.filter(item => item.status !== "done" && item.status !== "error").length > 0 && (
              <span className="absolute top-1 right-1 bg-blue-600 w-2.5 h-2.5 rounded-full animate-pulse" />
            )}
          </button>
        </div>

        <div className="flex-1 flex flex-row overflow-hidden relative">
          <div className="flex-1 overflow-y-auto p-4 md:p-8">
          
          {/* Onboarding Welcome Banner */}
          {showOnboardingBanner && (
            <div className="mb-6 p-4 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/50 rounded-xl relative flex gap-3 shadow-sm">
              <Sparkles className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
              <div className="pr-8">
                <h3 className="text-sm font-bold text-indigo-950 dark:text-indigo-200">
                  Welcome to Voication Studio!
                </h3>
                <p className="text-xs text-indigo-900 dark:text-indigo-300 mt-1 leading-relaxed">
                  To get started:
                  <span className="block mt-1">1. Configure or clone voices in the <strong>Playground / Voice Creator</strong> and save them to your <strong>Curated Library</strong>.</span>
                  <span className="block mt-0.5">2. Switch to <strong>Script & Dialogue</strong>, write your story, and assign curated speakers to lines.</span>
                  <span className="block mt-0.5">3. Render the lines, load them into the <strong>Multitrack Mixer</strong>, and export your final audio format!</span>
                </p>
              </div>
              <button
                onClick={() => {
                  setShowOnboardingBanner(false);
                  localStorage.setItem("voication_show_onboarding", "false");
                }}
                className="absolute top-3 right-3 text-indigo-400 hover:text-indigo-600 transition"
                title="Dismiss Welcome Banner"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
      {activeMainTab === "experiment" && (
        <div className="max-w-6xl mx-auto mt-8 px-4 sm:px-6 lg:px-8 pb-16">
          {/* Sub-tab Navigation */}
          <div className="flex border-b border-gray-150 dark:border-slate-800 mb-6">
            <button
              onClick={() => setPlaygroundSubTab("voice")}
              className={`py-3 px-6 font-semibold text-sm flex items-center gap-2 border-b-2 transition-all duration-200 ${
                playgroundSubTab === "voice"
                  ? "border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400"
                  : "border-transparent text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
            >
              <Mic className="h-4 w-4" />
              Voice Sandbox
            </button>
            <button
              onClick={() => setPlaygroundSubTab("sound")}
              className={`py-3 px-6 font-semibold text-sm flex items-center gap-2 border-b-2 transition-all duration-200 ${
                playgroundSubTab === "sound"
                  ? "border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400"
                  : "border-transparent text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
            >
              <ListMusic className="h-4 w-4" />
              Sound & Music Library
            </button>
          </div>

          {playgroundSubTab === "voice" ? (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Experiment Controls (8 cols) */}
          <div className="lg:col-span-8 p-6 bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 shadow-xl rounded-2xl relative text-gray-900 dark:text-slate-100">
            <h1 className="text-2xl font-bold text-center text-gray-900 dark:text-white">Voication Playground</h1>
            <p className="text-xs text-gray-500 text-center mt-1 mb-6">
              Experiment with text-to-speech models, tune voice settings, and save curated speakers.
            </p>

          {/* Voice selector */}
          <div className="mb-6">
            <label className="block text-sm font-semibold mb-2">
              Choose a Voice Model
            </label>
            <div className="flex gap-4 overflow-x-auto pb-2 pr-4">
              {voices.filter(v => v.name !== "vibevoice" && isModelEnabled(v.name)).map((v) => {
              const displayName = v.name === "bark" 
                ? "Bark (Expressive)" 
                : v.name === "tts_models/en/vctk/vits" 
                ? "VITS (English)" 
                : v.name === "tts_models/multilingual/multi-dataset/xtts_v2" 
                ? "XTTS v2 (Cloning)" 
                : v.name === "kokoro"
                ? "Kokoro-82M"
                : v.name === "qwen3-tts"
                ? "Qwen3-TTS"
                : v.name === "chatterbox-turbo"
                ? "Chatterbox Turbo"
                : v.name === "vibevoice"
                ? "VibeVoice"
                : v.name === "cosyvoice2-styletts2"
                ? "CosyVoice 2"
                : v.name === "chattts"
                ? "ChatTTS"
                : v.name === "fish-audio"
                ? "Fish Audio"
                : v.name;

              const isExperimental = v.name !== "tts_models/en/vctk/vits" && v.name !== "tts_models/multilingual/multi-dataset/xtts_v2";

              return (
                <label
                  key={v.name}
                  className={`border rounded-xl cursor-pointer w-[180px] shrink-0 whitespace-normal break-words transition-all duration-200 relative p-4 ${
                    selectedVoice === v.name
                      ? "border-blue-500 ring-2 ring-blue-300 dark:ring-blue-900 bg-blue-50/10 dark:bg-blue-950/20"
                      : "bg-white dark:bg-slate-800 border-gray-100 dark:border-slate-800 hover:border-gray-400 dark:hover:border-slate-700"
                  }`}
                >
                  <input
                    type="radio"
                    name="voice"
                    value={v.name}
                    checked={selectedVoice === v.name}
                    onChange={() => {
                      setSelectedVoice(v.name);
                      setLoadedCuratedVoiceId(null);
                    }}
                    className="hidden"
                  />
                  <h3 className="font-semibold whitespace-normal break-words flex flex-wrap items-center gap-1.5 text-sm mb-1">
                    {displayName}
                  </h3>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {v.is_simulator && (
                      <span className="bg-yellow-100 dark:bg-yellow-950/40 text-yellow-800 dark:text-yellow-350 text-[8px] font-bold px-1.5 py-0.5 rounded-md uppercase tracking-wider">
                        Simulator
                      </span>
                    )}
                    {isExperimental && (
                      <span className="bg-amber-100 dark:bg-amber-950/40 text-amber-800 dark:text-amber-350 text-[8px] font-bold px-1.5 py-0.5 rounded-md uppercase tracking-wider">
                        Experimental
                      </span>
                    )}
                    {v.requires_speaker_wav ? (
                      <span className="bg-rose-105 dark:bg-rose-955/40 text-rose-800 dark:text-rose-350 text-[8px] font-bold px-1.5 py-0.5 rounded-md uppercase tracking-wider flex items-center gap-1">
                        <Fingerprint className="h-2.5 w-2.5" />
                        Requires Voice Upload
                      </span>
                    ) : v.features?.includes("cloning") ? (
                      <span className="bg-green-105 dark:bg-green-955/40 text-green-800 dark:text-green-350 text-[8px] font-bold px-1.5 py-0.5 rounded-md uppercase tracking-wider flex items-center gap-1">
                        <Fingerprint className="h-2.5 w-2.5" />
                        Supports Cloning
                      </span>
                    ) : null}
                  </div>
                  {v.description && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 whitespace-normal break-words leading-relaxed">
                      {v.description}
                    </p>
                  )}
                  {selectedVoice === v.name && v.features && v.features.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-100 dark:border-slate-800">
                      <p className="text-[10px] uppercase tracking-wider text-gray-400 font-bold mb-1">Features:</p>
                      <ul className="text-[10px] text-gray-500 dark:text-gray-400 list-disc ml-3 space-y-0.5">
                        {v.features.map(f => (
                          <li key={f} className="capitalize">{f.replace('_', ' ')}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </label>
              );
            })}
          </div>
        </div>

        {/* --- Text input and AI Enhance toggle grouped --- */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <p className="text-base font-semibold">Add your text</p>
            <label className="px-2.5 py-1 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-950/20 rounded-lg transition flex items-center gap-1 border border-indigo-100 dark:border-indigo-900/30 cursor-pointer shadow-sm">
              <Upload className="h-3 w-3 shrink-0" />
              Import DOCX/Text/MD
              <input
                type="file"
                accept=".docx,.txt,.md,.markdown"
                className="hidden"
                onChange={(e) => handleDocumentImport(e, "playground")}
              />
            </label>
          </div>
          <div
            className={`enhance-input-container${isEnhancing ? " dimmed" : ""}`}
          >
            {selectedVoiceData?.features?.includes("tags") || selectedVoiceData?.features?.includes("multi_speaker") ? (
              <TagEditor
                ref={tagEditorRef}
                value={text}
                onChange={setText}
                tokens={tokensList}
                speakerColorsMap={speakerColorsMap}
                disabled={isEnhancing}
                isLoading={isEnhancing}
                loadingMessage="AI Script Polish & Enhancement in Progress..."
                placeholder="Type or paste your narration here – type “[” or “/” to see token suggestions…"
                className="w-full p-4 border border-gray-200 dark:border-slate-800 rounded-xl focus:ring-2 focus:ring-blue-500 min-h-[8rem] bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
              />
            ) : (
              <div className="relative">
                {isEnhancing && (
                  <div className="absolute inset-0 z-10 bg-gray-100 dark:bg-slate-900/60 bg-opacity-60 dark:bg-opacity-80 animate-pulse rounded-xl flex flex-col justify-center p-4 pointer-events-none">
                    <div className="h-4 bg-gray-300 dark:bg-slate-700 rounded mb-2 w-3/4"></div>
                    <div className="h-4 bg-gray-300 dark:bg-slate-700 rounded mb-2 w-full"></div>
                    <div className="h-4 bg-gray-300 dark:bg-slate-700 rounded w-1/2"></div>
                  </div>
                )}
                <textarea
                  ref={textareaRef}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Type or paste your narration here"
                  className="w-full p-4 border border-gray-200 dark:border-slate-800 rounded-xl focus:ring-2 focus:ring-blue-500 min-h-[8rem] resize-none relative z-0 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                />
              </div>
            )}

            {/* Consolidated Dynamic Tag Tray */}
            {selectedVoiceData && (
              <div className="mt-2 p-3 bg-gray-50 dark:bg-slate-900 border border-gray-100 dark:border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mr-1">
                    Insert Tag:
                  </span>
                  {selectedVoice === "chattts" && chatttsRefineText && (
                    <span className="text-[9px] text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 px-2 py-0.5 rounded-md font-semibold">
                      Auto-Refine is active (manual tags will be ignored)
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  {selectedVoice === "chattts" ? (
                    chatttsRefineText ? (
                      <div className="text-[10px] text-gray-400 dark:text-slate-500 italic py-0.5">
                        Tag insertion locked. Disable "Auto-Refine Phrasing" in the tuning panel to manually insert prosody tags.
                      </div>
                    ) : (
                      <div className="flex flex-wrap gap-2 items-center w-full text-xs">
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] font-semibold text-gray-500 dark:text-slate-400 mr-1">Laughter:</span>
                          {[0, 1, 2].map((i) => (
                            <button
                              key={`laugh_${i}`}
                              type="button"
                              onClick={() => insertChatttsTag(`[laugh_${i}]`)}
                              className="px-2 py-0.5 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-[11px] font-medium hover:border-blue-400 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400 transition"
                            >
                              [laugh_{i}]
                            </button>
                          ))}
                        </div>
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] font-semibold text-gray-500 dark:text-slate-400 mr-1">Pauses:</span>
                          {[0, 1, 3, 7].map((i) => (
                            <button
                              key={`break_${i}`}
                              type="button"
                              onClick={() => insertChatttsTag(`[break_${i}]`)}
                              className="px-2 py-0.5 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-[11px] font-medium hover:border-blue-400 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400 transition"
                            >
                              [break_{i}]
                            </button>
                          ))}
                        </div>
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] font-semibold text-gray-500 dark:text-slate-400 mr-1">Oral:</span>
                          {[0, 2, 5, 9].map((i) => (
                            <button
                              key={`oral_${i}`}
                              type="button"
                              onClick={() => insertChatttsTag(`[oral_${i}]`)}
                              className="px-2 py-0.5 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-[11px] font-medium hover:border-blue-400 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400 transition"
                            >
                              [oral_{i}]
                            </button>
                          ))}
                        </div>
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] font-semibold text-gray-500 dark:text-slate-400 mr-1">Word-Level:</span>
                          {["uv_break", "lbreak"].map((tag) => (
                            <button
                              key={tag}
                              type="button"
                              onClick={() => insertChatttsTag(`[${tag}]`)}
                              className="px-2 py-0.5 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-[11px] font-medium hover:border-blue-400 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400 transition"
                            >
                              [{tag}]
                            </button>
                          ))}
                        </div>
                      </div>
                    )
                  ) : (
                    selectedVoiceData?.features?.includes("tags") && tokensList.length > 0 ? (
                      tokensList.map((tag) => (
                        <button
                          key={tag}
                          type="button"
                          onClick={() => insertTextAtCursor(`[${tag}]`)}
                          className="px-2 py-1 text-xs font-semibold bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg hover:border-blue-400 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400 transition shadow-sm active:scale-95 text-gray-900 dark:text-slate-200"
                        >
                          [{tag}]
                        </button>
                      ))
                    ) : (
                      <span className="text-[10px] text-gray-400 italic">No tag controls available for this model.</span>
                    )
                  )}
                </div>
              </div>
            )}

            {/* Speaker Tag Insertion Pills (VibeVoice) */}
            {selectedVoiceData?.features?.includes("multi_speaker") && (
              <div className="mt-2 flex flex-wrap items-center gap-1.5 p-2 bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100/50 dark:border-indigo-900/30 rounded-xl">
                <span className="text-[10px] font-bold text-indigo-400 dark:text-indigo-400 uppercase tracking-wider mr-1">Insert Speaker Tag:</span>
                {[1, 2, 3, 4].map((num) => (
                  <button
                    key={num}
                    type="button"
                    onClick={() => insertTextAtCursor(`[Speaker ${num}]`)}
                    className="px-2.5 py-1 text-xs font-bold bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-900 rounded-lg hover:border-indigo-400 dark:hover:border-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 transition shadow-sm active:scale-95"
                    title={`Click to insert [Speaker {num}]`}
                  >
                    [Speaker {num}]
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* AI Enhance options for Bark models (restored original layout) */}
          {(selectedVoiceData?.model?.toLowerCase().includes("bark") || selectedVoiceData?.features?.includes("tags")) && (
            <div className="mt-3 mb-2">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 cursor-pointer select-none mb-0">
                  <span className="text-sm font-medium">Enable AI Enhance</span>
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
              </div>

              {!smartEnhance && selectedVoiceData?.model?.toLowerCase().includes("bark") && (
                <div className="text-[11px] text-amber-600 bg-amber-50 rounded p-2 border border-amber-100 mt-2 leading-relaxed">
                  ⚠ Disabling AI Enhance for Bark models may result in more random/unstable output.
                </div>
              )}
            </div>
          )}

          {/* Show Enhance button and prompt only if enabled */}
          {smartEnhance && (
            <div className="mt-4">
              <div className="flex items-end gap-2">
                <input
                  id="ai-enhance-prompt"
                  type="text"
                  value={enhancePrompt}
                  onChange={(e) => setEnhancePrompt(e.target.value)}
                  placeholder="Enhancement prompt (e.g. Dramatic, add sighs, etc.)"
                  className="flex-1 p-2 border rounded focus:ring-2 focus:ring-blue-500 text-sm"
                />
                <button
                  type="button"
                  className="px-4 py-2 bg-blue-600 text-white text-sm rounded shadow hover:bg-blue-700 transition disabled:bg-gray-400"
                  disabled={isEnhancing || !text.trim()}
                  onClick={runEnhancement}
                  title="Apply AI enhancement to your text"
                >
                  {isEnhancing ? "Enhancing..." : "Enhance"}
                </button>
              </div>
              <div className="mt-2">
                <label
                  htmlFor="creativity"
                  className="block mb-1 text-sm font-medium text-gray-900"
                >
                  Creativity
                </label>
                <input
                  id="creativity"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={enhanceCreativity}
                  onChange={(e) =>
                    setEnhanceCreativity(parseFloat(e.target.value))
                  }
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Conservative</span>
                  <span>Creative</span>
                </div>
              </div>
            </div>
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
            playVoicePreview={playVoicePreview}
            playingPreview={playingPreview}
          />
        )}

        {/* --- Speaker ID selection (dynamic for any model with preset speakers) --- */}
        {selectedVoiceData?.supported_speakers?.length > 0 && !selectedVoiceData?.features?.includes("multi_speaker") && (
          <div className="mb-6">
            <label className="block text-sm font-semibold mb-2 flex items-center gap-1.5">
              <User className="h-4 w-4 text-indigo-400 shrink-0" />
              <span>Speaker ID</span>
            </label>
            <div className="flex items-center gap-2">
              <select
                className="flex-1 p-3 border rounded-xl focus:ring-2 focus:ring-blue-500 text-sm bg-white"
                value={speaker}
                onChange={(e) => setSpeaker(e.target.value)}
              >
                <option value="">-- Select speaker --</option>
                {selectedVoiceData.supported_speakers.map((spk) => (
                  <option key={spk} value={spk}>
                    {spk}
                  </option>
                ))}
              </select>
              {speaker && (
                <button
                  onClick={() => playVoicePreview(speaker, selectedVoice)}
                  className="p-3 border rounded-xl hover:bg-gray-100 flex items-center justify-center text-lg bg-white shrink-0"
                  title="Preview Voice"
                  type="button"
                >
                  {playingPreview === speaker ? "⏸️" : "▶️"}
                </button>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Select speaker profile or preset voice.
            </p>
          </div>
        )}

        {/* --- Multi-speaker Voice Assignment Panel (for VibeVoice and other multi-speaker models) --- */}
        {selectedVoiceData?.features?.includes("multi_speaker") && (
          <div className="mb-6 p-4 bg-indigo-50/20 border border-indigo-100/50 rounded-2xl">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-1.5 text-indigo-900 dark:text-indigo-400">
              <Users className="h-4 w-4 text-indigo-500 dark:text-indigo-400 shrink-0" />
              <span>Multi-Speaker Voice Assignment</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[1, 2, 3, 4].map((num) => {
                const rawSpk = speakerMapping[`speaker_${num}`] || `p${224 + num}`;
                const spkVal = rawSpk.includes(":") ? rawSpk.split(":").pop() : rawSpk;
                return (
                  <div key={num} className="flex flex-col">
                    <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Speaker {num}</label>
                    <div className="flex items-center gap-1.5">
                      <select
                        className="flex-1 p-2 border rounded-xl bg-white text-xs focus:ring-2 focus:ring-blue-500"
                        value={spkVal}
                        onChange={(e) => {
                          const val = e.target.value;
                          const combinedVal = val.startsWith("p") ? `tts_models/en/vctk/vits:${val}` : val;
                          handleTrackVoiceChange(`speaker_${num}`, combinedVal);
                        }}
                      >
                        {Array.from({ length: 56 }, (_, i) => `p${225 + i}`).map((spk) => (
                          <option key={spk} value={spk}>{spk}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => playVoicePreview(spkVal, "vits")}
                        className="p-2 border rounded-xl hover:bg-gray-100 flex items-center justify-center text-xs bg-white shrink-0"
                        title="Preview Voice"
                        type="button"
                      >
                        {playingPreview === spkVal ? "⏸️" : "▶️"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-[10px] text-gray-500 mt-2 leading-relaxed">
              Assign a VITS voice identity to each speaker. Speaker tags like <code>[Speaker 1]</code> will render with their assigned voice.
            </p>
          </div>
        )}

        {/* --- VITS-specific tuning sliders --- */}
        {selectedVoiceData?.model?.toLowerCase().includes("vits") && (
          <div className="mb-6 space-y-3 pt-4 border-t border-gray-100">
            <label className="block text-sm font-semibold">Noise Scale</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={vitsNoiseScale}
              onChange={(e) => setVitsNoiseScale(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm border rounded-xl"
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
              className="w-full px-3 py-2 text-sm border rounded-xl"
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


        {/* Helper warning if reference wav is required but not uploaded */}
        {selectedVoiceData?.requires_speaker_wav && !recordedBlob && (
          <div className="mb-6 p-4 bg-blue-50/10 dark:bg-blue-950/20 border border-blue-200/50 dark:border-blue-900/30 rounded-xl flex flex-col items-center text-center">
            <p className="text-xs text-blue-800 dark:text-blue-200 mb-3 font-semibold">
              This model requires a speaker reference WAV file to clone a voice.
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setShowVoiceCreatorModal(true)}
                className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] text-white text-xs font-bold rounded-lg transition"
              >
                Upload/Record Voice Here
              </button>
              <button
                type="button"
                onClick={() => setActiveMainTab("creator")}
                className="px-3 py-1.5 border border-gray-300 dark:border-slate-800 hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-700 dark:text-slate-350 text-xs font-bold rounded-lg transition"
              >
                Go to Voice Creator Tab
              </button>
            </div>
          </div>
        )}

        {(selectedVoiceData?.requires_speaker_wav || selectedVoiceData?.features?.includes("cloning")) && (recordedBlob || (activeCloneProfile && (activeCloneProfile.type === "library" || activeCloneProfile.type === "reference"))) && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl flex items-center gap-3">
            <Fingerprint className="h-5 w-5 text-green-600 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-green-800">
                {(activeCloneProfile?.type === "library" || activeCloneProfile?.type === "reference") ? "Library reference voice active" : "Custom voice clone loaded & active"}
              </p>
              <p className="text-[10px] text-green-600 truncate mt-0.5">
                {(activeCloneProfile?.type === "library" || activeCloneProfile?.type === "reference") ? `Voice: ${activeCloneProfile.name}` : `Reference Audio: ${recordedBlob?.name || "speaker_reference.wav"}`}
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setRecordedBlob(null);
                setActiveCloneProfile(null);
                formDataRef.current.delete("speaker_wav");
              }}
              className="px-2 py-1 text-[10px] font-bold text-red-600 hover:text-red-800 border border-red-200 hover:border-red-300 rounded bg-white transition"
            >
              Clear Reference
            </button>
          </div>
        )}

        {/* Style Instructions (for models supporting instructions, like Qwen3-TTS) */}
        {selectedVoiceData?.features?.includes("instructions") && (
          <div className="mb-6">
            <label className="block text-sm font-semibold mb-2">
              Voice Style / Prompt Instructions:
            </label>
            <input
              type="text"
              value={voiceDirection}
              onChange={(e) => setVoiceDirection(e.target.value)}
              placeholder="e.g. Speak excitedly in a whispers tone, or Neutral narrator style..."
              className="w-full p-3 border rounded-xl focus:ring-2 focus:ring-blue-500"
            />
            
            {/* Style Suggestion Chips */}
            <div className="mt-2 flex flex-wrap gap-1.5">
              {[
                "Excited & High Pitch",
                "Soft Whispers",
                "Slow Dramatic Narrator",
                "Deep & authoritative",
                "Fast-paced news anchor"
              ].map((styleOption) => (
                <button
                  key={styleOption}
                  type="button"
                  onClick={() => setVoiceDirection(styleOption)}
                  className={`px-2.5 py-1 text-xs rounded-full border transition active:scale-95 ${
                    voiceDirection === styleOption
                      ? "bg-blue-50 border-blue-500 text-blue-600 font-medium"
                      : "bg-white border-gray-200 text-gray-600 hover:border-gray-300"
                  }`}
                >
                  {styleOption}
                </button>
              ))}
            </div>

            <p className="text-xs text-gray-500 mt-2">
              Provide natural language guidance to style the synthesized voice.
            </p>
          </div>
        )}

        {/* Real-time latency controls (for models supporting streaming, like CosyVoice 2) */}
        {selectedVoiceData?.features?.includes("streaming") && (
          <div className="mb-6">
            <h3 className="text-sm font-semibold mb-2">
              Latency Options [Experimental]
            </h3>
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={streamingLatency}
                onChange={(e) => setStreamingLatency(e.target.checked)}
                className="rounded text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm font-medium">Enable sub-200ms streaming (low-latency)</span>
            </label>
            <p className="text-xs text-gray-500 mt-1">
              Reduces latency for real-time applications by processing in smaller streams.
            </p>
          </div>
        )}

        {/* ChatTTS Settings Card */}
        {selectedVoice === "chattts" && (
          <div className="mb-6 p-4 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-slate-900/40 dark:to-indigo-950/20 rounded-xl border border-indigo-100/50 dark:border-slate-800 space-y-4 text-gray-900 dark:text-slate-100">
            <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-400">
              💬 ChatTTS Tuning Parameters
            </h3>

            {/* 1. Text Processing Gatekeeper (State Dependency Control) */}
            <div className="space-y-3 pb-3 border-b border-indigo-100/50 dark:border-slate-800/60">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <label className="text-xs font-bold text-gray-700 dark:text-slate-300">Auto-Refine Phrasing</label>
                  <div className="group relative inline-block">
                    <HelpCircle className="h-3.5 w-3.5 text-gray-400 cursor-pointer hover:text-indigo-600 transition" />
                    <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2 rounded-lg opacity-0 group-hover:opacity-100 transition duration-200 pointer-events-none z-50 shadow-lg text-center leading-normal">
                      Automatically refines dialogue pacing by inserting natural oral markers, sighs, and breaths. If active, manual tag buttons are locked as the refiner overrides them.
                    </span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={chatttsRefineText}
                  onChange={(e) => setChatttsRefineText(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500"
                />
              </div>

              {/* Text Temp (only exposed if Auto-Refine Phrasing is True) */}
              {chatttsRefineText && (
                <div className="animate-fadeIn">
                  <div className="flex justify-between items-center mb-1">
                    <div className="flex items-center gap-1">
                      <label className="text-[11px] font-semibold text-gray-500 dark:text-slate-400">Text Temp</label>
                      <div className="group relative inline-block">
                        <HelpCircle className="h-3 w-3 text-gray-400 cursor-pointer hover:text-indigo-600 transition" />
                        <span className="absolute bottom-full left-0 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2 rounded-lg opacity-0 group-hover:opacity-100 transition duration-200 pointer-events-none z-50 shadow-lg text-center leading-normal font-semibold">
                          Controls the pacing randomness and speed. Higher values introduce more stutters, pauses, and speech variation.
                        </span>
                      </div>
                    </div>
                    <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400">{chatttsTextTemp}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={chatttsTextTemp}
                    onChange={(e) => setChatttsTextTemp(parseFloat(e.target.value))}
                    className="w-full h-1 bg-indigo-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600"
                  />
                </div>
              )}
            </div>

            {/* 2. Vocal Core Identity */}
            <div className="space-y-2 pb-3 border-b border-indigo-100/50 dark:border-slate-800/60">
              <div className="flex items-center gap-1.5">
                <label className="text-xs font-bold text-gray-700 dark:text-slate-300">Speaker Seed / Pitch ID</label>
                <div className="group relative inline-block">
                  <HelpCircle className="h-3.5 w-3.5 text-gray-400 cursor-pointer hover:text-indigo-600 transition" />
                  <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2 rounded-lg opacity-0 group-hover:opacity-100 transition duration-200 pointer-events-none z-50 shadow-lg text-center leading-normal">
                    Enter an integer (e.g. 42, 108). The same seed will generate the exact same vocal timbre/pitch profile. Leave empty to use default.
                  </span>
                </div>
              </div>
              <div className="flex gap-2 items-center">
                <input
                  type="number"
                  placeholder="Random seed (e.g. 42)"
                  value={chatttsSpkSeed}
                  onChange={(e) => setChatttsSpkSeed(e.target.value)}
                  className="flex-1 p-2 border border-indigo-200 dark:border-slate-800 rounded-lg text-xs focus:ring-2 focus:ring-indigo-400 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                />
                <button
                  type="button"
                  onClick={() => {
                    const randSeed = Math.floor(Math.random() * 999999) + 1;
                    setChatttsSpkSeed(randSeed.toString());
                  }}
                  className="py-2 px-3 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition shrink-0 flex items-center gap-1"
                  title="Generate Random Speaker Seed"
                >
                  🎲 Roll
                </button>
              </div>
            </div>

            {/* 3. Advanced Sampling Controls (Collapsible Accordion) */}
            <div className="border-b border-indigo-100/50 dark:border-slate-800/60 pb-3">
              <button
                type="button"
                onClick={() => setIsAdvancedAccordionOpen(prev => !prev)}
                className="w-full flex items-center justify-between py-1.5 text-xs font-bold text-gray-700 dark:text-slate-350 hover:text-indigo-600 dark:hover:text-indigo-400 transition"
              >
                <span>⚙️ Advanced Sampling Controls</span>
                <span className="text-[10px]">{isAdvancedAccordionOpen ? "▲ Hide" : "▼ Show"}</span>
              </button>

              {isAdvancedAccordionOpen && (
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-indigo-50/50 dark:border-slate-800/40 animate-fadeIn">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-[10px] font-semibold text-gray-500 dark:text-slate-400">Speaker Temp</label>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={chatttsSpkTemp}
                      onChange={(e) => setChatttsSpkTemp(parseFloat(e.target.value))}
                      className="w-full h-1 bg-indigo-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600 dark:accent-indigo-400"
                    />
                    <div className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 text-right mt-1">{chatttsSpkTemp}</div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-[10px] font-semibold text-gray-500 dark:text-slate-400">Top_P</label>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={chatttsTopP}
                      onChange={(e) => setChatttsTopP(parseFloat(e.target.value))}
                      className="w-full h-1 bg-indigo-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600 dark:accent-indigo-400"
                    />
                    <div className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 text-right mt-1">{chatttsTopP}</div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-[10px] font-semibold text-gray-500 dark:text-slate-400">Top_K</label>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={chatttsTopK}
                      onChange={(e) => setChatttsTopK(parseInt(e.target.value))}
                      className="w-full h-1 bg-indigo-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600 dark:accent-indigo-400"
                    />
                    <div className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 text-right mt-1">{chatttsTopK}</div>
                  </div>
                </div>
              )}
            </div>

            {/* 4. Performance & Latency */}
            <div className="pt-1">
              <label className="inline-flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={streamingLatency}
                  onChange={(e) => setStreamingLatency(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-xs font-bold text-gray-700 dark:text-slate-300">Enable sub-200ms streaming (low-latency)</span>
              </label>
              <p className="text-[10px] text-gray-500 dark:text-slate-400 mt-0.5 leading-normal">
                Reduces latency for real-time applications by processing in smaller streams.
              </p>
            </div>
          </div>
        )}

        {/* Fish Audio Settings Card */}
        {selectedVoice === "fish-audio" && (
          <div className="mb-6 p-4 bg-gradient-to-r from-teal-50/50 to-emerald-50/50 rounded-xl border border-teal-100/50 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-teal-700">
              🐟 Fish Audio Pro Parameters
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Engine Version</label>
                <select
                  value={fishEngine}
                  onChange={(e) => setFishEngine(e.target.value)}
                  className="w-full p-2 border border-teal-200 rounded-lg text-xs focus:ring-2 focus:ring-teal-400 bg-white"
                >
                  <option value="s2">S2 Model (Fast)</option>
                  <option value="s2_1_pro">S2.1 Pro Engine (High Quality)</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <div className="flex items-center gap-1">
                    <label className="text-xs font-medium text-gray-600">Similarity Weight</label>
                    <div className="group relative inline-block">
                      <HelpCircle className="h-3 w-3 text-gray-400 cursor-pointer hover:text-teal-600 transition" />
                      <span className="absolute bottom-full right-0 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2 rounded-lg opacity-0 group-hover:opacity-100 transition duration-200 pointer-events-none z-50 shadow-lg text-center leading-normal">
                        Determines how strongly the speech synthesis output matches the source speaker clone sample rather than default voice parameters.
                      </span>
                    </div>
                  </div>
                  <span className="text-[10px] font-bold text-teal-600">{Math.round(fishSimilarityWeight * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1.0"
                  step="0.05"
                  value={fishSimilarityWeight}
                  onChange={(e) => setFishSimilarityWeight(parseFloat(e.target.value))}
                  className="w-full h-1 bg-teal-200 rounded appearance-none cursor-pointer accent-teal-600"
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-semibold text-gray-700">Text Normalization</label>
                <div className="group relative inline-block">
                  <HelpCircle className="h-3.5 w-3.5 text-gray-400 cursor-pointer hover:text-teal-600 transition" />
                  <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2 rounded-lg opacity-0 group-hover:opacity-100 transition duration-200 pointer-events-none z-50 shadow-lg text-center leading-normal">
                    Converts numbers, dates, and abbreviations to their spoken word forms before synthesis.
                  </span>
                </div>
              </div>
              <input
                type="checkbox"
                checked={fishNormalize}
                onChange={(e) => setFishNormalize(e.target.checked)}
                className="rounded text-teal-600 focus:ring-teal-500"
              />
            </div>

            {selectedVoiceData?.features?.includes("cloning") && (
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Reference Audio Transcript (Prompt)</label>
                <textarea
                  rows={2}
                  placeholder="Optional prompt transcript matching the speaker reference WAV..."
                  value={fishPromptText}
                  onChange={(e) => setFishPromptText(e.target.value)}
                  className="w-full p-2 border border-teal-200 rounded-lg text-xs focus:ring-2 focus:ring-teal-400 bg-white"
                />
                <p className="text-[9px] text-gray-400">
                  Providing the text spoken in your clone reference audio significantly improves speech intelligibility.
                </p>
              </div>
            )}
          </div>
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



        {/* Execution Actions pinned at the bottom */}
        <div className="mt-8 pt-6 border-t border-gray-100 dark:border-slate-800/80 space-y-3">
          {(() => {
            const isBark = selectedVoiceData?.model?.toLowerCase() === "bark";
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
                className={`w-full py-3 px-6 rounded-xl text-white font-bold text-sm transition-all duration-200 shadow-md flex items-center justify-center gap-2 ${
                  !canGenerate
                    ? "bg-gray-400 dark:bg-slate-700 cursor-not-allowed opacity-50 text-gray-200"
                    : "bg-blue-600 hover:bg-blue-700 hover:shadow-lg active:scale-[0.99]"
                }`}
              >
                🚀 Add to Queue
              </button>
            );
          })()}
          
          {loadedCuratedVoiceId ? (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={updateCuratedVoice}
                className="flex-1 py-2 px-3 bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs rounded-xl transition duration-200 flex items-center justify-center gap-1 shadow-sm active:scale-[0.99]"
              >
                <Save className="h-3.5 w-3.5" />
                <span>Save</span>
              </button>
              <button
                type="button"
                onClick={saveAsCuratedVoice}
                className="flex-1 py-2 px-3 bg-indigo-50 dark:bg-slate-800 hover:bg-indigo-100 dark:hover:bg-slate-700 text-indigo-700 dark:text-indigo-300 font-bold text-xs rounded-xl border border-indigo-200 dark:border-slate-700 transition duration-200 flex items-center justify-center gap-1 active:scale-[0.99]"
              >
                <Copy className="h-3.5 w-3.5" />
                <span>Save As</span>
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={saveAsCuratedVoice}
              className="w-full py-2.5 px-4 bg-indigo-50 dark:bg-slate-800 hover:bg-indigo-100 dark:hover:bg-slate-700 text-indigo-700 dark:text-indigo-300 font-bold text-xs rounded-xl border border-indigo-200 dark:border-slate-700 transition-all duration-200 flex items-center justify-center gap-1.5 active:scale-[0.99]"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Save as Curated Voice</span>
            </button>
          )}
        </div>
      </div>

      {/* Right Column: Curated & Cloned Library (4 cols) */}
      <div className="lg:col-span-4 p-6 bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 shadow-xl rounded-2xl flex flex-col max-h-[850px] text-gray-900 dark:text-slate-100 gap-6">
          
          {/* Section 1: Saved Presets */}
          <details open className="group flex flex-col flex-1 min-h-0">
            <summary className="text-sm font-semibold text-gray-800 dark:text-slate-100 border-b border-gray-100 dark:border-slate-800 pb-2 mb-3 flex items-center justify-between cursor-pointer list-none select-none">
              <div className="flex items-center gap-1.5">
                <Library className="h-4 w-4 text-indigo-500 shrink-0" />
                <span>Saved Presets ({curatedVoices.length})</span>
              </div>
              <ChevronDown className="h-4 w-4 text-gray-400 group-open:rotate-180 transition-transform shrink-0" />
            </summary>
            <div className="space-y-3 overflow-y-auto flex-1 max-h-[250px] pr-1 mt-1">
              {curatedVoices.length === 0 ? (
                <div className="text-center text-xs text-gray-400 py-6 italic bg-gray-50 dark:bg-slate-800/40 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl">
                  No curated voices saved yet. Configure a voice on the left and click "Save as Curated Voice".
                </div>
              ) : (
                curatedVoices.map((v) => {
                  const isActive = v.id === loadedCuratedVoiceId;
                  return (
                    <div key={v.id} className={`p-3 border rounded-xl flex items-center justify-between transition duration-200 text-gray-900 dark:text-slate-100 font-sans ${
                      isActive 
                        ? "bg-indigo-50/40 dark:bg-indigo-950/20 border-indigo-500 dark:border-indigo-400 shadow-sm" 
                        : "bg-gray-50 dark:bg-slate-950 border-gray-200 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-indigo-500/50"
                    }`}>
                      <div>
                        <div className="flex items-center gap-1.5">
                          <div className="font-bold text-xs text-indigo-900 dark:text-indigo-400">{v.name}</div>
                          {isActive && (
                            <span className="bg-indigo-100 dark:bg-indigo-950/40 text-indigo-800 dark:text-indigo-300 text-[8px] font-extrabold px-1.5 py-0.5 rounded-full uppercase tracking-wider">
                              Active
                            </span>
                          )}
                        </div>
                        <div className="text-[9px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mt-0.5">
                          {v.model.replace("tts_models/en/vctk/", "").replace("tts_models/multilingual/multi-dataset/", "")} ({v.voice || "default"})
                        </div>
                      </div>
                      <div className="flex gap-1.5">
                        <button
                          type="button"
                          onClick={() => loadCuratedVoice(v)}
                          className={`px-2 py-1 border rounded-lg text-[10px] font-bold shadow-sm transition ${
                            isActive
                              ? "bg-indigo-600 hover:bg-indigo-700 text-white border-transparent"
                              : "bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700 hover:border-indigo-400 dark:hover:border-indigo-500 hover:text-indigo-600 dark:hover:text-indigo-300 text-gray-950 dark:text-slate-200"
                          }`}
                        >
                          {isActive ? "Loaded" : "Load"}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            const newName = prompt("Rename voice to:", v.name);
                            if (newName && newName.trim()) {
                              setCuratedVoices(prev => prev.map(item => item.id === v.id ? { ...item, name: newName.trim() } : item));
                            }
                          }}
                          className="px-2 py-1 bg-amber-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-amber-400 hover:text-amber-600 dark:hover:text-amber-300 rounded-lg text-[10px] font-bold shadow-sm transition text-gray-950 dark:text-slate-200"
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (window.confirm(`Delete curated voice "${v.name}"?`)) {
                              setCuratedVoices(prev => prev.filter(item => item.id !== v.id));
                            }
                          }}
                          className="px-2 py-1 bg-red-50 dark:bg-red-955/20 text-red-655 hover:bg-red-100 hover:text-red-750 dark:hover:bg-red-900/40 border border-red-100 dark:border-red-900/50 rounded-lg text-[10px] font-bold transition flex items-center justify-center"
                          title="Delete Voice"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </details>

          {/* Section 2: Voice Clones Library */}
          <div className="flex flex-col flex-1 min-h-0 border-t border-gray-100 dark:border-slate-800 pt-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-slate-800 pb-2 mb-3">
              <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-100 flex items-center gap-1.5">
                <Fingerprint className="h-4 w-4 text-green-550 shrink-0" />
                <span>Voice Clones Library</span>
              </h3>
              <button
                type="button"
                onClick={() => setShowVoiceCreatorModal(true)}
                className="px-2 py-1 bg-indigo-50 dark:bg-slate-800 text-indigo-700 dark:text-indigo-350 hover:bg-indigo-100 hover:text-indigo-800 rounded-lg text-[10px] font-extrabold transition flex items-center gap-1"
              >
                <Plus className="h-3 w-3" />
                Create Clone
              </button>
            </div>
            
            <div className="space-y-4 overflow-y-auto flex-1 pr-1">
              {/* My Voice Clones Sub-section */}
              <div className="space-y-2">
                <div className="text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider px-1">
                  My Voice Clones
                </div>
                {clonedProfiles.filter(p => p.type === "clone").length === 0 ? (
                  <div className="text-center text-[10px] text-gray-400 py-4 italic bg-gray-50/50 dark:bg-slate-900/20 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl">
                    No custom voice clones yet. Record or upload one above!
                  </div>
                ) : (
                  clonedProfiles.map((p, idx) => {
                    if (p.type !== "clone") return null;
                    const isActive = activeCloneProfile && activeCloneProfile.name === p.name;
                    const isEditing = editingProfileIdx === idx;
                    return (
                      <div key={idx} className={`p-2.5 rounded-xl border flex flex-col transition duration-200 text-gray-900 dark:text-slate-100 font-sans ${
                        isActive 
                          ? "border-green-500 bg-green-50/10 dark:bg-green-950/10" 
                          : "bg-gray-50 dark:bg-slate-950 border-gray-200 dark:border-slate-800 hover:border-gray-300"
                      }`}>
                        <div className="flex w-full items-center justify-between">
                          <div>
                            <div className="font-bold text-xs flex items-center gap-1">
                              <Mic className="h-3.5 w-3.5 text-green-400 shrink-0" />
                              <span className={isActive ? "text-green-700 dark:text-green-400" : ""}>{p.name}</span>
                            </div>
                            <div className="text-[9px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mt-0.5">
                              Custom Clone
                            </div>
                          </div>
                          <div className="flex gap-1.5 items-center">
                            <button
                              type="button"
                              onClick={() => {
                                setActiveCloneProfile(p);
                                setRecordedBlob(p.file || null);
                                const supportsCloning = selectedVoiceData?.features?.includes("cloning") || selectedVoiceData?.requires_speaker_wav;
                                if (!supportsCloning) {
                                  const qwen = voices.find(v => v.name === "qwen3-tts");
                                  if (qwen && isModelEnabled("qwen3-tts")) {
                                    setSelectedVoice("qwen3-tts");
                                  } else {
                                    const xtts = voices.find(v => v.name === "tts_models/multilingual/multi-dataset/xtts_v2");
                                    if (xtts && isModelEnabled(xtts.name)) {
                                      setSelectedVoice(xtts.name);
                                    }
                                  }
                                }
                              }}
                              className={`px-2 py-1 rounded-lg text-[10px] font-bold shadow-sm transition border ${
                                isActive 
                                  ? "bg-green-600 text-white border-green-606 hover:bg-green-700" 
                                  : "bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700 hover:border-indigo-400 dark:hover:border-indigo-500 text-gray-950 dark:text-slate-200"
                              }`}
                            >
                              {isActive ? "Active" : "Use"}
                            </button>
                            
                            <button
                              type="button"
                              onClick={() => {
                                if (isEditing) {
                                  setEditingProfileIdx(null);
                                } else {
                                  setEditingProfileIdx(idx);
                                  setEditingProfileName(p.name);
                                  setEditingProfileTranscript(p.transcript || "");
                                }
                              }}
                              className="px-2 py-1 bg-amber-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-amber-400 hover:text-amber-600 dark:hover:text-amber-300 rounded-lg text-[10px] font-bold shadow-sm transition text-gray-950 dark:text-slate-200"
                              title="Edit profile & transcript"
                            >
                              <Edit3 className="h-3 w-3" />
                            </button>

                            <button
                              type="button"
                              onClick={() => {
                                if (window.confirm(`Delete custom voice clone "${p.name}"?`)) {
                                  setClonedProfiles(prev => prev.filter((_, i) => i !== idx));
                                  if (isActive) {
                                    setActiveCloneProfile(null);
                                    setRecordedBlob(null);
                                  }
                                }
                              }}
                              className="px-2 py-1 bg-red-50 dark:bg-red-955/20 text-red-600 hover:bg-red-100 hover:text-red-700 dark:hover:bg-red-900/40 border border-red-100 dark:border-red-900/50 rounded-lg text-[10px] font-bold transition flex items-center justify-center"
                              title="Delete custom voice clone"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>

                        {isEditing && (
                          <div className="space-y-2 mt-2 pt-2 border-t border-gray-200 dark:border-slate-800 w-full text-xs">
                            <div className="space-y-1">
                              <label className="block text-[10px] font-bold text-gray-500 uppercase">Voice Name</label>
                              <input 
                                type="text" 
                                value={editingProfileName} 
                                onChange={(e) => setEditingProfileName(e.target.value)}
                                className="w-full p-1.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400"
                              />
                            </div>
                            <div className="space-y-1">
                              <label className="block text-[10px] font-bold text-gray-500 uppercase">Reference Transcript (Highly Recommended for Accents)</label>
                              <textarea 
                                value={editingProfileTranscript} 
                                onChange={(e) => setEditingProfileTranscript(e.target.value)}
                                className="w-full p-1.5 border border-gray-200 dark:border-slate-805 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 min-h-[3.5rem] resize-none focus:ring-1 focus:ring-indigo-400"
                                placeholder="Type the spoken reference words to enable accent alignment..."
                              />
                            </div>
                            <div className="flex gap-1.5 justify-end">
                              <button 
                                type="button" 
                                onClick={() => setEditingProfileIdx(null)}
                                className="px-2 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-slate-855 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg text-[10px] font-bold transition"
                              >
                                Cancel
                              </button>
                              <button 
                                type="button" 
                                onClick={() => {
                                  setClonedProfiles(prev => prev.map((item, i) => i === idx ? { ...item, name: editingProfileName.trim(), transcript: editingProfileTranscript.trim() } : item));
                                  setEditingProfileIdx(null);
                                }}
                                className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-[10px] font-bold transition"
                              >
                                Save
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>

              {/* Reference Voices Sub-section */}
              <div className="space-y-2 pt-3 border-t border-gray-100 dark:border-slate-800">
                <div className="text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider px-1 flex items-center justify-between">
                  <span>Reference Voices</span>
                  <span className="text-[9px] lowercase font-normal text-slate-400 dark:text-slate-500">Preset Dialects</span>
                </div>
                {clonedProfiles.map((p, idx) => {
                  if (p.type !== "reference" && p.type !== "library") return null;
                  const isActive = activeCloneProfile && activeCloneProfile.name === p.name;
                  const isEditing = editingProfileIdx === idx;
                  return (
                    <div key={idx} className={`p-2.5 rounded-xl border flex flex-col transition duration-200 text-gray-900 dark:text-slate-100 font-sans ${
                      isActive 
                        ? "border-green-500 bg-green-50/10 dark:bg-green-950/10" 
                        : "bg-gray-50 dark:bg-slate-950 border-gray-200 dark:border-slate-800 hover:border-gray-300"
                    }`}>
                      <div className="flex w-full items-center justify-between">
                        <div>
                          <div className="font-bold text-xs flex items-center gap-1">
                            <BookOpen className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                            <span className={isActive ? "text-green-700 dark:text-green-400" : ""}>{p.name}</span>
                          </div>
                          <div className="text-[9px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mt-0.5">
                            Reference Voice
                          </div>
                        </div>
                        <div className="flex gap-1.5 items-center">
                          <button
                            type="button"
                            onClick={() => {
                              setActiveCloneProfile(p);
                              setRecordedBlob(null);
                              const supportsCloning = selectedVoiceData?.features?.includes("cloning") || selectedVoiceData?.requires_speaker_wav;
                              if (!supportsCloning) {
                                const qwen = voices.find(v => v.name === "qwen3-tts");
                                if (qwen && isModelEnabled("qwen3-tts")) {
                                  setSelectedVoice("qwen3-tts");
                                } else {
                                  const xtts = voices.find(v => v.name === "tts_models/multilingual/multi-dataset/xtts_v2");
                                  if (xtts && isModelEnabled(xtts.name)) {
                                    setSelectedVoice(xtts.name);
                                  }
                                }
                              }
                            }}
                            className={`px-2 py-1 rounded-lg text-[10px] font-bold shadow-sm transition border ${
                              isActive 
                                ? "bg-green-600 text-white border-green-606 hover:bg-green-700" 
                                : "bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700 hover:border-indigo-400 dark:hover:border-indigo-500 text-gray-950 dark:text-slate-200"
                            }`}
                          >
                            {isActive ? "Active" : "Use"}
                          </button>
                          
                          <button
                            type="button"
                            onClick={() => {
                              if (isEditing) {
                                setEditingProfileIdx(null);
                              } else {
                                setEditingProfileIdx(idx);
                                setEditingProfileName(p.name);
                                setEditingProfileTranscript(p.transcript || "");
                              }
                            }}
                            className="px-2 py-1 bg-amber-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-amber-400 hover:text-amber-600 dark:hover:text-amber-300 rounded-lg text-[10px] font-bold shadow-sm transition text-gray-950 dark:text-slate-200"
                            title="Edit profile & transcript"
                          >
                            <Edit3 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>

                      {isEditing && (
                        <div className="space-y-2 mt-2 pt-2 border-t border-gray-200 dark:border-slate-800 w-full text-xs">
                          <div className="space-y-1">
                            <label className="block text-[10px] font-bold text-gray-500 uppercase">Voice Name</label>
                            <input 
                              type="text" 
                              value={editingProfileName} 
                              onChange={(e) => setEditingProfileName(e.target.value)}
                              className="w-full p-1.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400"
                            />
                          </div>
                          <div className="space-y-1">
                            <label className="block text-[10px] font-bold text-gray-500 uppercase">Reference Transcript (Highly Recommended for Accents)</label>
                            <textarea 
                              value={editingProfileTranscript} 
                              onChange={(e) => setEditingProfileTranscript(e.target.value)}
                              className="w-full p-1.5 border border-gray-200 dark:border-slate-805 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 min-h-[3.5rem] resize-none focus:ring-1 focus:ring-indigo-400"
                              placeholder="Type the spoken reference words to enable accent alignment..."
                            />
                          </div>
                          <div className="flex gap-1.5 justify-end">
                            <button 
                              type="button" 
                              onClick={() => setEditingProfileIdx(null)}
                              className="px-2 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg text-[10px] font-bold transition"
                            >
                              Cancel
                            </button>
                            <button 
                              type="button" 
                              onClick={() => {
                                setClonedProfiles(prev => prev.map((item, i) => i === idx ? { ...item, name: editingProfileName.trim(), transcript: editingProfileTranscript.trim() } : item));
                                setEditingProfileIdx(null);
                              }}
                              className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-[10px] font-bold transition"
                            >
                              Save
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 text-gray-900 dark:text-slate-100">
              {/* Left Column: MusicGen and Upload/Freesound */}
              <div className="lg:col-span-6 space-y-8 bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 shadow-xl rounded-2xl p-6">
                <div>
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-amber-500" />
                    AI Music Generator
                  </h2>
                  <p className="text-xs text-gray-500 mt-1 leading-normal">
                    Generate unique ambient music or cinematic backing tracks using Meta s MusicGen model.
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                      Music Gen Prompt
                    </label>
                    <textarea
                      rows={3}
                      value={musicPrompt}
                      onChange={(e) => setMusicPrompt(e.target.value)}
                      placeholder="e.g. lo-fi hip hop beat with calm acoustic guitar, soft drums, relaxed vibe"
                      className="w-full p-3 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-800 text-sm rounded-xl focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                      <span>Duration (seconds)</span>
                      <span className="text-blue-500 font-semibold">{musicDuration}s</span>
                    </div>
                    <input
                      type="range"
                      min={5}
                      max={30}
                      step={1}
                      value={musicDuration}
                      onChange={(e) => setMusicDuration(parseInt(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                  </div>

                  <button
                    onClick={() => {
                      if (!musicPrompt.trim()) return;
                      setIsGeneratingMusic(true);
                      fetch("http://localhost:5000/api/sound-library/generate-music", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ prompt: musicPrompt, duration: musicDuration })
                      })
                      .then(r => r.json())
                      .then(res => {
                        setIsGeneratingMusic(false);
                        if (res.success) {
                          setMusicPrompt("");
                          fetchSoundAssets();
                        } else {
                          alert(res.error || "Failed to generate music");
                        }
                      })
                      .catch(err => {
                        setIsGeneratingMusic(false);
                        console.error(err);
                      });
                    }}
                    disabled={isGeneratingMusic || !musicPrompt.trim()}
                    className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-350 disabled:dark:bg-slate-800 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition"
                  >
                    {isGeneratingMusic ? (
                      <>
                        <span className="animate-spin h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full" />
                        Generating Music Track...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-3.5 w-3.5" />
                        Generate AI backing track
                      </>
                    )}
                  </button>
                </div>

                <div className="border-t border-gray-100 dark:border-slate-800 pt-6 space-y-6">
                  <div>
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                      <Search className="h-5 w-5 text-blue-500" />
                      Search Public Sounds (Freesound)
                    </h2>
                    <p className="text-xs text-gray-500 mt-1 leading-normal">
                      Search public library sounds. To authenticate, provide a Freesound API Token in settings.
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={freesoundQuery}
                      onChange={(e) => setFreesoundQuery(e.target.value)}
                      placeholder="e.g. dog barking, wind chimes, typing..."
                      className="flex-1 p-2.5 border border-gray-250 dark:border-slate-800 bg-white dark:bg-slate-800 text-xs rounded-xl focus:ring-2 focus:ring-blue-500"
                    />
                    <select
                      value={freesoundType}
                      onChange={(e) => setFreesoundType(e.target.value)}
                      className="p-2.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-800 text-xs rounded-xl focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="sfx">SFX</option>
                      <option value="music">Music</option>
                    </select>
                    <button
                      onClick={() => {
                        if (!freesoundQuery.trim()) return;
                        setIsSearchingFreesound(true);
                        fetch("http://localhost:5000/api/sound-library/search", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ query: freesoundQuery, type: freesoundType, token: freesoundToken })
                        })
                        .then(r => r.json())
                        .then(res => {
                          setIsSearchingFreesound(false);
                          if (res.results) {
                            setFreesoundResults(res.results);
                          } else if (res.error) {
                            alert(res.error);
                          }
                        })
                        .catch(err => {
                          setIsSearchingFreesound(false);
                          console.error(err);
                        });
                      }}
                      className="px-4 bg-gray-900 hover:bg-gray-800 text-white rounded-xl text-xs font-bold transition"
                    >
                      {isSearchingFreesound ? "Searching..." : "Search"}
                    </button>
                  </div>

                  {freesoundResults.length > 0 && (
                    <div className="space-y-2 max-h-[220px] overflow-y-auto border border-gray-150 dark:border-slate-800 p-3 rounded-xl bg-gray-50/50 dark:bg-slate-950/20">
                      {freesoundResults.map((item) => (
                        <div key={item.id} className="flex justify-between items-center text-xs p-2 hover:bg-gray-100/50 dark:hover:bg-slate-800 rounded-lg">
                          <div className="flex-1 truncate pr-4">
                            <span className="font-semibold text-gray-800 dark:text-slate-200 block truncate">{item.name}</span>
                            <span className="text-[10px] text-gray-500">{item.duration.toFixed(1)}s</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => {
                                if (playingSoundUrl === item.preview_url) {
                                  setPlayingSoundUrl(null);
                                } else {
                                  setPlayingSoundUrl(item.preview_url);
                                }
                              }}
                              className="p-1 text-gray-500 hover:text-blue-500"
                            >
                              {playingSoundUrl === item.preview_url ? (
                                <Square className="h-3.5 w-3.5 fill-current" />
                              ) : (
                                <Play className="h-3.5 w-3.5 fill-current" />
                              )}
                            </button>
                            <button
                              onClick={() => {
                                fetch("http://localhost:5000/api/sound-library/download-freesound", {
                                  method: "POST",
                                  headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify({
                                    preview_url: item.preview_url,
                                    name: item.name,
                                    type: freesoundType
                                  })
                                })
                                .then(r => r.json())
                                .then(res => {
                                  if (res.success) {
                                    fetchSoundAssets();
                                    alert("Sound imported successfully!");
                                  } else {
                                    alert(res.error || "Failed to import sound");
                                  }
                                });
                              }}
                              className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-[10px] font-bold"
                            >
                              Import
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="border-t border-gray-100 dark:border-slate-800 pt-6 space-y-4">
                  <div>
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                      <Upload className="h-5 w-5 text-indigo-500" />
                      Upload Sound Asset
                    </h2>
                    <p className="text-xs text-gray-500 mt-1 leading-normal">
                      Upload local MP3 or WAV sounds to your playground library.
                    </p>
                  </div>

                  <div className="flex flex-col gap-3">
                    <div className="flex gap-4">
                      <label className="flex-1 border-2 border-dashed border-gray-200 dark:border-slate-800 hover:border-blue-500 hover:bg-blue-50/5 p-4 rounded-xl cursor-pointer text-center flex flex-col items-center justify-center gap-1 transition">
                        <Upload className="h-6 w-6 text-gray-400" />
                        <span className="text-xs font-semibold text-gray-600 dark:text-slate-300">Choose custom audio file</span>
                        <span className="text-[10px] text-gray-400">WAV or MP3</span>
                        <input
                          type="file"
                          accept="audio/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files[0];
                            if (!file) return;
                            setIsUploadingSound(true);
                            const formData = new FormData();
                            formData.append("file", file);
                            formData.append("type", freesoundType);
                            fetch("http://localhost:5000/api/sound-library/upload", {
                              method: "POST",
                              body: formData
                            })
                            .then(r => r.json())
                            .then(res => {
                              setIsUploadingSound(false);
                              if (res.success) {
                                fetchSoundAssets();
                                alert("Sound uploaded successfully!");
                              } else {
                                alert(res.error || "Failed to upload file");
                              }
                            })
                            .catch(err => {
                              setIsUploadingSound(false);
                              console.error(err);
                            });
                          }}
                        />
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Library List & Curation */}
              <div className="lg:col-span-6 space-y-8 bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 shadow-xl rounded-2xl p-6">
                <div>
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <ListMusic className="h-5 w-5 text-indigo-500" />
                    Library Curation
                  </h2>
                  <p className="text-xs text-gray-500 mt-1 leading-normal">
                    Manage and preview your sound effects (SFX) and music backing tracks. Use bracket tags like <code>[music: track_name]</code> or <code>[sfx: effect_name]</code> in the script editor to match them.
                  </p>
                </div>

                <div className="space-y-6">
                  {/* Music list */}
                  <div>
                    <h3 className="text-xs font-bold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-3">Music Backing Tracks</h3>
                    <div className="space-y-2 max-h-[250px] overflow-y-auto pr-1">
                      {soundAssets.filter(s => s.type === "music").length === 0 ? (
                        <div className="text-xs text-gray-400 py-4 text-center border border-dashed border-gray-150 dark:border-slate-800 rounded-xl">No music tracks available.</div>
                      ) : (
                        soundAssets.filter(s => s.type === "music").map((item) => (
                          <div key={item.key} className="flex justify-between items-center text-xs p-2.5 bg-gray-50 dark:bg-slate-950/20 border border-gray-100 dark:border-slate-800 hover:border-gray-250 dark:hover:border-slate-700 rounded-xl">
                            <div className="flex-1 truncate pr-4">
                              <span className="font-bold text-gray-800 dark:text-slate-200 block truncate">{item.name}</span>
                              <span className="text-[10px] text-gray-500">Key: <code className="bg-gray-100 dark:bg-slate-800 px-1 py-0.5 rounded">{item.key}</code> • {item.duration.toFixed(1)}s • {item.source}</span>
                            </div>
                            <div className="flex items-center gap-2.5">
                              <button
                                onClick={() => {
                                  if (playingSoundUrl === item.url) {
                                    setPlayingSoundUrl(null);
                                  } else {
                                    setPlayingSoundUrl(item.url);
                                  }
                                }}
                                className="p-1 text-gray-500 hover:text-indigo-500 transition"
                              >
                                {playingSoundUrl === item.url ? (
                                  <Square className="h-4 w-4 fill-current" />
                                ) : (
                                  <Play className="h-4 w-4 fill-current" />
                                )}
                              </button>
                              
                              {item.source !== "built-in" && (
                                <>
                                  <button
                                    onClick={() => {
                                      const newName = prompt("Rename music track:", item.name);
                                      if (newName && newName.trim()) {
                                        fetch("http://localhost:5000/api/sound-library/rename", {
                                          method: "POST",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ key: item.key, type: "music", name: newName.trim() })
                                        })
                                        .then(r => r.json())
                                        .then(res => {
                                          if (res.success) fetchSoundAssets();
                                        });
                                      }
                                    }}
                                    className="p-1 text-gray-400 hover:text-amber-500 transition"
                                  >
                                    <Edit3 className="h-3.5 w-3.5" />
                                  </button>
                                  <button
                                    onClick={() => {
                                      if (confirm(`Delete sound asset "${item.name}"?`)) {
                                        fetch("http://localhost:5000/api/sound-library/delete", {
                                          method: "POST",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ key: item.key, type: "music" })
                                        })
                                        .then(r => r.json())
                                        .then(res => {
                                          if (res.success) fetchSoundAssets();
                                        });
                                      }
                                    }}
                                    className="p-1 text-gray-400 hover:text-red-500 transition"
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </button>
                                </>
                              )}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* SFX list */}
                  <div>
                    <h3 className="text-xs font-bold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-3">Sound Effects (SFX)</h3>
                    <div className="space-y-2 max-h-[250px] overflow-y-auto pr-1">
                      {soundAssets.filter(s => s.type === "sfx").length === 0 ? (
                        <div className="text-xs text-gray-400 py-4 text-center border border-dashed border-gray-150 dark:border-slate-800 rounded-xl">No sound effects available.</div>
                      ) : (
                        soundAssets.filter(s => s.type === "sfx").map((item) => (
                          <div key={item.key} className="flex justify-between items-center text-xs p-2.5 bg-gray-50 dark:bg-slate-950/20 border border-gray-100 dark:border-slate-800 hover:border-gray-250 dark:hover:border-slate-700 rounded-xl">
                            <div className="flex-1 truncate pr-4">
                              <span className="font-bold text-gray-800 dark:text-slate-200 block truncate">{item.name}</span>
                              <span className="text-[10px] text-gray-500">Key: <code className="bg-gray-100 dark:bg-slate-800 px-1 py-0.5 rounded">{item.key}</code> • {item.duration.toFixed(1)}s • {item.source}</span>
                            </div>
                            <div className="flex items-center gap-2.5">
                              <button
                                onClick={() => {
                                  if (playingSoundUrl === item.url) {
                                    setPlayingSoundUrl(null);
                                  } else {
                                    setPlayingSoundUrl(item.url);
                                  }
                                }}
                                className="p-1 text-gray-500 hover:text-indigo-500 transition"
                              >
                                {playingSoundUrl === item.url ? (
                                  <Square className="h-4 w-4 fill-current" />
                                ) : (
                                  <Play className="h-4 w-4 fill-current" />
                                )}
                              </button>
                              
                              {item.source !== "built-in" && (
                                <>
                                  <button
                                    onClick={() => {
                                      const newName = prompt("Rename sound effect:", item.name);
                                      if (newName && newName.trim()) {
                                        fetch("http://localhost:5000/api/sound-library/rename", {
                                          method: "POST",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ key: item.key, type: "sfx", name: newName.trim() })
                                        })
                                        .then(r => r.json())
                                        .then(res => {
                                          if (res.success) fetchSoundAssets();
                                        });
                                      }
                                    }}
                                    className="p-1 text-gray-400 hover:text-amber-500 transition"
                                  >
                                    <Edit3 className="h-3.5 w-3.5" />
                                  </button>
                                  <button
                                    onClick={() => {
                                      if (confirm(`Delete sound asset "${item.name}"?`)) {
                                        fetch("http://localhost:5000/api/sound-library/delete", {
                                          method: "POST",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ key: item.key, type: "sfx" })
                                        })
                                        .then(r => r.json())
                                        .then(res => {
                                          if (res.success) fetchSoundAssets();
                                        });
                                      }
                                    }}
                                    className="p-1 text-gray-400 hover:text-red-500 transition"
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </button>
                                </>
                              )}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {playingSoundUrl && (
            <audio
              src={playingSoundUrl}
              autoPlay
              onEnded={() => setPlayingSoundUrl(null)}
              className="hidden"
            />
          )}
        </div>
      )}

      {activeMainTab === "project-settings" && (
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 mt-8 space-y-8 pb-16">
          <div className="bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 p-8 rounded-3xl shadow-xl space-y-6 text-gray-900 dark:text-slate-100">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                <Settings2 className="h-6 w-6 text-indigo-500" />
                Project Settings
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Configure voice models, naming, and properties for the active workspace.
              </p>
            </div>

            <div className="space-y-6 border-t border-gray-100 dark:border-slate-800 pt-6">
              {/* Project Rename */}
              <div className="space-y-2">
                <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                  Project Title
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={activeProjectName}
                    onChange={(e) => setActiveProjectName(e.target.value)}
                    className="flex-1 p-2.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-800 text-sm focus:ring-2 focus:ring-indigo-500 rounded-xl dark:text-slate-100"
                    placeholder="e.g. Dialogue Scene 1"
                  />
                </div>
              </div>

              {/* Model Activations */}
              <div className="space-y-3">
                <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                  Enable / Disable Voice Technologies
                </label>
                <p className="text-[11px] text-gray-500 dark:text-gray-400">
                  Select which text-to-speech models are active for this project. Disabled technologies will be hidden from selectors to declutter the workspace.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                  {[
                    { key: "kokoro", name: "Kokoro-82M", desc: "Ultra-fast, high-quality offline model. Excellent for standard dialogue." },
                    { key: "vits", name: "VITS English", desc: "Lightweight, robust synthesis. Very low CPU overhead." },
                    { key: "bark", name: "Bark (Expressive)", desc: "Supports expressive sounds (laughter, gasps, sighs) and emotional tags." },
                    { key: "chattts", name: "ChatTTS", desc: "Conversational text-to-speech with natural rhythm, speed, and pause refinement." },
                    { key: "fish-audio", name: "Fish Audio", desc: "Reference-based speech generation and high fidelity cloning." },
                    { key: "qwen3-tts", name: "Qwen3-TTS ✧", desc: "Describe the voice you want in plain English. Instruction-following synthesis. First use downloads ~1.7GB." }
                  ].map((m) => {
                    const isChecked = enabledModels.includes(m.key);
                    return (
                      <label
                        key={m.key}
                        className={`flex items-start gap-3 p-4 rounded-2xl border cursor-pointer transition ${
                          isChecked
                            ? "bg-indigo-50/20 dark:bg-indigo-950/20 border-indigo-200 dark:border-indigo-800"
                            : "bg-white dark:bg-slate-800 border-gray-100 dark:border-slate-800 hover:border-gray-200 dark:hover:border-slate-700"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(e) => {
                            const active = e.target.checked;
                            setEnabledModels(prev => {
                              let updated;
                              if (active) {
                                updated = [...prev, m.key];
                              } else {
                                if (prev.length <= 1) {
                                  alert("You must leave at least one model enabled!");
                                  return prev;
                                }
                                updated = prev.filter(k => k !== m.key);
                              }
                              return updated;
                            });
                          }}
                          className="mt-1 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        />
                        <div>
                          <div className="text-xs font-bold text-gray-800">
                            {m.name}
                          </div>
                          <div className="text-[10px] text-gray-500 mt-1 leading-normal">
                            {m.desc}
                          </div>
                        </div>
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Obsidian Integration */}
              <div className="space-y-2 border-t border-gray-100 dark:border-slate-800 pt-6">
                <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                  Obsidian Vault Integration
                </label>
                <p className="text-[11px] text-gray-500 dark:text-gray-400">
                  Specify the absolute folder path where novel/book chapters should be saved directly to your Obsidian Vault.
                </p>
                <input
                  type="text"
                  value={obsidianVaultPath}
                  onChange={(e) => setObsidianVaultPath(e.target.value)}
                  className="w-full p-2.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-800 text-sm focus:ring-2 focus:ring-indigo-500 rounded-xl dark:text-slate-100 font-semibold"
                  placeholder="e.g. /Users/username/Documents/Obsidian/NovelVault"
                />
              </div>

              {/* Freesound API Token Integration */}
              <div className="space-y-2 border-t border-gray-100 dark:border-slate-800 pt-6">
                <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                  Freesound API Token
                </label>
                <p className="text-[11px] text-gray-500 dark:text-gray-400">
                  Provide your Freesound API Client Secret/Token to enable public search and curation of sound effects and music tracks.
                </p>
                <input
                  type="password"
                  value={freesoundToken}
                  onChange={(e) => {
                    setFreesoundToken(e.target.value);
                    localStorage.setItem("voication_freesound_token", e.target.value);
                  }}
                  className="w-full p-2.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-800 text-sm focus:ring-2 focus:ring-indigo-500 rounded-xl dark:text-slate-100 font-semibold"
                  placeholder="Paste Freesound API Token/Key here..."
                />
              </div>

              {/* Project Stats & Actions */}
              <div className="border-t border-gray-100 dark:border-slate-800 pt-6 space-y-4">
                <label className="block text-xs font-bold text-gray-700 dark:text-slate-200 uppercase tracking-wider">
                  Project Workspace Actions
                </label>
                
                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={() => {
                      const activeProj = projects.find(p => p.id === currentProjectId);
                      if (activeProj) {
                        try {
                          const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(activeProj, null, 2));
                          const downloadAnchor = document.createElement('a');
                          downloadAnchor.setAttribute("href", dataStr);
                          downloadAnchor.setAttribute("download", `${activeProj.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_project.json`);
                          document.body.appendChild(downloadAnchor);
                          downloadAnchor.click();
                          downloadAnchor.remove();
                        } catch (e) {
                          alert("Failed to export project!");
                        }
                      }
                    }}
                    className="px-4 py-2 bg-indigo-50 dark:bg-indigo-950/40 hover:bg-indigo-100 dark:hover:bg-indigo-900/40 text-indigo-655 dark:text-indigo-400 rounded-xl text-xs font-bold transition flex items-center gap-1.5"
                  >
                    <Upload className="h-4 w-4" />
                    Export Project Workspace
                  </button>

                  <button
                    onClick={() => {
                      if (window.confirm("Are you sure you want to delete this project? This will erase all script contents and tracks.")) {
                        setProjects(prev => prev.filter(p => p.id !== currentProjectId));
                        const remaining = projects.filter(p => p.id !== currentProjectId);
                        if (remaining.length > 0) {
                          loadProject(remaining[0]);
                        } else {
                          setCurrentProjectId("");
                          setActiveProjectName("Untitled Project");
                          window.location.reload();
                        }
                      }
                    }}
                    className="px-4 py-2 bg-red-50 dark:bg-red-950/30 hover:bg-red-100 dark:hover:bg-red-900/40 text-red-655 dark:text-red-400 rounded-xl text-xs font-bold transition flex items-center gap-1.5"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete Project
                  </button>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Voice Creator */}
      {activeMainTab === "creator" && (
        <div className="relative p-6 max-w-2xl mx-auto bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 shadow-xl rounded-2xl mt-8 text-gray-900 dark:text-slate-100">
          <h2 className="text-2xl font-bold mb-2 text-center text-gray-900 dark:text-white">Voice Creator</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center mb-6">
            Generate unique voice personas or clone your own voices using audio reference files.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-gray-100 dark:border-slate-800">
            {/* Left Column: Cloner Controls */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-805 dark:text-slate-100 border-b border-gray-100 dark:border-slate-800 pb-2 flex items-center gap-1.5">
                <Mic className="h-4 w-4 text-indigo-500" /> Clone a Voice
              </h3>
              <p className="text-xs text-gray-600 leading-relaxed">
                Upload or record a short (5-10 seconds) audio sample. For best results, speak clearly with minimal background noise.
              </p>
              
              <div className="text-xs italic bg-gray-50 border border-gray-100 p-3 rounded-xl text-gray-500">
                “The sun sets behind the hills, and the sky turns orange. I really enjoy storytelling and character voices.”
              </div>

              {/* Upload Speaker Reference (WAV) input */}
              <div className="space-y-1.5">
                <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                  Upload WAV Sample
                </label>
                <input
                  type="file"
                  accept=".wav"
                  onChange={(e) => {
                    if (e.target.files?.[0]) {
                      const file = e.target.files[0];
                      formDataRef.current.set("speaker_wav", file);
                      setRecordedBlob(file);
                      if (!customCloneName) {
                        setCustomCloneName(file.name.replace(/\.[^/.]+$/, ""));
                      }
                    }
                  }}
                  className="block w-full text-xs text-gray-600 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border file:border-gray-200 file:text-xs file:font-semibold file:bg-gray-50 file:text-gray-700 hover:file:bg-gray-100 cursor-pointer"
                />
              </div>

              {/* Record interface */}
              <div className="space-y-2 border-t border-gray-100 pt-3">
                <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                  Or Record Reference
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold text-white transition ${
                      isRecording ? "bg-red-500 animate-pulse" : "bg-blue-600 hover:bg-blue-700"
                    }`}
                    onClick={startRecording}
                    disabled={isRecording}
                  >
                    {isRecording ? (
                      <span className="flex items-center gap-1.5 justify-center">
                        <span className="w-2 h-2 bg-white rounded-full animate-ping shrink-0" />
                        Recording...
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 justify-center">
                        <Mic className="h-3.5 w-3.5" />
                        Record
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="py-1.5 px-4 bg-gray-700 text-white text-xs font-semibold rounded-lg hover:bg-gray-800 transition disabled:opacity-50"
                    onClick={stopRecording}
                    disabled={!isRecording}
                  >
                    Stop
                  </button>
                </div>
              </div>

              {recordedBlob && (
                <div className="space-y-2 pt-2">
                  <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                    Clone Preview
                  </label>
                  <audio
                    controls
                    src={URL.createObjectURL(recordedBlob)}
                    className="w-full h-8"
                  />
                  
                  <div className="flex gap-2 items-center">
                    <input
                      type="text"
                      value={customCloneName}
                      onChange={(e) => setCustomCloneName(e.target.value)}
                      placeholder="Give this voice a name (e.g. My Cloned Voice)"
                      className="flex-1 p-2 border border-gray-200 rounded-lg text-xs"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        if (!recordedBlob) return;
                        const name = customCloneName.trim() || `Cloned Voice ${clonedProfiles.length + 1}`;
                        const reader = new FileReader();
                        reader.readAsDataURL(recordedBlob);
                        reader.onloadend = () => {
                          const base64data = reader.result;
                          setClonedProfiles(prev => [
                            ...prev,
                            { name, type: "clone", voice: "custom_clone", file: recordedBlob, fileBase64: base64data }
                          ]);
                          setCustomCloneName("");
                          alert(`Voice profile "${name}" saved to active session and local storage!`);
                        };
                      }}
                      className="py-2 px-3 bg-green-600 hover:bg-green-700 text-white text-xs font-bold rounded-lg transition"
                    >
                      Save Voice Profile
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Voice Profile List */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-800 border-b border-gray-100 pb-2 flex items-center gap-1.5">
                <Library className="h-4 w-4 text-indigo-500" /> Active Voice Profiles
              </h3>
              <p className="text-xs text-gray-600">
                These profiles can be used in your single-speaker experiments and multi-speaker storytelling.
              </p>
              <div className="space-y-4 max-h-[450px] overflow-y-auto pr-1">
                {/* My Voice Clones collapsible */}
                <details open className="group space-y-2">
                  <summary className="text-xs font-bold text-gray-500 uppercase tracking-wider px-1 cursor-pointer list-none select-none flex items-center justify-between hover:text-gray-700">
                    <span>My Voice Clones ({clonedProfiles.filter(p => p.type === "clone").length})</span>
                    <ChevronDown className="h-4 w-4 text-gray-400 group-open:rotate-180 transition-transform shrink-0" />
                  </summary>
                  <div className="space-y-2 mt-2">
                    {clonedProfiles.filter(p => p.type === "clone").length === 0 ? (
                      <div className="text-center text-[10px] text-gray-400 py-4 italic bg-gray-50 dark:bg-slate-800/40 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl">
                        No custom voice clones yet. Record or upload one above!
                      </div>
                    ) : (
                      clonedProfiles.map((p, idx) => {
                        if (p.type !== "clone") return null;
                        const isEditing = editingProfileIdx === idx;
                        return (
                          <div 
                            key={idx}
                            className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-xl flex flex-col hover:border-blue-300 dark:hover:border-blue-500/50 transition duration-200 text-gray-900 dark:text-slate-100"
                          >
                            <div className="flex items-center justify-between w-full">
                              <div>
                                <div className="font-bold text-xs text-gray-800 dark:text-slate-100">{p.name}</div>
                                <div className="text-[10px] text-gray-400 dark:text-slate-500 font-semibold uppercase tracking-wider mt-0.5">
                                  🟢 Custom Clone
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  type="button"
                                  onClick={() => {
                                    formDataRef.current.set("speaker_wav", p.file);
                                    setRecordedBlob(p.file);
                                    setActiveCloneProfile(p);
                                    alert(`Loaded clone reference "${p.name}" as active speaker.`);
                                  }}
                                  className="px-2 py-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 dark:text-slate-200 rounded-lg text-xs font-semibold shadow-sm transition"
                                >
                                  Load Clone
                                </button>
                                
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (isEditing) {
                                      setEditingProfileIdx(null);
                                    } else {
                                      setEditingProfileIdx(idx);
                                      setEditingProfileName(p.name);
                                      setEditingProfileTranscript(p.transcript || "");
                                    }
                                  }}
                                  className="px-2 py-1 bg-amber-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-amber-400 hover:text-amber-600 dark:hover:text-amber-300 rounded-lg text-xs font-semibold shadow-sm transition text-gray-950 dark:text-slate-200"
                                  title="Edit profile & transcript"
                                >
                                  <Edit3 className="h-3.5 w-3.5" />
                                </button>

                                <button
                                  type="button"
                                  onClick={() => {
                                    if (window.confirm(`Are you sure you want to delete the cloned voice "${p.name}"?`)) {
                                      setClonedProfiles(prev => prev.filter((_, i) => i !== idx));
                                    }
                                  }}
                                  className="px-2 py-1 bg-red-50 dark:bg-red-900/35 hover:bg-red-100 dark:hover:bg-red-900/40 text-red-655 dark:text-red-400 border border-red-200 dark:border-red-900/50 hover:border-red-300 dark:hover:border-red-800 rounded-lg text-xs font-semibold shadow-sm transition flex items-center justify-center"
                                  title="Delete Cloned Voice"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            </div>

                            {isEditing && (
                              <div className="space-y-2 mt-2 pt-2 border-t border-gray-200 dark:border-slate-800 w-full text-xs">
                                <div className="space-y-1">
                                  <label className="block text-[10px] font-bold text-gray-500 uppercase">Voice Name</label>
                                  <input 
                                    type="text" 
                                    value={editingProfileName} 
                                    onChange={(e) => setEditingProfileName(e.target.value)}
                                    className="w-full p-1.5 border border-gray-200 dark:border-slate-855 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <label className="block text-[10px] font-bold text-gray-500 uppercase">Reference Transcript (Highly Recommended for Accents)</label>
                                  <textarea 
                                    value={editingProfileTranscript} 
                                    onChange={(e) => setEditingProfileTranscript(e.target.value)}
                                    className="w-full p-1.5 border border-gray-200 dark:border-slate-855 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 min-h-[3.5rem] resize-none focus:ring-1 focus:ring-indigo-400"
                                    placeholder="Type the spoken reference words to enable accent alignment..."
                                  />
                                </div>
                                <div className="flex gap-1.5 justify-end">
                                  <button 
                                    type="button" 
                                    onClick={() => setEditingProfileIdx(null)}
                                    className="px-2 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-slate-850 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg text-xs font-semibold transition shadow-sm"
                                  >
                                    Cancel
                                  </button>
                                  <button 
                                    type="button" 
                                    onClick={() => {
                                      setClonedProfiles(prev => prev.map((item, i) => i === idx ? { ...item, name: editingProfileName.trim(), transcript: editingProfileTranscript.trim() } : item));
                                      setEditingProfileIdx(null);
                                    }}
                                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold transition shadow-sm"
                                  >
                                    Save
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </details>

                {/* Reference Voices collapsible */}
                <details open className="group space-y-2 pt-3 border-t border-gray-100">
                  <summary className="text-xs font-bold text-gray-500 uppercase tracking-wider px-1 cursor-pointer list-none select-none flex items-center justify-between hover:text-gray-700">
                    <span>Reference Voices ({clonedProfiles.filter(p => p.type === "reference" || p.type === "library").length})</span>
                    <ChevronDown className="h-4 w-4 text-gray-400 group-open:rotate-180 transition-transform shrink-0" />
                  </summary>
                  <div className="space-y-2 mt-2">
                    {clonedProfiles.map((p, idx) => {
                      if (p.type !== "reference" && p.type !== "library") return null;
                      const isEditing = editingProfileIdx === idx;
                      return (
                        <div 
                          key={idx}
                          className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-xl flex flex-col hover:border-blue-300 dark:hover:border-blue-500/50 transition duration-200 text-gray-900 dark:text-slate-100"
                        >
                          <div className="flex items-center justify-between w-full">
                            <div>
                              <div className="font-bold text-xs text-gray-800 dark:text-slate-100">{p.name}</div>
                              <div className="text-[10px] text-gray-400 dark:text-slate-500 font-semibold uppercase tracking-wider mt-0.5">
                                📚 Reference Voice
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => {
                                  formDataRef.current.delete("speaker_wav");
                                  setRecordedBlob(null);
                                  setActiveCloneProfile(p);
                                  alert(`Loaded library voice preset "${p.name}" as active speaker.`);
                                }}
                                className="px-2 py-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 dark:text-slate-200 rounded-lg text-xs font-semibold shadow-sm transition"
                              >
                                Load Clone
                              </button>
                              
                              <button
                                type="button"
                                onClick={() => {
                                  if (isEditing) {
                                    setEditingProfileIdx(null);
                                  } else {
                                    setEditingProfileIdx(idx);
                                    setEditingProfileName(p.name);
                                    setEditingProfileTranscript(p.transcript || "");
                                  }
                                }}
                                className="px-2 py-1 bg-amber-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 hover:border-amber-400 hover:text-amber-600 dark:hover:text-amber-300 rounded-lg text-xs font-semibold shadow-sm transition text-gray-950 dark:text-slate-200"
                                title="Edit profile & transcript"
                              >
                                <Edit3 className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </div>

                          {isEditing && (
                            <div className="space-y-2 mt-2 pt-2 border-t border-gray-200 dark:border-slate-800 w-full text-xs">
                              <div className="space-y-1">
                                <label className="block text-[10px] font-bold text-gray-500 uppercase">Voice Name</label>
                                <input 
                                  type="text" 
                                  value={editingProfileName} 
                                  onChange={(e) => setEditingProfileName(e.target.value)}
                                  className="w-full p-1.5 border border-gray-200 dark:border-slate-855 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400"
                                />
                              </div>
                              <div className="space-y-1">
                                <label className="block text-[10px] font-bold text-gray-500 uppercase">Reference Transcript (Highly Recommended for Accents)</label>
                                <textarea 
                                  value={editingProfileTranscript} 
                                  onChange={(e) => setEditingProfileTranscript(e.target.value)}
                                  className="w-full p-1.5 border border-gray-200 dark:border-slate-855 bg-white dark:bg-slate-900 rounded-lg text-xs text-gray-900 dark:text-slate-100 min-h-[3.5rem] resize-none focus:ring-1 focus:ring-indigo-400"
                                  placeholder="Type the spoken reference words to enable accent alignment..."
                                />
                              </div>
                              <div className="flex gap-1.5 justify-end">
                                <button 
                                  type="button" 
                                  onClick={() => setEditingProfileIdx(null)}
                                  className="px-2 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-slate-850 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg text-xs font-semibold transition shadow-sm"
                                >
                                  Cancel
                                </button>
                                <button 
                                  type="button" 
                                  onClick={() => {
                                    setClonedProfiles(prev => prev.map((item, i) => i === idx ? { ...item, name: editingProfileName.trim(), transcript: editingProfileTranscript.trim() } : item));
                                    setEditingProfileIdx(null);
                                  }}
                                  className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold transition shadow-sm"
                                >
                                  Save
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </details>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeMainTab === "storyteller" && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8 space-y-8 pb-16">
          
          {storytellerViewMode === "overview" ? (
            /* --- PARENT SCREEN: OVERVIEW --- */
            <div className="space-y-6 text-gray-900 dark:text-slate-100">
              
              {/* Togglable Tabs at Project Level */}
              <div className="flex justify-center mb-6 overflow-x-auto px-2">
                <div className="inline-flex p-1 bg-gray-150 dark:bg-slate-950 rounded-xl border border-gray-200 dark:border-slate-800 min-w-max">
                  <button
                    type="button"
                    onClick={() => setProjectViewMode("overview")}
                    className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                      projectViewMode === "overview"
                        ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                        : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                    }`}
                  >
                    <Folder className="h-3.5 w-3.5" />
                    <span>Chapters Outline</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setProjectViewMode("multitrack")}
                    className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                      projectViewMode === "multitrack"
                        ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                        : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                    }`}
                  >
                    <Sliders className="h-3.5 w-3.5" />
                    <span>Project Multitrack</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setProjectViewMode("phonetic")}
                    className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                      projectViewMode === "phonetic"
                        ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                        : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                    }`}
                  >
                    <BookOpen className="h-3.5 w-3.5" />
                    <span>Phonetic Dictionary</span>
                  </button>
                </div>
              </div>

              {projectViewMode === "overview" ? (
                <>
                  {/* Header Title */}
                  <div className="text-center">
                    <h2 className="text-3xl font-extrabold text-gray-900 dark:text-white tracking-tight">
                      {mediaFormat === "podcast" ? "Storyteller Outline (Podcast)" : mediaFormat === "audiobook" ? "Storyteller Outline (Audiobook)" : "Storyteller Outline (Video)"}
                    </h2>
                    <p className="mt-2 text-sm text-gray-500 dark:text-slate-400 font-medium">
                      Outline Center: Coordinate chapters, manage global configurations, and batch process timelines.
                    </p>
                  </div>

                  {/* Project Stats & Summary Row */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Stats Card */}
                    <div className="bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 p-6 rounded-2xl shadow-md space-y-3">
                      <h3 className="text-xs font-bold text-gray-400 dark:text-slate-400 uppercase tracking-wider">Project Outline Stats</h3>
                      <div className="grid grid-cols-2 gap-3 sm:gap-4">
                        <div>
                          <span className="text-[10px] text-gray-400 font-medium">Chapters</span>
                          <p className="text-2xl font-black text-indigo-600">{chapters.length}</p>
                        </div>
                        <div>
                          <span className="text-[10px] text-gray-400 font-medium">Format</span>
                          <p className="text-xs font-bold text-gray-700 dark:text-slate-200 capitalize mt-1">{mediaFormat}</p>
                        </div>
                      </div>
                      <div className="pt-2 border-t border-gray-50 dark:border-slate-800">
                        <span className="text-[10px] text-gray-400 font-medium block">Obsidian Path</span>
                        <p className="text-[11px] font-semibold truncate text-gray-600 dark:text-slate-400">{obsidianVaultPath || "Not configured (Settings)"}</p>
                      </div>
                    </div>

                    {/* Global Novel Outline / Summary */}
                    <div className="md:col-span-2 bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 p-6 rounded-2xl shadow-md space-y-2.5">
                      <div className="flex items-center justify-between">
                        <h3 className="text-xs font-bold text-gray-400 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                          <BookOpen className="h-4 w-4 text-indigo-500" />
                          Global Novel Summary / Shared Context
                        </h3>
                      </div>
                      <textarea
                        value={globalSummary}
                        onChange={(e) => setGlobalSummary(e.target.value)}
                        placeholder="Enter global character profiles, style sheets, outline, or plot summaries shared across Ollama revisions..."
                        className="w-full p-3 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-955 rounded-xl text-xs resize-none h-[4.5rem] focus:ring-1 focus:ring-indigo-400 font-semibold text-gray-800 dark:text-slate-200"
                      />
                    </div>
                  </div>

                  {/* Chapters List Table */}
                  <div className="bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 rounded-2xl shadow-md overflow-hidden">
                    <div className="p-5 border-b border-gray-100 dark:border-slate-800 flex justify-between items-center bg-gray-50/50 dark:bg-slate-955/25">
                      <h3 className="text-sm font-bold text-indigo-900 dark:text-indigo-200">
                        Chapters List
                      </h3>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            createNewChapter();
                            setStorytellerViewMode("editor");
                          }}
                          className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition flex items-center gap-1 shadow-sm"
                          type="button"
                        >
                          <Plus className="h-3.5 w-3.5" />
                          Add Chapter
                        </button>
                        <button
                          onClick={triggerBatchSynthesis}
                          className="px-3.5 py-1.5 bg-green-600 hover:bg-green-700 active:scale-[0.98] text-white text-xs font-bold rounded-lg transition flex items-center gap-1 shadow-sm"
                          type="button"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                          Batch Render All
                        </button>
                        <button
                          onClick={() => exportMarkdownScript("project")}
                          disabled={chapters.length === 0}
                          className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-205 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-250 text-xs font-bold rounded-lg transition flex items-center gap-1 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                          type="button"
                          title="Export the entire project transcript as markdown"
                        >
                          <Download className="h-3.5 w-3.5" />
                          Export Transcript (.md)
                        </button>
                      </div>
                    </div>

                    <div className="divide-y divide-gray-100 dark:divide-slate-800 text-gray-900 dark:text-slate-100">
                      {chapters.map((ch, idx) => (
                        <div key={ch.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-gray-50/40 dark:hover:bg-slate-955/10 transition">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-black text-gray-300 dark:text-slate-500">
                                #{idx + 1}
                              </span>
                              <span className="text-sm font-bold text-gray-800 dark:text-slate-100 truncate">
                                {ch.name}
                              </span>
                            </div>
                            <div className="text-[10px] text-gray-400 mt-1 flex gap-4">
                              <span>Timeline Clips: <strong className="text-gray-650 dark:text-slate-350">{(ch.playlistClips || []).length}</strong></span>
                              <span>Text Length: <strong className="text-gray-650 dark:text-slate-350">{(ch.podcastText || "").length} chars</strong></span>
                            </div>
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <button
                              onClick={() => {
                                switchChapter(ch.id);
                                setStorytellerViewMode("editor");
                              }}
                              className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-950 dark:hover:bg-indigo-900/60 text-indigo-700 dark:text-indigo-300 text-xs font-bold rounded-lg transition"
                              type="button"
                            >
                              Open Editor
                            </button>
                            <button
                              onClick={() => {
                                const newName = prompt("Enter new name for chapter:", ch.name);
                                if (newName && newName.trim()) {
                                  setChapters(prev => prev.map(c => c.id === ch.id ? { ...c, name: newName.trim() } : c));
                                  setTimeout(() => saveProjectSync(), 100);
                                }
                              }}
                              className="px-3 py-1.5 border border-gray-200 dark:border-slate-800 hover:bg-gray-50 dark:hover:bg-slate-800 text-gray-700 dark:text-slate-300 text-xs font-bold rounded-lg transition"
                              type="button"
                            >
                              Rename
                            </button>
                            <button
                              onClick={() => deleteChapter(ch.id)}
                              className="p-1.5 text-red-500 hover:bg-red-50 dark:hover:bg-red-955/20 rounded-lg transition"
                              title="Delete Chapter"
                              type="button"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : projectViewMode === "multitrack" ? (
                renderProjectMultitrackTimeline()
              ) : (
                renderPhoneticDictionaryView()
              )}

            </div>
          ) : (
            /* --- PART 2: CHAPTER EDITOR --- */
            <div className="space-y-6 text-gray-900 dark:text-slate-100">
              
              {/* Back to Chapters Outline Button */}
              <div className="flex justify-between items-center bg-white dark:bg-slate-900 p-4 rounded-xl border border-gray-100 dark:border-slate-800 shadow-sm text-gray-900 dark:text-slate-100">
                <button
                  onClick={goBackToOverview}
                  className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-1"
                type="button"
                >
                  ← Back to Chapters Outline
                </button>
                <div className="flex items-center gap-4 flex-wrap">
                  <div className="flex items-center gap-1.5 text-xs text-gray-700 dark:text-slate-350 font-semibold">
                    <input
                      type="checkbox"
                      id="chapter_use_phonetics"
                      checked={chapters.find(c => c.id === currentChapterId)?.usePhoneticSettings ?? true}
                      onChange={(e) => {
                        const activeVal = e.target.checked;
                        setChapters(prev => prev.map(c => c.id === currentChapterId ? { ...c, usePhoneticSettings: activeVal } : c));
                        setTimeout(() => saveProjectSync(currentProjectId), 100);
                      }}
                      className="w-3.5 h-3.5 rounded text-indigo-600 border-gray-300 dark:border-slate-800 focus:ring-indigo-500"
                    />
                    <label htmlFor="chapter_use_phonetics" className="cursor-pointer select-none">
                      Apply Dictionary & Dials
                    </label>
                  </div>
                  <span className="text-gray-350 dark:text-slate-600">|</span>
                  <div className="text-xs text-gray-500 dark:text-slate-400 font-semibold">
                    Active Chapter: <span className="text-indigo-600 dark:text-indigo-400 font-bold">{chapters.find(c => c.id === currentChapterId)?.name}</span>
                  </div>
                </div>
              </div>

              {/* Header Title inside Editor */}
              <div className="text-center">
                <h2 className="text-2xl font-extrabold text-gray-900 dark:text-white tracking-tight">
                  {chapters.find(c => c.id === currentChapterId)?.name || "Active Chapter"} Editor
                </h2>
                
                {/* Togglable Tabs */}
                <div className="flex justify-center my-4">
                  <div className="inline-flex p-1 bg-gray-150 dark:bg-slate-950 rounded-xl border border-gray-200 dark:border-slate-800">
                    <button
                      type="button"
                      onClick={() => {
                        syncTimelineToScript();
                        setChapterEditorTab("script");
                      }}
                      className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                        chapterEditorTab === "script"
                          ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                          : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                      }`}
                    >
                      <FileText className="h-3.5 w-3.5" />
                      <span>Script Editor</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        syncScriptToTimeline();
                        setChapterEditorTab("multitrack");
                      }}
                      className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                        chapterEditorTab === "multitrack"
                          ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                          : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                      }`}
                    >
                      <Sliders className="h-3.5 w-3.5" />
                      <span>Multitrack Timeline</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        loadActiveChapterMix();
                        setChapterEditorTab("reviewer");
                      }}
                      className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-200 flex items-center gap-1.5 ${
                        chapterEditorTab === "reviewer"
                          ? "bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm"
                          : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                      }`}
                    >
                      <MessageSquare className="h-3.5 w-3.5" />
                      <span>Audio & Text Reviewer</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Top Panel: Script Writer & Voice Settings */}
              {chapterEditorTab === "script" && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Left/Voice Settings Column (5 cols) */}
            <div className="lg:col-span-5 space-y-6 bg-white dark:bg-slate-900 p-6 rounded-2xl shadow-md border border-gray-100 dark:border-slate-800 text-gray-900 dark:text-slate-100">
              
              {/* Media Format Segmented Selector */}
              <div className="p-1 bg-gray-100 dark:bg-slate-950 rounded-xl flex gap-1 shadow-inner border border-gray-200/20 dark:border-slate-800">
                {[
                  { id: "podcast", label: "Podcast", icon: Mic, desc: "Co-hosts & guests" },
                  { id: "audiobook", label: "Audiobook", icon: BookOpen, desc: "Narrators" },
                  { id: "video", label: "Video/Script", icon: Sliders, desc: "Teasers & Promos" }
                ].map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => {
                        setMediaFormat(item.id);
                        // Set default number of speakers appropriate for media type
                        if (item.id === "audiobook") {
                          setNumberOfSpeakers(3);
                        } else if (item.id === "video") {
                          setNumberOfSpeakers(2);
                        } else {
                          setNumberOfSpeakers(4);
                        }
                      }}
                      className={`flex-1 py-2 px-3 rounded-lg text-center transition duration-200 flex flex-col items-center justify-center gap-1 ${
                        mediaFormat === item.id
                          ? "bg-white dark:bg-slate-900 text-indigo-900 dark:text-indigo-400 shadow-sm border border-gray-200/50 dark:border-slate-700 font-bold"
                          : "text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200 hover:bg-white/40 dark:hover:bg-slate-800/40"
                      }`}
                      type="button"
                    >
                      <span className="flex items-center gap-1.5 text-xs font-semibold">
                        <Icon className="h-3.5 w-3.5" />
                        {item.label}
                      </span>
                      <span className="text-[9px] opacity-75 mt-0.5 hidden sm:inline">{item.desc}</span>
                    </button>
                  );
                })}
              </div>

              {/* Voice Recipes */}
              <div className="p-4 bg-purple-50/10 dark:bg-slate-900 border border-purple-100/50 dark:border-slate-800 rounded-xl">
                <div className="flex items-center justify-between mb-3 border-b border-purple-100/20 dark:border-slate-800 pb-2 flex-wrap gap-2">
                  <h4 className="font-semibold text-purple-955 dark:text-purple-200 text-xs flex items-center gap-1.5">
                    <Sparkles className="h-4 w-4 text-purple-500 shrink-0" />
                    {mediaFormat === "podcast" ? "Podcast" : mediaFormat === "audiobook" ? "Audiobook" : "Video"} Voice Recipes
                  </h4>
                  <button
                    type="button"
                    onClick={() => {
                      const name = prompt("Enter a name for this custom recipe:");
                      if (!name || !name.trim()) return;
                      const description = prompt("Enter a description for this custom recipe:");
                      const newRecipe = {
                        id: `custom_recipe_${Date.now()}`,
                        name: name.trim(),
                        mediaType: mediaFormat,
                        description: (description || "").trim(),
                        mapping: { ...speakerMapping },
                        colors: { ...speakerColors },
                        names: { ...speakerNames },
                        speakerCount: numberOfSpeakers
                      };
                      setCustomRecipes(prev => [...prev, newRecipe]);
                      alert(`Custom recipe "${name.trim()}" saved!`);
                    }}
                    className="flex items-center gap-1 px-2.5 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-[10px] font-bold shadow-sm transition"
                  >
                    <Save className="h-3.5 w-3.5" />
                    Save Current as Custom Recipe
                  </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {allRecipes.filter((r) => r.mediaType === mediaFormat).map((recipe) => {
                    const active = isRecipeActive(recipe.mapping);
                    const isCustom = !!recipe.id;
                    return (
                      <div
                        key={recipe.id || recipe.name}
                        className={`relative p-2 rounded-xl border text-left transition duration-205 flex flex-col justify-between min-h-[4.5rem] group ${
                          active
                            ? "bg-gradient-to-br from-indigo-600 to-purple-600 text-white border-transparent shadow-md transform -translate-y-0.5"
                            : "bg-white dark:bg-slate-950 border-purple-100 dark:border-slate-800 text-purple-900 dark:text-purple-300 hover:border-purple-300 dark:hover:border-purple-700 hover:bg-purple-100/30 dark:hover:bg-slate-900"
                        }`}
                      >
                        <div
                          onClick={() => applyRecipe(recipe)}
                          className="flex-1 cursor-pointer flex flex-col justify-between"
                        >
                          <div className="font-bold text-[10px] uppercase tracking-wider mb-0.5 pr-6 truncate">{recipe.name}</div>
                          <div className={`text-[9px] leading-snug ${active ? "text-purple-100" : "text-gray-500 dark:text-slate-400"}`}>
                            {recipe.description}
                          </div>
                        </div>
                        {isCustom && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (window.confirm(`Delete custom recipe "${recipe.name}"?`)) {
                                setCustomRecipes(prev => prev.filter(r => r.id !== recipe.id));
                              }
                            }}
                            className={`absolute top-2 right-2 p-1 rounded hover:bg-red-500/20 text-red-500 transition-colors duration-150`}
                            title="Delete custom recipe"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Multi-Speaker Voice Assignment */}
              <div className="p-4 bg-indigo-50/30 dark:bg-slate-900 border border-indigo-100/50 dark:border-slate-800 rounded-xl text-gray-900 dark:text-slate-100">
                <div className="flex items-center justify-between mb-3 border-b border-indigo-100/40 dark:border-slate-800 pb-2">
                  <h3 className="text-sm font-semibold flex items-center gap-1.5 text-indigo-900 dark:text-slate-100">
                    <Users className="h-4 w-4 text-indigo-500 shrink-0" />
                    Character Assignment & Voice Mapping
                  </h3>
                  <div className="flex items-center gap-1.5 bg-white dark:bg-slate-950 border border-indigo-100 dark:border-slate-800 rounded-lg px-2 py-1 shadow-sm shrink-0">
                    <span className="text-[10px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Speakers:</span>
                    <select
                      value={numberOfSpeakers}
                      onChange={(e) => {
                        const count = parseInt(e.target.value);
                        setNumberOfSpeakers(count);
                        try {
                          localStorage.setItem("voication_num_speakers", count.toString());
                        } catch (err) {}
                        setTimeout(() => saveProjectSync(), 100);
                      }}
                      className="bg-transparent border-0 text-xs font-bold text-indigo-900 dark:text-indigo-200 focus:ring-0 p-0 cursor-pointer dark:[&>option]:bg-slate-900 dark:[&>option]:text-slate-100"
                    >
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                      <option value={3}>3</option>
                      <option value={4}>4</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {Array.from({ length: numberOfSpeakers }, (_, i) => i + 1).map((num) => {
                    const spkKey = `speaker_${num}`;
                    const spkVal = speakerMapping[spkKey] || `tts_models/en/vctk/vits:p225`;
                    const charName = speakerNames[spkKey] !== undefined ? speakerNames[spkKey] : `Speaker ${num}`;
                    
                    const isCuratedVal = spkVal && spkVal.startsWith("curated:");
                    const curatedId = isCuratedVal ? spkVal.split(":")[1] : "";

                    // Parse mapping value
                    const { model: currentModel, voice: currentVoice } = getModelAndVoiceFromMapping(spkVal);
                    
                    // Populate voice options grouped by taxonomy
                    const modelInfo = voices.find(v => v.name === currentModel);
                    
                    const curatedOpts = curatedVoices
                      .filter(cv => cv.model === currentModel)
                      .map(cv => ({ id: `curated:${cv.id}`, label: cv.name }));

                    const customCloneOpts = [];
                    const referenceVoiceOpts = [];
                    if (modelInfo && modelInfo.features?.includes("cloning")) {
                      clonedProfiles.forEach(p => {
                        if (p.type === "clone") {
                          customCloneOpts.push({ id: `clone:${p.name}`, label: p.name });
                        } else if (p.type === "reference" || p.type === "library") {
                          referenceVoiceOpts.push({ id: `clone:${p.name}`, label: p.name });
                        }
                      });
                    }

                    const builtInOpts = [];
                    if (modelInfo) {
                      const spks = modelInfo.supported_speakers?.length
                        ? modelInfo.supported_speakers
                        : modelInfo.speakers?.length
                        ? modelInfo.speakers
                        : modelInfo.speaker_list?.length
                        ? modelInfo.speaker_list
                        : modelInfo.speaker_ids?.length
                        ? modelInfo.speaker_ids
                        : [];
                      spks.forEach(s => {
                        builtInOpts.push({ id: s, label: s });
                      });
                      
                      const presets = modelInfo.presets || [];
                      presets.forEach(p => {
                        builtInOpts.push({ id: p, label: p });
                      });
                    }

                    const voiceOptions = [...curatedOpts, ...customCloneOpts, ...referenceVoiceOpts, ...builtInOpts];
                    if (voiceOptions.length === 0) {
                      voiceOptions.push({ id: "no_clones", label: "[!] Create Clone in Creator" });
                    }
                    
                    const targetVoiceVal = isCuratedVal ? "curated:" + curatedId : (currentVoice.startsWith("clone:") ? "clone:" + currentVoice.replace("clone:", "") : currentVoice);
                    const isVoiceValid = voiceOptions.some(opt => opt.id === targetVoiceVal);
                    const activeSelectVoiceVal = isVoiceValid ? targetVoiceVal : voiceOptions[0]?.id || "no_clones";

                    return (
                      <div
                        key={num}
                        style={{
                          borderColor: speakerColors[spkKey] || "#4f46e5",
                          borderWidth: "2px",
                          borderStyle: "solid"
                        }}
                        className="flex flex-col p-3 bg-white/60 dark:bg-slate-950/40 rounded-xl space-y-2.5 min-w-0"
                      >
                        <div className="flex flex-col space-y-1">
                          <label className="text-[10px] font-bold text-indigo-900 dark:text-indigo-400 uppercase tracking-wider">
                            Speaker {num}
                          </label>
                          
                          {/* Premium Color Picker */}
                          <div className="flex items-center gap-1">
                            {["#4f46e5", "#059669", "#d97706", "#e11d48", "#2563eb", "#db2777"].map((color) => (
                              <button
                                key={color}
                                type="button"
                                onClick={() => {
                                  setSpeakerColors(prev => ({ ...prev, [spkKey]: color }));
                                  setTimeout(() => saveProjectSync(), 100);
                                }}
                                style={{ backgroundColor: color }}
                                className={`w-3 h-3 rounded-full border transition-all ${
                                  (speakerColors[spkKey] || "#4f46e5") === color
                                    ? "border-black dark:border-white scale-110 shadow-sm"
                                    : "border-transparent hover:scale-105"
                                }`}
                                title={`Select speaker color`}
                              />
                            ))}
                          </div>
                        </div>
                        
                        <input
                          type="text"
                          value={charName}
                          onChange={(e) => {
                            const val = e.target.value;
                            const oldName = speakerNames[spkKey] || `Speaker ${num}`;
                            setSpeakerNames(prev => {
                              const updated = { ...prev, [spkKey]: val };
                              localStorage.setItem("voication_speaker_names", JSON.stringify(updated));
                              return updated;
                            });
                            if (oldName && val && oldName.trim() !== val.trim()) {
                              setPodcastText(prev => {
                                if (!prev) return prev;
                                return prev.split(`[${oldName.trim()}]`).join(`[${val.trim()}]`);
                              });
                            }
                            setTimeout(() => saveProjectSync(), 100);
                          }}
                          placeholder={`Name (e.g. Alice)`}
                          className="w-full p-1.5 border border-indigo-200 dark:border-slate-800 rounded-lg text-xs bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400 font-semibold"
                        />

                        <div className="flex flex-col space-y-1">
                           <label className="text-[9px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Model</label>
                           <select
                             className="w-full p-1.5 border border-indigo-200 dark:border-slate-800 rounded-lg bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-100 text-[11px] focus:ring-1 focus:ring-indigo-400 truncate dark:[&>option]:bg-slate-900 dark:[&>option]:text-slate-100 font-semibold"
                             value={currentModel}
                             onChange={(e) => {
                               const newModel = e.target.value;
                               const newModelInfo = voices.find(v => v.name === newModel);
                               let defVoice = "";
                               if (newModelInfo) {
                                 const hasClones = newModelInfo.features?.includes("cloning") && clonedProfiles.some(p => p.type === "clone" || p.type === "library" || p.type === "reference");
                                 if (hasClones) {
                                   const firstClone = clonedProfiles.find(p => p.type === "clone" || p.type === "library" || p.type === "reference");
                                   defVoice = `clone:${firstClone.name}`;
                                 } else {
                                   const firstSpk = newModelInfo.supported_speakers?.[0] || newModelInfo.speakers?.[0] || newModelInfo.presets?.[0];
                                   if (firstSpk) defVoice = firstSpk;
                                 }
                               }
                               if (!defVoice) defVoice = "no_clones";
                               
                               const combinedVal = defVoice.startsWith("clone:")
                                 ? `${newModel}:clone:${defVoice.replace("clone:", "")}`
                                 : `${newModel}:${defVoice}`;
                                 
                               setSpeakerMapping(prev => {
                                 const updated = { ...prev, [spkKey]: combinedVal };
                                 try {
                                   localStorage.setItem("vibevoice_speaker_mapping", JSON.stringify(updated));
                                 } catch (err) {}
                                 return updated;
                               });
                               setTimeout(() => saveProjectSync(), 100);
                             }}
                           >
                             {voices.filter(v => v.name !== "vibevoice" && isModelEnabled(v.name)).map((v) => {
                               const display = v.name === "bark" 
                                 ? "Bark" 
                                 : v.name === "tts_models/en/vctk/vits" 
                                 ? "VITS (Baseline)" 
                                 : v.name === "tts_models/multilingual/multi-dataset/xtts_v2" 
                                 ? "XTTS v2" 
                                 : v.name === "kokoro"
                                 ? "Kokoro-82M"
                                 : v.name === "qwen3-tts"
                                 ? "Qwen3"
                                 : v.name === "chatterbox-turbo"
                                 ? "Chatterbox"
                                 : v.name === "cosyvoice2-styletts2"
                                 ? "CosyVoice 2"
                                 : v.name === "chattts"
                                 ? "ChatTTS"
                                 : v.name === "fish-audio"
                                 ? "Fish Audio"
                                 : v.name;
                               return (
                                 <option key={v.name} value={v.name}>
                                   {display}
                                 </option>
                               );
                             })}
                           </select>
                         </div>

                         <div className="flex flex-col space-y-1">
                           <label className="text-[9px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Voice / Preset</label>
                           <div className="flex items-center gap-1.5 min-w-0 w-full">
                             <select
                               className="flex-1 min-w-0 p-1.5 border border-indigo-200 dark:border-slate-800 rounded-lg bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-100 text-[11px] focus:ring-1 focus:ring-indigo-400 truncate dark:[&>option]:bg-slate-900 dark:[&>option]:text-slate-100 font-semibold"
                               value={isCuratedVal ? `curated:${curatedId}` : (currentVoice.startsWith("clone:") ? `clone:${currentVoice.replace("clone:", "")}` : currentVoice)}
                               onChange={(e) => {
                                 const val = e.target.value;
                                 if (val === "action_clone_voice") {
                                   setShowVoiceCreatorModal(true);
                                   return;
                                 }
                                 let combinedVal;
                                 if (val.startsWith("curated:")) {
                                   combinedVal = val;
                                 } else if (val.startsWith("clone:")) {
                                   combinedVal = `${currentModel}:clone:${val.replace("clone:", "")}`;
                                 } else {
                                   combinedVal = `${currentModel}:${val}`;
                                 }
                                 setSpeakerMapping(prev => {
                                   const updated = { ...prev, [spkKey]: combinedVal };
                                   try {
                                     localStorage.setItem("vibevoice_speaker_mapping", JSON.stringify(updated));
                                   } catch (err) {}
                                   return updated;
                                 });
                                 setTimeout(() => saveProjectSync(), 100);
                               }}
                             >
                                {curatedOpts.length > 0 && (
                                  <optgroup label="Saved Presets">
                                    {curatedOpts.map(opt => (
                                      <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                  </optgroup>
                                )}
                                {customCloneOpts.length > 0 && (
                                  <optgroup label="My Voice Clones">
                                    {customCloneOpts.map(opt => (
                                      <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                  </optgroup>
                                )}
                                {referenceVoiceOpts.length > 0 && (
                                  <optgroup label="Reference Voices">
                                    {referenceVoiceOpts.map(opt => (
                                      <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                  </optgroup>
                                )}
                                {builtInOpts.length > 0 && (
                                  <optgroup label="Built-in Voices">
                                    {builtInOpts.map(opt => {
                                      let cleanLabel = opt.label;
                                      if (cleanLabel && cleanLabel.includes(":") && !cleanLabel.includes(" (") && !cleanLabel.startsWith("http")) {
                                        const parts = cleanLabel.split(":");
                                        cleanLabel = parts[parts.length - 1];
                                      }
                                      return (
                                        <option key={opt.id} value={opt.id}>{cleanLabel}</option>
                                      );
                                    })}
                                  </optgroup>
                                )}
                                {voiceOptions.length === 1 && voiceOptions[0].id === "no_clones" && (
                                  <option value="no_clones">[!] No voices available</option>
                                )}
                                <option value="action_clone_voice">+ Clone/Upload New Voice...</option>
                             </select>
                             <button
                               onClick={() => playVoicePreview(spkVal, currentModel)}
                               className="p-1.5 border border-indigo-200 dark:border-slate-800 rounded-lg hover:bg-gray-150 dark:hover:bg-slate-800 flex items-center justify-center text-xs bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-200 shrink-0"
                               title="Preview Voice"
                               disabled={activeSelectVoiceVal === "no_clones" && !isCuratedVal}
                               type="button"
                             >
                               {playingPreview === spkVal ? "⏸" : "▶"}
                             </button>
                           </div>
                           
                           <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-1.5 pt-1.5 border-t border-indigo-50/60 dark:border-slate-800/80">
                             <button
                               onClick={() => setShowVoiceCreatorModal(true)}
                               className="text-[9px] font-bold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center gap-0.5"
                               type="button"
                             >
                               <Plus className="h-2.5 w-2.5 shrink-0" /> Clone Voice
                             </button>
                             
                             <span className="text-[9px] text-gray-300 dark:text-slate-700">|</span>
                             
                             {isCuratedVal ? (
                               <button
                                 onClick={() => {
                                   const curated = curatedVoices.find(v => v.id === curatedId);
                                   if (curated) {
                                     loadCuratedVoice(curated);
                                     setLoadedCuratedVoiceId(curated.id);
                                     setActiveMainTab("experiment");
                                   }
                                 }}
                                 className="text-[9px] font-bold text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-0.5"
                                 type="button"
                               >
                                 <Sparkles className="h-2.5 w-2.5 shrink-0 text-blue-500" /> Edit in Playground
                               </button>
                             ) : (
                               <button
                                 onClick={() => {
                                   const voiceName = isClone ? currentVoice : currentVoice;
                                   const modelData = voices.find(v => v.name === currentModel);
                                   if (modelData) {
                                     setSelectedVoice(voiceName);
                                     setSelectedVoiceData(modelData);
                                     setRecordedBlob(null);
                                     setLoadedCuratedVoiceId(null);
                                     setActiveMainTab("experiment");
                                   }
                                 }}
                                 className="text-[9px] font-bold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center gap-0.5"
                                 type="button"
                               >
                                 <Sparkles className="h-2.5 w-2.5 shrink-0" /> Customize in Playground
                               </button>
                             )}
                           </div>
                         </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Generative Emotion & Steerage Controls */}
              <div className="p-4 bg-purple-50/10 dark:bg-slate-900 border border-purple-100/50 dark:border-slate-800 rounded-xl space-y-4 text-gray-900 dark:text-slate-100">


                {/* Nervousness/Stutter and Amusement/Laughter sliders removed — these injected
                    random [sighs]/[laughter] tags that did not function reliably cross-model. */}

                {/* Default Spacing Between Clips (Overlap) */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-semibold text-purple-950 dark:text-purple-200 flex items-center gap-1.5">
                      <Sliders className="h-4 w-4 text-purple-500 shrink-0" />
                      Default Spacing (Overlap)
                    </label>
                    <span className="text-xs font-bold text-purple-650 dark:text-purple-300 bg-purple-50 dark:bg-purple-950/40 px-2 py-0.5 rounded">
                      {defaultClipSpacing < 0 ? `${defaultClipSpacing.toFixed(2)}s (Overlap)` : `+${defaultClipSpacing.toFixed(2)}s (Gap)`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-0.5"
                    max="1.0"
                    step="0.05"
                    value={defaultClipSpacing}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setDefaultClipSpacing(val);
                      localStorage.setItem("voication_default_clip_spacing", val.toString());
                    }}
                    className="w-full h-1.5 bg-purple-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-600"
                  />
                  <p className="text-[9px] text-purple-900/60 dark:text-slate-400 mt-1 font-medium leading-relaxed">
                    Adjust this slider to control the default spacing between timeline clips. Negative values create a cross-fade / overlap, resolving awkward pauses.
                  </p>
                </div>
              </div>
            </div>
                {/* Right Column: Script writer & controls (7 cols) */}
                <div className="lg:col-span-7 space-y-6 bg-white dark:bg-slate-900 p-6 rounded-2xl shadow-md border border-gray-100 dark:border-slate-800 text-gray-900 dark:text-slate-100">
                  {/* AI Scriptwriter */}
                  <div className="p-4 bg-purple-50/10 dark:bg-purple-950/10 border border-purple-100/50 dark:border-purple-900/30 rounded-xl space-y-3">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-200 text-xs flex items-center gap-1.5">
                      <Cpu className="h-4 w-4 text-purple-500 shrink-0" />
                      AI {mediaFormat === "podcast" ? "Podcast" : mediaFormat === "audiobook" ? "Audio Book" : "Video"} Scriptwriter
                    </h4>
                    <div className="space-y-2">
                      <div>
                        <label className="block text-[9px] font-bold text-purple-700 uppercase tracking-wider mb-1">Source Material (Articles, notes, transcripts)</label>
                        <textarea
                          value={podcastSource}
                          onChange={(e) => setPodcastSource(e.target.value)}
                          placeholder="Paste source text here..."
                          className="w-full p-2 border border-purple-200 dark:border-purple-900 rounded-lg text-xs bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-purple-400 min-h-[4rem] resize-y"
                        />
                      </div>
                      <div>
                        <label className="block text-[9px] font-bold text-purple-700 uppercase tracking-wider mb-1">Guidelines / Directives (Tone, roles, style)</label>
                        <input
                          type="text"
                          value={podcastPrompt}
                          onChange={(e) => setPodcastPrompt(e.target.value)}
                          placeholder={mediaFormat === "podcast" ? "e.g. Host A and Host B argue humorously about technology..." : mediaFormat === "audiobook" ? "e.g. Dramatic sci-fi story narrator with quick-paced dialog..." : "e.g. Booming documentary voiceover intro with two witness lines..."}
                          className="w-full p-2 border border-purple-200 dark:border-purple-900 rounded-lg text-xs bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-purple-400"
                        />
                      </div>
                      
                      <div className="mb-4">
                        <button
                          onClick={generatePodcastScript}
                          disabled={isGeneratingPodcast}
                          className={`w-full py-2 px-3 rounded-lg text-xs font-bold text-white transition flex items-center justify-center gap-1.5 ${
                            isGeneratingPodcast
                              ? "bg-purple-400 cursor-not-allowed"
                              : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 active:scale-[0.98]"
                          }`}
                          type="button"
                        >
                          {isGeneratingPodcast ? (
                            <>
                              <span className="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full" />
                              Writing Script...
                            </>
                          ) : (
                            <>
                              {mediaFormat === "podcast" ? (
                                <Mic className="h-3.5 w-3.5 shrink-0" />
                              ) : mediaFormat === "audiobook" ? (
                                <BookOpen className="h-3.5 w-3.5 shrink-0" />
                              ) : (
                                <Sliders className="h-3.5 w-3.5 shrink-0" />
                              )}
                              <span>
                                {mediaFormat === "podcast" ? "Create Podcast Script" : mediaFormat === "audiobook" ? "Create Book Script" : "Create Video Script"}
                              </span>
                            </>
                          )}
                        </button>
                      </div>

                      {/* Separator line */}
                      <div className="border-t border-purple-100 dark:border-purple-900/50 my-4" />

                      {/* AI Assisted Tagging & Enhancement Block */}
                      <div className="space-y-3">
                        <label className="block text-[10px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
                          AI Assisted Tagging & Script Enhancements
                        </label>
                        
                        <div className="flex flex-col sm:flex-row gap-2">
                          <button
                            onClick={autoTagScriptWithAI}
                            disabled={isAutoTagging || !podcastText.trim()}
                            className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold text-white transition flex items-center justify-center gap-1.5 ${
                              isAutoTagging || !podcastText.trim()
                                ? "bg-indigo-300 cursor-not-allowed"
                                : "bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98]"
                            }`}
                            type="button"
                          >
                            {isAutoTagging ? (
                              <>
                                <span className="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full" />
                                Tagging Emotes...
                              </>
                            ) : (
                              <>
                                <Sparkles className="h-3.5 w-3.5 shrink-0" />
                                <span>AI Assisted Emotive Tagging</span>
                              </>
                            )}
                          </button>

                          <button
                            onClick={identifySpeakers}
                            disabled={isIdentifyingSpeakers || !podcastText.trim()}
                            className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold text-white transition flex items-center justify-center gap-1.5 ${
                              isIdentifyingSpeakers || !podcastText.trim()
                                ? "bg-purple-300 dark:bg-purple-900/50 cursor-not-allowed text-purple-100"
                                : "bg-purple-600 hover:bg-purple-700 active:scale-[0.98]"
                            }`}
                            type="button"
                          >
                            {isIdentifyingSpeakers ? (
                              <>
                                <span className="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full" />
                                Identifying Characters...
                              </>
                            ) : (
                              <>
                                <Users className="h-3.5 w-3.5 shrink-0" />
                                <span>AI Assisted Character Tagging</span>
                              </>
                            )}
                          </button>
                          
                          <button
                            onClick={() => setShowSoundTagger(prev => !prev)}
                            disabled={!podcastText.trim()}
                            className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold text-white transition flex items-center justify-center gap-1.5 ${
                              !podcastText.trim()
                                ? "bg-emerald-300 dark:bg-emerald-900/30 cursor-not-allowed text-emerald-100"
                                : showSoundTagger
                                ? "bg-emerald-700 ring-2 ring-emerald-500 shadow-inner"
                                : "bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98]"
                            }`}
                            type="button"
                          >
                            <Music className="h-3.5 w-3.5 shrink-0" />
                            <span>AI Assisted Sound/Music</span>
                          </button>
                        </div>
                        
                        {showSoundTagger && (
                          <div className="bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-100 dark:border-emerald-900/30 rounded-xl p-3.5 space-y-3.5 animate-fadeIn">
                            <div>
                              <label className="block text-[10px] font-bold text-emerald-800 dark:text-emerald-400 uppercase tracking-wider mb-1.5">
                                Sound / Music Directive Prompt
                              </label>
                              <textarea
                                value={soundTaggerPrompt}
                                onChange={(e) => setSoundTaggerPrompt(e.target.value)}
                                placeholder="e.g. 1990s b-film vibe including music and atmospheric background sounds, and some minor sound effects"
                                className="w-full p-2.5 border border-emerald-200 dark:border-emerald-900 rounded-lg text-xs bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-emerald-400 font-semibold"
                                rows={2}
                              />
                            </div>
                            <div className="flex justify-end">
                              <button
                                onClick={autoTagSoundWithAI}
                                disabled={isAutoTaggingSound || !soundTaggerPrompt.trim()}
                                className={`py-1.5 px-4 rounded-lg text-xs font-bold text-white transition flex items-center gap-1.5 ${
                                  isAutoTaggingSound || !soundTaggerPrompt.trim()
                                    ? "bg-emerald-400 cursor-not-allowed"
                                    : "bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98]"
                                }`}
                                type="button"
                              >
                                {isAutoTaggingSound ? (
                                  <>
                                    <span className="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full" />
                                    Analyzing & Tagging Sound...
                                  </>
                                ) : (
                                  <>
                                    <Sparkles className="h-3.5 w-3.5 shrink-0" />
                                    Run Sound Auto-Tagger
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                {/* Right/Script Editor Column (7 cols) */}
                <div className="lg:col-span-7 space-y-6 bg-white dark:bg-slate-900 p-6 rounded-2xl shadow-md border border-gray-100 dark:border-slate-800 text-gray-900 dark:text-slate-100">
                  {/* Script Editor */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="block text-sm font-semibold text-gray-800 dark:text-slate-100 flex items-center gap-1.5">
                        <FileEdit className="h-4 w-4 text-indigo-500 shrink-0" />
                        {mediaFormat === "podcast" ? "Podcast" : mediaFormat === "audiobook" ? "Audio Book" : "Video"} Script Editor
                      </label>
                      <div className="flex items-center gap-2">
                        <label className="px-2.5 py-1 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-950/20 rounded-lg transition flex items-center gap-1 border border-indigo-100 dark:border-indigo-900/30 cursor-pointer">
                          <Upload className="h-3 w-3 shrink-0" />
                          Import DOCX/Text/MD
                          <input
                            type="file"
                            accept=".docx,.txt,.md,.markdown"
                            className="hidden"
                            onChange={handleDocumentImport}
                          />
                        </label>
                        
                        <button
                          onClick={clearSpeakerTags}
                          disabled={!podcastText.trim()}
                          className="px-2.5 py-1 text-[10px] font-bold text-red-655 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition flex items-center gap-1 border border-red-100 dark:border-red-900/30 disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Remove all speaker tags from the script"
                          type="button"
                        >
                          <Trash2 className="h-3 w-3" />
                          Clear Speaker Tags
                        </button>

                        {scriptHistory.length > 0 && (
                          <button
                            onClick={undoScriptChange}
                            className="px-2.5 py-1 text-[10px] font-bold text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition flex items-center gap-1 border border-amber-100 dark:border-amber-900/30"
                            title="Undo the last script change"
                            type="button"
                          >
                            <Undo className="h-3 w-3" />
                            Undo Change ({scriptHistory.length})
                          </button>
                        )}

                        <button
                          onClick={() => exportMarkdownScript("chapter")}
                          disabled={!podcastText.trim()}
                          className="px-2.5 py-1 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-950/20 rounded-lg transition flex items-center gap-1 border border-indigo-100 dark:border-indigo-900/30 disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Export this chapter's script as markdown"
                          type="button"
                        >
                          <Download className="h-3 w-3" />
                          Export Script (.md)
                        </button>
                      </div>
                    </div>
                    <div className={`enhance-input-container${isEnhancing || isScriptEditorBusy ? " dimmed" : ""}`}>
                      <TagEditor
                        ref={tagEditorRef}
                        value={podcastText}
                        onChange={setPodcastText}
                        tokens={podcastEditorTokens}
                        speakerColorsMap={speakerColorsMap}
                        onCaretChange={handleCaretChange}
                        disabled={isScriptEditorBusy}
                        isLoading={isScriptEditorBusy}
                        loadingMessage={scriptEditorProcessName}
                        placeholder="Paste or write your script here... Use speaker tags like [Speaker 1] to set voicing."
                        className="w-full p-4 border border-gray-200 dark:border-slate-800 rounded-xl focus:ring-2 focus:ring-blue-500 min-h-[8rem] bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-100"
                      />
                    </div>
                
                {/* Visual Dialogue Flow */}
                <div className="mt-4 space-y-3">
                  {(() => {
                    const allSegments = parseScriptTextToSegments(podcastText, numberOfSpeakers, speakerNames);
                    const hasLongSegments = allSegments.some(seg => !seg.isPause && seg.text.split(/\s+/).filter(Boolean).length > 150);
                    return (
                      <div className="flex justify-between items-center mb-1">
                        <h4 className="text-xs font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider">Visual Dialogue Flow</h4>
                        {hasLongSegments && (
                          <button
                            onClick={autoChunkAllSegments}
                            className="px-2 py-0.5 text-[9px] font-bold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-955/20 hover:bg-amber-100 dark:hover:bg-amber-900/30 rounded border border-amber-200 dark:border-amber-900/40 transition flex items-center gap-1"
                            type="button"
                          >
                            <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0 animate-pulse" />
                            Auto-Chunk All Long Clips
                          </button>
                        )}
                      </div>
                    );
                  })()}
                  <div className="space-y-2 max-h-[300px] overflow-y-auto p-2 bg-gray-50/50 dark:bg-slate-950/40 rounded-xl border border-gray-100 dark:border-slate-800/80">
                    {parseScriptTextToSegments(podcastText, numberOfSpeakers, speakerNames).map((seg, sIdx) => {
                      if (!seg.text.trim()) return null;
                      const spkColor = seg.speakerKey ? (speakerColors[seg.speakerKey] || "#4f46e5") : "#94a3b8";
                      const speakerLabel = seg.speakerKey ? (speakerNames[seg.speakerKey] || "Speaker " + seg.speakerKey.split("_")[1]) : (seg.isPause ? "Pause" : "Unassigned");
                      const wordCount = seg.text.split(/\s+/).filter(Boolean).length;
                      const showWarning = !seg.isPause && wordCount > 150;
                      
                      return (
                        <div 
                          key={sIdx} 
                          className="p-3 rounded-lg border text-xs transition duration-200"
                          style={{
                            borderColor: showWarning ? "#f59e0b" : spkColor,
                            backgroundColor: seg.isPause ? "rgba(217, 119, 6, 0.05)" : (showWarning ? "rgba(245, 158, 11, 0.03)" : spkColor + "0a"),
                            borderLeftWidth: "4px"
                          }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-extrabold" style={{ color: showWarning ? "#f59e0b" : (seg.isPause ? "#d97706" : spkColor) }}>
                              {speakerLabel} {showWarning && <span className="text-[9px] font-bold text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-955 px-1 py-0.5 rounded ml-1">Warning: Long Clip</span>}
                            </span>
                            {seg.isPause && (
                              <span className="text-[10px] bg-amber-100 dark:bg-amber-955 text-amber-800 dark:text-amber-300 font-bold px-1.5 py-0.5 rounded">
                                {seg.duration}s
                              </span>
                            )}
                          </div>
                          <p className="text-gray-750 dark:text-slate-200 font-medium leading-relaxed">
                            {seg.text}
                          </p>
                          {showWarning && (
                            <div className="mt-2 p-2 bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200/50 dark:border-amber-900/30 rounded-lg flex items-center justify-between gap-3 text-[10px] text-amber-800 dark:text-amber-300 leading-normal">
                              <span className="flex items-center gap-1.5 font-semibold">
                                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 animate-pulse" />
                                <span>Dialogue is long ({wordCount} words). High risk of voice drifting.</span>
                              </span>
                              <button
                                onClick={() => autoChunkSegment(seg)}
                                className="px-2 py-0.5 bg-amber-100 hover:bg-amber-200 dark:bg-amber-900/60 dark:hover:bg-amber-900/80 text-[10px] font-bold rounded transition border border-amber-300 dark:border-amber-800 shrink-0 text-amber-800 dark:text-amber-200"
                                type="button"
                              >
                                Auto-Chunk
                              </button>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Active Speaker Status Helper Bar */}
                {(() => {
                  const num = activeSpeakerKey.split("_")[1];
                  const customName = speakerNames[activeSpeakerKey] || `Speaker ${num}`;
                  
                  // Get model name
                  const assignedVoiceId = speakerMapping[activeSpeakerKey];
                  let modelName = "VITS (Default)";
                  let hasTags = false;
                  let tagsList = [];
                  
                  if (assignedVoiceId) {
                    if (assignedVoiceId.startsWith("curated:")) {
                      const curatedId = assignedVoiceId.split(":")[1];
                      const curated = curatedVoices.find(v => v.id === curatedId);
                      if (curated) {
                        modelName = `${curated.model.toUpperCase()} (Curated: ${curated.name})`;
                        const voiceData = voices.find(v => v.name === curated.model);
                        if (voiceData && voiceData.features?.includes("tags")) {
                          hasTags = true;
                          tagsList = voiceData.tokens || [];
                        }
                      }
                    } else if (assignedVoiceId.includes(":")) {
                      const modelKey = assignedVoiceId.split(":")[0];
                      modelName = modelKey.toUpperCase();
                      const voiceData = voices.find(v => v.name === modelKey);
                      if (voiceData && voiceData.features?.includes("tags")) {
                        hasTags = true;
                        tagsList = voiceData.tokens || [];
                      }
                    }
                  }
                  
                  return (
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 bg-purple-50/50 border border-purple-100/70 rounded-xl text-xs">
                      <div className="flex items-center flex-wrap gap-x-2 gap-y-1">
                        <span className="font-semibold text-purple-900 inline-flex items-center gap-1">
                          <Mic className="h-3.5 w-3.5 text-purple-700" />
                          <span>Active Line Speaker:</span>
                        </span>{" "}
                        <span className="font-extrabold text-indigo-700">[{customName}]</span>{" "}
                        <span className="text-[10px] text-gray-500 font-medium">({modelName})</span>
                      </div>
                      {hasTags && tagsList.length > 0 ? (
                        <div className="text-[10px] text-purple-700 flex flex-wrap gap-1 items-center max-w-[400px]">
                          <span className="font-bold text-purple-900">Allowed tags:</span>
                          {tagsList.map(t => (
                            <button
                              key={t}
                              type="button"
                              onClick={() => {
                                if (tagEditorRef.current) {
                                  const cleanTag = t.replace(/^\[|\]$/g, "");
                                  tagEditorRef.current.insertToken(cleanTag);
                                }
                              }}
                              className="px-1.5 py-0.5 bg-purple-100 hover:bg-purple-200 text-purple-800 rounded font-semibold transition"
                            >
                              {t}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <div className="text-[10px] text-gray-400 italic">No vocalization tags supported by this model.</div>
                      )}
                    </div>
                  );
                })()}

                {/* Speaker Tag insertion shortcuts */}
                <div className="flex flex-wrap items-center gap-1.5 p-2 bg-gray-50 border border-gray-100 rounded-xl">
                  <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mr-1">Insert Tag:</span>
                  {Array.from({ length: numberOfSpeakers }, (_, i) => i + 1).map((num) => {
                    const spkKey = `speaker_${num}`;
                    const customName = speakerNames[spkKey] || `Speaker ${num}`;
                    return (
                      <button
                        key={num}
                        type="button"
                        onClick={() => {
                          if (tagEditorRef.current) {
                            tagEditorRef.current.insertToken(customName);
                          } else {
                            setPodcastText(prev => prev + `[${customName}] `);
                          }
                        }}
                        className="px-2.5 py-1 text-xs font-bold bg-white text-indigo-600 border border-indigo-200 rounded-lg hover:border-indigo-400 hover:bg-indigo-50 transition shadow-sm active:scale-95 flex items-center gap-1"
                        title={`Insert [${customName}] tag`}
                      >
                        <span>[{customName}]</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Sync and View Multitrack Button */}
              <div className="pt-2">
                <button
                  onClick={() => {
                    syncScriptToTimeline();
                    setChapterEditorTab("multitrack");
                  }}
                  className="w-full py-2.5 px-4 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-md active:scale-[0.98] transition flex items-center justify-center gap-1.5"
                  type="button"
                >
                  <Sliders className="h-4 w-4" />
                  View Multitrack
                </button>
              </div>
            </div>
          </div>
            </div>
        )}
      {chapterEditorTab === "multitrack" && (
        <div className="space-y-6">
          {renderMultitrackTimeline()}
        </div>
      )}
      {chapterEditorTab === "reviewer" && (
        <div className="space-y-6 animate-fadeIn">
          {/* Header Banner & Obsidian Export Actions */}
          <div className="bg-gradient-to-r from-indigo-900 via-slate-900 to-purple-950 p-6 rounded-2xl border border-indigo-500/20 shadow-xl text-white flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-indigo-400" />
                <h3 className="text-lg font-extrabold tracking-tight">Audio & Text Reviewer</h3>
                <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-indigo-500/30 border border-indigo-400/40 text-indigo-200">ADHD Proofing Workspace</span>
              </div>
              <p className="text-xs text-indigo-200/80 mt-1 max-w-2xl">
                Listen to exported chapter audio, mark timecoded text changes you hear, request AI rewrites, and export revised notes directly back to your Obsidian Vault.
              </p>
            </div>
            
            <div className="flex items-center gap-2 shrink-0 flex-wrap">
              <button
                type="button"
                onClick={() => publishToObsidian("clean")}
                className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs rounded-xl transition shadow-md flex items-center gap-1.5 border border-indigo-400/40"
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span>Export Clean to Obsidian</span>
              </button>
              <button
                type="button"
                onClick={() => publishToObsidian("commented")}
                className="px-3.5 py-2 bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs rounded-xl transition shadow-md flex items-center gap-1.5 border border-purple-400/40"
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>Export Commented (%%)</span>
              </button>
              <button
                type="button"
                onClick={loadActiveChapterMix}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs rounded-xl transition border border-slate-700 flex items-center gap-1.5"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Reload Mix</span>
              </button>
            </div>
          </div>

          {/* 2-Column Responsive Review Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* LEFT / MAIN PANEL: Interactive Audio Scrubber & Add Markup Form (7 cols) */}
            <div className="lg:col-span-7 space-y-6">
              {/* Interactive Audio Player & Scrubber with Pins */}
              <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-extrabold text-gray-800 dark:text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <Volume2 className="h-4 w-4 text-indigo-500" />
                    <span>Chapter Master Audio Playback</span>
                  </label>
                  <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/50 px-2.5 py-1 rounded-lg border border-indigo-100 dark:border-indigo-900/40">
                    {Math.floor(reviewCurrentTime / 60)}:{(reviewCurrentTime % 60).toFixed(1).padStart(4, "0")} / {Math.floor(reviewAudioDuration / 60)}:{(reviewAudioDuration % 60).toFixed(1).padStart(4, "0")}
                  </span>
                </div>

                {activeReviewAudioUrl ? (
                  <>
                    <audio
                      ref={reviewAudioRef}
                      src={activeReviewAudioUrl}
                      onTimeUpdate={(e) => setReviewCurrentTime(e.target.currentTime)}
                      onLoadedMetadata={(e) => setReviewAudioDuration(e.target.duration)}
                      onEnded={() => setReviewCurrentTime(0)}
                      className="hidden"
                    />

                    {/* Custom Timeline Scrubber & Pin Markers Bar */}
                    <div className="space-y-1.5 pt-2">
                      <div 
                        className="relative w-full h-8 bg-gray-100 dark:bg-slate-950 rounded-xl border border-gray-200 dark:border-slate-800 cursor-pointer overflow-hidden group shadow-inner"
                        onClick={(e) => {
                          if (reviewAudioRef.current && reviewAudioDuration > 0) {
                            const rect = e.currentTarget.getBoundingClientRect();
                            const pos = (e.clientX - rect.left) / rect.width;
                            const newTime = pos * reviewAudioDuration;
                            reviewAudioRef.current.currentTime = newTime;
                            setReviewCurrentTime(newTime);
                          }
                        }}
                      >
                        {/* Progress Fill */}
                        <div 
                          className="absolute inset-y-0 left-0 bg-indigo-500/20 dark:bg-indigo-600/30 border-r-2 border-indigo-500 transition-all duration-75"
                          style={{ width: `${reviewAudioDuration > 0 ? (reviewCurrentTime / reviewAudioDuration) * 100 : 0}%` }}
                        />

                        {/* Timecoded Pin Markers */}
                        {reviewNotes.map((note) => {
                          const pinPos = reviewAudioDuration > 0 ? (note.timecode / reviewAudioDuration) * 100 : 0;
                          return (
                            <div
                              key={note.id}
                              className="absolute top-0 bottom-0 w-1.5 z-20 group/pin cursor-pointer hover:w-3 transition-all"
                              style={{ left: `${pinPos}%` }}
                              onClick={(e) => {
                                e.stopPropagation();
                                if (reviewAudioRef.current) {
                                  reviewAudioRef.current.currentTime = note.timecode;
                                  reviewAudioRef.current.play().catch(() => {});
                                }
                              }}
                              title={`[${Math.floor(note.timecode / 60)}:${(note.timecode % 60).toFixed(1).padStart(4, "0")}] ${note.text}`}
                            >
                              <div className={`w-full h-full ${note.status === "approved" ? "bg-emerald-500" : note.status === "rejected" ? "bg-gray-400" : "bg-purple-600 animate-pulse"}`} />
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex justify-between text-[10px] text-gray-400 font-bold px-1">
                        <span>0:00</span>
                        <span>Click timeline to jump audio playback</span>
                        <span>{Math.floor(reviewAudioDuration / 60)}:{(reviewAudioDuration % 60).toFixed(0).padStart(2, "0")}</span>
                      </div>
                    </div>

                    {/* Transport Controls */}
                    <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-slate-800 flex-wrap gap-2">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            if (reviewAudioRef.current) {
                              if (reviewAudioRef.current.paused) {
                                reviewAudioRef.current.play().catch(() => {});
                              } else {
                                reviewAudioRef.current.pause();
                              }
                            }
                          }}
                          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-xl shadow transition flex items-center gap-1.5"
                        >
                          <Play className="h-3.5 w-3.5 fill-current" />
                          <span>Play / Pause</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (reviewAudioRef.current) {
                              reviewAudioRef.current.currentTime = Math.max(0, reviewAudioRef.current.currentTime - 5);
                            }
                          }}
                          className="px-3 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-200 font-bold text-xs rounded-xl transition"
                        >
                          -5s
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (reviewAudioRef.current) {
                              reviewAudioRef.current.currentTime = Math.min(reviewAudioDuration, reviewAudioRef.current.currentTime + 5);
                            }
                          }}
                          className="px-3 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-200 font-bold text-xs rounded-xl transition"
                        >
                          +5s
                        </button>
                      </div>

                      {/* Speed Control */}
                      <div className="flex items-center gap-1.5 text-xs font-bold text-gray-500">
                        <span>Speed:</span>
                        {[0.75, 1.0, 1.25, 1.5].map((spd) => (
                          <button
                            key={spd}
                            type="button"
                            onClick={() => {
                              setReviewPlaybackRate(spd);
                              if (reviewAudioRef.current) reviewAudioRef.current.playbackRate = spd;
                            }}
                            className={`px-2 py-1 rounded-lg text-[10px] font-bold transition ${
                              reviewPlaybackRate === spd
                                ? "bg-indigo-600 text-white"
                                : "bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-400 hover:bg-gray-200"
                            }`}
                          >
                            {spd}x
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 bg-gray-50 dark:bg-slate-950 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl space-y-3">
                    <Volume2 className="h-8 w-8 text-gray-300 dark:text-slate-700 mx-auto" />
                    <div className="text-xs text-gray-500 dark:text-slate-400 font-medium">
                      No master chapter audio active. Click <strong>"Reload Mix"</strong> or export a timeline mixdown first.
                    </div>
                    <button
                      type="button"
                      onClick={loadActiveChapterMix}
                      className="px-3.5 py-1.5 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-300 font-bold text-xs rounded-lg hover:bg-indigo-100 transition border border-indigo-200 dark:border-indigo-900/30"
                    >
                      Load Active Chapter Audio
                    </button>
                  </div>
                )}
              </div>

              {/* Add New Markup / Note Form */}
              <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md space-y-4">
                <div className="flex justify-between items-center border-b border-gray-100 dark:border-slate-800 pb-3">
                  <h4 className="text-xs font-bold text-gray-800 dark:text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <Edit3 className="h-4 w-4 text-purple-500" />
                    <span>Add Timecoded Note / Correction Markup</span>
                  </h4>
                  <span className="text-[11px] font-bold text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-950/50 px-2 py-0.5 rounded border border-purple-200 dark:border-purple-900/40">
                    Timestamp: {Math.floor(reviewCurrentTime / 60)}:{(reviewCurrentTime % 60).toFixed(1).padStart(4, "0")}
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="block text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mb-1">
                      Target Text / Mispronounced Phrase
                    </label>
                    <input
                      type="text"
                      value={newNoteTarget}
                      onChange={(e) => {
                        setNewNoteTarget(e.target.value);
                        if (autoPauseOnType && reviewAudioRef.current && !reviewAudioRef.current.paused) {
                          reviewAudioRef.current.pause();
                        }
                      }}
                      placeholder="e.g. COP or Dr. Smith"
                      className="w-full p-3 text-xs border border-gray-200 dark:border-slate-800 rounded-xl bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-slate-100 focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mb-1">
                      Correction Note / Desired Pronunciation
                    </label>
                    <textarea
                      rows="3"
                      value={newNoteText}
                      onChange={(e) => {
                        setNewNoteText(e.target.value);
                        if (autoPauseOnType && reviewAudioRef.current && !reviewAudioRef.current.paused) {
                          reviewAudioRef.current.pause();
                        }
                      }}
                      placeholder="e.g. Pronounce as C.O.P. letter by letter, or change to Doctor Smith"
                      className="w-full p-3 text-xs border border-gray-200 dark:border-slate-800 rounded-xl bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-slate-100 focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div className="flex justify-between items-center pt-2">
                    <label className="flex items-center gap-2 text-xs font-semibold text-gray-600 dark:text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoPauseOnType}
                        onChange={(e) => setAutoPauseOnType(e.target.checked)}
                        className="rounded text-purple-600 focus:ring-purple-500"
                      />
                      <span>Auto-pause audio when typing note</span>
                    </label>

                    <button
                      type="button"
                      disabled={!newNoteText.trim()}
                      onClick={() => {
                        addReviewNote(newNoteText, newNoteTarget);
                        setNewNoteText("");
                        setNewNoteTarget("");
                      }}
                      className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow transition flex items-center gap-1.5"
                    >
                      <Plus className="h-3.5 w-3.5" />
                      <span>Add Note to Timeline</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT PANEL: Scrollable Comment Feed & AI Rewrites (5 cols) */}
            <div className="lg:col-span-5 space-y-6">
              <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-gray-100 dark:border-slate-800 shadow-md space-y-4">
                <div className="flex justify-between items-center border-b border-gray-100 dark:border-slate-800 pb-3">
                  <h4 className="text-xs font-bold text-gray-800 dark:text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-indigo-500" />
                    <span>Timecoded Notes Feed ({reviewNotes.length})</span>
                  </h4>
                  <span className="text-[10px] font-bold text-gray-400">Click timecode to seek</span>
                </div>

                {reviewNotes.length === 0 ? (
                  <div className="text-center py-12 bg-gray-50 dark:bg-slate-950 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl space-y-2">
                    <MessageSquare className="h-8 w-8 text-gray-300 dark:text-slate-700 mx-auto" />
                    <div className="text-xs text-gray-500 dark:text-slate-400 font-medium">
                      No proofing notes added yet.
                    </div>
                    <div className="text-[10px] text-gray-400">
                      Listen to audio on the left and submit correction notes to mark timecoded changes.
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4 max-h-[600px] overflow-y-auto pr-1">
                    {reviewNotes.map((note) => (
                      <div
                        key={note.id}
                        className={`p-4 rounded-xl border transition space-y-3 ${
                          note.status === "approved"
                            ? "bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/40"
                            : note.status === "rejected"
                            ? "bg-gray-50 dark:bg-slate-950 border-gray-200 dark:border-slate-800 opacity-60"
                            : "bg-white dark:bg-slate-950 border-indigo-100 dark:border-slate-800 shadow-sm"
                        }`}
                      >
                        {/* Note Header */}
                        <div className="flex justify-between items-center">
                          <button
                            type="button"
                            onClick={() => {
                              if (reviewAudioRef.current) {
                                reviewAudioRef.current.currentTime = note.timecode;
                                reviewAudioRef.current.play().catch(() => {});
                              }
                            }}
                            className="px-2.5 py-1 bg-indigo-100 hover:bg-indigo-200 dark:bg-indigo-950 dark:hover:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-xs font-bold rounded-lg transition flex items-center gap-1"
                            title="Seek audio to timecode"
                          >
                            <span>⏱️ {Math.floor(note.timecode / 60)}:{(note.timecode % 60).toFixed(1).padStart(4, "0")}</span>
                          </button>

                          <div className="flex items-center gap-2">
                            <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded ${
                              note.status === "approved"
                                ? "bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300"
                                : "bg-purple-100 dark:bg-purple-950 text-purple-700 dark:text-purple-300"
                            }`}
                            >
                              {note.status}
                            </span>
                            <button
                              type="button"
                              onClick={() => deleteReviewNote(note.id)}
                              className="text-gray-400 hover:text-red-500 transition text-xs font-bold px-1.5 py-0.5"
                              title="Delete note"
                            >
                              ✕
                            </button>
                          </div>
                        </div>

                        <div className="text-xs text-gray-800 dark:text-slate-200 font-medium leading-relaxed">
                          {note.text}
                        </div>

                        {note.targetPara && (
                          <div className="text-[11px] text-gray-600 dark:text-slate-400 bg-gray-50 dark:bg-slate-900 p-2.5 rounded-lg border border-gray-100 dark:border-slate-800 font-mono">
                            Target: <span className="font-bold text-indigo-600 dark:text-indigo-400">"{note.targetPara}"</span>
                          </div>
                        )}

                        {/* AI Rewrite Result / Trigger */}
                        {note.suggestion ? (
                          <div className="p-3 bg-purple-50/60 dark:bg-purple-950/30 border border-purple-200/60 dark:border-purple-900/40 rounded-xl space-y-2">
                            <div className="text-[10px] font-bold text-purple-800 dark:text-purple-300 uppercase tracking-wider flex items-center gap-1">
                              <Sparkles className="h-3 w-3 text-purple-500" />
                              <span>Suggested AI Revision:</span>
                            </div>
                            <div className="text-xs font-mono text-gray-900 dark:text-slate-100 font-bold bg-white dark:bg-slate-900 p-2 rounded-lg border border-purple-100 dark:border-purple-900/30">
                              {note.suggestion}
                            </div>
                            <div className="flex gap-2 pt-1">
                              <button
                                type="button"
                                onClick={() => approveRewrite(note.id)}
                                className="flex-1 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-[11px] rounded-lg transition shadow-sm flex items-center justify-center gap-1"
                              >
                                <Check className="h-3.5 w-3.5" /> Accept Change
                              </button>
                              <button
                                type="button"
                                onClick={() => rejectRewrite(note.id)}
                                className="px-3 py-1.5 border border-gray-200 dark:border-slate-800 text-gray-600 dark:text-slate-400 text-[11px] font-bold rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 transition"
                              >
                                Dismiss
                              </button>
                            </div>
                          </div>
                        ) : (
                          note.targetPara && (
                            <button
                              type="button"
                              onClick={() => requestLlmRewrite(note.id)}
                              disabled={isSuggestingEdit}
                              className="w-full py-2 bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-950/40 dark:hover:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 font-bold text-xs rounded-xl transition border border-indigo-200 dark:border-indigo-900/30 flex items-center justify-center gap-1.5 disabled:opacity-50"
                            >
                              <Sparkles className="h-3.5 w-3.5 text-indigo-500" />
                              <span>{isSuggestingEdit ? "Generating AI Rewrite..." : "Suggest AI Rewrite"}</span>
                            </button>
                          )
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )}





          {/* Dialog Clip Inspector Modal */}
          {isClipModalOpen && selectedTimelineClip && (
            <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="bg-white dark:bg-gray-900 rounded-3xl p-6 w-full max-w-lg shadow-2xl border border-gray-200 transform scale-100 transition-all flex flex-col space-y-4">
                
                <div className="flex justify-between items-center pb-2 border-b border-gray-100 dark:border-gray-800">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    🔧 Edit Clip: {selectedTimelineClip.id.substring(0, 8)}
                  </h3>
                  <button
                    onClick={() => setIsClipModalOpen(false)}
                    className="text-gray-400 hover:text-gray-650 text-lg font-bold"
                  >
                    ✕
                  </button>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center bg-gray-50 dark:bg-gray-800 p-2.5 rounded-xl text-xs font-medium text-gray-600 dark:text-gray-300">
                    <span>Track Lane: <strong className="text-purple-600">{selectedTimelineClip.trackId}</strong></span>
                    <span>Status: <strong className="uppercase">{selectedTimelineClip.status}</strong></span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                        Timeline Start Offset (s)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        className="w-full p-2 border border-gray-200 rounded-xl text-xs focus:ring-1 focus:ring-purple-400 bg-white text-gray-900"
                        value={selectedTimelineClip.startTime}
                        onChange={(e) => {
                          const val = Math.max(0, parseFloat(e.target.value) || 0);
                          setSelectedTimelineClip(prev => ({ ...prev, startTime: val }));
                          setPlaylistClips(prev => {
                            const updated = prev.map(c => c.id === selectedTimelineClip.id ? { ...c, startTime: val, manuallyMoved: true } : c);
                            return reflowClips(updated);
                          });
                        }}
                      />
                    </div>

                    <div>
                      <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                        Clip Duration (s)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min="0.5"
                        disabled={selectedTimelineClip.status === "done"}
                        className="w-full p-2 border border-gray-200 rounded-xl text-xs bg-gray-50 text-gray-500"
                        value={selectedTimelineClip.duration}
                        onChange={(e) => {
                          const val = Math.max(0.5, parseFloat(e.target.value) || 2.0);
                          setSelectedTimelineClip(prev => ({ ...prev, duration: val }));
                          setPlaylistClips(prev => {
                            const updated = prev.map(c => c.id === selectedTimelineClip.id ? { ...c, duration: val } : c);
                            return reflowClips(updated);
                          });
                        }}
                      />
                    </div>
                  </div>

                  {selectedTimelineClip.trackId.startsWith("speaker_") && (
                    <>
                      <div>
                        <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                          Dialogue Text
                        </label>
                        <textarea
                          rows="4"
                          className="w-full p-3 border border-gray-200 rounded-xl text-xs focus:ring-1 focus:ring-purple-400 bg-white text-gray-900"
                          value={selectedTimelineClip.text}
                          onChange={(e) => {
                            const val = e.target.value;
                            setSelectedTimelineClip(prev => ({ ...prev, text: val }));
                            setPlaylistClips(prev => prev.map(c => c.id === selectedTimelineClip.id ? { ...c, text: val } : c));
                          }}
                        />
                      </div>

                      <div>
                        <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                          Voice Direction (e.g. whispering, excitedly)
                        </label>
                        <input
                          type="text"
                          className="w-full p-2 border border-gray-200 rounded-xl text-xs focus:ring-1 focus:ring-purple-400 bg-white text-gray-900"
                          value={selectedTimelineClip.voiceDirection}
                          placeholder="Leave blank for default tone"
                          onChange={(e) => {
                            const val = e.target.value;
                            setSelectedTimelineClip(prev => ({ ...prev, voiceDirection: val }));
                            setPlaylistClips(prev => prev.map(c => c.id === selectedTimelineClip.id ? { ...c, voiceDirection: val } : c));
                          }}
                        />
                      </div>
                    </>
                  )}

                  {selectedTimelineClip.trackId === "music" && (
                    <div>
                      <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                        Music Preset Track
                      </label>
                      <select
                        className="w-full p-2 border border-gray-200 rounded-xl text-xs bg-white text-gray-900"
                        value={selectedTimelineClip.musicKey}
                        onChange={(e) => {
                          const val = e.target.value;
                          const newUrl = `http://localhost:5000/audio/music_${val}`;
                          setSelectedTimelineClip(prev => ({ ...prev, musicKey: val, text: `Music: ${val}`, audioUrl: newUrl }));
                          setPlaylistClips(prev => prev.map(c => c.id === selectedTimelineClip.id ? { ...c, musicKey: val, text: `Music: ${val}`, audioUrl: newUrl } : c));
                          if (audioBuffersCache.current[selectedTimelineClip.id]) {
                            delete audioBuffersCache.current[selectedTimelineClip.id];
                          }
                        }}
                      >
                        <option value="lofi">Ambient Lo-Fi</option>
                        <option value="intro">Tech Talk Intro</option>
                        <option value="suspense">Dramatic Suspense</option>
                        <option value="acoustic">Happy Acoustic</option>
                      </select>
                    </div>
                  )}

                  {selectedTimelineClip.trackId === "sfx" && (
                    <div>
                      <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">
                        SFX Cue Preset
                      </label>
                      <select
                        className="w-full p-2 border border-gray-200 rounded-xl text-xs bg-white text-gray-900"
                        value={selectedTimelineClip.sfxKey}
                        onChange={(e) => {
                          const val = e.target.value;
                          const newUrl = `http://localhost:5000/audio/sfx_${val}`;
                          setSelectedTimelineClip(prev => ({ ...prev, sfxKey: val, text: `SFX: ${val}`, audioUrl: newUrl }));
                          setPlaylistClips(prev => prev.map(c => c.id === selectedTimelineClip.id ? { ...c, sfxKey: val, text: `SFX: ${val}`, audioUrl: newUrl } : c));
                          if (audioBuffersCache.current[selectedTimelineClip.id]) {
                            delete audioBuffersCache.current[selectedTimelineClip.id];
                          }
                        }}
                      >
                        <option value="phone">Phone Ring</option>
                        <option value="applause">Applause</option>
                        <option value="jazz">Jazz Chord</option>
                        <option value="scratch">Record Scratch</option>
                        <option value="cafe">Cafe Ambience</option>
                        <option value="birds">Birds Chirping</option>
                      </select>
                    </div>
                  )}
                </div>

                <div className="flex justify-between items-center pt-4 border-t border-gray-100 dark:border-gray-800">
                  <button
                    onClick={() => {
                      setPlaylistClips(prev => prev.filter(c => c.id !== selectedTimelineClip.id));
                      if (audioBuffersCache.current[selectedTimelineClip.id]) {
                        delete audioBuffersCache.current[selectedTimelineClip.id];
                      }
                      setIsClipModalOpen(false);
                    }}
                    className="bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 font-bold text-xs py-2 px-4 rounded-xl transition flex items-center justify-center gap-1.5"
                    type="button"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete Clip
                  </button>

                  <div className="flex gap-2">
                    {selectedTimelineClip.trackId.startsWith("speaker_") && (
                      <button
                        onClick={() => {
                          generateClipAudio(selectedTimelineClip.id);
                          setIsClipModalOpen(false);
                        }}
                        className="bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs py-2 px-4 rounded-xl shadow transition"
                        type="button"
                      >
                        🔄 Re-generate Clip
                      </button>
                    )}
                    
                    <button
                      onClick={() => setIsClipModalOpen(false)}
                      className="bg-gray-100 hover:bg-gray-200 border border-gray-200 text-gray-700 font-bold text-xs py-2 px-4 rounded-xl transition"
                      type="button"
                    >
                      Save & Close
                    </button>
                  </div>
                </div>

              </div>
            </div>
          )}
        </div>
      )}

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-white dark:bg-slate-900 w-[700px] max-w-full rounded-2xl border border-gray-200 dark:border-slate-800 shadow-2xl flex relative text-gray-900 dark:text-slate-100 overflow-hidden">
            <button
              onClick={() => setShowSettings(false)}
              className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 transition"
              title="Close"
            >
              ✕
            </button>

            {/* Sidebar */}
            <div className="w-1/4 bg-gray-50 dark:bg-slate-950 p-4 border-r border-gray-200 dark:border-slate-800 flex flex-col justify-between">
              <div>
                <h3 className="text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider mb-4">
                  Voication
                </h3>
                <ul className="space-y-2">
                  <li>
                    <button
                      onClick={() => setSettingsTab("general")}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition ${
                        settingsTab === "general"
                          ? "bg-white dark:bg-slate-900 shadow-sm text-gray-950 dark:text-white"
                          : "text-gray-500 dark:text-slate-400 hover:bg-gray-200/50 dark:hover:bg-slate-800/40"
                      }`}
                    >
                      General
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => setSettingsTab("ollama")}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition ${
                        settingsTab === "ollama"
                          ? "bg-white dark:bg-slate-900 shadow-sm text-gray-950 dark:text-white"
                          : "text-gray-500 dark:text-slate-400 hover:bg-gray-200/50 dark:hover:bg-slate-800/40"
                      }`}
                    >
                      AI Preprocessing
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => setSettingsTab("cloud")}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition ${
                        settingsTab === "cloud"
                          ? "bg-white dark:bg-slate-900 shadow-sm text-gray-950 dark:text-white"
                          : "text-gray-500 dark:text-slate-400 hover:bg-gray-200/50 dark:hover:bg-slate-800/40"
                      }`}
                    >
                      Cloud Sync (Drive)
                    </button>
                  </li>
                </ul>
                <h3 className="text-[10px] font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider my-4">
                  Voice Models
                </h3>
                <ul className="space-y-2">
                  <li>
                    <button
                      onClick={() => setSettingsTab("bark")}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition ${
                        settingsTab === "bark"
                          ? "bg-white dark:bg-slate-900 shadow-sm text-gray-950 dark:text-white"
                          : "text-gray-500 dark:text-slate-400 hover:bg-gray-200/50 dark:hover:bg-slate-800/40"
                      }`}
                    >
                      Bark Settings
                    </button>
                  </li>
                </ul>
              </div>
            </div>

            {/* Content Area */}
            <div className="w-3/4 p-6 overflow-y-auto max-h-[500px]">
              
              {/* General Settings Tab */}
              {settingsTab === "general" && (
                <>
                  <h2 className="text-lg font-bold text-gray-800 dark:text-white mb-4">General Settings</h2>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">
                        Device & Acceleration Target
                      </label>
                      <select
                        value={device}
                        onChange={(e) => setDevice(e.target.value)}
                        className="w-full p-2 border border-gray-200 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                      >
                        <option value="auto">Auto-Detect Accelerator</option>
                        <option value="mps">Apple Silicon (MPS)</option>
                        <option value="cuda">NVIDIA CUDA GPU</option>
                        <option value="cpu">CPU Only (No acceleration)</option>
                      </select>
                      <p className="text-[10px] text-gray-500 dark:text-gray-400 mt-1">
                        Apple MPS is recommended for Mac M1/M2/M3. CUDA is best for NVIDIA cards.
                      </p>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">
                        Audio Output Folder
                      </label>
                      <input
                        type="text"
                        value={outputFolder}
                        onChange={(e) => setOutputFolder(e.target.value)}
                        className="w-full p-2 border border-gray-200 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                        placeholder="e.g. output"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">
                        Freesound API Token
                      </label>
                      <input
                        type="password"
                        value={freesoundToken}
                        onChange={(e) => {
                          setFreesoundToken(e.target.value);
                          localStorage.setItem("voication_freesound_token", e.target.value);
                        }}
                        className="w-full p-2 border border-gray-200 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 font-semibold"
                        placeholder="Paste Freesound API Token/Key here..."
                      />
                      <p className="text-[10px] text-gray-500 dark:text-gray-400 mt-1">
                        Provide your Freesound API Client Secret/Token to enable public search and curation of sound effects and music tracks.
                      </p>
                    </div>



                    <label className="flex items-center space-x-2 cursor-pointer pt-2">
                      <input
                        type="checkbox"
                        className="rounded border-gray-300 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                        checked={darkMode}
                        onChange={() => {
                          const updated = !darkMode;
                          setDarkMode(updated);
                          localStorage.setItem(
                            "voicationSetting",
                            JSON.stringify({ darkMode: updated })
                          );
                        }}
                      />
                      <span className="text-sm font-medium text-gray-700 dark:text-slate-300">Enable Dark Mode</span>
                    </label>

                    <div className="pt-4 border-t border-gray-200 dark:border-slate-800 space-y-3">
                      <h4 className="text-xs font-bold text-gray-400 dark:text-slate-500 uppercase tracking-wider">Script & Dialogue Editor Settings</h4>
                      <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                          checked={autoRippleOnSync}
                          onChange={(e) => {
                            const val = e.target.checked;
                            setAutoRippleOnSync(val);
                            localStorage.setItem("voication_auto_ripple", val ? "true" : "false");
                          }}
                        />
                        <span className="text-sm font-medium text-gray-700 dark:text-slate-300">
                          Automatic Ripple Edit on Script Sync (Back-to-Back Clips)
                        </span>
                      </label>
                    </div>

                    <div className="pt-4 border-t border-gray-200 dark:border-slate-800">
                      <button
                        onClick={saveGeneralSettings}
                        className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 transition"
                      >
                        Save General Settings
                      </button>
                    </div>
                  </div>
                </>
              )}

              {/* AI Preprocessing Tab */}
              {settingsTab === "ollama" && (
                <>
                  <h2 className="text-lg font-bold text-gray-800 dark:text-white mb-2 flex items-center">
                    AI Preprocessing (Ollama)
                    <span className="ml-2 bg-amber-100 text-amber-800 dark:bg-amber-955/40 dark:text-amber-400 dark:border dark:border-amber-900/50 text-[10px] font-bold px-2 py-0.5 rounded-full">Experimental</span>
                  </h2>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-4 leading-normal">
                    Specify the local Ollama LLM endpoint to enable the experimental automatic narrative emotional tagging features.
                  </p>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">Ollama API URL</label>
                      <input
                        type="text"
                        value={ollamaUrl}
                        onChange={(e) => setOllamaUrl(e.target.value)}
                        className="w-full p-2 border border-gray-200 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">Ollama Model</label>
                      <input
                        type="text"
                        value={ollamaModel}
                        onChange={(e) => setOllamaModel(e.target.value)}
                        className="w-full p-2 border border-gray-200 dark:border-slate-800 rounded focus:ring-2 focus:ring-blue-500 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                      />
                    </div>

                    <div className="flex items-center gap-4 py-2">
                      <button
                        type="button"
                        onClick={testOllamaSettings}
                        disabled={testingOllama}
                        className="px-3 py-1.5 border border-blue-600 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-xs font-bold rounded transition disabled:opacity-50"
                      >
                        {testingOllama ? "Testing Connection..." : "Test Connection"}
                      </button>

                      {ollamaTestResult && (
                        <div className="text-xs">
                          {ollamaTestResult.connected ? (
                            ollamaTestResult.model_available ? (
                              <span className="text-green-600 dark:text-green-400 font-medium">✓ Connected! Model '{ollamaModel}' ready.</span>
                            ) : (
                              <span className="text-amber-600 dark:text-amber-400 font-medium">⚠ Connected, but '{ollamaModel}' not found. Download it in terminal using `ollama run {ollamaModel}`.</span>
                            )
                          ) : (
                            <span className="text-red-500 dark:text-red-400 font-medium">✗ Connection failed: {ollamaTestResult.message}</span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="pt-4 border-t border-gray-200 dark:border-slate-800">
                      <button
                        onClick={saveGeneralSettings}
                        className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 transition"
                      >
                        Save AI Settings
                      </button>
                    </div>
                  </div>
                </>
              )}

              {/* Bark Settings Tab */}
              {settingsTab === "bark" && (
                <>
                  <h2 className="text-lg font-bold text-gray-800 dark:text-white mb-4 flex items-center">
                    Bark Synthesis Settings
                    <span className="ml-2 bg-amber-100 text-amber-800 dark:bg-amber-955/40 dark:text-amber-400 dark:border dark:border-amber-900/50 text-[10px] font-bold px-2 py-0.5 rounded-full">Experimental</span>
                  </h2>
                  <div className="space-y-4">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        className="rounded border-gray-305 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                        checked={barkSplitSentences}
                        onChange={(e) => {
                          setBarkSplitSentences(e.target.checked);
                          localStorage.setItem(
                            "barkSplitSentences",
                            e.target.checked.toString()
                          );
                        }}
                      />
                      <span className="text-sm text-gray-700 dark:text-slate-300 font-medium">Split long sentences by max duration</span>
                    </label>

                    <div>
                      <label className="block text-xs font-semibold text-gray-700 dark:text-slate-350 mb-1">
                        Max sentence duration (seconds)
                      </label>
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
                        className="block w-24 border border-gray-200 dark:border-slate-800 rounded p-2 text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
                      />
                    </div>

                    {/* Small Bark models */}
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        className="rounded border-gray-305 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                        checked={barkSettings.small_models}
                        onChange={(e) => {
                          const val = e.target.checked;
                          setBarkSettings((prev) => ({
                            ...prev,
                            small_models: val,
                          }));
                          localStorage.setItem(
                            "barkSmallModels",
                            val.toString()
                          );
                        }}
                      />
                      <span className="text-sm text-gray-700 dark:text-slate-300 font-medium">Use Small Bark models (faster draft, lower VRAM)</span>
                    </label>

                    {/* Skip fine stage */}
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        className="rounded border-gray-305 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                        checked={barkSettings.skip_fine}
                        onChange={(e) => {
                          const val = e.target.checked;
                          setBarkSettings((prev) => ({
                            ...prev,
                            skip_fine: val,
                          }));
                          localStorage.setItem(
                            "barkSkipFine",
                            val.toString()
                          );
                        }}
                      />
                      <span className="text-sm text-gray-700 dark:text-slate-300 font-medium">Skip fine stage (significant draft speed boost)</span>
                    </label>

                    {/* Smart Enhance (Post-Processing) */}
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        className="rounded border-gray-305 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
                        checked={barkSettings.smart_enhance}
                        onChange={(e) => {
                          const val = e.target.checked;
                          setBarkSettings((prev) => ({
                            ...prev,
                            smart_enhance: val,
                          }));
                          try {
                            localStorage.setItem(
                              "barkSmartEnhance",
                              val.toString()
                            );
                          } catch {
                            /* ignore quota errors */
                          }
                        }}
                      />
                      <span className="text-sm text-gray-700 dark:text-slate-300 font-medium">Smart Enhance (Narrative post-processing)</span>
                    </label>

                    <div className="text-[10px] text-gray-500 dark:text-gray-400 mt-1 leading-normal">
                      Sentences longer than the limit will be split using natural punctuation boundaries when possible.
                    </div>

                    <div className="pt-4 border-t border-gray-200 dark:border-slate-800 flex gap-4">
                      <button
                        className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 transition"
                        onClick={() => {
                          localStorage.setItem(
                            "barkSplitSentences",
                            barkSplitSentences.toString()
                          );
                          localStorage.setItem(
                            "barkMaxDuration",
                            barkMaxDuration.toString()
                          );
                          saveGeneralSettings();
                        }}
                      >
                        Save Bark Settings
                      </button>
                    </div>
                  </div>
                </>
              )}
              {settingsTab === "cloud" && (
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-gray-900 dark:text-slate-100 flex items-center gap-2">
                    <Cloud className="h-4 w-4 text-blue-500" />
                    Google Drive Integration
                  </h4>
                  <p className="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
                    Voication Studio connects directly to your Google Drive via browser OAuth using the safe <code className="bg-gray-100 dark:bg-slate-800 px-1 py-0.5 rounded text-blue-600 dark:text-blue-400">drive.file</code> permission scope. It only accesses files created by this application.
                  </p>

                  <div className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-100 dark:border-slate-800 rounded-xl space-y-2">
                    <label className="block text-xs font-semibold text-gray-700 dark:text-slate-300">
                      Custom Google OAuth Client ID (Optional)
                    </label>
                    <input
                      type="text"
                      value={googleClientId}
                      onChange={(e) => {
                        const val = e.target.value.trim();
                        setGoogleClientId(val);
                        localStorage.setItem("voication_google_client_id", val);
                      }}
                      placeholder="1041584982401-example.apps.googleusercontent.com"
                      className="w-full border border-gray-200 dark:border-slate-800 rounded-lg p-2.5 text-xs bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 font-mono"
                    />
                    <p className="text-[10px] text-gray-400">
                      Leave blank to use Voication Studio standard OAuth configuration.
                    </p>
                  </div>

                  <div className="pt-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-gray-600 dark:text-slate-300">
                      Connection Status: {driveAccessToken ? <strong className="text-green-600 dark:text-green-400">Connected ({driveUserEmail})</strong> : <strong className="text-amber-600 dark:text-amber-400">Not Connected</strong>}
                    </span>
                    <button
                      type="button"
                      onClick={handleConnectDrive}
                      disabled={isDriveSyncing}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-lg shadow transition"
                    >
                      {driveAccessToken ? "Reconnect Account" : "Connect Google Drive"}
                    </button>
                  </div>
                </div>
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

      {/* Project Manager Modal */}
      {showProjectManager && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm">
          <div className="bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 w-[600px] max-w-full rounded-2xl shadow-2xl flex flex-col relative max-h-[85vh] overflow-hidden animate-fadeIn text-gray-900 dark:text-slate-100">
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-gray-100 dark:border-slate-800 flex items-center justify-between bg-gradient-to-r from-indigo-50/60 to-purple-50/60 dark:from-slate-950 dark:to-slate-900">
              <div>
                <h2 className="text-lg font-bold text-indigo-950 dark:text-indigo-200 flex items-center gap-2">
                  <Folder className="h-5 w-5 text-indigo-500 shrink-0" />
                  Studio Project Manager
                </h2>
                <p className="text-xs text-gray-505 dark:text-slate-400 mt-0.5">
                  Organize scripts, multitracks, character profiles, and curated sandbox voices.
                </p>
              </div>
              <button
                onClick={() => setShowProjectManager(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 transition p-1 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full"
                title="Close"
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-6 flex-1">
              {/* Project Creation & Importing Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    createNewProject();
                    alert("Created and loaded a new project!");
                  }}
                  className="flex-1 py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-semibold shadow-md transition duration-200 flex items-center justify-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  New Project
                </button>
                
                {/* File Import Button */}
                <label className="flex-1 py-3 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-xl text-xs font-semibold shadow-md transition duration-200 flex items-center justify-center gap-2 cursor-pointer text-center">
                  <Download className="h-4 w-4" />
                  Import Project
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (!file) return;
                      const reader = new FileReader();
                      reader.onload = (evt) => {
                        try {
                          const project = JSON.parse(evt.target.result);
                          if (!project.id || !project.name) {
                            alert("Invalid project format! Must contain id and name properties.");
                            return;
                          }
                          const exists = projects.some(p => p.id === project.id);
                          let importProject = { ...project };
                          if (exists) {
                            if (confirm(`Project "${project.name}" already exists. Would you like to overwrite it?`)) {
                              setProjects(prev => prev.map(p => p.id === project.id ? importProject : p));
                            } else {
                              importProject.id = `project_${Date.now()}`;
                              importProject.name = `${project.name} (Imported)`;
                              setProjects(prev => [importProject, ...prev]);
                            }
                          } else {
                            setProjects(prev => [importProject, ...prev]);
                          }
                          loadProject(importProject);
                          alert(`Project "${importProject.name}" successfully imported and loaded.`);
                        } catch (err) {
                          alert("Failed to parse project file! Ensure it is a valid JSON file.");
                        }
                      };
                      reader.readAsText(file);
                      e.target.value = null; // reset input
                    }}
                  />
                </label>
              </div>

              {/* Projects List */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">
                  Saved Projects ({projects.length})
                </h3>
                {projects.length === 0 ? (
                  <div className="text-center py-6 text-sm text-gray-400 border-2 border-dashed border-gray-200 rounded-xl">
                    No saved projects found. Click "New Project" to start.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {projects.map((proj) => {
                      const isActive = proj.id === currentProjectId;
                      return (
                        <div
                          key={proj.id}
                          className={`p-4 rounded-xl border flex items-center justify-between transition-all duration-200 ${
                            isActive
                              ? "bg-indigo-50/50 dark:bg-indigo-950/20 border-indigo-200 dark:border-indigo-800 shadow-sm"
                              : "bg-white dark:bg-slate-800 border-gray-100 dark:border-slate-800 hover:border-gray-300 dark:hover:border-slate-700"
                          }`}
                        >
                          <div className="flex-1 pr-4">
                            <div className="flex items-center gap-2">
                              {isActive ? (
                                <span className="bg-indigo-600 text-white text-[9px] font-bold px-2 py-0.5 rounded-full">
                                  Active
                                </span>
                              ) : null}
                              <span className="text-sm font-bold text-gray-800 dark:text-slate-100">
                                {proj.name}
                              </span>
                            </div>
                            <div className="text-[10px] text-gray-400 mt-1 flex items-center gap-3">
                              <span>Format: <strong className="text-gray-500 dark:text-slate-300 uppercase">{proj.mediaFormat || "podcast"}</strong></span>
                              <span>Speakers: <strong className="text-gray-500 dark:text-slate-300">{proj.numberOfSpeakers || 4}</strong></span>
                              <span>Updated: <strong className="text-gray-500 dark:text-slate-300">{new Date(proj.updatedAt || proj.createdAt).toLocaleString()}</strong></span>
                            </div>
                          </div>

                          {/* Project Actions */}
                          <div className="flex items-center gap-1.5 shrink-0">
                            {/* Rename */}
                            <button
                              onClick={() => {
                                const newName = prompt("Rename project to:", proj.name);
                                if (newName && newName.trim()) {
                                  setProjects(prev =>
                                    prev.map(p =>
                                      p.id === proj.id
                                        ? { ...p, name: newName.trim(), updatedAt: new Date().toISOString() }
                                        : p
                                    )
                                  );
                                  if (isActive) {
                                    setActiveProjectName(newName.trim());
                                  }
                                }
                              }}
                              className="p-2 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-lg text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200 transition"
                              title="Rename Project"
                            >
                              <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline">Rename</span>
                            </button>

                            {/* Load / Activate */}
                            {!isActive && (
                              <button
                                onClick={() => {
                                  loadProject(proj);
                                  alert(`Loaded project: ${proj.name}`);
                                }}
                                className="px-3 py-1.5 bg-indigo-50 dark:bg-indigo-950/40 hover:bg-indigo-100 dark:hover:bg-indigo-900/40 text-indigo-700 dark:text-indigo-400 font-bold text-xs rounded-lg transition"
                                title="Open Project"
                              >
                                Open
                              </button>
                            )}

                            {/* Export */}
                            <button
                              onClick={() => {
                                try {
                                  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(proj, null, 2));
                                  const downloadAnchor = document.createElement('a');
                                  downloadAnchor.setAttribute("href", dataStr);
                                  downloadAnchor.setAttribute("download", `${proj.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_project.json`);
                                  document.body.appendChild(downloadAnchor);
                                  downloadAnchor.click();
                                  downloadAnchor.remove();
                                } catch (e) {
                                  alert("Failed to export project!");
                                }
                              }}
                              className="p-2 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-lg text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200 transition"
                              title="Export Project File"
                            >
                              <Upload className="h-3.5 w-3.5" />
                            </button>

                            {/* Delete */}
                            <button
                              onClick={() => {
                                if (confirm(`Are you sure you want to delete the project "${proj.name}"?`)) {
                                  setProjects(prev => prev.filter(p => p.id !== proj.id));
                                  if (isActive) {
                                    const remaining = projects.filter(p => p.id !== proj.id);
                                    if (remaining.length > 0) {
                                      loadProject(remaining[0]);
                                    } else {
                                      setCurrentProjectId("");
                                      setActiveProjectName("Untitled Project");
                                      window.location.reload();
                                    }
                                  }
                                }
                              }}
                              className="p-2 hover:bg-red-50 dark:hover:bg-red-950/30 hover:text-red-600 dark:hover:text-red-400 rounded-lg text-gray-400 dark:text-slate-400 transition flex items-center justify-center"
                              title="Delete Project"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {projectManagerTab === "cloud" && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold text-gray-700 dark:text-slate-300 uppercase tracking-wider">
                      Drive Backups in "Voication_Studio_Backups"
                    </h3>
                    {driveAccessToken && (
                      <button
                        onClick={() => handleRefreshDriveBackups()}
                        disabled={isDriveSyncing}
                        className="text-xs text-blue-600 hover:underline font-semibold flex items-center gap-1"
                      >
                        <RefreshCw className={`h-3 w-3 ${isDriveSyncing ? "animate-spin" : ""}`} />
                        Refresh List
                      </button>
                    )}
                  </div>

                  {!driveAccessToken ? (
                    <div className="text-center py-8 text-sm text-gray-400 border-2 border-dashed border-gray-200 dark:border-slate-800 rounded-xl space-y-3">
                      <Cloud className="h-8 w-8 mx-auto text-gray-300 dark:text-slate-600" />
                      <p>Connect your Google Drive account above to view and restore cloud backups.</p>
                      <button
                        onClick={handleConnectDrive}
                        className="py-2 px-4 bg-blue-600 text-white font-bold text-xs rounded-xl shadow hover:bg-blue-700 transition"
                      >
                        Connect Google Drive
                      </button>
                    </div>
                  ) : driveBackupsList.length === 0 ? (
                    <div className="text-center py-8 text-sm text-gray-400 border-2 border-dashed border-gray-200 dark:border-slate-800 rounded-xl">
                      No cloud backups found in Google Drive folder "Voication_Studio_Backups".
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {driveBackupsList.map((file) => (
                        <div
                          key={file.id}
                          className="p-3.5 bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-800 rounded-xl flex items-center justify-between"
                        >
                          <div>
                            <h4 className="text-sm font-bold text-gray-800 dark:text-slate-100 flex items-center gap-2">
                              <FileJson className="h-4 w-4 text-blue-500" />
                              {file.name}
                            </h4>
                            <div className="text-[10px] text-gray-400 mt-0.5 flex items-center gap-3">
                              <span>Modified: {new Date(file.modifiedTime).toLocaleString()}</span>
                              {file.size && <span>Size: {(file.size / 1024).toFixed(1)} KB</span>}
                            </div>
                          </div>

                          <button
                            onClick={() => handleRestoreFromDrive(file)}
                            disabled={isDriveSyncing}
                            className="py-1.5 px-3 bg-blue-50 dark:bg-blue-950/40 hover:bg-blue-100 dark:hover:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-bold text-xs rounded-lg transition flex items-center gap-1.5"
                          >
                            <CloudDownload className="h-3.5 w-3.5" />
                            Restore
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {/* Voice Creator Modal */}
      {showVoiceCreatorModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm">
          <div className="bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 w-[500px] max-w-full rounded-2xl shadow-2xl flex flex-col relative max-h-[85vh] overflow-hidden text-gray-900 dark:text-slate-100">
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-gray-100 dark:border-slate-800 flex items-center justify-between bg-gradient-to-r from-indigo-50/60 to-purple-50/60 dark:from-slate-950 dark:to-slate-900">
              <div>
                <h2 className="text-lg font-bold text-indigo-900 dark:text-indigo-200 flex items-center gap-2">
                  <Mic className="h-5 w-5 text-indigo-500 shrink-0" />
                  Quick Voice Creator / Cloner
                </h2>
                <p className="text-xs text-gray-505 dark:text-slate-400 mt-0.5 font-semibold">
                  Record or upload a 5-10s audio clip to create a voice profile.
                </p>
              </div>
              <button
                onClick={() => {
                  setShowVoiceCreatorModal(false);
                  setCustomCloneTranscript("");
                }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 transition p-1 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full"
                title="Close"
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-4 flex-1">
              <div className="text-xs italic bg-gray-50 dark:bg-slate-950 border border-gray-100 dark:border-slate-800 p-3 rounded-xl text-gray-500 dark:text-slate-400">
                “The sun sets behind the hills, and the sky turns orange. I really enjoy storytelling and character voices.”
              </div>

              {/* Upload WAV */}
              <div className="space-y-1.5">
                <label className="block text-[11px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                  Upload WAV Sample
                </label>
                <input
                  type="file"
                  accept=".wav"
                  onChange={(e) => {
                    if (e.target.files?.[0]) {
                      const file = e.target.files[0];
                      formDataRef.current.set("speaker_wav", file);
                      setRecordedBlob(file);
                      if (!customCloneName) {
                        setCustomCloneName(file.name.replace(/\.[^/.]+$/, ""));
                      }
                    }
                  }}
                  className="block w-full text-xs text-gray-600 dark:text-slate-400 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border file:border-gray-200 dark:file:border-slate-800 file:text-xs file:font-semibold file:bg-gray-50 dark:file:bg-slate-800 file:text-gray-700 dark:file:text-slate-350 hover:file:bg-gray-100 dark:hover:file:bg-slate-700 cursor-pointer"
                />
              </div>

              {/* Record */}
              <div className="space-y-2 border-t border-gray-100 dark:border-slate-800 pt-3">
                <label className="block text-[11px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                  Or Record Reference
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold text-white transition ${
                      isRecording ? "bg-red-500 animate-pulse" : "bg-blue-600 hover:bg-blue-700"
                    }`}
                    onClick={startRecording}
                    disabled={isRecording}
                  >
                    {isRecording ? (
                      <span className="flex items-center gap-1.5 justify-center">
                        <span className="w-2 h-2 bg-white rounded-full animate-ping shrink-0" />
                        Recording...
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 justify-center">
                        <Mic className="h-3.5 w-3.5" />
                        Record
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="py-1.5 px-4 bg-gray-700 dark:bg-slate-800 text-white text-xs font-semibold rounded-lg hover:bg-gray-800 dark:hover:bg-slate-700 transition disabled:opacity-50"
                    onClick={stopRecording}
                    disabled={!isRecording}
                  >
                    Stop
                  </button>
                </div>
              </div>

              {/* Preview & Save */}
              {recordedBlob && (
                <div className="space-y-3 pt-3 border-t border-gray-100 dark:border-slate-800">
                  <label className="block text-[11px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                    Clone Preview & Save
                  </label>
                  <audio
                    controls
                    src={URL.createObjectURL(recordedBlob)}
                    className="w-full h-8"
                  />
                  
                  <div className="space-y-1.5">
                    <label className="block text-[11px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                      Reference Audio Transcript (Highly Recommended for Accents)
                    </label>
                    <textarea
                      value={customCloneTranscript}
                      onChange={(e) => setCustomCloneTranscript(e.target.value)}
                      placeholder="Optional. Type the words spoken in this reference clip to enable high-fidelity accent cloning..."
                      className="w-full p-2 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-950 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400 min-h-[3.5rem] resize-none"
                    />
                  </div>
                  
                  <div className="flex gap-2 items-center">
                    <input
                      type="text"
                      value={customCloneName}
                      onChange={(e) => setCustomCloneName(e.target.value)}
                      placeholder="Give this voice a name (e.g. My Cloned Voice)"
                      className="flex-1 p-2 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-950 rounded-lg text-xs text-gray-900 dark:text-slate-100 focus:ring-1 focus:ring-indigo-400"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        if (!recordedBlob) return;
                        const name = customCloneName.trim() || `Cloned Voice ${clonedProfiles.length + 1}`;
                        const reader = new FileReader();
                        reader.readAsDataURL(recordedBlob);
                        reader.onloadend = () => {
                          const base64data = reader.result;
                          setClonedProfiles(prev => [
                            ...prev,
                            { 
                              name, 
                              type: "clone", 
                              voice: "custom_clone", 
                              file: recordedBlob, 
                              fileBase64: base64data,
                              transcript: customCloneTranscript.trim()
                            }
                          ]);
                          setCustomCloneName("");
                          setCustomCloneTranscript("");
                          setShowVoiceCreatorModal(false);
                          alert(`Voice profile "${name}" successfully saved!`);
                        };
                      }}
                      className="py-2 px-3 bg-green-600 hover:bg-green-700 text-white text-xs font-bold rounded-lg transition shrink-0"
                    >
                      Save & Use Voice
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Queue mobile overlay backdrop */}

      {/* Queue mobile overlay backdrop */}
      {showQueue && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden transition-opacity duration-300"
          onClick={() => setShowQueue(false)}
        />
      )}

      {/* RHS Render Queue Column */}
      <div className={`fixed inset-y-0 right-0 z-50 w-80 sm:w-96 bg-white dark:bg-slate-900 border-l border-gray-200 dark:border-slate-800 flex flex-col h-full shrink-0 shadow-2xl text-gray-900 dark:text-slate-100 overflow-hidden transition-all duration-300 transform ${
        showQueue 
          ? "translate-x-0 opacity-100 pointer-events-auto" 
          : "translate-x-full opacity-0 pointer-events-none"
      }`}>
        {showQueue && (
          <>
            {/* Panel Tabs */}
            <div className="p-2 border-b border-gray-100 dark:border-slate-800 flex items-center justify-between bg-gray-50 dark:bg-slate-800/50 gap-2">
            <div className="flex gap-1">
              <button
                onClick={() => setRightPanelTab("queue")}
                className={`px-3 py-1 text-xs font-bold rounded-lg transition flex items-center gap-1 ${
                  rightPanelTab === "queue"
                    ? "bg-slate-800 text-white dark:bg-slate-800"
                    : "text-gray-500 hover:text-gray-700 dark:hover:text-slate-350"
                }`}
                type="button"
              >
                <ListMusic className="h-3.5 w-3.5" />
                Queue
              </button>
              <button
                onClick={() => setRightPanelTab("reviewer")}
                className={`px-3 py-1 text-xs font-bold rounded-lg transition flex items-center gap-1 ${
                  rightPanelTab === "reviewer"
                    ? "bg-slate-800 text-white dark:bg-slate-800"
                    : "text-gray-500 hover:text-gray-700 dark:hover:text-slate-350"
                }`}
                type="button"
              >
                <Mic className="h-3.5 w-3.5" />
                Reviewer
              </button>
            </div>
            <button
              onClick={() => setShowQueue(false)}
              className="text-gray-400 hover:text-gray-650 p-1 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition"
              title="Close Panel"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {rightPanelTab === "queue" ? (
              queue.length === 0 ? (
                <div className="text-center text-xs text-gray-405 py-12 px-4 italic bg-gray-50 border border-dashed border-gray-200 rounded-xl flex flex-col items-center justify-center gap-2">
                  <ListMusic className="h-6 w-6 text-gray-300" />
                  <span>Queue is empty. Generate previews or scripts to see items here.</span>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex gap-2 mb-2">
                    {queue.some(item => item.status === "done" || item.status === "error") && (
                      <button
                        onClick={clearCompletedQueueItems}
                        className="flex-1 py-1.5 px-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-[10px] font-bold text-slate-650 dark:text-slate-300 rounded-lg transition flex items-center justify-center gap-1 border border-slate-200 dark:border-slate-700 shadow-sm"
                        type="button"
                      >
                        <Trash2 className="h-3 w-3" />
                        Clear Completed
                      </button>
                    )}
                    {queue.some(item => ["generating", "queued", "processing"].includes(item.status)) && (
                      <button
                        onClick={cancelAllQueue}
                        className="flex-1 py-1.5 px-2 bg-red-50 hover:bg-red-100 dark:bg-red-950/40 dark:hover:bg-red-950/60 text-[10px] font-bold text-red-650 dark:text-red-400 rounded-lg transition flex items-center justify-center gap-1 border border-red-200 dark:border-red-900/40 shadow-sm"
                        type="button"
                      >
                        <X className="h-3 w-3" />
                        Cancel All
                      </button>
                    )}
                  </div>
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
                      <div key={item.id} className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-xl relative shadow-sm hover:border-blue-300 transition duration-200">
                        <div className="flex justify-between items-start gap-1">
                          <p className="text-xs font-semibold text-gray-800 dark:text-slate-200 truncate flex-1">{item.text}</p>
                          {item.status === "done" && (
                            <button
                              onClick={() => {
                                setQueue((prev) =>
                                  prev.filter((q) => q.id !== item.id)
                                );
                                const saved =
                                  JSON.parse(localStorage.getItem("savedQueue")) ||
                                  [];
                                localStorage.setItem(
                                  "savedQueue",
                                  JSON.stringify(
                                    saved.filter((j) => j.id !== item.id)
                                  )
                                );
                              }}
                              className="text-gray-400 hover:text-gray-600 transition"
                              title="Clear Item"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          )}
                          {["generating", "queued", "processing"].includes(item.status) && (
                            <button
                              onClick={() => cancelQueueItem(item.id)}
                              className="text-gray-405 hover:text-red-500 transition"
                              title="Cancel Generation"
                            >
                              <X className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                        
                        <div className="mt-2 flex items-center justify-between text-[10px] text-gray-500">
                          <span>Status: <span className="font-bold capitalize text-blue-600">{item.status}</span></span>
                          {item.model && (
                            <span className="font-medium text-gray-400 flex items-center gap-1">
                              <Cpu className="h-3 w-3 text-gray-400 shrink-0" />
                              {item.model}
                            </span>
                          )}
                        </div>
                        
                        {item.message && <div className="mt-1 text-[10px] text-gray-500 font-normal leading-normal">{item.message}</div>}
  
                        {item.status !== "done" && item.status !== "error" && (
                          <div className="mt-2 space-y-1">
                            <div className="flex justify-between text-[10px] text-gray-500 font-semibold">
                              {item.chunkIndex != null && item.totalChunks != null ? (
                                <span>
                                  Chunk {item.chunkIndex + 1} of {item.totalChunks}
                                </span>
                              ) : (
                                <span>Progress</span>
                              )}
                              <span>{progress}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div
                                className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                            <button
                              onClick={() => cancelQueueItem(item.id)}
                              className="text-[10px] text-red-600 hover:underline font-bold mt-1 inline-block"
                            >
                              Cancel Generation
                            </button>
                          </div>
                        )}
                        
                        {item.status === "done" && item.downloadUrl && (
                          <div className="mt-2 pt-2 border-t border-gray-155 dark:border-slate-800 space-y-2">
                            <audio
                              controls
                              src={item.downloadUrl}
                              className="w-full h-8"
                            />
                            <div className="flex gap-1.5 w-full">
                              <button
                                onClick={() => {
                                  const safeName = `${(item.text || "audio").replace(/[^\w\-_]/g, "_")}.wav`;
                                  triggerDownload(item.downloadUrl, safeName);
                                }}
                                className="flex-1 py-1.5 bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] text-white rounded-lg text-[10px] font-bold transition flex items-center justify-center gap-1"
                              >
                                <Download className="h-3 w-3 shrink-0" />
                                Download
                              </button>
                              <button
                                onClick={() => {
                                  setActiveReviewAudioUrl(item.downloadUrl);
                                  setRightPanelTab("reviewer");
                                }}
                                className="flex-1 py-1.5 border border-indigo-200 dark:border-slate-800 text-indigo-750 dark:text-indigo-300 rounded-lg text-[10px] font-bold transition flex items-center justify-center gap-1 bg-white dark:bg-slate-900"
                              >
                                <Mic className="h-3 w-3 shrink-0" />
                                Review
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )
            ) : (
              <div className="space-y-4">
                {/* Audio selector & player */}
                <div className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-xl space-y-2.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[10px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
                      Review Audio Source
                    </label>
                    <button
                      onClick={loadActiveChapterMix}
                      className="text-[9px] font-bold text-indigo-600 hover:underline"
                      type="button"
                    >
                      Load Chapter Mix
                    </button>
                  </div>
                  
                  {activeReviewAudioUrl ? (
                    <>
                      <audio
                        ref={reviewAudioRef}
                        controls
                        src={activeReviewAudioUrl}
                        className="w-full h-8"
                      />
                      <div className="text-[9px] text-gray-500 font-semibold truncate bg-white dark:bg-slate-900 border border-gray-100 dark:border-slate-800 p-1.5 rounded-md">
                        File: {activeReviewAudioUrl.split("/").pop()}
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-4 text-xs text-gray-400 italic font-medium">
                      No review file active. Use "Load Chapter Mix" or review a generated clip.
                    </div>
                  )}
                </div>

                {/* Notes List */}
                <div className="space-y-2">
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                    Timecoded Notes ({reviewNotes.length})
                  </label>
                  
                  {reviewNotes.length === 0 ? (
                    <div className="text-center py-6 text-xs text-gray-400 italic font-medium bg-gray-50 dark:bg-slate-950 border border-dashed border-gray-200 dark:border-slate-800 rounded-xl">
                      No notes added yet. Use form below to tag audio timestamps.
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
                      {reviewNotes.map((note) => (
                        <div
                          key={note.id}
                          className="p-3 bg-gray-50 dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-xl space-y-2 relative"
                        >
                          {/* Note Header */}
                          <div className="flex justify-between items-start gap-1">
                            <button
                              onClick={() => {
                                if (reviewAudioRef.current) {
                                  reviewAudioRef.current.currentTime = note.timecode;
                                  reviewAudioRef.current.play().catch(() => {});
                                }
                              }}
                              className="px-1.5 py-0.5 bg-indigo-100 hover:bg-indigo-200 dark:bg-indigo-950 dark:hover:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-[10px] font-bold rounded"
                              title="Seek audio to timecode"
                              type="button"
                            >
                              ⏱️ {Math.floor(note.timecode / 60)}:{(note.timecode % 60).toFixed(1).padStart(4, "0")}
                            </button>
                            <button
                              onClick={() => deleteReviewNote(note.id)}
                              className="text-gray-400 hover:text-red-500 transition text-[9px]"
                              title="Delete note"
                              type="button"
                            >
                              ✕
                            </button>
                          </div>
                          
                          <p className="text-xs text-gray-800 dark:text-slate-200 font-semibold">{note.text}</p>
                          
                          {note.targetPara && (
                            <div className="text-[10px] text-gray-500 bg-white dark:bg-slate-900/60 p-2 rounded-lg border border-gray-100 dark:border-slate-800 line-clamp-2">
                              Target: "{note.targetPara}"
                            </div>
                          )}

                          {/* Rewrite Section */}
                          {note.suggestion ? (
                            <div className="pt-2 border-t border-gray-200 dark:border-slate-800 space-y-2">
                              <div className="text-[10px] font-bold text-purple-700 dark:text-purple-400 uppercase tracking-wider flex items-center gap-1">
                                <Sparkles className="h-3 w-3" /> Suggested Revision:
                              </div>
                              <div className="text-[10px] p-2 rounded-lg bg-green-50/50 dark:bg-emerald-950/20 border border-green-200/50 dark:border-emerald-900/30 text-emerald-800 dark:text-emerald-300 font-medium">
                                "{note.suggestion}"
                              </div>
                              
                              {note.status === "approved" ? (
                                <span className="inline-block text-[9px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
                                  ✓ Approved & Updated
                                </span>
                              ) : (
                                <div className="flex gap-1.5 pt-1">
                                  <button
                                    onClick={() => approveRewrite(note.id)}
                                    className="px-2 py-1 bg-green-600 hover:bg-green-700 text-white text-[10px] font-bold rounded transition"
                                    type="button"
                                  >
                                    Approve
                                  </button>
                                  <button
                                    onClick={() => rejectRewrite(note.id)}
                                    className="px-2 py-1 border border-gray-300 dark:border-slate-800 text-gray-700 dark:text-slate-350 text-[10px] font-bold rounded transition"
                                    type="button"
                                  >
                                    Reject
                                  </button>
                                </div>
                              )}
                            </div>
                          ) : (
                            note.targetPara && note.status !== "approved" && (
                              <button
                                onClick={() => requestLlmRewrite(note.id)}
                                disabled={isSuggestingEdit}
                                className="w-full py-1 bg-purple-600 hover:bg-purple-700 text-white text-[10px] font-bold rounded-lg transition flex items-center justify-center gap-1 disabled:opacity-50"
                                type="button"
                              >
                                {isSuggestingEdit ? (
                                  <>
                                    <span className="w-2.5 h-2.5 border border-white border-t-transparent rounded-full animate-spin" />
                                    Rewriting...
                                  </>
                                ) : (
                                  <>
                                    <Sparkles className="h-3 w-3" />
                                    Suggest Revision
                                  </>
                                )}
                              </button>
                            )
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Add Review Note Form */}
                <div className="pt-3 border-t border-gray-100 dark:border-slate-800 space-y-3">
                  <label className="block text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                    Add Timecoded Note
                  </label>
                  
                  <textarea
                    id="newNoteText"
                    placeholder="Describe issue (e.g. Speak slower, change 'terrible' to 'eerie'...)"
                    className="w-full p-2.5 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 rounded-xl text-xs resize-none h-[4rem] focus:ring-1 focus:ring-indigo-400"
                  />

                  {/* Select target paragraph */}
                  <div className="space-y-1">
                    <label className="block text-[9px] font-bold text-gray-400 uppercase tracking-wider">
                      Target Text / Paragraph
                    </label>
                    <select
                      id="newNoteTarget"
                      className="w-full p-2 border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 text-xs rounded-xl focus:ring-1 focus:ring-indigo-400 dark:[&>option]:bg-slate-950"
                    >
                      <option value="">-- No target paragraph --</option>
                      {podcastText.split("\n\n").filter(p => p.trim()).map((para, idx) => (
                        <option key={idx} value={para.trim()}>
                          Para {idx + 1}: {para.trim().substring(0, 45)}...
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    onClick={() => {
                      const txtEl = document.getElementById("newNoteText");
                      const tgtEl = document.getElementById("newNoteTarget");
                      if (txtEl && txtEl.value.trim()) {
                        addReviewNote(txtEl.value.trim(), tgtEl ? tgtEl.value : "");
                        txtEl.value = "";
                        if (tgtEl) tgtEl.value = "";
                      } else {
                        alert("Please enter a note description.");
                      }
                    }}
                    className="w-full py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition"
                    type="button"
                  >
                    Tag Timecode & Add Note
                  </button>
                </div>

                {/* Publish to Obsidian */}
                <div className="pt-2">
                  <button
                    onClick={publishToObsidian}
                    className="w-full py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white text-xs font-bold rounded-xl transition flex items-center justify-center gap-1.5 shadow-md active:scale-[0.98]"
                    type="button"
                  >
                    <BookOpen className="h-3.5 w-3.5 shrink-0" />
                    Publish Version to Obsidian
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}
        </div>
        </div>
      </div>
    </div>
  </div>
  );
}
// Bark voice presets are now loaded dynamically from backend for consistency.
function AppInnerWrapper() {
  const [barkPresets, setBarkPresets] = useState({});
  const [darkMode, setDarkMode] = useState(false);
  const [setupCompleted, setSetupCompleted] = useState(true);
  const [config, setConfig] = useState(null);

  // Check setup status and load presets
  useEffect(() => {
    fetch("http://localhost:5000/voices")
      .then((res) => res.json())
      .then((data) => setBarkPresets(data))
      .catch((err) => console.error("Failed to fetch voice presets", err));

    axios.get("http://localhost:5000/config")
      .then((res) => {
        const cfg = res.data;
        setConfig(cfg);
        setSetupCompleted(cfg.setup_completed === true);
      })
      .catch((err) => {
        console.error("Failed to fetch backend configuration", err);
        const localSetup = localStorage.getItem("voicationSetupCompleted") === "true";
        setSetupCompleted(localSetup);
      });
  }, []);

  // Load dark mode from localStorage "voicationSetting" on mount
  useEffect(() => {
    const stored = localStorage.getItem("voicationSetting");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (typeof parsed.darkMode === "boolean") {
          setDarkMode(parsed.darkMode);
        }
      } catch (e) {
        console.warn("Failed to parse voicationSetting:", e);
      }
    }
  }, []);

  // Sync dark mode class with root html element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const handleSetupComplete = (newConfig) => {
    axios.post("http://localhost:5000/config", newConfig)
      .then((res) => {
        setConfig(res.data.config);
        setSetupCompleted(true);
        localStorage.setItem("voicationSetupCompleted", "true");
        window.location.reload();
      })
      .catch((err) => {
        console.error("Failed to save config", err);
        alert("Failed to save configuration to backend.");
      });
  };

  if (!setupCompleted) {
    return <SetupWizard onComplete={handleSetupComplete} />;
  }

  return (
    <AppInner
      barkPresets={barkPresets}
      setBarkPresets={setBarkPresets}
      darkMode={darkMode}
      setDarkMode={setDarkMode}
      initialConfig={config}
    />
  );
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
function VoiceProfilePanel({ presetList, onApplyProfile, playVoicePreview, playingPreview }) {
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
                className="px-3 py-1 text-xs rounded-lg bg-gray-200 dark:bg-slate-800 hover:bg-gray-300 dark:hover:bg-slate-700 text-gray-900 dark:text-slate-100 transition"
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
          className="px-2 py-1 text-xs rounded bg-blue-500 text-white hover:bg-blue-600 transition"
        >
          + New Profile
        </button>
      </div>

      {showProfileEditor && (
        <div className="mb-4 space-y-2">
          {/* Save message feedback */}
          {saveMessage && (
            <div className="text-green-600 dark:text-green-400 text-sm mb-2">{saveMessage}</div>
          )}
          <input
            type="text"
            placeholder="Profile Name"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-slate-800 rounded bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
          />
          <label className="flex items-center space-x-2 text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              checked={isPinned}
              onChange={(e) => setIsPinned(e.target.checked)}
              className="rounded border-gray-300 dark:border-slate-700 text-blue-600 focus:ring-blue-500 bg-white dark:bg-slate-800"
            />
            <span>Pin profile</span>
          </label>
          {/* Bark preset dropdown */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Select Bark Voice
            </label>
            <div className="flex items-center gap-2 mb-4">
              <select
                className="flex-1 px-3 py-1 border border-gray-300 dark:border-slate-800 rounded text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
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
              {voicePreset && playVoicePreview && (
                <button
                  onClick={() => playVoicePreview(voicePreset, "bark")}
                  className="px-3 py-1 border border-gray-300 dark:border-slate-800 rounded hover:bg-gray-100 dark:hover:bg-slate-800 flex items-center justify-center text-xs bg-white dark:bg-slate-900 shrink-0 text-gray-900 dark:text-slate-100"
                  title="Preview Voice"
                  type="button"
                >
                  {playingPreview === voicePreset ? "⏸️" : "▶️"}
                </button>
              )}
            </div>
          </div>
          {/* Bark tuning fields inserted here */}
          <h4 className="text-xs font-medium text-gray-600 dark:text-slate-400 mt-4">
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
                className="w-full px-3 py-1 border border-gray-300 dark:border-slate-800 rounded text-sm bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100"
              />
              <button
                onClick={randomiseSeed}
                className="px-2 py-1 text-xs bg-gray-200 dark:bg-slate-800 hover:bg-gray-300 dark:hover:bg-slate-700 rounded text-gray-900 dark:text-slate-100 transition"
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
                className="px-3 py-1 text-xs rounded-full border border-gray-300 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-900 dark:text-slate-100 transition shadow-sm"
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
                className="px-3 py-1 text-xs rounded-full border border-gray-300 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-900 dark:text-slate-100 transition shadow-sm"
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
                className="px-3 py-1 text-xs rounded-full border border-gray-300 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-900 dark:text-slate-100 transition shadow-sm"
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
              className="w-full h-2 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
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
              className="w-full h-2 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
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
              className="w-full h-2 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
          </div>
          <div className="flex gap-2 pt-2">
            <button
              onClick={handleSaveProfile}
              className="text-xs px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-bold rounded shadow transition"
            >
              Save
            </button>
            <button
              onClick={handleDeleteProfile}
              className="text-xs px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow transition"
            >
              Delete
            </button>
            <button
              type="button"
              onClick={() => {
                setShowProfileEditor(false);
                setSelectedProfileName("");
              }}
              className="text-xs px-4 py-2 bg-gray-200 dark:bg-slate-800 hover:bg-gray-300 dark:hover:bg-slate-700 text-gray-900 dark:text-slate-100 font-bold rounded transition shadow-sm"
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

// --- Skeleton loader and dimmed styles ---
// You may move this to a CSS file as needed
const style = document.createElement("style");
style.innerHTML = `
.skeleton-loader {
  margin-bottom: 1rem;
  padding: 0.5rem;
  animation: pulse 1.5s infinite;
}
.skeleton-line {
  height: 10px;
  background: #ddd;
  border-radius: 5px;
  margin: 6px 0;
}
.skeleton-line.short {
  width: 40%;
}
.skeleton-line.long {
  width: 80%;
}
.dimmed {
  opacity: 0.6;
  pointer-events: none;
}
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.4; }
  100% { opacity: 1; }
}
`;
if (
  typeof window !== "undefined" &&
  !document.getElementById("skeleton-loader-style")
) {
  style.id = "skeleton-loader-style";
  document.head.appendChild(style);
}
