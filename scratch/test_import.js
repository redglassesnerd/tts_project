const text = `---
uuid: "fdbf9dbb-0af7-43fa-a98f-6dc8676fd18d"
title: "Loop 6,260.76 , 5 years -  Mother in the sky - READY FOR LISTEN"
status: "draft"
order: 17
project_id: "a7ca92cd-a6aa-4472-963f-b58c3eab07cf"
last_updated: 1761887035019
tts:
  model: "xtts"
  voice: "tts_models/multilingual/multi-dataset/xtts_v2"
  speaker: ""
  preset: "default"
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  speed: 1.0
  last_rendered_hash: ""
  audio_path: ""
revisions:
  current_version: 1
  review_pending: false
---

**5 years in - Some of Sara’s madness starting to show.**

"Hello down there?” Sarah’s long nails and soft hands curls around the microphone of the short wave radio. She release the press-to-talk button and sniffs in the recycled air of the craft while she waits for a reply. *Farts and incense.*`;

const cleanMarkdownContent = (rawText) => {
  if (!rawText) return "";
  let txt = rawText.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  
  const lines = txt.split("\n");
  let startIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== "") {
      if (lines[i].trim().startsWith("---") || lines[i].trim().startsWith("+++")) {
        startIdx = i;
      }
      break;
    }
  }
  if (startIdx !== -1) {
    let endIdx = -1;
    for (let j = startIdx + 1; j < Math.min(lines.length, 100); j++) {
      if (lines[j].trim().startsWith("---") || lines[j].trim().startsWith("+++")) {
        endIdx = j;
        break;
      }
    }
    if (endIdx !== -1) {
      txt = lines.slice(endIdx + 1).join("\n");
    }
  }

  // Strip Markdown headers (# Heading -> Heading)
  txt = txt.replace(/^#{1,6}\s+/gm, "");

  // Strip Markdown bold and italics (**text** or *text* or __text__)
  txt = txt.replace(/\*\*([^*]+)\*\*/g, "$1");
  txt = txt.replace(/__([^_]+)__/g, "$1");
  txt = txt.replace(/\*([^*]+)\*/g, "$1");
  txt = txt.replace(/_([^_]+)_/g, "$1");

  // Strip Markdown link formatting [text](url) -> text
  txt = txt.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

  return txt.trim();
};

console.log("Cleaned Output:\n", cleanMarkdownContent(text));
