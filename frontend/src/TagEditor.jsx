/* src/TagEditor.jsx */
import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from "react";

/**
 * TagEditor
 * =========
 * • Type “[” or “/” → dropdown of tokens
 * • Dynamic filter matching as you type
 * • Keyboard navigation (ArrowUp, ArrowDown, Enter, Escape)
 * • Inserts tag as a pill with a trailing space
 * • Correctly decorates manually typed tags and tags with spaces (like [clears throat])
 */
const TagEditor = forwardRef(({
  value,
  onChange,
  tokens = [],
  speakerColorsMap = {},
  placeholder = "",
  className = "",
  onCaretChange,
  disabled = false,
  isLoading = false,
  loadingMessage = "",
}, ref) => {
  const edRef = useRef(null);
  const [popup, setPopup] = useState(false);
  const [filter, setFilter] = useState("");
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [activeIndex, setActiveIndex] = useState(0);

  const CARET_MARKER_SENTINEL = "\uE000CARET_MARKER\uE000";

  const elementToPlainText = (element) => {
    let result = "";
    for (const node of element.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        result += node.nodeValue;
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        if (node.id === "editor-caret-marker") {
          result += CARET_MARKER_SENTINEL;
          continue;
        }

        const dataPlain = node.getAttribute("data-plain");
        if (dataPlain !== null) {
          result += dataPlain;
          continue;
        }

        if (node.classList.contains("bark-token")) {
          const raw = node.textContent.trim();
          const clean = raw.replace(/×$/, "").trim();
          const txt = clean.replace(/\][^\]]*$/, "]").trim();
          result += txt.startsWith("[") ? txt : `[${txt}]`;
          continue;
        }
        if (node.classList.contains("music-token")) {
          const raw = node.textContent.trim();
          const clean = raw.replace(/×$/, "").trim().replace(/^🎵\s*/, "").trim();
          result += clean.startsWith("[") ? clean : `[${clean}]`;
          continue;
        }
        if (node.classList.contains("sfx-token")) {
          const raw = node.textContent.trim();
          const clean = raw.replace(/×$/, "").trim().replace(/^🔊\s*/, "").trim();
          result += clean.startsWith("[") ? clean : `[${clean}]`;
          continue;
        }
        if (node.classList.contains("pause-token")) {
          const text = node.textContent.trim();
          const match = text.match(/Pause:\s*([0-9.]+)\s*s/i);
          const sec = match ? match[1] : "1.0";
          result += `<Pause: ${sec} seconds>`;
          continue;
        }

        const tagName = node.tagName.toLowerCase();
        if (tagName === "br") {
          result += "\n";
        } else {
          const childText = elementToPlainText(node);
          const isBlock = ["div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"].includes(tagName);
          if (isBlock) {
            if (result && !result.endsWith("\n")) {
              result += "\n";
            }
            result += childText;
            if (!result.endsWith("\n")) {
              result += "\n";
            }
          } else {
            result += childText;
          }
        }
      }
    }
    return result;
  };

  const htmlToPlainText = (html) => {
    if (!html) return "";
    const div = document.createElement("div");
    div.innerHTML = html;
    return elementToPlainText(div);
  };

  const getPlainTextBeforeCaret = () => {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return "";
    const range = sel.getRangeAt(0).cloneRange();
    if (!edRef.current || !edRef.current.contains(range.startContainer)) return "";
    range.setStart(edRef.current, 0);
    const tempDiv = document.createElement("div");
    tempDiv.appendChild(range.cloneContents());
    return plain(tempDiv.innerHTML);
  };

  const triggerCaretCallback = () => {
    if (onCaretChange) {
      onCaretChange(getPlainTextBeforeCaret());
    }
  };

  const savedRange = useRef(null);

  // Keep track of active selection/range in the contenteditable
  const saveRange = () => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0) {
      const range = sel.getRangeAt(0);
      if (edRef.current && edRef.current.contains(range.startContainer)) {
        savedRange.current = range.cloneRange();
      }
    }
  };

  // Helper to compute caret coordinates relative to the container
  const getRelativeCaretPos = () => {
    const sel = window.getSelection();
    if (!sel.rangeCount) return { x: 0, y: 0 };
    const range = sel.getRangeAt(0).cloneRange();
    range.collapse(false);
    const rects = range.getClientRects();
    const containerRect = edRef.current.getBoundingClientRect();
    if (rects.length) {
      const caretRect = rects[0];
      return {
        x: caretRect.left - containerRect.left + edRef.current.scrollLeft,
        y: caretRect.bottom - containerRect.top + edRef.current.scrollTop,
      };
    }
    return { x: 0, y: containerRect.height + edRef.current.scrollTop };
  };

  // Analyze text node at selection to get trigger position and filter text
  const getFilterAndTriggerPos = () => {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    const node = range.startContainer;
    const offset = range.startOffset;

    if (node && node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.substring(0, offset);
      const lastBracket = text.lastIndexOf("[");
      const lastSlash = text.lastIndexOf("/");
      const lastIdx = Math.max(lastBracket, lastSlash);

      if (lastIdx !== -1) {
        const triggerChar = text[lastIdx];
        const textAfterTrigger = text.substring(lastIdx + 1);

        // '/' trigger: no spaces or newlines allowed, must follow whitespace
        if (triggerChar === "/") {
          if (textAfterTrigger.includes(" ") || textAfterTrigger.includes("\n")) {
            return null;
          }
          if (lastIdx > 0) {
            const charBefore = text[lastIdx - 1];
            if (charBefore !== " " && charBefore !== "\n") {
              return null;
            }
          }
        }

        // '[' trigger: allow spaces (multi-word tokens like Speaker names),
        // but close on newlines or if a ']' already closes the bracket
        if (triggerChar === "[") {
          if (textAfterTrigger.includes("\n") || textAfterTrigger.includes("]")) {
            return null;
          }
        }

        return {
          trigger: triggerChar,
          filter: textAfterTrigger,
          node,
          offset: lastIdx
        };
      }
    }
    return null;
  };

  const decorate = (txt) => {
    if (!txt) return "";
    let html = txt.replace(/\[([^\[\]]+?)\]/g, (_, t) => {
      const tokenLower = t?.toLowerCase().trim();
      
      const isMusic = (tokenLower.startsWith("music:") && tokenLower.substring(6).trim().length > 0) || 
                      (tokenLower.startsWith("music ") && tokenLower.substring(6).trim().length > 0);
      const isSfx = (tokenLower.startsWith("sfx:") && tokenLower.substring(4).trim().length > 0) || 
                    (tokenLower.startsWith("sfx ") && tokenLower.substring(4).trim().length > 0) || 
                    (tokenLower.startsWith("sound effect:") && tokenLower.substring(13).trim().length > 0) || 
                    (tokenLower.startsWith("sound effect ") && tokenLower.substring(13).trim().length > 0);
      
      if (isMusic) {
        return `<span contenteditable="false"
                      data-plain="[${t}]"
                      class="music-token inline-flex items-center gap-1.5 whitespace-nowrap py-0.5 px-2 bg-purple-100 text-purple-700 dark:bg-purple-950/40 dark:text-purple-300 border border-purple-200 dark:border-purple-900 rounded-full text-xs leading-none mr-1 font-bold shadow-sm">
                   <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="shrink-0"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
                   ${t}
                   <span data-x="remove-music" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>
                 </span>`;
      }
      
      if (isSfx) {
        return `<span contenteditable="false"
                      data-plain="[${t}]"
                      class="sfx-token inline-flex items-center gap-1.5 whitespace-nowrap py-0.5 px-2 bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300 border border-rose-200 dark:border-rose-900 rounded-full text-xs leading-none mr-1 font-bold shadow-sm">
                   <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="shrink-0"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>
                   ${t}
                   <span data-x="remove-sfx" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>
                 </span>`;
      }

      const matchedToken = tokens.find(tok => tok.toLowerCase().trim() === tokenLower);
      if (!t || !matchedToken) {
        return `[${t}]`;
      }
      
      let color = "#4f46e5"; // default fallback
      if (speakerColorsMap) {
        const matchedKey = Object.keys(speakerColorsMap).find(k => k.toLowerCase() === tokenLower);
        if (matchedKey) {
          color = speakerColorsMap[matchedKey];
        }
      }
      
      // Soft / Faded Pill type style: 10% opacity color background, 30% border, full opacity text
      const inlineStyle = `background-color: ${color}1a; color: ${color}; border: 1px solid ${color}4d;`;

      return `<span contenteditable="false"
                    data-plain="[${t}]"
                    style="${inlineStyle}"
                    class="bark-token inline-flex items-center whitespace-nowrap py-0.5 px-2 rounded-full text-xs leading-none mr-1 font-extrabold shadow-sm">
                 [${t}]<span data-x="remove" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>
               </span>`;
    });

    html = html.replace(/&lt;Pause:\s*(\d+(?:\.\d+)?)\s*seconds&gt;|<Pause:\s*(\d+(?:\.\d+)?)\s*seconds>/gi, (_, sec1, sec2) => {
      const sec = sec1 || sec2;
      return `<span contenteditable="false"
                    data-plain="<Pause: ${sec} seconds>"
                    class="pause-token inline-flex items-center gap-1.5 whitespace-nowrap py-0.5 px-2 bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300 border border-amber-200 dark:border-amber-900 rounded-full text-xs leading-none mr-1 font-bold">
                 <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="shrink-0"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                 Pause: ${sec}s
                 <span data-x="remove-pause" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>
               </span>`;
    });

    // Render Markdown headings visually while maintaining plain text compatibility
    html = html.replace(/^(#{1,3})\s+(.+)$/gm, (_, hash, content) => {
      const level = hash.length;
      const sizeClass = level === 1 ? "text-xl font-black text-indigo-600 dark:text-indigo-400 my-2 block" : level === 2 ? "text-lg font-bold text-slate-800 dark:text-slate-100 my-1.5 block" : "text-base font-semibold text-slate-700 dark:text-slate-200 my-1 block";
      return `<span class="md-heading ${sizeClass}" data-md="${hash}">${hash} ${content}</span>`;
    });

    // Restore caret marker element if present
    html = html.replace(new RegExp(CARET_MARKER_SENTINEL, "g"), '<span id="editor-caret-marker"></span>');

    return html;
  };

  const plain = (html) => {
    if (!html) return "";
    let txt = htmlToPlainText(html);
    txt = txt.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    return txt;
  };

  // Sync external value when changed outside
  useEffect(() => {
    if (!edRef.current) return;
    const current = plain(edRef.current.innerHTML);
    if (current !== value) {
      edRef.current.innerHTML = decorate(value);
    }
  }, [value]);

  // Re-decorate content when available tokens or speaker colors change
  useEffect(() => {
    if (!edRef.current) return;
    const plainText = plain(edRef.current.innerHTML);
    if (plainText) {
      const marker = saveCaretMarker();
      edRef.current.innerHTML = decorate(plainText);
      restoreCaretMarker(marker);
    }
  }, [tokens, speakerColorsMap]);

  const commit = (render = true) => {
    if (!edRef.current) return;
    const newText = plain(edRef.current.innerHTML);
    onChange(newText);
    if (render) {
      const marker = saveCaretMarker();
      edRef.current.innerHTML = decorate(plain(edRef.current.innerHTML));
      restoreCaretMarker(marker);
    }
  };

  const saveCaretMarker = () => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0) {
      const range = sel.getRangeAt(0);
      if (edRef.current && edRef.current.contains(range.startContainer)) {
        const marker = document.createElement("span");
        marker.id = "editor-caret-marker";
        marker.style.display = "none";
        range.insertNode(marker);
        return marker;
      }
    }
    return null;
  };

  const restoreCaretMarker = (marker) => {
    if (!marker || !edRef.current) return;
    const realMarker = edRef.current.querySelector("#editor-caret-marker");
    if (realMarker) {
      const range = document.createRange();
      const sel = window.getSelection();
      range.selectNode(realMarker);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
      realMarker.remove();
    }
  };

  const insertToken = (tok) => {
    const triggerInfo = getFilterAndTriggerPos();
    if (triggerInfo) {
      const { node, offset, filter } = triggerInfo;
      const sel = window.getSelection();
      const range = sel.getRangeAt(0);

      // Select trigger character plus typed filter characters to replace
      range.setStart(node, offset);
      range.setEnd(node, offset + filter.length + 1);
      range.deleteContents();

      if (tok === "sfx:" || tok === "music:" || tok === "Pause:") {
        let textToInsert = "";
        let caretOffsetFromEnd = 0;
        if (tok === "sfx:") {
          textToInsert = "[sfx: ]";
          caretOffsetFromEnd = 1;
        } else if (tok === "music:") {
          textToInsert = "[music: ]";
          caretOffsetFromEnd = 1;
        } else {
          textToInsert = "<Pause: 1.0 seconds>";
          caretOffsetFromEnd = 0;
        }
        
        const textNode = document.createTextNode(textToInsert);
        range.insertNode(textNode);
        
        // Position caret before closing bracket/tag if editable
        range.setStart(textNode, textToInsert.length - caretOffsetFromEnd);
        range.collapse(true);
        sel.removeAllRanges();
        sel.addRange(range);
        
        commit(false);
        setPopup(false);
        edRef.current.focus();
        return;
      }

      const pill = document.createElement("span");
      pill.setAttribute("contenteditable", "false");
      pill.setAttribute("data-plain", `[${tok}]`);
      
      const tokenLower = tok.toLowerCase().trim();
      let color = "#4f46e5";
      if (speakerColorsMap) {
        const matchedKey = Object.keys(speakerColorsMap).find(k => k.toLowerCase() === tokenLower);
        if (matchedKey) {
          color = speakerColorsMap[matchedKey];
        }
      }
      pill.style.cssText = `background-color: ${color}1a; color: ${color}; border: 1px solid ${color}4d;`;
      pill.className = "bark-token inline-flex items-center whitespace-nowrap py-0.5 px-2 rounded-full text-xs leading-none mr-1 font-extrabold shadow-sm";
      pill.innerHTML = `[${tok}]<span data-x="remove" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>`;
      range.insertNode(pill);

      // Insert a trailing space after the pill for seamless typing flow
      const spaceNode = document.createTextNode(" ");
      pill.after(spaceNode);

      range.setStartAfter(spaceNode);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);

      commit(false);
      setPopup(false);
      edRef.current.focus();
    }
  };

  // Expose token insertion method for parent layout buttons
  useImperativeHandle(ref, () => ({
    insertToken(tok) {
      if (!edRef.current) return;
      
      let range = null;
      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0) {
        const activeRange = sel.getRangeAt(0);
        if (edRef.current.contains(activeRange.startContainer)) {
          range = activeRange;
        }
      }

      if (!range && savedRange.current) {
        range = savedRange.current;
      }

      if (!range) {
        edRef.current.focus();
        const selection = window.getSelection();
        range = document.createRange();
        range.selectNodeContents(edRef.current);
        range.collapse(false); // collapse to end
        selection.removeAllRanges();
        selection.addRange(range);
      }

      const pill = document.createElement("span");
      pill.setAttribute("contenteditable", "false");
      pill.setAttribute("data-plain", `[${tok}]`);
      
      const tokenLower = tok.toLowerCase().trim();
      let color = "#4f46e5";
      if (speakerColorsMap) {
        const matchedKey = Object.keys(speakerColorsMap).find(k => k.toLowerCase() === tokenLower);
        if (matchedKey) {
          color = speakerColorsMap[matchedKey];
        }
      }
      pill.style.cssText = `background-color: ${color}1a; color: ${color}; border: 1px solid ${color}4d;`;
      pill.className = "bark-token inline-flex items-center whitespace-nowrap py-0.5 px-2 rounded-full text-xs leading-none mr-1 font-extrabold shadow-sm";
      pill.innerHTML = `[${tok}]<span data-x="remove" class="cursor-pointer ml-1 font-normal opacity-70 hover:opacity-100">×</span>`;

      range.deleteContents();
      range.insertNode(pill);

      const spaceNode = document.createTextNode(" ");
      pill.after(spaceNode);

      range.setStartAfter(spaceNode);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);

      savedRange.current = range;
      commit(false);
      edRef.current.focus();
    }
  }));

  const handleInput = () => {
    // Process content change cleanly without kicking caret out mid-typing
    commit(false);
    triggerCaretCallback();
    
    // Check if autocomplete trigger state matches
    const triggerInfo = getFilterAndTriggerPos();
    if (triggerInfo) {
      setFilter(triggerInfo.filter);
      const caretCoords = getRelativeCaretPos();
      setPos(caretCoords);
      setPopup(true);
    } else {
      setPopup(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "]") {
      setTimeout(() => commit(true), 10);
    }
    // Backspace into a pill → convert back to editable "[" and reopen autocomplete
    // so the user can change tag type (speaker / sfx / music) purely with keyboard
    if (e.key === "Backspace" && !popup) {
      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0) {
        const range = sel.getRangeAt(0);
        if (range.collapsed) {
          const node = range.startContainer;
          const offset = range.startOffset;
          let pillToConvert = null;

          // Caret at start of a text node – check previous sibling for a pill
          if (node.nodeType === Node.TEXT_NODE && offset === 0) {
            const prev = node.previousSibling;
            if (prev && prev.nodeType === Node.ELEMENT_NODE) {
              const pillClasses = ["bark-token", "music-token", "sfx-token", "pause-token"];
              if (pillClasses.some(cls => prev.classList?.contains(cls))) {
                pillToConvert = prev;
              }
            }
          // Caret inside the container element – check child at offset-1
          } else if (node.nodeType === Node.ELEMENT_NODE && edRef.current.contains(node) && offset > 0) {
            const prev = node.childNodes[offset - 1];
            if (prev && prev.nodeType === Node.ELEMENT_NODE) {
              const pillClasses = ["bark-token", "music-token", "sfx-token", "pause-token"];
              if (pillClasses.some(cls => prev.classList?.contains(cls))) {
                pillToConvert = prev;
              }
            }
          }

          if (pillToConvert) {
            e.preventDefault();
            // Replace the pill with a plain "[" to reopen the full autocomplete dropdown
            const editableText = document.createTextNode("[");
            pillToConvert.replaceWith(editableText);

            // Position caret right after "["
            const newRange = document.createRange();
            newRange.setStart(editableText, 1);
            newRange.collapse(true);
            sel.removeAllRanges();
            sel.addRange(newRange);

            // Sync parent value without re-rendering
            commit(false);

            // Open the autocomplete dropdown
            setFilter("");
            const caretCoords = getRelativeCaretPos();
            setPos(caretCoords);
            setPopup(true);
            return;
          }
        }
      }
    }
    if (popup) {
      if (e.key === "Escape") {
        e.preventDefault();
        setPopup(false);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        if (filtered.length > 0) {
          insertToken(filtered[activeIndex]);
        } else {
          setPopup(false);
        }
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((prev) => (prev + 1) % filtered.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((prev) => (prev - 1 + filtered.length) % filtered.length);
        return;
      }
    }
  };

  const handleKeyUp = () => {
    saveRange();
    triggerCaretCallback();
    const triggerInfo = getFilterAndTriggerPos();
    if (triggerInfo) {
      setFilter(triggerInfo.filter);
      const caretCoords = getRelativeCaretPos();
      setPos(caretCoords);
      setPopup(true);
    } else {
      setPopup(false);
    }
  };

  const click = (e) => {
    // Map close-button data attributes to their pill container selectors
    const removeActions = {
      "remove": ".bark-token",
      "remove-pause": ".pause-token",
      "remove-music": ".music-token",
      "remove-sfx": ".sfx-token",
    };
    const removeType = e.target.dataset?.x;
    if (removeType && removeActions[removeType]) {
      const pill = e.target.closest(removeActions[removeType]);
      if (pill) {
        // Delete via execCommand so the browser's undo stack records it.
        // Ctrl+Z / Cmd+Z will restore the pill; the input event fires
        // automatically afterwards → handleInput → commit(false).
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNode(pill);
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("delete", false, null);
      }
    }
    saveRange();
    triggerCaretCallback();
  };

  const handleMouseUp = () => {
    saveRange();
    triggerCaretCallback();
  };

  const handleCopy = (e) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !edRef.current) return;

    if (!edRef.current.contains(sel.anchorNode) && !edRef.current.contains(sel.focusNode)) return;

    const range = sel.getRangeAt(0);
    const container = document.createElement("div");
    container.appendChild(range.cloneContents());

    const cleanText = plain(container.innerHTML);
    e.clipboardData.setData("text/plain", cleanText);
    e.preventDefault();
  };

  const handleCut = (e) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !edRef.current) return;

    if (!edRef.current.contains(sel.anchorNode) && !edRef.current.contains(sel.focusNode)) return;

    const range = sel.getRangeAt(0);
    const container = document.createElement("div");
    container.appendChild(range.cloneContents());

    const cleanText = plain(container.innerHTML);
    e.clipboardData.setData("text/plain", cleanText);
    e.preventDefault();

    document.execCommand("delete", false, null);
    commit(true);
    triggerCaretCallback();
  };

  const handlePaste = (e) => {
    e.preventDefault();
    let pastedText = e.clipboardData.getData("text/plain") || e.clipboardData.getData("text");
    if (!pastedText) return;

    // Check single token paste shortcut
    const m = pastedText.trim().match(/^\[([^\[\]]+)\]$/);
    if (m) {
      const matchedToken = tokens.find(tok => tok.toLowerCase().trim() === m[1].toLowerCase().trim());
      if (matchedToken) {
        insertToken(matchedToken);
        return;
      }
    }

    // Clean CRLF to LF
    pastedText = pastedText.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

    document.execCommand("insertText", false, pastedText);
    commit(true);
    triggerCaretCallback();
  };

  const handleBlur = () => {
    saveRange();
    commit(true);
    // Allow minor delay for mouse clicks on dropdown options
    setTimeout(() => {
      setPopup(false);
    }, 200);
  };

  const getFilteredOptions = () => {
    const triggerInfo = getFilterAndTriggerPos();
    if (!triggerInfo) return [];
    const { trigger, filter: triggerFilter } = triggerInfo;
    if (trigger === "/") {
      const options = ["sfx:", "music:", "Pause:"];
      return options.filter(opt => opt.toLowerCase().startsWith(triggerFilter.toLowerCase()));
    } else {
      // "[" trigger: speaker tokens plus sfx/music tag helpers
      const soundOptions = ["sfx:", "music:"];
      const allOptions = [...tokens, ...soundOptions];
      return allOptions.filter((t) =>
        t.toLowerCase().startsWith(triggerFilter.toLowerCase())
      );
    }
  };
  const filtered = getFilteredOptions();

  // Reset active dropdown index when filter scope changes
  useEffect(() => {
    setActiveIndex(0);
  }, [filter]);

  const isBlocked = disabled || isLoading;

  return (
    <div className="relative">
      <div
        ref={edRef}
        contentEditable={!isBlocked}
        suppressContentEditableWarning
        className={
          "focus:outline-none whitespace-pre-wrap break-words transition-opacity duration-200 " +
          className +
          (isBlocked ? " pointer-events-none select-none opacity-45 cursor-not-allowed" : "")
        }
        onKeyDown={isBlocked ? undefined : handleKeyDown}
        onKeyUp={isBlocked ? undefined : handleKeyUp}
        onInput={isBlocked ? undefined : handleInput}
        onClick={isBlocked ? undefined : click}
        onMouseUp={isBlocked ? undefined : handleMouseUp}
        onFocus={isBlocked ? undefined : triggerCaretCallback}
        onBlur={isBlocked ? undefined : handleBlur}
        onCopy={isBlocked ? undefined : handleCopy}
        onCut={isBlocked ? undefined : handleCut}
        onPaste={isBlocked ? undefined : handlePaste}
      ></div>

      {/* Loading & Disabled Spinner Overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-40 bg-white/80 dark:bg-slate-900/85 backdrop-blur-xs rounded-xl flex flex-col items-center justify-center gap-3 p-4 select-none animate-fadeIn">
          <div className="flex items-center gap-3.5 px-4 py-3 bg-white dark:bg-slate-800 border border-indigo-200 dark:border-indigo-800/80 rounded-2xl shadow-xl shadow-indigo-500/10">
            <svg className="animate-spin h-5 w-5 text-indigo-600 dark:text-indigo-400 shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-indigo-500 animate-ping shrink-0" />
                {loadingMessage || "AI Task in Progress..."}
              </span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">
                Script Editor is paused while processing
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Placeholder display */}
      {!value && !isLoading && (
        <div className="absolute inset-0 p-4 text-gray-400 pointer-events-none select-none whitespace-pre-wrap">
          {placeholder}
        </div>
      )}

      {!isBlocked && popup && filtered.length > 0 && (
        <ul
          className="absolute z-50 w-52 max-h-48 overflow-auto 
                     bg-white dark:bg-slate-850 border border-gray-200 dark:border-slate-800 rounded-md shadow text-gray-950 dark:text-slate-100"
          style={{ left: pos.x, top: pos.y + 18 }} // offset downwards slightly for caret clearance
        >
          {filtered.map((tok, idx) => (
            <li
              key={tok}
              className={`px-2 py-1.5 text-sm cursor-pointer border-b border-gray-50 dark:border-slate-800 last:border-0 ${
                idx === activeIndex ? "bg-blue-100 dark:bg-blue-900/60 text-blue-900 dark:text-blue-200 font-medium" : "hover:bg-blue-50 dark:hover:bg-slate-800"
              }`}
              onMouseDown={(e) => {
                e.preventDefault();
                insertToken(tok);
              }}
            >
              {tok === "sfx:" ? "🔊 sfx: sound effect" : tok === "music:" ? "🎵 music: background" : tok === "Pause:" ? "⏸ Pause: silence" : `[${tok}]`}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
});

export default TagEditor;
