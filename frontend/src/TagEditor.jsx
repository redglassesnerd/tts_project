/* src/TagEditor.jsx */
import React, { useState, useRef, useEffect } from "react";

/**
 * TagEditor
 * =========
 * • Type “[” → dropdown of tokens
 * • Enter / click inserts coloured-pill token
 * • Each pill shows “×” to remove
 * • `value` prop = plain text containing “[token]”
 */
export default function TagEditor({
  value,
  onChange,
  tokens = [],
  placeholder = "",
  className = "",
}) {
  const edRef   = useRef(null);
  const [popup, setPopup]     = useState(false);   // show dropdown?

  // whenever we open the dropdown, clear previous filter
  useEffect(() => {
    if (popup) setFilter("");
  }, [popup]);

  const [filter, setFilter]   = useState("");      // text after “[”
  const [pos, setPos]         = useState({ x: 0, y: 0 });

  // Compute caret coordinates relative to the editable container
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
        y: caretRect.bottom - containerRect.top + edRef.current.scrollTop
      };
    }
    // fallback to bottom-left of container
    return { x: 0, y: containerRect.height + edRef.current.scrollTop };
  };

  /* ---------- helpers ---------- */
  const decorate = (txt) =>
    txt.replace(/\[([^\[\]\s]+?)\]/g, (_, t) => {
      // only wrap if t is non-empty and in the allowed list
      if (!t || !tokens.includes(t)) {
        return `[${t}]`;
      }
      return `<span contenteditable="false"
                    class="bark-token inline-flex items-center whitespace-nowrap py-0.5 px-2 bg-indigo-100 text-indigo-700 rounded-full text-xs leading-none mr-1">
                 [${t}]<span data-x="remove" class="cursor-pointer ml-1">×</span>
               </span>`;
    });

  const plain = (html) => {
    const div = document.createElement("div");
    div.innerHTML = html;
    div.querySelectorAll(".bark-token").forEach((s) => {
      const txt = s.textContent.replace(/×$/, "").trim();
      s.replaceWith(txt);
    });
    return div.textContent || "";
  };

  /* ---------- sync external value ---------- */
  useEffect(() => {
    if (!edRef.current) return;
    const current = plain(edRef.current.innerHTML);
    if (current !== value) edRef.current.innerHTML = decorate(value);
  }, [value]);

  /* ---------- events ---------- */
  const commit = (render = true) => {
    if (!edRef.current) return;
    const newText = plain(edRef.current.innerHTML);
    onChange(newText);
    // only re-render if requested
    if (render) {
      edRef.current.innerHTML = decorate(newText);
    }
  };

  const insertToken = (tok) => {
    const sel = window.getSelection();
    if (!sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    /* delete the lone "[" that triggered autocomplete */
    range.setStart(range.startContainer, range.startOffset - 1);
    range.deleteContents();

    const pill = document.createElement("span");
    pill.setAttribute("contenteditable", "false");
    pill.className =
      "bark-token inline-flex items-center whitespace-nowrap py-0.5 px-2 bg-indigo-100 text-indigo-700 rounded-full text-xs leading-none mr-1";
    pill.innerHTML = `[${tok}]<span data-x="remove" class="cursor-pointer ml-1">×</span>`;
    range.insertNode(pill);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
    commit(false);     // only update parent value, keep current DOM to avoid stray “x”
    setPopup(false);
  };

  const keyDown = (e) => {
    if (popup) {
      if (e.key === "Escape") {
        setPopup(false);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        insertToken(filtered[0]);
        return;
      }
      if (e.key === "Backspace") {
        e.preventDefault();
        setFilter(f => f.slice(0, -1));
        setPopup(true);
        return;
      }
      if (e.key.length === 1 && /[A-Za-z0-9_-]/.test(e.key)) {
        e.preventDefault();
        setFilter(f => f + e.key);
        setPopup(true);
        return;
      }
    }
    if (e.key === "[") {
      // record caret for dropdown placement
      const { x, y } = getRelativeCaretPos();
      setPos({ x, y });
      setFilter("");
      setPopup(true);
    }
  };

  const click = (e) => {
    if (e.target.dataset.x === "remove") {
      try {
        e.target.closest(".bark-token").remove();
      } catch {}
      commit();
    }
  };

  const handlePaste = (e) => {
    e.preventDefault();
    const pastedText = e.clipboardData.getData('text');
    // if user pasted exactly “[token]”, insert as a pill
    const m = pastedText.match(/^\[([^\[\]\s]+)\]$/);
    if (m && tokens.includes(m[1])) {
      insertToken(m[1]);
      return;
    }
    // otherwise fallback to plain insertion and update value
    document.execCommand('insertText', false, pastedText);
    commit(false);
  };

  /* ---------- filter list ---------- */
  const filtered = tokens.filter((t) =>
    t.toLowerCase().startsWith(filter.toLowerCase())
  );

  return (
    <div className="relative">
      <div
        ref={edRef}
        contentEditable
        suppressContentEditableWarning
        className={
          "focus:outline-none whitespace-pre-wrap break-words " + className
        }
        onKeyDown={keyDown}
        onClick={click}
        onPaste={handlePaste}
        onInput={() => commit(false)}
      >
      </div>

      {/* overlay placeholder (outside contentEditable to avoid React DOM conflicts) */}
      {!value && (
        <div
          className="absolute inset-0 p-4 text-gray-400 pointer-events-none select-none whitespace-pre-wrap"
        >
          {placeholder}
        </div>
      )}

      {popup && filtered.length > 0 && (
        <ul
          className="absolute z-50 w-52 max-h-48 overflow-auto 
                     bg-white border border-gray-200 rounded-md shadow"
          style={{ left: pos.x, top: pos.y }}
        >
          {filtered.map((tok) => (
            <li
              key={tok}
              className="px-2 py-1 text-sm cursor-pointer hover:bg-blue-100"
              onMouseDown={(e) => {
                e.preventDefault();
                insertToken(tok);
              }}
            >
              [{tok}]
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}