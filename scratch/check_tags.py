import re

with open("frontend/src/App.jsx", "r") as f:
    lines = f.readlines()

block = lines[2626:3480] # 0-indexed: lines 2627 to 3480

tag_regex = re.compile(r"<(/?[a-zA-Z0-9\-]+)(?:\s+[^>]*?)?>")

stack = []

for idx, line in enumerate(block):
    line_num = 2627 + idx
    # Simple regex parsing
    # strip quotes content to avoid matching tags in strings
    cleaned_line = re.sub(r'"[^"]*"', '""', line)
    cleaned_line = re.sub(r"'[^']*'", "''", cleaned_line)
    cleaned_line = re.sub(r"`[^`]*`", "``", cleaned_line)
    
    matches = tag_regex.finditer(cleaned_line)
    for match in matches:
        tag = match.group(1)
        # ignore self-closing tags like <input />, <img />, <circle />, <path />, <textarea />, <svg /> if they are self-closing
        # wait, let's check if the match in the line has a closing '/'
        match_str = match.group(0)
        is_self_closing = match_str.endswith("/>") or tag.lower() in ["input", "img", "br", "hr", "circle", "path", "rect"]
        
        if is_self_closing and not tag.startswith("/"):
            continue
            
        if tag.startswith("/"):
            close_tag = tag[1:]
            if stack:
                top_tag, top_line = stack[-1]
                if top_tag == close_tag:
                    stack.pop()
                else:
                    print(f"Mismatch at line {line_num}: closing </{close_tag}> but expected </{top_tag}> (opened at line {top_line})")
            else:
                print(f"Mismatch at line {line_num}: closing </{close_tag}> but stack is empty")
        else:
            stack.append((tag, line_num))

print("\n--- Remaining open tags ---")
for tag, line in stack:
    print(f"Tag <{tag}> opened at line {line} remains open")
