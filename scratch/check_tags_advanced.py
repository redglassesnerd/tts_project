import re

with open("frontend/src/App.jsx", "r") as f:
    content = f.read()

# Let's extract the return block of AppInner.
# It starts with "return (" and ends before "function AppInnerWrapper"
start_match = list(re.finditer(r"return\s*\(\s*<>", content))
if not start_match:
    print("Could not find start of return block")
    exit(1)
    
start_pos = start_match[-1].start()
end_match = re.search(r"function AppInnerWrapper", content)
if not end_match:
    print("Could not find AppInnerWrapper")
    exit(1)
    
end_pos = end_match.start()

sub_content = content[start_pos:end_pos]

# Function to recursively remove curly braces contents
def remove_braces(s):
    # We find the outermost { and } and remove everything in between
    # To handle nesting, we loop and find matching pairs
    out = []
    level = 0
    in_quote = None
    escape = False
    
    i = 0
    n = len(s)
    while i < n:
        char = s[i]
        
        if escape:
            escape = False
            if level == 0:
                out.append(char)
            i += 1
            continue
            
        if char == '\\':
            escape = True
            if level == 0:
                out.append(char)
            i += 1
            continue
            
        if in_quote:
            if char == in_quote:
                in_quote = None
            if level == 0:
                out.append(char)
            i += 1
            continue
            
        if char in ['"', "'", '`']:
            in_quote = char
            if level == 0:
                out.append(char)
            i += 1
            continue
            
        if char == '{':
            level += 1
            i += 1
            continue
        elif char == '}':
            level = max(0, level - 1)
            i += 1
            continue
            
        if level == 0:
            out.append(char)
        i += 1
        
    return "".join(out)

cleaned = remove_braces(sub_content)

# Now count <div and </div> in the cleaned markup
div_opens = [m.start() for m in re.finditer(r"<div[\s>]", cleaned)]
div_closes = [m.start() for m in re.finditer(r"</div\s*>", cleaned)]

print(f"Cleaned block length: {len(cleaned)}")
print(f"Number of opening <div: {len(div_opens)}")
print(f"Number of closing </div: {len(div_closes)}")

# Let's trace tags in the cleaned block
tag_regex = re.compile(r"<(/?[a-zA-Z0-9\-]+)(?:\s+[^>]*?)?>")
stack = []
lines = cleaned.split("\n")

for line_idx, line in enumerate(lines):
    line_num = line_idx + 1
    for match in tag_regex.finditer(line):
        tag = match.group(1)
        match_str = match.group(0)
        is_self_closing = match_str.endswith("/>") or tag.lower() in ["input", "img", "br", "hr", "circle", "path", "rect", "svg"]
        
        if is_self_closing and not tag.startswith("/"):
            continue
            
        if tag.startswith("/"):
            close_tag = tag[1:]
            if stack:
                top_tag, top_line_num, top_line_content = stack[-1]
                if top_tag == close_tag:
                    stack.pop()
                else:
                    print(f"Mismatch: closing </{close_tag}> but expected </{top_tag}> (opened at line {top_line_num}: {top_line_content.strip()})")
            else:
                print(f"Mismatch: closing </{close_tag}> but stack is empty on line {line_num}: {line.strip()}")
        else:
            stack.append((tag, line_num, line))

print("\n--- Remaining open tags ---")
for tag, line_num, line in stack:
    print(f"Tag <{tag}> opened at line {line_num} remains open: {line.strip()}")
