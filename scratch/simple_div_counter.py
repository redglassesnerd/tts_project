with open("frontend/src/App.jsx", "r") as f:
    lines = f.readlines()

stack = []

for idx, line in enumerate(lines):
    line_num = idx + 1
    # Find all <div and </div> on this line
    i = 0
    while i < len(line):
        if line[i:i+4] == "<div":
            # Check if self-closing
            rest = line[i:]
            end_angle = rest.find(">")
            is_self = False
            if end_angle != -1:
                if rest[:end_angle].endswith("/"):
                    is_self = True
            if not is_self:
                stack.append((line_num, line.strip()))
            i += 4
        elif line[i:i+6] == "</div>":
            if stack:
                stack.pop()
            else:
                print(f"Extra closing </div> at line {line_num}: {line.strip()}")
            i += 6
        else:
            i += 1

print("\n--- Unclosed divs at end of file ---")
for line_num, content in stack:
    print(f"Line {line_num}: {content}")
