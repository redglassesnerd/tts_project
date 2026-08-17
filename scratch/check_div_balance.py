import re

with open("frontend/src/App.jsx", "r") as f:
    lines = f.readlines()

# Let's track curly braces nesting per line.
# Whenever we are inside a curly brace, we ignore HTML tags.
# This mimics how JSX behaves.
level = 0
in_quote = None
escape = False

div_stack = []

for idx, line in enumerate(lines):
    line_num = idx + 1
    
    # Process characters on this line to keep track of level
    i = 0
    n = len(line)
    
    # We want to find HTML tags on this line ONLY when level == 0
    # To do that, we scan the line and check when we are at level 0.
    # When at level 0, we can match <div or </div>.
    
    while i < n:
        char = line[i]
        
        if escape:
            escape = False
            i += 1
            continue
            
        if char == '\\':
            escape = True
            i += 1
            continue
            
        if in_quote:
            if char == in_quote:
                in_quote = None
            i += 1
            continue
            
        if char in ['"', "'", '`']:
            in_quote = char
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
            
        # If we are at level 0, we look for div tags
        if level == 0:
            if line[i:i+4] == "<div":
                # check if it's self-closing (though div is not normally self closing)
                # let's look for matching > on the rest of the line
                rest = line[i:]
                closing_slash = False
                end_angle = rest.find(">")
                if end_angle != -1:
                    tag_head = rest[:end_angle]
                    if tag_head.endswith("/"):
                        closing_slash = True
                
                if not closing_slash:
                    div_stack.append(line_num)
                    # print(f"Open <div> at line {line_num}")
                i += 4
                continue
            elif line[i:i+6] == "</div>":
                if div_stack:
                    opened_line = div_stack.pop()
                    # print(f"Close </div> at line {line_num} (matches line {opened_line})")
                else:
                    print(f"Error: unmatched </div> at line {line_num}")
                i += 6
                continue
                
        i += 1

print("\n--- Remaining open divs in App.jsx ---")
for opened_line in div_stack:
    print(f"div opened at line {opened_line} remains open: {lines[opened_line-1].strip()}")
