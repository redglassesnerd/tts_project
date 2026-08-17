import docx

def convert_docx_to_markdown(file_stream_or_path):
    try:
        doc = docx.Document(file_stream_or_path)
        markdown_lines = []
        
        for element in doc.element.body:
            if element.tag.endswith('p'):
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()
                if not text:
                    continue
                
                style_name = p.style.name.lower()
                if 'heading 1' in style_name:
                    markdown_lines.append(f"# {text}\n")
                elif 'heading 2' in style_name:
                    markdown_lines.append(f"## {text}\n")
                elif 'heading 3' in style_name:
                    markdown_lines.append(f"### {text}\n")
                elif 'heading' in style_name:
                    markdown_lines.append(f"#### {text}\n")
                elif 'list bullet' in style_name:
                    markdown_lines.append(f"- {text}")
                elif 'list number' in style_name:
                    markdown_lines.append(f"1. {text}")
                else:
                    # Parse formatting runs (bold/italic)
                    formatted_text = ""
                    for run in p.runs:
                        run_text = run.text
                        if not run_text:
                            continue
                        if run.bold and run.italic:
                            formatted_text += f"***{run_text}***"
                        elif run.bold:
                            formatted_text += f"**{run_text}**"
                        elif run.italic:
                            formatted_text += f"*{run_text}*"
                        else:
                            formatted_text += run_text
                    
                    markdown_lines.append(formatted_text + "\n")
                    
            elif element.tag.endswith('tbl'):
                table = docx.table.Table(element, doc)
                table_lines = []
                for row_idx, row in enumerate(table.rows):
                    row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    table_lines.append("| " + " | ".join(row_cells) + " |")
                    if row_idx == 0:
                        table_lines.append("| " + " | ".join(["---"] * len(row_cells)) + " |")
                markdown_lines.append("\n".join(table_lines) + "\n")
                
        return "\n".join(markdown_lines).strip()
    except Exception as e:
        print(f"Error converting docx to markdown: {e}")
        raise e
