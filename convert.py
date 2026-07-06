import nbformat
import os
import base64

def convert_notebook(notebook_path, output_dir):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    md_content = []
    frontmatter = ""

    for idx, cell in enumerate(nb.cells):
        # 1. Grab Frontmatter
        if idx == 0 and cell.cell_type == 'markdown' and cell.source.strip().startswith('---'):
            frontmatter = cell.source + "\n\n"
            continue
        
        # 2. Markdown cells
        if cell.cell_type == 'markdown':
            md_content.append(cell.source + "\n\n")
        
        # 3. Process Code Cells
        elif cell.cell_type == 'code':
            md_content.append(f"```python\n{cell.source}\n```\n")
            
            if cell.outputs:
                text_outputs = []
                image_outputs = []
                
                # Separate text logs from image plots
                for o_idx, out in enumerate(cell.outputs):
                    if out.output_type == 'stream':
                        text = out.text.replace('<', '&lt;').replace('>', '&gt;')
                        text_outputs.append(f"<pre><code>{text}</code></pre>\n")
                    
                    elif out.output_type in ['execute_result', 'display_data']:
                        if 'image/png' in out.data:
                            img_data = out.data['image/png']
                            base_name = os.path.basename(notebook_path).replace('.ipynb', '')
                            img_filename = f"{base_name}_plot_{idx}_{o_idx}.png"
                            img_path = os.path.join(output_dir, img_filename)
                            
                            with open(img_path, 'wb') as img_f:
                                img_f.write(base64.b64decode(img_data))
                            
                            # Add to image list instead of text list
                            image_outputs.append(f"\n![plot]({img_filename})\n\n")
                        
                        elif 'text/plain' in out.data:
                            text = out.data['text/plain'].replace('<', '&lt;').replace('>', '&gt;')
                            text_outputs.append(f"<pre><code>{text}</code></pre>\n")
                    
                    elif out.output_type == 'error':
                        text_outputs.append(f"<pre><code style='color:#d9534f;'>{out.ename}: {out.evalue}</code></pre>\n")
                
                # Wrap ONLY the text logs in the dropdown
                if text_outputs:
                    md_content.append('<details class="output-wrapper"><summary>▶ View Output</summary>\n')
                    md_content.extend(text_outputs)
                    md_content.append('</details>\n\n')
                
                # Append the images completely outside the dropdown
                if image_outputs:
                    md_content.extend(image_outputs)

    # 4. Assemble and Save
    final_md = frontmatter + "".join(md_content)
    
    filename = os.path.basename(notebook_path).replace('.ipynb', '.md')
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(final_md)
    
    print(f"Successfully constructed perfect HTML for: {filename}")

if __name__ == "__main__":
    convert_notebook('notebooks/PINN_Glioblastoma.ipynb', 'src/content/lab/')