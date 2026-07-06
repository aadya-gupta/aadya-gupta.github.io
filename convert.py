import nbformat
from nbconvert import MarkdownExporter
import os
import re

def convert_notebook(notebook_path, output_dir):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(nb)
    if 'outputs' in resources:
        for filename, data in resources['outputs'].items():
            with open(os.path.join(output_dir, filename), 'wb') as f:
                f.write(data)

    # 1. Ensure the frontmatter is at the very top
    # This regex looks for the block between --- and ---
    frontmatter_match = re.search(r'---.*?---', body, re.DOTALL)
    
    if frontmatter_match:
        frontmatter = frontmatter_match.group(0)
        # Remove the frontmatter from the body to reposition it
        body = body.replace(frontmatter, "")
        # Add it back at the very top
        body = frontmatter + "\n\n" + body

    # 2. Wrap output cells in <details>
    # This targets the standard output block pattern
    body = body.replace('<div class="output_subarea">', 
                        '<details><summary class="cursor-pointer font-bold opacity-60 hover:opacity-100">▶ View Output</summary>\n<div class="output_subarea">')
    body = body.replace('</div>', '</div></details>')

    # Save
    filename = os.path.basename(notebook_path).replace('.ipynb', '.md')
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"Metadata injected and converted: {filename}")

if __name__ == "__main__":
    # Point this to your actual notebook file
    convert_notebook('notebooks/PINN_Gliostama.ipynb', 'src/content/lab/')