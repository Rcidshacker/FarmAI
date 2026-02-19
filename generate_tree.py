import os

# Config: Folders and files to strictly ignore
IGNORE_NAMES = {'node_modules', 'venv', '.venv', '.git', '__pycache__', 'dist', 'build', '.DS_Store'}
IGNORE_FILES = {'project_structure.md', 'generate_tree.py'}

def write_tree(dir_path, prefix="", f=None):
    """
    Recursive function to write the tree structure.
    """
    try:
        # Get valid list of files and folders
        items = os.listdir(dir_path)
    except PermissionError:
        return

    # Filter and sort items (directories first, then files is a common preference, 
    # but alphabetical is standard for 'tree')
    filtered_items = sorted([
        item for item in items 
        if item not in IGNORE_NAMES and item not in IGNORE_FILES
    ])
    
    count = len(filtered_items)
    
    for index, item in enumerate(filtered_items):
        path = os.path.join(dir_path, item)
        is_last = (index == count - 1)
        
        # dynamic connector: └── for the last item, ├── for others
        connector = "└── " if is_last else "├── "
        
        # Write the current item
        if f:
            f.write(f"{prefix}{connector}{item}\n")
        
        # If it's a directory, recurse into it
        if os.path.isdir(path):
            # Prepare the prefix for the children
            # If we are the last item, children don't need the vertical bar │
            new_prefix = prefix + ("    " if is_last else "│   ")
            write_tree(path, new_prefix, f)

def generate_file():
    output_file = 'project_structure.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        root_name = os.path.basename(os.getcwd())
        f.write(f"# Project Structure: {root_name}\n\n```text\n.\n")
        
        # Start recursion from current directory
        write_tree('.', "", f)
        
        f.write("```\n")
        
    print(f"✅ Tree generated successfully in '{output_file}'")

if __name__ == "__main__":
    generate_file()