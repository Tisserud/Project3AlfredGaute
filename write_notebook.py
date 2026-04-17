import jupytext
import nbformat
import os

FILE_NAME = ".src/oppg2.ipynb"
IGNORE_TAG = "imports"

def get_cells_from_file(filepath: str) -> list[any]:
    """
    Reads markdown `.md` and python `.py` files to
    `.ipynb` cells from file at filepath `filepath`.

    For formatting see `jupytext`'s documentation.

    Parameters:
        filepath (str): relative filepath of file to read

    Returns:
        file_cells(list[any]): contents of file if file is a `.py` or a `.md`
    """
    with open(filepath, 'r') as file:
        content = file.read() + "\n"

    file_cells = []
    if filepath.endswith(".py"):
        file_cells = jupytext.reads(content, fmt = 'py:percent').cells
    elif filepath.endswith(".md"):
        file_cells = jupytext.reads(content, fmt = 'md:myst').cells

    return file_cells

def get_cells_from_dir(dirpath: str) -> list[any]:
    """
    Loops through directory alphabetically, adding every
    file within as a list of `.ipynb` cells.
    
    Directories are included, and handled recursively.

    For formatting see `jupytext`'s documentation.

    Parameters:
        dirpath (str): relative filepath directory

    Returns:
        file_cells(list[any]): contents of all `.py` or a `.md` in directory
    """
    dir_cells = []
    file_names = sorted(os.listdir(dirpath))
    for file_name in file_names:
        filepath = dirpath + file_name
        if os.path.isdir(filepath):
            if file_name != '__pycache__':
                file_cells = get_cells_from_dir(dirpath = filepath + '/')       
        else:
            file_cells = get_cells_from_file(filepath = filepath)
        dir_cells.extend(file_cells)
    return dir_cells

def remove_cells_by_tag(cells: list[any], remove_tag: str) -> list[any]:
    """
    Removes every `cell` tagged with `remove_tag`

    Parameters:
        cells (list): list of ipynb cells
        remove_tag (str): tag of cells to remove

    Returns:
        remaining_cells (list): remaining cells
    """
    remaining_cells = []
    for cell in cells:
        if 'metadata' in cell:
            if 'tags' in cell['metadata']:
                if remove_tag in cell['metadata']['tags']:
                    continue
        remaining_cells.append(cell)
    return remaining_cells


notebook_cells = get_cells_from_dir(dirpath = 'tasks/')
notebook_cells = remove_cells_by_tag(notebook_cells, IGNORE_TAG)

# Write to nbformat's dict representation of a notebook
new_notebook = nbformat.v4.new_notebook(cells = notebook_cells)

# Check if there is any difference between old and new (to stop pre-commit from preventing all commits)
if os.path.isfile(FILE_NAME):
    old_notebook = jupytext.read(FILE_NAME)

    # Read to a string for comparison (as dicts aren't directly comparible)
    old_notebook_serialised = jupytext.writes(old_notebook, version=nbformat.NO_CONVERT, fmt = 'py:percent')
    new_notebook_serialised = jupytext.writes(new_notebook, version=nbformat.NO_CONVERT, fmt = 'py:percent')
    if old_notebook_serialised != new_notebook_serialised:
        jupytext.write(new_notebook, FILE_NAME)
else:
    jupytext.write(new_notebook, FILE_NAME)