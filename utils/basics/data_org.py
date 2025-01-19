import platform
import os
from pathlib import Path
import shutil
import pandas as pd

def curr_computer():
    if platform.system() == "Darwin":  # macOS
        root = "/Volumes/cooper/"
        # root = '/Volumes/bbari1/'  # Uncomment if needed
        # sep = "/"
    elif platform.system() == "Windows":
        root = r"F:\\"  # Windows
        # root = 'Z:\\'  # Uncomment if needed
        # root = 'C:\\Users\\zhixi\\Documents\\data\\'  # Uncomment if needed
        # root = 'D:\\'  # Uncomment if needed
        # sep = "\\"
    else:
        raise RuntimeError("Unsupported operating system")
    return root


def parse_session_string(file_or_folder):
    """
    Parses input string to generate corresponding pathData outputs.

    Args:
        file_or_folder (str): Session name or name of .asc file.
                              e.g., 'mBB041d20161006' or 'mBB041d20161006.asc'
        root (str): Root folder, e.g., 'G:\\'
        sep (str): Separator, e.g., '\\' or '/'

    Returns:
        dict: A dictionary containing session-related path data.
    """
    root = curr_computer()
    filename = file_or_folder
    # Split animal name and date
    animal_name, date = filename.split("d", 1)
    animal_name = animal_name[1:]  # Remove leading 'm'
    date = date[:9]
    session_folder = f"m{animal_name}d{date}"

    if ".asc" in file_or_folder:  # Input is an .asc file
        behavioral_data_path = os.path.join(root, animal_name, session_folder, "behavior", filename)
        suptitle_name = filename.split(".asc")[0]
        save_fig_name = suptitle_name
        videopath = os.path.join(root, animal_name, session_folder, "pupil")
        sorted_folder_location = lick_path = None
    else:  # Input is the folder
        file_path = os.path.join(root, animal_name, session_folder, "behavior")
        all_files = list(Path(file_path).iterdir())
        file_ind = [
            file.name for file in all_files if file_or_folder + ".asc" in file.name
        ]

        if file_ind:
            behavioral_data_path = os.path.join(file_path, file_ind[0])
            suptitle_name = file_ind[0][:-4]  # Remove '.asc'
        else:  # If looking at a folder without behavioral data
            suptitle_name = None
            behavioral_data_path = None

        save_fig_name = suptitle_name
        videopath = os.path.join(root, animal_name, session_folder, "pupil")

        if file_or_folder[-1].isalpha():  # If last character is alphabetical
            sorted_folder_location = os.path.join(
                root, animal_name, session_folder, "sorted", f"session {file_or_folder[-1]}"
            )
            lick_path = os.path.join(root, animal_name, session_folder, "lick", file_or_folder[-1])
        else:
            sorted_folder_location = os.path.join(root, animal_name, session_folder, "sorted", "session")
            lick_path = os.path.join(root, animal_name, session_folder, "lick", "session")
        
        photometry_path = os.path.join(root, animal_name, session_folder, "photometry")

    # Construct output dictionary
    path_data = {
        "aniName": animal_name,
        "suptitleName": suptitle_name,
        "sessionFolder": session_folder,
        "sortedFolder": sorted_folder_location,
        "animalName": animal_name,
        "saveFigName": save_fig_name,
        "saveFigFolder": os.path.join(root, animal_name, session_folder, "figures"),
        "baseFolder": os.path.join(root, animal_name, session_folder),
        "behavioralDataPath": behavioral_data_path,
        "date": date,
        "videopath": videopath,
        "lickPath": lick_path,
        "photometryPath": photometry_path,
    }

    # Check for neuralynx folders
    base_folder = path_data["baseFolder"]
    nlynx_folder = Path(base_folder, "neuralynx")
    if nlynx_folder.is_dir():
        path_data.update(
            {
                'nlynx_folder': os.path.join(base_folder, "neuralynx"),
                'nlynxFolderOpto': os.path.join(base_folder, "neuralynx", "opto"),
                'nlynxFolderSession': os.path.join(base_folder, "neuralynx", "session"),
            }
        )

    for key, value in path_data.items():
        if ('Path' in value or 'Folder' in value) and not os.path.exists(value):
            os.mkdir(value)
    return path_data

def move_subfolders(dir1, dir2, subfolders=None):
    """
    Moves specified subfolders from dir1 to dir2 if they exist.
    If no subfolders are specified, moves all subfolders.

    :param dir1: Source directory
    :param dir2: Destination directory
    :param subfolders: List of subfolder names to move (optional)
    """
    # Ensure the destination directory exists
    os.makedirs(dir2, exist_ok=True)
    
    # Get the list of subfolders to move
    if subfolders is None:
        # Move all subfolders
        subfolders = [d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))]
    
    # Move each subfolder if it exists
    for subfolder in subfolders:
        src_path = os.path.join(dir1, subfolder)
        dest_path = os.path.join(dir2, subfolder)
        if os.path.exists(src_path) and os.path.isdir(src_path):
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                print(f"Copied: {src_path} -> {dest_path}")
            else:
                print(f"Subfolder already exists: {dest_path}")
        else:
            print(f"Subfolder not found: {src_path}")

def move_animals(animal_list, target_root = curr_computer(), subfolders=['behavior', 'pupil', 'photometry']):
    root = curr_computer()
    for animal in animal_list:
        curr_sessions = [session for session in os.listdir(os.path.join(root, animal)) if os.path.isdir(os.path.join(root, animal, session))]
        for session in curr_sessions:
            curr_dir = os.path.join(root, animal, session)
            target_dir = os.path.join(target_root, animal, session)
            print(f"Current session: {session}")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            move_subfolders(curr_dir, target_dir, subfolders=subfolders)

def get_session_list(xlFile, sheetName, col):
    root = curr_computer()
    xlFile_path = os.path.join(root, xlFile+'.xlsx')
    df = pd.read_excel(xlFile_path, sheet_name=sheetName)
    session_list = df[col].tolist()
    # remove nan values
    session_list = [x for x in session_list if str(x) != 'nan']
    return session_list
