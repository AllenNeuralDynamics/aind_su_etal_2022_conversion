import platform
import os
from pathlib import Path


def curr_computer():
    if platform.system() == "Darwin":  # macOS
        root = "/Volumes/cooper/"
        # root = '/Volumes/bbari1/'  # Uncomment if needed
        # sep = "/"
    elif platform.system() == "Windows":
        root = "F:\\"
        # root = 'Z:\\'  # Uncomment if needed
        # root = 'C:\\Users\\zhixi\\Documents\\data\\'  # Uncomment if needed
        # root = 'D:\\'  # Uncomment if needed
        # sep = "\\"
    else:
        raise RuntimeError("Unsupported operating system")
    return root


def parse_session_string(file_or_folder, root):
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
        filepath = os.path.join(root, animal_name, session_folder, "behavior")
        all_files = list(Path(filepath).iterdir())
        file_ind = [
            file.name for file in all_files if file_or_folder + ".asc" in file.name
        ]

        if file_ind:
            behavioral_data_path = os.path.join(filepath, file_ind[0])
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

    return path_data
