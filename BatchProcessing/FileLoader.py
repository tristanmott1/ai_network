import json
from pathlib import Path
import shutil  # New import for copying files

def save_analysis(foldername, win_distribution, result_distribution, result_str, jsonl_file_path=None):
    """
    Saves two dictionaries, a result string, and optionally a JSONL file into specified files within a structured folder.

    Parameters:
    - foldername (str): Name of the subfolder inside 'Analysis'.
    - win_distribution (dict): Dictionary mapping states/parties to win percents.
    - result_distribution (dict): Dictionary mapping potential results to probabilities.
    - result_str (str): The string to write into 'results.txt'.
    - jsonl_file_path (str, optional): Path to the JSONL file to be copied into the target folder.
    """

    # Define the base 'Analysis' directory
    analysis_dir = Path('Analysis')

    # Define the target subfolder path
    target_dir = analysis_dir / foldername

    try:
        # Create 'Analysis' directory if it doesn't exist
        analysis_dir.mkdir(exist_ok=True)
        print(f"Ensured that the directory '{analysis_dir}' exists.")

        # Create 'Analysis/foldername' directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)
        print(f"Ensured that the directory '{target_dir}' exists.")
    except Exception as e:
        print(f"Error creating directories: {e}")
        raise

    # Define file paths
    dict1_path = target_dir / 'win_distribution.json'
    dict2_path = target_dir / 'result_distribution.json'
    results_path = target_dir / 'results.txt'

    try:
        # Save dict1 as JSON
        with dict1_path.open('w', encoding='utf-8') as f:
            json.dump(win_distribution, f, indent=4)

        # Save dict2 as JSON
        with dict2_path.open('w', encoding='utf-8') as f:
            json.dump(result_distribution, f, indent=4)

        # Save the result string
        with results_path.open('w', encoding='utf-8') as f:
            f.write(result_str)

        # If a JSONL file path is provided, copy it into the target directory
        if jsonl_file_path:
            jsonl_filename = Path(jsonl_file_path).name  # Get the filename from the provided path
            destination_path = target_dir / jsonl_filename
            shutil.copy(jsonl_file_path, destination_path)
            print(f"Copied JSONL file to '{destination_path}'.")
    except Exception as e:
        print(f"Error saving files: {e}")
        raise

def load_analysis(foldername):
    """
    Loads two dictionaries from specified files within a structured folder.

    Parameters:
    - foldername (str): Name of the subfolder inside 'Analysis' to load data from.

    Returns:
    - tuple: A tuple containing two dictionaries (dict1, dict2).
    """

    # Define the base 'Analysis' directory
    analysis_dir = Path('Analysis')

    # Define the target subfolder path
    target_dir = analysis_dir / foldername

    # Define file paths
    dict1_path = target_dir / 'win_distribution.json'
    dict2_path = target_dir / 'result_distribution.json'

    try:
        # Load dict1 from JSON
        with dict1_path.open('r', encoding='utf-8') as f:
            dict1 = json.load(f)
        print(f"Loaded win_distribution from '{dict1_path}'.")

        # Load dict2 from JSON
        with dict2_path.open('r', encoding='utf-8') as f:
            dict2 = json.load(f)
        print(f"Loaded result_distribution from '{dict2_path}'.")

        return dict1, dict2
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
        raise 
    except json.JSONDecodeError as json_error:
        print(f"Error decoding JSON: {json_error}")
        raise
    except Exception as e:
        print(f"Error loading files: {e}")
        raise

