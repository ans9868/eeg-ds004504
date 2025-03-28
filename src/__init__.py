__version__ = "1.0.0"

from pathlib import Path

# Define the root directory of your project as one level above the src folder
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def getSubPath(sub, derivatives=True):
    if derivatives:
        path = PROJECT_ROOT / 'ds004504' / "derivatives" / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
    else:
        path = PROJECT_ROOT / 'ds004504' / "sub" / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
    
    if not path.exists():
        raise FileNotFoundError(f"The path was not found for {sub}, path: {path}")
    
    return path

def participantsInfoPath():
    return PROJECT_ROOT / "participants.tsv"
