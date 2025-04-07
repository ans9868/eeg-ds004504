__version__ = "1.0.0"

import os 

# Get the path to the src directory
SRC_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the path to the parent directory (one level up from src)
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
