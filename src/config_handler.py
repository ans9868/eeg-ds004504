import yaml
import os

_CONFIG = None  # internal singleton

def initiate_config(config_path="config.yaml", interactive=False):
    """
    Loads config.yaml into memory. If it doesn't exist, prompt user for values and save new file.
    If interactive=True, allow editing of loaded/default config.
    """
    global _CONFIG

    if not os.path.exists(config_path):
        print(f"[INFO] Config file not found at '{config_path}'.")
        print("→ Starting interactive setup.")
        _CONFIG = _interactive_config()
        with open(config_path, "w") as f:
            yaml.safe_dump(_CONFIG, f)
        print(f"[INFO] New config file created at '{config_path}'")
    else:
        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)

    if interactive:
        print("[INFO] Entering interactive mode to update config.")
        _CONFIG = _interactive_config(_CONFIG)
        with open(config_path, "w") as f:
            yaml.safe_dump(_CONFIG, f)
        print(f"[INFO] Updated config saved to '{config_path}'")

    return _CONFIG

def _interactive_config(default_config=None):
    """
    Prompt user for config values. Requires non-empty data_path.
    """
    if default_config is None:
        default_config = {
            "data_path": "",
            "derivatives": True,
            "method": "welch",
            "windowLength": 3,
            "stepSize": 1.50,
            "freqBands": {
                "Delta": [0.5, 4],
                "Theta": [4, 8],
                "Alpha": [8, 12],
                "Beta": [12, 30],
            }
        }

    print("\n--- CONFIG SETUP ---")
    print("The 'data_path' is mandatory and should be the **FOLDER** where the 'ds004504' directory is located.\n")

    config = {}
    for key, default in default_config.items():
        # Special case for data_path
        if key == "data_path":
            while True:
                user_input = input(f"{key} [{default}]: ").strip()
                if user_input:
                    config[key] = user_input
                    break
                elif default:
                    config[key] = default
                    break
                else:
                    print("❗ ERROR: 'data_path' is required and cannot be empty.")
        
        # Special case for freqBands
        elif key == "freqBands":
            print("\nDefine frequency bands of interest.")
            use_defaults = input(f"Use default freqBands? (y/n) [{default}]: ").strip().lower()
            if use_defaults in ("", "y", "yes"):
                config[key] = default
            else:
                custom_bands = {}
                print("Enter frequency bands (name + min/max Hz). Leave name empty to finish.")
                while True:
                    name = input("Band name (e.g. Alpha): ").strip()
                    if not name:
                        break
                    try:
                        fmin = float(input(f"  {name} min freq: "))
                        fmax = float(input(f"  {name} max freq: "))
                        if fmin >= fmax:
                            print("min must be less than max. Try again.")
                            continue
                        custom_bands[name] = [fmin, fmax]
                    except ValueError:
                        print("Please enter valid numbers for frequencies.")
                config[key] = custom_bands if custom_bands else default
        # special case for welch / multitaper 
        elif key == "method":
            valid_methods = ["welch", "multitaper"]
            while True:
                user_input = input(f"{key} [{default}] (welch or multitaper): ").strip().lower()
                if not user_input and default:
                    config["method"] = default.lower()
                    break
                elif user_input in valid_methods:
                    config["method"] = user_input
                    break
                else:
                    print("ERROR: Method must be either 'welch' or 'multitaper'. Please try again.")
        # Everything else (derivatives, windowLength, stepSize)
        else:
            while True:
                user_input = input(f"{key} [{default}]: ").strip()
                if not user_input:
                    config[key] = default
                    break
                try:
                    config[key] = type(default)(user_input)
                    break
                except ValueError:
                    print(f"Could not convert '{user_input}' to {type(default).__name__}. Please try again.")

    return config

def load_config():
    if _CONFIG is None:
        raise RuntimeError("Config not initialized. Call initiate_config() first.")
    return _CONFIG
