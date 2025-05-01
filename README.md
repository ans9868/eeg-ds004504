Welcome to Our project!
```text
config.yaml
    ↓
config_handler.py  -->  provides global settings and parameters
    ↓
preprocess_sets.py  -->  loads raw EEG (.set) files
                       --> applies filters, re-referencing, epoching
                       --> outputs MNE Epochs objects
    ↓
feature_extraction.py
    --> iterates over subjects, epochs, electrodes
    --> uses feature_extraction_helper.py to compute:
        - band power (delta, theta, etc.)
        - entropy, RMS, skewness, Hjorth params, etc.
    --> outputs feature dictionaries
    ↓
populate_schemas.py
    --> uses schema_definition.py to define structure
    --> organizes extracted features into:
        - per-epoch tables
        - per-electrode-per-epoch tables
        - per-band-per-electrode-per-epoch tables
    ↓
dimensionality_reduction.py
    --> optionally reduces feature dimensions using PCA
    --> returns transformed feature sets for modeling or visualization
```


Here is a [full report](https://drive.google.com/file/d/14tasoNzfkrh8TtQJ5RnaQljPXRAV1kAX/view?usp=sharing).
