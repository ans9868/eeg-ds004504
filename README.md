# eeg-ds004504

Here we are automating the pipeline from the jupyter notebooks into highly expandable, readable and efficient code.

\*\* Finished pipeline should look like this
eeg-feature-pipeline/
│
├── notebooks/ # Your existing notebooks go here (still usable)
│ ├── Baseline1.4.ipynb
│ ├── Baseline-AD-CN-2.0PCA.ipynb
│
├── src/ # Python codebase
│ ├── **init**.py
│ ├── schema_definition.py # ← defines the Subject, Epoch, Feature schemas
│ ├── populate_schemas.py # ← populates those tables from EEG data
│ ├── feature_extraction.py # ← general feature pipeline (already done!)
│ ├── feature_methods.py # ← PCA, ICA, t-SNE, etc. (dimensionality reduction)
│ ├── preprocess_sets.py # ← loading/preprocessing EEG (already done!)
│ └── modeling.py # ← KNN, SVM, trees, cross-validation, etc.
│
├── main_pipeline.py # ← entry script to run full pipeline
├── requirements.txt # ← dependencies for portability
└── README.md # ← overview, setup instructions
