🧠 Project Summary: EEG Feature Extraction & Classification for Alzheimer's Detection

This project uses data from the OpenNeuro dataset ds004504 (version 1.0.8), which contains resting-state EEG recordings from individuals with Alzheimer’s disease and age-matched healthy controls. The data is preprocessed and segmented into short epochs, enabling time-resolved analysis of spectral power features across various frequency bands (Delta, Theta, Alpha, Beta, and Total).
📦 Dataset Description:

    Subjects: Individuals with Alzheimer's and healthy controls

    Modality: EEG recordings from multiple scalp electrodes

    Sampling: Preprocessed into epochs (time segments)

    Groups:

        Group A: Alzheimer's

        Group C: Control

Each row in the processed dataset represents an (EpochID, SubjectID) combination with computed power values across electrodes and frequency bands.
🛠️ Tools & Technologies Used:

    Apache Spark (PySpark):
    For distributed data handling, preprocessing, and scalable ML workflows

    pandas:
    For lightweight data manipulation and .pkl I/O

    NumPy:
    For PCA and numerical analysis

    scikit-learn:
    For evaluation metrics like classification_report and accuracy

    Matplotlib:
    For plotting PCA explained variance and diagnostics

    PySpark MLlib:
    Used for dimensionality reduction (PCA) and classification (Multilayer Perceptron)

🧪 Workflow Highlights:

    Group Filtering: Alzheimer's and control groups are separated and labeled.

    Feature Normalization: Power features are z-scored using training-set statistics.

    Dimensionality Reduction: PCA is applied to reduce feature dimensionality while preserving variance.

    Classification: A PySpark MultilayerPerceptronClassifier is trained to distinguish between groups.

    Train/Test Splitting: Custom subject-based splitting ensures no overlap in evaluation.

    Evaluation: Classification performance is assessed using AUC and classification_report.

📓 Notebooks

Two example notebooks are included to illustrate the workflow:

    Example_Data_Creation.ipynb – Prepares features, labels, and saves datasets as .pkl

    Example_Data_Preprocessing+ML.ipynb – Performs PCA, trains a classifier, and evaluates model performance.
