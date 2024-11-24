# Anomaly Detection in Operating System Logs

## Project Overview

The  **Anomaly-Detection**  project is designed to identify unusual patterns or behaviors within system and network logs. It leverages machine learning models to analyze and classify events, enabling the detection of potential security threats or system issues.

## Components

### 1. Data Processing

-   **`interpret.py`**
    -   **Purpose:**  Processes packet capture (PCAP) files and converts them into a structured JSON format.
    -   **Functionality:**
        -   Iterates through training and testing directories.
        -   Extracts and processes TCP payloads from PCAP files.
        -   Writes the processed data to  `train.json`  and  `test.json`.

### 2. Machine Learning Models

-   **`linear_svc.py`**
    -   **Purpose:**  Trains and evaluates a Linear Support Vector Classifier for anomaly detection.
    -   **Functionality:**
        -   Loads and preprocesses JSON data.
        -   Removes outliers using Isolation Forest.
        -   Encodes categorical variables and standardizes features.
        -   Trains the LinearSVC model with class weighting.
        -   Evaluates model performance and saves predictions.
        - 
-   **`pytorch_nn.py`**
    -   **Purpose:**  Builds and trains a neural network using PyTorch for binary classification.
    -   **Functionality:**
        -   Loads and preprocesses JSON data.
        -   Removes outliers and encodes features.
        -   Defines an enhanced neural network architecture with multiple layers, batch normalization, and dropout.
        -   Trains the model with early stopping based on validation accuracy.
        -   Evaluates model performance and saves predictions.

-   **`tensorflow_nn.py`**
    -   **Purpose:**  Constructs and trains a neural network using TensorFlow and Keras for binary classification.
    -   **Functionality:**
        -   Loads and preprocesses JSON data.
        -   Removes outliers using Local Outlier Factor and applies PCA for dimensionality reduction.
        -   Defines a neural network architecture with dense layers, batch normalization, and dropout.
        -   Compiles and trains the model, making predictions on the test set.
        -   Evaluates model performance and saves predictions.

# linear_svc

## Workflow

1.  **Data Loading:**
    
    -   Opens and reads  `train.json`  and  `test.json`  files.
    -   Transposes the DataFrames to align data correctly.
2.  **Data Preprocessing:**
    
    -   Converts dictionary-type columns to JSON strings.
    -   Handles missing values by imputing the most frequent value in each column.
    -   Encodes categorical variables using  LabelEncoder.
3.  **Handling Missing Columns:**
    
    -   Ensures that the test dataset has the same columns as the training dataset by adding any missing columns with the most frequent values.
4.  **Outlier Detection and Removal:**
    
    -   Applies Isolation Forest to remove outliers from both training and testing datasets.
5.  **Feature Selection:**
    
    -   Selects feature columns excluding the target label.
6.  **Class Weight Calculation:**
    
    -   Computes class weights to handle imbalanced classes in the target variable.
7.  **Data Standardization:**
    
    -   Standardizes the feature data using  StandardScaler  to ensure that each feature contributes equally to the model training.
8.  **Model Training:**
    
    -   Initializes and trains a  LinearSVC  model with the computed class weights.
9.  **Model Prediction and Evaluation:**
    
    -   Predicts labels for the test dataset.
    -   Evaluates model performance using accuracy score, confusion matrix, and classification report.
10.   **Prediction Results:**
      -   Saved as a JSON file in the  `output/labels`  directory, mapping each instance to its predicted label.


## Output

-   **Evaluation Metrics:**
    
    -   **Accuracy Score:**  Measures the proportion of correctly classified instances.
    -   **Confusion Matrix:**  Provides a summary of prediction results on the classification problem.
    -   **Classification Report:**  Includes precision, recall, f1-score for each class.
-   **Prediction Results:**
    
    -   Saved as a JSON file in the  `output/labels`  directory, mapping each instance to its predicted label.

# pytorch_nn.py

The  pytorch_nn.py  script is designed to train and evaluate a neural network using PyTorch for binary classification tasks. It processes training and testing data, preprocesses the data, removes outliers, trains the model with early stopping, evaluates its performance, and saves the prediction results.

## Workflow

1.  **Data Loading:**
    
    -   Opens and reads  train.json  and  `test.json`  files.
    -   Transposes the DataFrames to align data correctly.
2.  **Data Preprocessing:**
    
    -   Converts dictionary-type columns to JSON strings.
    -   Handles missing values by imputing the most frequent value in each column.
    -   Encodes categorical variables using  LabelEncoder.
3.  **Handling Missing Columns:**
    
    -   Ensures that the test dataset has the same columns as the training dataset by adding any missing columns with the most frequent values.
4.  **Outlier Detection and Removal:**
    
    -   Applies Isolation Forest to remove outliers from both training and testing datasets.
5.  **Feature Selection:**
    
    -   Selects feature columns excluding the target label.
6.  **Class Weight Calculation:**
    
    -   Computes class weights to handle imbalanced classes in the target variable.
7.  **Data Standardization:**
    
    -   Standardizes the feature data using  StandardScaler  to ensure that each feature contributes equally to the model training.
8.  **Model Training:**
    
    -   Initializes and trains the  EnhancedNN  model with the computed class weights.
9.  **Model Prediction and Evaluation:**
    
    -   Predicts labels for the test dataset.
    -   Evaluates model performance using accuracy score, confusion matrix, and classification report.
10.  **Saving Predictions:**
    
     -   Saves the prediction results to a JSON file in the  `output/labels`  directory.

## Output

-   **Evaluation Metrics:**
    
    -   **Accuracy Score:**  Measures the proportion of correctly classified instances.
    -   **Confusion Matrix:**  Provides a summary of prediction results on the classification problem.
    -   **Classification Report:**  Includes precision, recall, f1-score for each class.
-   **Prediction Results:**
    
    -   Saved as a JSON file in the  `output/labels`  directory, mapping each instance to its predicted label.


# tensorflow_nn.py

The  tensorflow_nn.py  script is designed to train and evaluate a neural network using TensorFlow and Keras for binary classification tasks. It processes training and testing data, preprocesses the data, removes outliers, trains the model, evaluates its performance, and saves the prediction results.

## Workflow

1.  **Data Loading:**
    
    -   Opens and reads  `train.json`  and  `test.json`  files.
    -   Transposes the DataFrames to align data correctly.
2.  **Data Preprocessing:**
    
    -   Converts dictionary-type columns to JSON strings.
    -   Handles missing values by imputing the most frequent value in each column.
    -   Encodes categorical variables using  LabelEncoder.
3.  **Handling Missing Columns:**
    
    -   Ensures that the test dataset has the same columns as the training dataset by adding any missing columns with the most frequent values.
4.  **Outlier Detection and Removal:**
    
    -   Applies Local Outlier Factor to remove outliers from both training and testing datasets.
5.  **Feature Selection and Scaling:**
    
    -   Selects feature columns excluding the target label.
    -   Standardizes the feature data using  StandardScaler  to ensure that each feature contributes equally to the model training.
6.  **Dimensionality Reduction:**
    
    -   Applies PCA to reduce the number of features while retaining 95% of the variance.
7.  **Data Conversion:**
    
    -   Converts the processed data into TensorFlow tensors for model training.
8.  **Model Compilation and Training:**
    
    -   Compiles the neural network model with the Adam optimizer and binary cross-entropy loss.
    -   Trains the model for 50 epochs with a batch size of 32, using the test dataset for validation.
9.  **Model Prediction and Evaluation:**
    
    -   Makes predictions on the test dataset.
    -   Converts predictions to binary values based on a threshold of 0.5.
    -   Evaluates model performance using accuracy score, confusion matrix, and classification report.
10.  **Saving Predictions:**
    
     -   Saves the prediction results to a JSON file in the  `output/labels`  directory.


# interpret.py

The  interpret.py  script processes packet capture (PCAP) files and converts them into a human-readable format, which is then written to JSON files. The script processes files in both the  `train`  and  `test`  directories under  InputData.

## Functions

### concatenate_tcp_payloads(packets)
This function concatenates TCP payloads from packets.

### process_file(file_path)
This function reads a PCAP file, concatenates TCP payloads, decodes the payloads into a human-readable string, and removes newlines and extra whitespaces.

## Usage

The script performs the following steps:

1.  **Initialize JSON File for Training Data**
    
    -   Opens  `InputData/train/train.json`  in write mode and writes the opening brace  `{`.
2.  **Process Training Data Files**
    
    -   Iterates through all files in the  train  directory.
    -   Skips the  `InputData/train/train.json`  file.
    -   For each file, it:
        -   Reads the file and processes it using the  process_file  function.
        -   Appends the processed content to  `InputData/train/train.json`  in a human-readable format.
        -   Adds a comma  `,`  after each entry except the last one.
        -   Stops processing after 200 files.
3.  **Finalize JSON File for Training Data**
    
    -   Writes the closing brace  `}`  to  `InputData/train/train.json`.
4.  **Initialize JSON File for Test Data**
    
    -   Opens  `InputData/test/test.json`  in write mode and writes the opening brace  `{`.
5.  **Process Test Data Files**
    
    -   Iterates through all files in the  test  directory.
    -   Skips the  `InputData/test/test.json`  file.
    -   For each file, it:
        -   Reads the file and processes it using the  process_file  function.
        -   Appends the processed content to  `InputData/test/test.json`  in a human-readable format.
        -   Adds a comma  `,`  after each entry except the last one.
6.  **Finalize JSON File for Test Data**
    
    -   Writes the closing brace  `}`  to  `InputData/test/test.json`.

The  linear_svc.py  script is designed to train and evaluate a Linear Support Vector Classifier (LinearSVC) for anomaly detection. It processes training and testing data, preprocesses the data, trains the model, evaluates its performance, and saves the prediction results.

