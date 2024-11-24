import os
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


train_file = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "train", "train.json"))
test_file = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "test", "test.json"))
df = pd.read_json(train_file)
df2 = pd.read_json(test_file)

df = df.transpose()
df2 = df2.transpose()

# Preprocess the data
for col in df.columns:
    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

most_frequent_values = {}
for col in df.columns:
    mode_series = df[col][df[col].notnull()].mode()
    if not mode_series.empty:
        most_frequent_values[col] = mode_series.iloc[0]
    else:
        most_frequent_values[col] = None

for col_name in df.columns:
    most_frequent_value = most_frequent_values[col_name]
    if isinstance(most_frequent_value, list):
        most_frequent_value = most_frequent_value[0] if most_frequent_value else None
    df.loc[df[col_name].isnull(), col_name] = most_frequent_value

label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Match columns in df2 with df
missing_cols = set(df.columns) - set(df2.columns)
for col in missing_cols:
    df2[col] = most_frequent_values.get(col, np.nan)
df2 = df2[df.columns]

for col in df2.columns:
    df2[col] = label_encoder.fit_transform(df2[col].astype(str))

def remove_outliers_lof(df):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    preds = lof.fit_predict(df)
    mask = preds != -1
    return df[mask]

df = remove_outliers_lof(df)
df2 = remove_outliers_lof(df2)

features = df.columns[df.columns != 'label']
X_train = df[features].values
X_test = df2[features].values
y_train = df['label'].values
y_test = df2['label'].values

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert data to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train_pca, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_pca, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define a simpler neural network architecture
model = Sequential([
    Dense(128, input_dim=X_train_pca.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tensor, y_train_tensor, epochs=50, batch_size=32, validation_data=(X_test_tensor, y_test_tensor))

# Make predictions
predictions = model.predict(X_test_tensor)

# Convert predictions to binary values
predictions = (predictions > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Write predictions to a file
output_path = os.path.join(os.sep, "usr", "src", "app", "output", "labels")
with open(output_path, "w") as write_file:
    json.dump({str(i): int(pred[0]) for i, pred in enumerate(predictions)}, write_file)