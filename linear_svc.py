import os
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest 
from sklearn.utils import class_weight
from sklearn.svm import LinearSVC

# Load data
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

def remove_outliers_isolation_forest(df):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    preds = iso_forest.fit_predict(df)
    mask = preds != -1
    return df[mask]

df = remove_outliers_isolation_forest(df)
df2 = remove_outliers_isolation_forest(df2)

feature_cols = [col for col in df.columns if col != 'label']
X_train = df[feature_cols].values
y_train = df['label'].values
X_test = df2[feature_cols].values
y_test = df2['label'].values

classes = np.unique(y_train)
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights_values)}

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_svc = LinearSVC(class_weight=class_weights_dict, max_iter=10000, random_state=42)

# Train the Model on Encoded Data
linear_svc.fit(X_train, y_train)

# Predict on Test Set using Encoded Data
y_pred = linear_svc.predict(X_test)

# Evaluate the Model
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

output_path = os.path.join(os.sep, "usr", "src", "app", "output", "labels")
with open(output_path, "w") as write_file:
    json.dump({str(i): int(pred) for i, pred in enumerate(y_pred)}, write_file)


