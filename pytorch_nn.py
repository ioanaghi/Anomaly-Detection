import os
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest


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

# Train the neural network
features = df.columns[df.columns != 'label']
X_train = df[features].values
X_test = df2[features].values
y_train = df['label'].values
y_test = df2['label'].values

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the neural network architecture
class EnhancedNN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)  # Assuming binary classification
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Train the neural network
input_dim = X_train.shape[1]
model = EnhancedNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop with early stopping
num_epochs = 100
best_accuracy = 0
patience = 10
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    
    # Compute loss
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)
        accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    # Early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        trigger_times = 0
    else:
        trigger_times += 1
    
    if trigger_times >= patience:
        print('Early stopping!')
        break

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Final Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Write predictions to a file
output_path = os.path.join(os.sep, "usr", "src", "app", "output", "labels")
with open(output_path, "w") as write_file:
    json.dump({str(i): int(pred) for i, pred in enumerate(y_pred)}, write_file)
