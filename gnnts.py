# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve,
                             auc, cohen_kappa_score, accuracy_score)
import dgl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load sensor data into pandas dataframe
data = pd.read_csv("CQI.csv")

# Drop any unnecessary columns (like unnamed columns if they exist)
data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col], errors='ignore')

# Separate features and target
features = data.drop('cqi', axis=1)
target = data['cqi']

# One-hot encode the target variable
OH = OneHotEncoder()
trans_target = OH.fit_transform(target.values.reshape(-1, 1))
trans_target_np = trans_target.toarray()

# Define number of classes
num_classes = trans_target_np.shape[1]

# Impute missing values using the most frequent strategy
simple_imputer = SimpleImputer(strategy="most_frequent")
imputed_data = simple_imputer.fit_transform(features)

# Standardize the features
scaler = StandardScaler()
norm_data = scaler.fit_transform(imputed_data)

### Graph Construction ###
# Define graph structure as a simple chain and add self-loops
graph = dgl.graph([(i, i + 1) for i in range(len(norm_data) - 1)])
graph = dgl.add_self_loop(graph)

# Assign features and labels to the graph nodes
graph.ndata['feat'] = torch.tensor(norm_data, dtype=torch.float32)
graph.ndata['label'] = torch.tensor(trans_target_np.argmax(axis=1), dtype=torch.long)

# **Extract Triangles**: Using DGL's `triangles` function to capture local triangle structures
triangles = dgl.triangles(graph)  # This returns a tensor of node triplets that form triangles
print(f"Number of triangles: {len(triangles)}")

# Split the data into training and testing sets
train_idx, test_idx = train_test_split(range(len(norm_data)), test_size=0.2, random_state=42)

# Create a mask for train and test nodes
train_mask = torch.zeros(len(norm_data), dtype=torch.bool)
train_mask[train_idx] = True
test_mask = torch.zeros(len(norm_data), dtype=torch.bool)
test_mask[test_idx] = True

### Define Simple GCN Model with Attention ###
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.att_weight = nn.Parameter(torch.Tensor(input_dim, 1))
        self.att_bias = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_uniform_(self.att_weight)
        nn.init.zeros_(self.att_bias)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.ndata['a'] = torch.matmul(h, self.att_weight) + self.att_bias
            g.ndata['a'] = F.softmax(g.ndata['a'], dim=1)
            g.ndata['h'] = g.ndata['h'] * g.ndata['a']
            return g.ndata['h']

class SimpleGCNWithAttention(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(SimpleGCNWithAttention, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.attention = AttentionLayer(hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, num_classes)

    def forward(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = self.attention(g, h)
        h = self.conv2(g, h)
        return h

# Instantiate the model
n_features = norm_data.shape[1]
model = SimpleGCNWithAttention(in_feats=n_features, hidden_feats=64, num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training and Evaluation Metrics
def compute_metrics(y_true, y_pred, y_probs, num_classes):
    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # MCC and Cohen's kappa
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    # ROC AUC and PR AUC for each class
    roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    pr_auc = auc(*precision_recall_curve(y_true, y_probs[:, 1])[:2])

    return {
        "conf_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

# Training Function
def train(model, g, features, labels, train_mask, test_mask, optimizer, criterion, num_classes, num_epochs=50):
    metrics = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(g, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(g, features)
            test_loss = criterion(test_logits[test_mask], labels[test_mask])

            # Calculate predictions and accuracy
            test_preds = torch.argmax(test_logits[test_mask], dim=1)
            test_accuracy = (test_preds == labels[test_mask]).float().mean().item() * 100

            # Probabilities and True labels
            test_probs = F.softmax(test_logits[test_mask], dim=1).cpu().numpy()
            test_true = labels[test_mask].cpu().numpy()
            test_preds_np = test_preds.cpu().numpy()

            # Compute metrics
            results = compute_metrics(test_true, test_preds_np, test_probs, num_classes)

            # Log metrics
            metrics.append({
                "epoch": epoch + 1,
                "loss": test_loss.item(),
                "accuracy": test_accuracy,
                **results
            })

            # Print metrics for each epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                  f"ROC AUC: {results['roc_auc']:.4f}, PR AUC: {results['pr_auc']:.4f}, "
                  f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}, "
                  f"MCC: {results['mcc']:.4f}, Kappa: {results['kappa']:.4f}")
            print(f"Confusion Matrix:\n{results['conf_matrix']}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('performance_metrics.csv', index=False)

    return metrics_df

# Run training
save_dir = 'SimpleGCN_50epoch'
metrics_df = train(model, graph, graph.ndata['feat'], graph.ndata['label'], train_mask, test_mask, optimizer, criterion, num_classes, num_epochs=50)

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    If normalize=True, the matrix will show ratios instead of absolute numbers.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Row-wise normalization for true label proportions

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Display percentages with 2 decimal points if normalized
    fmt = '.2' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()  # Automatically adjust subplot params to minimize margins

    # Save the figure with minimal margins
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()

# Assuming final_confusion_matrix has been trained
# Define new class labels
new_labels = ['Steer', 'Do not Steer']

# Plot the final confusion matrix with updated labels and normalized ratios
if final_confusion_matrix is not None:
    plot_confusion_matrix(final_confusion_matrix, classes=new_labels, normalize=True, title='')


import os
import zipfile
from google.colab import files

def zip_folder(folder_path, zip_name, exclude_extensions=[], exclude_files=[]):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1]
                rel_path = os.path.relpath(file_path, os.path.join(folder_path, '..'))
                if not any(file_ext == ext for ext in exclude_extensions) and file not in exclude_files:
                    print(f"Adding {file_path} as {rel_path}")  # Debug print
                    zipf.write(file_path, rel_path)
                else:
                    print(f"Excluding {file_path} due to extension or specific file exclusion")  # Debug print