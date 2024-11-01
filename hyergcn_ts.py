# -*- coding: utf-8 -*-
"""HyerGCN-TS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10qMFCI11yhi5qqgAJwXAWV-LMtbF-XqQ
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve,
                             auc, cohen_kappa_score)
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load sensor data into pandas dataframe
data = pd.read_csv("CQI.csv")
data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col], errors='ignore')

# Separate features and target
features = data.drop('cqi', axis=1)
target = data['cqi']

# One-hot encode the target variable
OH = OneHotEncoder()
trans_target = OH.fit_transform(target.values.reshape(-1, 1))
trans_target_np = trans_target.toarray()

# Number of classes for CQI classification
num_classes = trans_target_np.shape[1]

# Impute missing values using the most frequent strategy and standardize the features
simple_imputer = SimpleImputer(strategy="most_frequent")
imputed_data = simple_imputer.fit_transform(features)
scaler = StandardScaler()
norm_data = scaler.fit_transform(imputed_data)

### Hypergraph Construction ###
# Hypergraph edges represent connections among multiple nodes at once.
# Each unique CQI value will form its own hyperedge, connecting nodes with the same CQI.
unique_cqi = np.unique(target)
hyperedges = {cqi_val: np.where(target == cqi_val)[0] for cqi_val in unique_cqi}

# Convert hyperedges into a DGL hypergraph format
hyper_src, hyper_dst = [], []
for hyperedge, nodes in hyperedges.items():
    hyper_src.extend(nodes)       # Each node connects to a hyperedge node
    hyper_dst.extend([hyperedge] * len(nodes))

# Construct a hypergraph with DGL
g = dgl.heterograph({('node', 'connects', 'hyperedge'): (hyper_src, hyper_dst),
                     ('hyperedge', 'connected_by', 'node'): (hyper_dst, hyper_src)})

# Add node features and labels to the hypergraph's nodes
g.nodes['node'].data['feat'] = torch.tensor(norm_data, dtype=torch.float32)
g.nodes['node'].data['label'] = torch.tensor(trans_target_np.argmax(axis=1), dtype=torch.long)

# Train/Test Split Masking - Use a split for prediction tasks
train_idx, test_idx = train_test_split(range(len(norm_data)), test_size=0.2, random_state=42)
train_mask = torch.zeros(len(norm_data), dtype=torch.bool)
train_mask[train_idx] = True
test_mask = torch.zeros(len(norm_data), dtype=torch.bool)
test_mask[test_idx] = True

### Define Hypergraph Neural Network with Attention ###
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(HypergraphAttentionLayer, self).__init__()
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

class SimpleHyperGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(SimpleHyperGCN, self).__init__()
        # Hypergraph convolution layers
        self.conv1 = dglnn.HeteroGraphConv({
            'connects': dglnn.GraphConv(in_feats, hidden_feats)
        }, aggregate='mean')

        self.attention = HypergraphAttentionLayer(hidden_feats)

        self.conv2 = dglnn.HeteroGraphConv({
            'connects': dglnn.GraphConv(hidden_feats, num_classes)
        }, aggregate='mean')

    def forward(self, g, features):
        h = {'node': features, 'hyperedge': torch.zeros(g.num_nodes('hyperedge'), features.shape[1])}
        h = self.conv1(g, h)['node']
        h = F.relu(h)
        h = self.attention(g, h)
        h = self.conv2(g, {'node': h})['node']
        return h

# Instantiate the hypergraph model
model = SimpleHyperGCN(in_feats=norm_data.shape[1], hidden_feats=64, num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Metric computation function
def compute_metrics(y_true, y_pred, y_probs, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
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

# Training function for hypergraph model
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

            test_preds = torch.argmax(test_logits[test_mask], dim=1)
            test_accuracy = (test_preds == labels[test_mask]).float().mean().item() * 100

            test_probs = F.softmax(test_logits[test_mask], dim=1).cpu().numpy()
            test_true = labels[test_mask].cpu().numpy()
            test_preds_np = test_preds.cpu().numpy()

            results = compute_metrics(test_true, test_preds_np, test_probs, num_classes)

            metrics.append({
                "epoch": epoch + 1,
                "loss": test_loss.item(),
                "accuracy": test_accuracy,
                **results
            })

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                  f"ROC AUC: {results['roc_auc']:.4f}, PR AUC: {results['pr_auc']:.4f}, "
                  f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}, "
                  f"MCC: {results['mcc']:.4f}, Kappa: {results['kappa']:.4f}")
            print(f"Confusion Matrix:\n{results['conf_matrix']}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('hypergraph_performance_metrics.csv', index=False)

    return metrics_df

# Run training on the hypergraph model
metrics_df = train(model, g, g.nodes['node'].data['feat'], g.nodes['node'].data['label'],
                   train_mask, test_mask, optimizer, criterion, num_classes, num_epochs=50)

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

# Define the folder and zip file names