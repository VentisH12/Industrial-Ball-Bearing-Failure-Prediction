# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:53:58 2025

@author: dache
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:15:25 2025

@author: dache
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, ConfusionMatrixDisplay, roc_auc_score, roc_curve, classification_report, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, InputLayer
from IPython.display import display
from tensorflow.keras.utils import to_categorical


def dict_to_flatlist(dict):
  for key in dict:
    if "DE" in key:
      olis = dict[key].flatten().tolist()
    elif "FE" in key:
      olis2 = dict[key].flatten().tolist()

  return olis + olis2

def stat_features_calc(data):
  RMS = np.sqrt(np.mean(data)**2)
  features = {
      "Min":np.min(data),
      "Max":np.max(data),
      "Mean":np.mean(data),
      "Std":np.std(data),
      "Var":np.var(data),
      "Skew":skew(data),
      "Kurtosis":kurtosis(data),
      "RMS": RMS,
      "Crest Factor":np.max(np.abs(data))/RMS if RMS != 0 else np.nan(),
      "Shape Factor":RMS/np.mean(np.abs(data)) if np.mean(np.abs(data)) != 0 else np.nan(),
  }
  return features

def time_features_calc(data):
  features = {
      "Peak Value":np.max(np.abs(data)),
      "Peak to Peak Value":np.ptp(data),
      "Impulse Factor":np.max(np.abs(data))/np.mean(np.abs(data)) if np.mean(np.abs(data)) != 0 else np.nan(),
      "Zero Crossing Rate":len(np.where(np.diff(np.sign(data)))[0])/len(data)
  }
  return features

def extract_block_features(data, cond_label):
  num_blocks = len(data) // block_size
  #print("num blocks = ", num_blocks)
  all_features = []

  for i in range(num_blocks):
    start_idx = i*block_size
    end_idx = i*block_size + block_size
    block_data = data[start_idx:end_idx]

    stat_features = stat_features_calc(block_data)
    time_features = time_features_calc(block_data)

    # merge features together
    stat_and_time_features = stat_features | time_features

    if cond_label == "IR":
        stat_and_time_features["Fault"] = "Inner_Ring"
    elif cond_label == "OR":
        stat_and_time_features["Fault"] = "Outer_Ring"
    elif cond_label == "B0":
        stat_and_time_features["Fault"] = "Ball"
    elif cond_label == "Normal":
        stat_and_time_features["Fault"] = "Normal"
    else:
        return

    #print(stat_and_time_features)
    all_features.append(stat_and_time_features)
  return all_features

def calculate_metrics(name, y_test, y_pred, y_prob):
  y_test_binarized = label_binarize(label_encoder.transform(y_test), classes=range(len(label_encoder.classes_)))
  cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
  average_cm = cm.mean()
  tp = np.diag(cm)
  fp = cm.sum(axis=0) - tp
  fn = cm.sum(axis=1) - tp
  tn = cm.sum() - (tp + fp + fn)
  tpr = np.nanmean(tp / (tp + fn))
  fpr = np.nanmean(fp / (fp + tn))
  auc_score = roc_auc_score(label_binarize(label_encoder.transform(y_test), classes = range(len(label_encoder.classes_))),
                                           y_prob, multi_class='ovr')
  
  results = {
      "Model" : name,
      "Avg Confusion Matrix Entries" : average_cm,
      "Accuracy" : accuracy_score(label_encoder.transform(y_test), label_encoder.transform(y_pred)),
      "True Positive Rate" : tpr,
      "False Positive Rate" : fpr,
      "Area under ROC Curve (AUC)" : auc_score
  }
  if y_prob is not None:
      # Binarize the true labels
      y_test_binarized = label_binarize(label_encoder.transform(y_test), classes=range(len(label_encoder.classes_)))

      plt.figure(figsize=(8, 6))
      for i in range(y_test_binarized.shape[1]):
          fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
          class_auc = auc(fpr, tpr)
          plt.plot(fpr, tpr, lw=2, label=f"{label_encoder.classes_[i]} (AUC = {class_auc:.2f})")

      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'{name} ROC Curve')
      plt.legend(loc='lower right')
      plt.grid(True)
      plt.show()
  return results


#===Step 1: Feature Engineering===
# import mat data files
IR = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/215_IR.mat")
B0 = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/228_B0.mat")
OR1 = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/240_OR1.mat")
OR2 = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/252_OR2.mat")
OR3 = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/264_OR3.mat")
Normal = scipy.io.loadmat("C:/Users/dache/.spyder-py3/CS4375MiniProj/99_NORMAL.mat")

# cleaning Normal
del Normal["X098_DE_time"]
del Normal["X098_FE_time"]

# converting arrays to lists inside dicts
IR_021_list = dict_to_flatlist(IR)
B0_021_list = dict_to_flatlist(B0)
OR1_021_list = dict_to_flatlist(OR1)
OR2_021_list = dict_to_flatlist(OR2)
OR3_021_list = dict_to_flatlist(OR3)
OR_021_list = OR1_021_list + OR2_021_list + OR3_021_list
Normal_021_list = dict_to_flatlist(Normal)

# Create Sampling Rate and RPM Variable
sampling_rate = 48000
rpm = 1750

# Calculate block size
block_size = int(round((sampling_rate * 60) / rpm, 0)) 

# Normalize data lengths
IR_duration = round(len(IR_021_list)/sampling_rate, 0)
OR_duration = round(len(OR_021_list)/sampling_rate, 0)
B0_duration = round(len(B0_021_list)/sampling_rate, 0)
Normal_duration = round(len(Normal_021_list)/sampling_rate, 0)

# min duration
short_duration = min(IR_duration, OR_duration, B0_duration, Normal_duration)


time = np.arange(0, short_duration, 1/sampling_rate)

IR_data = IR_021_list[:len(time)]
OR_data = OR_021_list[:len(time)]
B0_data = B0_021_list[:len(time)]
Normal_data = Normal_021_list[:len(time)]

IR_elements = extract_block_features(IR_data,"IR")
OR_elements = extract_block_features(OR_data,"OR")
B0_elements = extract_block_features(B0_data,"B0")
Normal_elements = extract_block_features(Normal_data,"Normal")
all_elements = IR_elements + OR_elements + B0_elements + Normal_elements
all_dataframe = pd.DataFrame(all_elements)
all_dataframe = all_dataframe.round(5)
all_dataframe.style
all_dataframe.to_csv("feature_data.csv", index=False)

# ====Step 2: Splitting the data====

# splitting the data
X = all_dataframe.drop('Fault', axis = 1)
y = all_dataframe['Fault']

# Label encode fault labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.25, shuffle=True, random_state=42, stratify=y_encoded
)

# ====Step 4: Defining Models====

# Initialize list to store metrics and plots
extended_metrics = []
confusion_matrices = {}
results_summary = []

# Define Dictionary of models
regModels = {"Logistic Regression" : LogisticRegression(max_iter = 10000),
          "Random Forest" : RandomForestClassifier(n_estimators = 100),
          "Support Vector Machine" : SVC(kernel = 'rbf', probability=True),
          "K Nearest Neighbors" : KNeighborsClassifier(n_neighbors = 5),
          "Gradient Boosting" : GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 3, random_state = 42)}

# ====Step 5: Predict with Traditional Models====

# Loop through and perform fit and predict for each model
for key in regModels:
  regModels[key].fit(X_train, y_train)
  y_pred = regModels[key].predict(X_test)
  y_pred = label_encoder.inverse_transform(y_pred)
  y_true = label_encoder.inverse_transform(y_test)
  y_prob = regModels[key].predict_proba(X_test) if hasattr(regModels[key], 'predict_proba') else None
  
  acc = accuracy_score(y_test, y_pred)
  results_summary.append({'Model': key, 'Accuracy': acc})

  extended_metrics.append(calculate_metrics(key, y_true, y_pred, y_prob))

  # plotting confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
  disp.plot(cmap='Blues')
  disp.ax_.set_title(f"{key}")
  confusion_matrices[key] = disp
  
  print(f"{key} Classification Report:\n", classification_report(y_true, y_pred))
  
  scores = cross_val_score(regModels[key], X_scaled, y_encoded, cv=5)
  print(f"{key} CV Accuracy: {scores.mean():.4f}")

# ====Step 6: Predict with CNN====
y_categorical = to_categorical(y_encoded)
y_train_dl = to_categorical(y_train)
y_test_dl = to_categorical(y_test)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convolutional Neural Networks (CNN)

CNN = Sequential([InputLayer(shape = (X_train_cnn.shape[1],1)),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(y_categorical.shape[1], activation='softmax')
                ])

CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN.fit(np.expand_dims(X_train_cnn, axis=2), y_train_dl, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

cnn_pred = CNN.predict(np.expand_dims(X_test_cnn, axis=2))
cnn_pred_labels = label_encoder.inverse_transform(cnn_pred.argmax(axis=1))
cnn_true_labels = label_encoder.inverse_transform(y_test_dl.argmax(axis=1))

# plotting CNN confusion matrix
cnn_cm = confusion_matrix(y_true, y_pred)
cnn_disp = ConfusionMatrixDisplay(cnn_cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
cnn_disp.plot(cmap='Blues')
cnn_disp.ax_.set_title("Convolutional Neural Networks(CNN)")
confusion_matrices["Convolutional Neural Network"] = cnn_disp

extended_metrics.append(calculate_metrics("CNN", cnn_true_labels, cnn_pred_labels, cnn_pred))

print("CNN Classification Report:\n", classification_report(cnn_true_labels, cnn_pred_labels))


# ====Step 7: Predict with LSTM====

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Long-Short Term Memory (LSTM)
LSTM_model = Sequential([LSTM(50, input_shape = (1, X_train.shape[1])),
                Dense(100, activation='relu'),
                Dense(y_categorical.shape[1], activation='softmax')
                ])

LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
LSTM_model.fit(X_train_lstm, y_train_dl, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

lstm_pred = LSTM_model.predict(X_test_lstm)
lstm_pred_labels = label_encoder.inverse_transform(lstm_pred.argmax(axis=1))
lstm_true_labels = label_encoder.inverse_transform(y_test_dl.argmax(axis=1))

# plotting LSTM confusion matrix
lstm_cm = confusion_matrix(lstm_true_labels, lstm_pred_labels)
lstm_disp = ConfusionMatrixDisplay(lstm_cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
lstm_disp.plot(cmap='Blues')
lstm_disp.ax_.set_title("Long Short-Term Memory(LSTM)")
confusion_matrices["Long Short-Term Memory"] = lstm_disp

extended_metrics.append(calculate_metrics("LSTM", lstm_true_labels, lstm_pred_labels, lstm_pred))

print("LSTM Classification Report:\n", classification_report(lstm_true_labels, lstm_pred_labels))


# Display extended metrics
extended_metrics_df = pd.DataFrame(extended_metrics)
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

table = ax.table(cellText=extended_metrics_df.round(5).values,
                 colLabels=extended_metrics_df.columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("📊 Summary of ML and DL Model Performance Metrics", fontsize=14, fontweight='bold')
plt.show()

# === Step 12: Predict Bearing Failures and Timeframe (Index-as-Time Assumption) ===
best_model_name = sorted(results_summary, key=lambda x: x['Accuracy'], reverse=True)[0]['Model']
print(f"\n📌 Predicting failures using best model: {best_model_name}")
X_full = scaler.transform(X)
true_labels = label_encoder.inverse_transform(y_encoded)
best_model = regModels[best_model_name]
predicted_labels = label_encoder.inverse_transform(best_model.predict(X_full))

failures = []
for i, (true_fault, pred_fault) in enumerate(zip(true_labels, predicted_labels)):
    # if pred_fault != "Normal" and true_fault == "Normal":
    if pred_fault != true_fault:
        failures.append((i, true_fault, pred_fault))

#earliest_failures = {}
#for idx, fault in failures:
#    if fault not in earliest_failures:
#        earliest_failures[fault] = idx

print("\n🕒 Predicted Bearing Failure Timeline (Assuming Index = Time Unit):")

# Print up to 10 failure or misclassification events
if failures:
    print("\n🕒 Misdiagnosed or Failure Transition Events Detected:")
    for idx, true_fault, pred_fault in failures[:10]:
        print(f"🔧 At time index {idx}: true = {true_fault}, predicted = {pred_fault}")
else:
    print("✅ All bearings classified correctly across timeline.")

#if earliest_failures:
#    for fault_type, time_idx in sorted(earliest_failures.items(), key=lambda x: x[1]):
#        print(f"🔧 Bearing predicted to fail with {fault_type} at time index {time_idx}")
#else:
#    print("✅ No abnormal bearing failures predicted on healthy bearings.")
