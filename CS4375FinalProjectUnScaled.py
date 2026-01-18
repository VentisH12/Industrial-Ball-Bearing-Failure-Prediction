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
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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
  return {
      "Model" : name,
      "Average Confusion Matrix Entries" : average_cm,
      "Accuracy" : accuracy_score(label_encoder.transform(y_test), label_encoder.transform(y_pred)),
      "True Positive Rate" : tpr,
      "False Positive Rate" : fpr,
      "Area under the ROC Curve (AUC)" : auc_score
  }


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

# displaying data dictionary keys
'''print("Inner Ring Keys:", IR.keys())
print("Ball Keys:", B0.keys())
print("Outer Ring 1 Keys:", OR1.keys())
print("Outer Ring 2 Keys:", OR2.keys())
print("Outer Ring 3 Keys:", OR3.keys())
print("Normal:", Normal.keys(), "\n")'''

# converting arrays to lists inside dicts
IR_021_list = dict_to_flatlist(IR)
B0_021_list = dict_to_flatlist(B0)
OR1_021_list = dict_to_flatlist(OR1)
OR2_021_list = dict_to_flatlist(OR2)
OR3_021_list = dict_to_flatlist(OR3)
OR_021_list = OR1_021_list + OR2_021_list + OR3_021_list
Normal_021_list = dict_to_flatlist(Normal)

# original data vs flattened list
'''print(len(B0_021_list))
print(IR["X215_DE_time"].shape[0], "\n")
print(IR["X215_FE_time"].shape[0], "\n")'''

#print(Normal["X098_DE_time"].shape[0])
#print(Normal["X098_FE_time"].shape[0])
#print(Normal["X099_DE_time"].shape[0])
#print(Normal["X099_FE_time"].shape[0])

'''print("OR List length: ", len(OR_021_list))
print("OR1 separate list length: ", len(OR1_021_list))
print("OR2 separate list length: ", len(OR2_021_list))
print("OR3 separate list length: ", len(OR3_021_list))'''

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
'''print(IR_duration)
print(OR_duration)
print(B0_duration)
print(Normal_duration, "\n")'''

# min duration
short_duration = min(IR_duration, OR_duration, B0_duration, Normal_duration)
#print("minimum duration:", short_duration)

time = np.arange(0, short_duration, 1/sampling_rate)
#print(time)

IR_data = IR_021_list[:len(time)]
#print(len(IR_data))

OR_data = OR_021_list[:len(time)]
#print(len(OR_data))

B0_data = B0_021_list[:len(time)]
#print(len(B0_data))

Normal_data = Normal_021_list[:len(time)]
#print(len(Normal_data))

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
X_train, X_test, y_train, y_test = train_test_split(all_dataframe.drop('Fault', axis = 1), all_dataframe['Fault'], test_size=.2, random_state=42, shuffle = True)

# ====Step 3: Encoding Labels====

# Encode Labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Debugging
#print("y_test", y_test)
#print("y_test_encoded", y_test_encoded)

# ====Step 4: Defining Models====

# Initialize list to store metrics
extended_metrics = []

# Define Dictionary of models
regModels = {"Logistic Regression" : LogisticRegression(max_iter = 10000),
          "Random Forest" : RandomForestClassifier(n_estimators = 100),
          "Support Vector Machine" : SVC(kernel = 'rbf', probability=True),
          "K Nearest Neighbors" : KNeighborsClassifier(n_neighbors = 5),
          "Gradient Boosting" : GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 3, random_state = 42)}

# ====Step 5: Predict with Traditional Models====

# Loop through and perform fit and predict for each model
for key in regModels:
  regModels[key].fit(X_train, y_train_encoded)
  y_pred = regModels[key].predict(X_test)
  y_pred = label_encoder.inverse_transform(y_pred)
  y_prob = regModels[key].predict_proba(X_test) if hasattr(regModels[key], 'predict_proba') else None

  extended_metrics.append(calculate_metrics(key, y_test, y_pred, y_prob))

  #accuracy = accuracy_score(y_test, y_pred)

  # Displaying first 10 predicted vs true results for each model
  #print(f"{key}: {accuracy}")
  #print("y_pred: ", y_pred[:10])
  #print("y_test: ", list(y_test)[:10])
  table_data = {
      "Model" : key,
      "True Fault": list(y_test)[:10],
      "Predicted Fault" : list(y_pred)[:10],
  }
  table_data_df = pd.DataFrame(table_data)
  #display(table_data_df)

  # plotting confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
  #disp.plot(cmap='Blues')
  #disp.ax_.set_title(f"{key}")

# ====Step 6: Predict with CNN====

#print("Deep Machine Learning Models:")

# Convolutional Neural Networks (CNN)
y_categorical = to_categorical(y_train_encoded)
CNN = Sequential([InputLayer(shape = (X_train.shape[1],1)),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(y_categorical.shape[1], activation='softmax')
                ])

CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN.fit(np.expand_dims(X_train, axis=2), y_categorical, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

y_pred_cnn = CNN.predict(np.expand_dims(X_test, axis=2))
y_pred = label_encoder.inverse_transform(np.argmax(y_pred_cnn, axis=1))

accuracy = accuracy_score(y_test, y_pred)
#print(f"CNN: {accuracy}")

cnn_table_data = {
    "Model" : "Convolutional Neural Network(CNN)",
    "True Fault": list(y_test)[:10],
    "Predicted Fault" : list(y_pred)[:10],
}
cnn_table_data_df = pd.DataFrame(cnn_table_data)
#print("CNN DataFrame for first ten predicted vs true faults")
#display(cnn_table_data_df)

# plotting CNN confusion matrix
cnn_cm = confusion_matrix(y_test, y_pred)
cnn_disp = ConfusionMatrixDisplay(cnn_cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
#cnn_disp.plot(cmap='Blues')
#cnn_disp.ax_.set_title(f"Convolutional Neural Networks(CNN)")

extended_metrics.append(calculate_metrics("Convolutional Neural Network(CNN)", y_test, y_pred, y_pred_cnn))

# ====Step 7: Predict with LSTM====

# Long-Short Term Memory (LSTM)
LSTM = Sequential([InputLayer(shape = (1, X_train.shape[1])),
                LSTM(50),
                Dense(100, activation='relu'),
                Dense(y_categorical.shape[1], activation='softmax')
                ])

LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
LSTM.fit(np.expand_dims(X_train, axis=1), y_categorical, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

y_pred_lstm = LSTM.predict(np.expand_dims(X_test, axis=1))
y_pred = label_encoder.inverse_transform(np.argmax(y_pred_lstm, axis=1))

accuracy = accuracy_score(y_test, y_pred)
#print(f"LSTM: {accuracy}")

lstm_table_data = {
    "Model" : "Long-Short Term Memory",
    "True Fault": list(y_test)[:10],
    "Predicted Fault" : list(y_pred)[:10],
}
lstm_table_data_df = pd.DataFrame(lstm_table_data)
#print("LSTM DataFrame for first ten predicted vs true faults")
#display(lstm_table_data_df)

# plotting LSTM confusion matrix
lstm_cm = confusion_matrix(y_test, y_pred)
lstm_disp = ConfusionMatrixDisplay(lstm_cm, display_labels = ["Inner Ring", "Outer Ring", "Ball", "Normal"])
#lstm_disp.plot(cmap='Blues')
#lstm_disp.ax_.set_title(f"Long Short Term Memory(LSTM)")

extended_metrics.append(calculate_metrics("Long Short Term Memory(LSTM)", y_test, y_pred, y_pred_lstm))

# Display extended metrics
pd.set_option('display.max_columns', None)
extended_metrics_df = pd.DataFrame(extended_metrics)
display(extended_metrics_df)

