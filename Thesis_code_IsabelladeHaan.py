#!/usr/bin/env python
# coding: utf-8

# Thesis code Isabella de Haan

# **Import Libraries**

# In[ ]:


pip install fast_ml


# In[ ]:


pip install mlxtend


# In[ ]:


# data manipulation
import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# data pre-processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# train, test, split and k-fold cross validation
from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# classification algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# feature selection
from mlxtend.feature_selection import SequentialFeatureSelector

# evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# statistical test
from statsmodels.stats.contingency_tables import mcnemar


# **Import Dataset**

# In[ ]:


# import dataset from csv
raw_df = pd.read_csv("data_airline_passenger_satisfaction_survey.csv")


# In[ ]:


# check for any duplicated rows in the dataset
raw_df.duplicated().sum()


# In[ ]:


# check if target variable is balanced
raw_df["Satisfaction"].value_counts()


# **Data pre-processing**

# In[ ]:


# visualize correlations
correlation_matrix = raw_df.corr()
plt.figure(figsize = (16,16))
sns.heatmap(correlation_matrix, annot = True, cmap = 'Blues', fmt = '.2g')
plt.savefig('Correlation_Matrix.png')
plt.show()


# **Eliminate Features**

# In[ ]:


# eliminate 'ID', since this feature is not important for classification
def_df = raw_df.drop(raw_df.iloc[:,[0]], axis = 1)


# In[ ]:


# eliminate 'Arrival Delay' because of high correlation with 'Departure Delay'
def_df = def_df.drop(def_df.iloc[:,[7]], axis = 1)


# **Label Encoding**

# In[ ]:


# function to encode categorical features in the dataframe
label_encoder = LabelEncoder()

def encode_df(df):    
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])
    df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
    df['Class'] = label_encoder.fit_transform(df['Class'])
    df['Satisfaction'] = label_encoder.fit_transform(df['Satisfaction'])
    return df

clean_df = encode_df(def_df)


# **Dataset train, validation, test split**

# In[ ]:


# create train (60%), validation (val) (20%), and test (20%) set.
X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(clean_df, target = 'Satisfaction', train_size = 0.6, valid_size = 0.2, test_size = 0.2)

print(X_train.shape), print(y_train.shape)
print(X_val.shape), print(y_val.shape)
print(X_test.shape), print(y_test.shape)


# ***Training***

# In[ ]:


# hyperparameter tuning and cross-validation on RF model
model_parameters = {'random_forest': {'model': RandomForestClassifier(), 'parameters' : {'n_estimators': [50, 100, 1000, 1500], 'min_samples_split': [2, 5, 10, 12, 15]} }}

performance = []

for i, z in model_parameters.items():
    classifier =  GridSearchCV(z['model'], z['parameters'], cv = 5, return_train_score = False)
    classifier.fit(X_train, y_train)
    performance.append({'model': i, 'best_score': classifier.best_score_, 'best_params': classifier.best_params_})
    
performance_df = pd.DataFrame(performance, columns=['model','best_score','best_params'])
performance_df


# In[ ]:


# hyperparameter tuning and cross-validation on SVM model
model_parameters = {'svm': {'model': SVC(), 'parameters' : {'C': [0.1, 0.2, 0.3], 'kernel': ['linear', 'rbf']} }}

performance = []

for i, z in model_parameters.items():
    classifier =  GridSearchCV(z['model'], z['parameters'], cv = 5, return_train_score = False)
    classifier.fit(X_train, y_train)
    performance.append({'model': i, 'best_score': classifier.best_score_, 'best_params': classifier.best_params_})
    
performance_df = pd.DataFrame(performance, columns=['model','best_score','best_params'])
performance_df


# In[ ]:


# wrapper RF backward feature selection
RF_backward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators = 1500, n_jobs = -1), k_features = (1,21), forward = False, floating = False, verbose = 2, scoring = "accuracy", cv = 5).fit(X_train, y_train)
print(RF_backward_feature_selector.k_score_)
print(RF_backward_feature_selector.k_feature_names_)


# In[ ]:


# make train subset RF wrapper
def make_subset_RF(df):    
    new_df = df.drop(["Ease of Online Booking"], axis = 1)
    new_df2 = new_df.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df3 = new_df2.drop(["Food and Drink"], axis = 1)
    new_df4 = new_df3.drop(["On-board Service"], axis = 1)
    return new_df4

X_train_subset_RF = make_subset_RF(X_train)


# In[ ]:


# make validation subset RF wrapper
def make_subset_RF(df):    
    new_df = df.drop(["Ease of Online Booking"], axis = 1)
    new_df2 = new_df.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df3 = new_df2.drop(["Food and Drink"], axis = 1)
    new_df4 = new_df3.drop(["On-board Service"], axis = 1)
    return new_df4

X_val_subset_RF = make_subset_RF(X_val)


# In[ ]:


# make test subset RF wrapper
def make_subset_RF(df):    
    new_df = df.drop(["Ease of Online Booking"], axis = 1)
    new_df2 = new_df.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df3 = new_df2.drop(["Food and Drink"], axis = 1)
    new_df4 = new_df3.drop(["On-board Service"], axis = 1)
    return new_df4

X_test_subset_RF = make_subset_RF(X_test)


# In[ ]:


# wrapper SVM backward feature selection
SVM_backward_feature_selector = SequentialFeatureSelector(SVC(C = 100), k_features = (1,21), forward = False, floating = False, verbose = 2, scoring = "accuracy", cv = 5).fit(X_train, y_train)
print(SVM_backward_feature_selector.k_score_)
print(SVM_backward_feature_selector.k_feature_names_)


# In[ ]:


# make train subset SVM wrapper:
def make_subset_SVM(df):    
    new_df = df.drop(["Age"], axis = 1)
    new_df2 = new_df.drop(["Class"], axis = 1)
    new_df3 = new_df2.drop(["Flight Distance"], axis = 1)
    new_df4 = new_df3.drop(["Departure Delay"], axis = 1)
    new_df5 = new_df4.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df6 = new_df5.drop(["Food and Drink"], axis = 1)
    return new_df6

X_train_subset_SVM = make_subset_SVM(X_train)


# In[ ]:


# make validation subset SVM wrapper:
def make_subset_SVM(df):    
    new_df = df.drop(["Age"], axis = 1)
    new_df2 = new_df.drop(["Class"], axis = 1)
    new_df3 = new_df2.drop(["Flight Distance"], axis = 1)
    new_df4 = new_df3.drop(["Departure Delay"], axis = 1)
    new_df5 = new_df4.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df6 = new_df5.drop(["Food and Drink"], axis = 1)
    return new_df6

X_val_subset_SVM = make_subset_SVM(X_val)


# In[ ]:


# make test subset SVM filter
def make_subset_SVM(df):    
    new_df = df.drop(["Age"], axis = 1)
    new_df2 = new_df.drop(["Class"], axis = 1)
    new_df3 = new_df2.drop(["Flight Distance"], axis = 1)
    new_df4 = new_df3.drop(["Departure Delay"], axis = 1)
    new_df5 = new_df4.drop(["Departure and Arrival Time Convenience"], axis = 1)
    new_df6 = new_df5.drop(["Food and Drink"], axis = 1)
    return new_df6

X_test_subset_SVM = make_subset_SVM(X_test)


# ***Validation***

# In[ ]:


# RF without feature selection
RF_model_1 = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_1.fit(X_train, y_train)
RF_model_1_predictions = RF_model_1.predict(X_val)

print("accuracy score:", accuracy_score(y_val, RF_model_1_predictions))
print("precision score:", precision_score(y_val, RF_model_1_predictions))
print("recall score:", recall_score(y_val, RF_model_1_predictions))
print("f1 score:", f1_score(y_val, RF_model_1_predictions))


# In[ ]:


# RF with feature selection (filter)
RF_model_filter = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_filter.fit(X_train, y_train)
RF_model_filter_predictions = RF_model_filter.predict(X_val)

print("accuracy score:", accuracy_score(y_val, RF_model_filter_predictions))
print("precision score:", precision_score(y_val, RF_model_filter_predictions))
print("recall score:", recall_score(y_val, RF_model_filter_predictions))
print("f1 score:", f1_score(y_val, RF_model_filter_predictions))


# In[ ]:


# RF with feature selection (wrapper)
RF_model_wrapper = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_wrapper.fit(X_train_subset_RF, y_train)
RF_model_wrapper_predictions = RF_model_wrapper.predict(X_val_subset_RF)

print("accuracy score:", accuracy_score(y_val, RF_model_wrapper_predictions))
print("precision score:", precision_score(y_val, RF_model_wrapper_predictions))
print("recall score:", recall_score(y_val, RF_model_wrapper_predictions))
print("f1 score:", f1_score(y_val, RF_model_wrapper_predictions))


# In[ ]:


# SVM without feature selection
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
SVM_model_filter = SVC(C = 100, kernel ='rbf')
SVM_model_filter.fit(X_train_scaled, y_train)
SVM_model_filter_predictions = SVM_model_filter.predict(X_val_scaled)

print("accuracy score:", accuracy_score(y_val, SVM_model_filter_predictions))
print("precision score:", precision_score(y_val, SVM_model_filter_predictions))
print("recall score:", recall_score(y_val, SVM_model_filter_predictions))
print("f1 score:", f1_score(y_val, SVM_model_filter_predictions))


# In[ ]:


# SVM with feature selection (filter)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
SVM_model_filter = SVC(C = 100, kernel ='rbf')
SVM_model_filter.fit(X_train_scaled, y_train)
SVM_model_filter_predictions = SVM_model_filter.predict(X_val_scaled)

print("accuracy score:", accuracy_score(y_val, SVM_model_filter_predictions))
print("precision score:", precision_score(y_val, SVM_model_filter_predictions))
print("recall score:", recall_score(y_val, SVM_model_filter_predictions))
print("f1 score:", f1_score(y_val, SVM_model_filter_predictions))


# In[ ]:


# SVM with feature selection (wrapper)
scaler = MinMaxScaler()
scaler.fit(X_train_subset_SVM)
X_train_scaled = scaler.transform(X_train_subset_SVM)
X_val_scaled = scaler.transform(X_val_subset_SVM)
SVM_model_wrapper = SVC(C = 100, kernel ='rbf')
SVM_model_wrapper.fit(X_train_scaled, y_train)
SVM_model_wrapper_predictions = SVM_model_wrapper.predict(X_val_scaled)

print("accuracy score:", accuracy_score(y_val, SVM_model_wrapper_predictions))
print("precision score:", precision_score(y_val, SVM_model_wrapper_predictions))
print("recall score:", recall_score(y_val, SVM_model_wrapper_predictions))
print("f1 score:", f1_score(y_val, SVM_model_wrapper_predictions))


# ***Testing***

# In[ ]:


# RF without feature selection
RF_model_1 = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_1.fit(X_train, y_train)
RF_model_1_predictions = RF_model_1.predict(X_test)

print("accuracy score:", accuracy_score(y_test, RF_model_1_predictions))
print("precision score:", precision_score(y_test, RF_model_1_predictions))
print("recall score:", recall_score(y_test, RF_model_1_predictions))
print("f1 score:", f1_score(y_test, RF_model_1_predictions))

# confusion matrix
Confusion_Matrix_RF_model_1 = confusion_matrix(y_test, RF_model_1_predictions)
print(Confusion_Matrix_RF_model_1)


# In[ ]:


# RF with feature selection (filter)
RF_model_filter = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_filter.fit(X_train, y_train)
RF_model_filter_predictions = RF_model_filter.predict(X_test)

print("accuracy score:", accuracy_score(y_test, RF_model_filter_predictions))
print("precision score:", precision_score(y_test, RF_model_filter_predictions))
print("recall score:", recall_score(y_test, RF_model_filter_predictions))
print("f1 score:", f1_score(y_test, RF_model_filter_predictions))

# confusion matrix
Confusion_Matrix_RF_model_filter = confusion_matrix(y_test, RF_model_filter_predictions)
print(Confusion_Matrix_RF_model_filter)


# In[ ]:


# RF with feature selection (wrapper)
RF_model_wrapper = RandomForestClassifier(n_estimators = 1500, min_samples_split = 5)
RF_model_wrapper.fit(X_train_subset_RF, y_train)
RF_model_wrapper_predictions = RF_model_wrapper.predict(X_test_subset_RF)

print("accuracy score:", accuracy_score(y_test, RF_model_wrapper_predictions))
print("precision score:", precision_score(y_test, RF_model_wrapper_predictions))
print("recall score:", recall_score(y_test, RF_model_wrapper_predictions))
print("f1 score:", f1_score(y_test, RF_model_wrapper_predictions))

# confusion matrix
Confusion_Matrix_RF_model_wrapper = confusion_matrix(y_test, RF_model_wrapper_predictions)
print(Confusion_Matrix_RF_model_wrapper)


# In[ ]:


# SVM without feature selection
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
SVM_model_filter = SVC(C = 100, kernel='rbf')
SVM_model_filter.fit(X_train_scaled, y_train)
SVM_model_filter_predictions = SVM_model_filter.predict(X_test_scaled)

print("accuracy score:", accuracy_score(y_test, SVM_model_filter_predictions))
print("precision score:", precision_score(y_test, SVM_model_filter_predictions))
print("recall score:", recall_score(y_test, SVM_model_filter_predictions))
print("f1 score:", f1_score(y_test, SVM_model_filter_predictions))

# confusion matrix
Confusion_Matrix_SVM_model_filter = confusion_matrix(y_test, SVM_model_filter_predictions)
print(Confusion_Matrix_SVM_model_filter)


# In[ ]:


# SVM with feature selection (filter)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
SVM_model_filter = SVC(C = 100, kernel='rbf')
SVM_model_filter.fit(X_train_scaled, y_train)
SVM_model_filter_predictions = SVM_model_filter.predict(X_test_scaled)

print("accuracy score:", accuracy_score(y_test, SVM_model_filter_predictions))
print("precision score:", precision_score(y_test, SVM_model_filter_predictions))
print("recall score:", recall_score(y_test, SVM_model_filter_predictions))
print("f1 score:", f1_score(y_test, SVM_model_filter_predictions))

# confusion matrix
Confusion_Matrix_SVM_model_filter = confusion_matrix(y_test, SVM_model_filter_predictions)
print(Confusion_Matrix_SVM_model_filter)


# In[ ]:


# SVM with feature selection (wrapper)
scaler = MinMaxScaler()
scaler.fit(X_train_subset_SVM)
X_train_scaled = scaler.transform(X_train_subset_SVM)
X_test_scaled = scaler.transform(X_test_subset_SVM)
SVM_model_wrapper = SVC(C = 100, kernel='rbf')
SVM_model_wrapper.fit(X_train_scaled, y_train)
SVM_model_wrapper_predictions = SVM_model_wrapper.predict(X_test_scaled)

print("accuracy score:", accuracy_score(y_test, SVM_model_wrapper_predictions))
print("precision score:", precision_score(y_test, SVM_model_wrapper_predictions))
print("recall score:", recall_score(y_test, SVM_model_wrapper_predictions))
print("f1 score:", f1_score(y_test, SVM_model_wrapper_predictions))

# confusion matrix
Confusion_Matrix_SVM_model_wrapper = confusion_matrix(y_test, SVM_model_wrapper_predictions)
print(Confusion_Matrix_SVM_model_wrapper)


# **Confusion Matrices RF and SVM models**

# In[ ]:


# RF without feature selection
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_RF_model_1_plot = ConfusionMatrixDisplay(Confusion_Matrix_RF_model_1, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_RF_model_1_plot.ax_.set_title("Confusion Matrix RF without feature selection")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_RF_model_1_plot.png')
plt.show()


# In[ ]:


# RF with feature selection (filter)
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_RF_filter = ConfusionMatrixDisplay(Confusion_Matrix_RF_model_filter, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_RF_filter.ax_.set_title("Confusion Matrix RF filter")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_RF_filter.png')
plt.show()


# In[ ]:


# RF with feature selection (wrapper)
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_RF_wrapper = ConfusionMatrixDisplay(Confusion_Matrix_RF_model_wrapper, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_RF_wrapper.ax_.set_title("Confusion Matrix RF wrapper")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_RF_wrapper.png')
plt.show()


# In[ ]:


# SVM without feature selection
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_SVM_model_1_plot = ConfusionMatrixDisplay(Confusion_Matrix_SVM_model_1, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_SVM_model_1_plot.ax_.set_title("Confusion Matrix SVM without feature selection")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_SVM_model_1_plot.png')
plt.show()


# In[ ]:


# SVM with feature selection (filter)
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_SVM_filter = ConfusionMatrixDisplay(Confusion_Matrix_SVM_model_filter, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_SVM_filter.ax_.set_title("Confusion Matrix SVM filter")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_SVM_filter.png')
plt.show()


# In[ ]:


# SVM with feature selection (wrapper)
target_names = ["Dissatisfied", "Satisfied"] 
Confusion_Matrix_SVM_wrapper = ConfusionMatrixDisplay(Confusion_Matrix_SVM_model_wrapper, display_labels = target_names).plot(cmap = 'Blues')
Confusion_Matrix_SVM_wrapper.ax_.set_title("Confusion Matrix SVM wrapper")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Confusion_Matrix_SVM_wrapper.png')
plt.show()


# **McNemar significance test**

# In[ ]:


# statistical test RF compared to RF filter
data_RF_filter = [[50029, 25941],
       [26011, 1923]]

print(mcnemar(data_RF_filter, exact = False, correction = False))


# In[ ]:


# statistical test RF compared to RF wrapper
data_RF_wrapper = [[50112, 26024],
       [25928, 1840]]

print(mcnemar(data_RF_wrapper, exact = False, correction = False))


# In[ ]:


# statistical test SVM compared to SVM filter
data_SVM_filter = [[49560, 26124],
       [25828, 2392]]

print(mcnemar(data_SVM_filter, exact = False, correction = False))


# In[ ]:


# statistical test SVM compared to SVM wrapper
data_SVM_wrapper = [[49558, 26122],
       [25830, 2394]]

print(mcnemar(data_SVM_wrapper, exact = False, correction = False))

