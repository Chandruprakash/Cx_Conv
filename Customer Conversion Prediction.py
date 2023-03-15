#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# **Collecting Dataset**
# 
Since the dataset is huge, adding file to google drive and importing the dataset. Based on Certain input parameters, we gonna predict whether the customer will opt for th insurance plan
# In[3]:


df = pd.read_excel("Customer Conversion Prediction.xlsx")
df


# In[4]:


df.head(5)


# # Data Cleaning
Since there are few text fields in my dataset, performing Data cleaning by using any of below methods

LabelEncoder(Inbuilt function) Using Map function( df.Sex = df.Sex.map({"male": 0, "female": 1}))
# In[6]:


#Since our ML model works on numeric data, using LabelEncoder to convert text dataset to numeric
# Let's perform label encoding on all object columns using MAP function
import numpy as np
objList = df.select_dtypes(include = "object").columns
for column1 in objList:
  ordinal_label = {k: i for i, k in enumerate(df[column1].unique(), 0)}
  df[column1] = df[column1].map(ordinal_label).astype(int)


# In[7]:


df.dtypes


# In[8]:


df["y"].value_counts()

Since the output class is biased towards one class(NO -> Didn't take insurance) we performing Stratified sampling for training our model


# In[9]:


#Segregating Dataset into X & Y

X = df.drop("y",axis=1)  #Feature Matrix
Y = df["y"] 


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
x_train.shape, x_test.shape

#other Ways to handle data imbalance --> SMOTE, ROSE


# In[11]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x_train)
tr_x_train = ss.transform(x_train)
tr_x_test = ss.transform(x_test)


# # Draw heatmap
Draw a heatmap using Pearson Correlation coeffecient
# In[12]:


import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
corr_map = x_train.corr()
sns.heatmap(corr_map, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[13]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[14]:


corr_features = correlation(x_train, 0.40)
len(set(corr_features))


# In[15]:


corr_features


# In[16]:


### Used the Variance threshold function to check, if there are any constant columns that can be removed. But there are no constant columns.


from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(df)
var_thres.get_support()


# In[18]:


import pandas as pd
dataset = pd.read_excel("Customer Conversion Prediction.xlsx")


# In[19]:


dataset.head()

From above datasets observation, duration is a continuous variable. Dropping in dataset: Duration, in order to use mutual_info_classify for determining the mutual information.
# In[20]:


df_mutual_info = x_train.drop("dur", axis=1)
df_mutual_info.describe()


# In[21]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information

mutual_info = mutual_info_classif(df_mutual_info, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = df_mutual_info.columns
# mutual_info.sort_values(ascending=False)
mutual_info.sort_values(ascending=False).plot.bar(figsize=(10, 5))
plt.show()


# In[22]:


#No we Will select the  top 5 important features

from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=5)
sel_five_cols.fit(x_train, y_train)
x_train.columns[sel_five_cols.get_support()]

From EDA and above checks, we see day is not contributing much to the output. Hence dropping the feature: day
# In[24]:


dataset = pd.read_excel("Customer Conversion Prediction.xlsx")
dataset


# In[26]:


df_final = dataset.drop("day", axis=1)
df_final


# In[27]:


df.head(5)


# # Label Encoding
Since there are multiclass text features in my dataset, applying OneHotEncoder to convert TEXT --> NUMERIC
# In[28]:


from sklearn.preprocessing import OneHotEncoder

#Map output class(Text) --> Numeric 
df_final["y"] = df_final["y"].map({"no": 0, "yes": 1}).astype(int)
objList = df_final.select_dtypes(include = "object").columns
df_final = pd.get_dummies(df, columns = objList.values)
df_final.columns


# In[31]:


#df_final["y"] = df_final["y"].map({"no": 0, "yes": 1}).astype(int)
df_final


# # Split the dataset into Input & Output
# 

# In[32]:


X = df_final.drop("y", axis=1)
Y = df_final["y"]


# # Sampling technique --> Stratified
# 
Since the output class is biased towards one class(NO -> Didn't take insurance) we performing Stratified sampling and split the datasets into Train and test for training our model.
# In[33]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
x_train.shape, x_test.shape


# In[34]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(x_train)
tr_x_train = ss.transform(x_train)
tr_x_test = ss.transform(x_test)


# In[35]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


rf_model = RandomForestClassifier()
lr_model = LogisticRegression()
ada_model = AdaBoostClassifier()
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()
models_accuracy = []

models = [rf_model, lr_model, ada_model, knn_model, nb_model]
models_name = [str(i) for i in models]

pred = []
for model in models:
  model.fit(tr_x_train, y_train)
  y_train_pred = model.predict_proba(tr_x_train)
  train_accuracy = roc_auc_score(y_train, y_train_pred[:,-1]) * 100
  y_test_pred = model.predict_proba(tr_x_test)
  test_accuracy = roc_auc_score(y_test, y_test_pred[:,-1]) * 100
  print("Accuracy of Model: {} for train: {} %, test: {} %".format(str(model), train_accuracy, test_accuracy))
  models_accuracy.append(test_accuracy)
  pred.append(pd.Series(y_test_pred[:,-1]))

df_models_output = pd.concat(pred, axis=1)
df_models_output.columns = models_name


#Create a dataframe of all the model accuracy for a paricular sample

df_models_output["final_prediction"] = df_models_output.mean(axis=1)
print("Ensemble test roc-auc: {}".format(roc_auc_score(y_test, df_models_output["final_prediction"])))
     


# In[36]:


from seaborn import colors
plt.figure(figsize=(8,4), dpi=85)
plt.bar(models_name, models_accuracy, color=list("rgbyc"))
plt.xlabel("Model Name")
plt.ylabel("Accuracy")
plt.title("ROC Accuracy Score of Various Classification models")
for index, data in enumerate(models_accuracy):
  plt.text(x=index, y=data+0.5, s=f"{data.round(2)}%", ha="center", fontsize=12)
plt.ylim(75, 95)
plt.xticks(rotation=60)
plt.show()


# # Hyperparameter Tuning
# 
RF classifier is the best model. 
Performed hyperparameter tuning for the RandomForestClassifier. Below are few observation

      1.RandomForestClassifier accuracy: 92.45 %
      2.After hyperparameter tuning: Accuracy for n_estimator=17,         min_sample_split=23: 92.60%.
# In[42]:


from sklearn.ensemble import RandomForestClassifier
for i in range(10,20):
  rf_hyper_model = RandomForestClassifier(random_state = 24, n_jobs = -1, n_estimators=i, max_features=None, min_samples_split=i+6)
  rf_hyper_model.fit(tr_x_train, y_train)
  y_test_hyper = rf_hyper_model.predict(tr_x_test)
  y_test_pred = rf_hyper_model.predict_proba(tr_x_test)
  test_accuracy = roc_auc_score(y_test, y_test_pred[:,-1]) * 100
  print("Accuracy for n_estimator={}, min_sample_split={} is {} & {}".format(i, i+6, accuracy_score(y_test_hyper, y_test) * 100, test_accuracy))
     


# # Adaboost hyperparameter tuning
# 
Performed hyperparameter tuning for the AdaBoostClassifier. Below are few observation

       1.AdaBoostClassifier accuracy: 90.35%
       2.After hyperparameter tuning: Accuracy for n_estimator=101: 90.58%
# In[43]:


from sklearn.ensemble import AdaBoostClassifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor

for i in range(95,105):
  ada_hyper_model = AdaBoostClassifier(n_estimators=i, random_state=0)
  ada_hyper_model.fit(tr_x_train, y_train)
  y_test_hyper = ada_hyper_model.predict(tr_x_test)
  y_test_pred = ada_hyper_model.predict_proba(tr_x_test)
  test_accuracy = roc_auc_score(y_test, y_test_pred[:,-1]) * 100
  print("Accuracy for n_estimator={} is {} & {}".format(i, accuracy_score(y_test_hyper, y_test) * 100, test_accuracy))
     


# # Confusion Matrix
# 
Calculate confusion matrix result for the predicted output to check accuracy of our model
# In[41]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = rf_model.predict(tr_x_test)
# cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp


# In[44]:


confusion_matrix(y_test, y_pred)


# In[45]:


df_models_output


# In[48]:


#Calculate ROC curve

fpr, tpr, thresholds = roc_curve(y_test,df_models_output["RandomForestClassifier()"].values)
len(thresholds)


# In[47]:


from sklearn.metrics import accuracy_score
accuracy_ls = []
for thres in thresholds:
  y_pred = np.where(df_models_output["final_prediction"].values > thres, 1, 0)
  accuracy = accuracy_score(y_test, y_pred)
  accuracy_ls.append(accuracy)

threshold_accuracy = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)], axis=1)
threshold_accuracy.columns = ["Threshold", "Accuracy"]
threshold_accuracy.sort_values(by="Accuracy", ascending=False, inplace=True)
threshold_accuracy.head()


# # Best threshold value for max Accuracy
# 
Based on the ROC score, we can conclude that the Accuracy of the model is maximum at Threshold: 0.59


# In[49]:


fpr, tpr, threshold = roc_curve(y_test, df_models_output["RandomForestClassifier()"])
fpr_fi, tpr_fi, threshold1 = roc_curve(y_test, df_models_output["final_prediction"])
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, "g-.v", mfc="r", mec="r", label="Random Forest Classifier")
plt.plot([0,1], [0,1], "b--o", label="Random Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Recursive Operation Characteristic(ROC) curve - RF Classifier")
plt.legend()
plt.subplot(1,2,2)
plt.plot(fpr_fi, tpr_fi, "b-.o", mfc="r", mec="r", label="final_prediction")
plt.plot([0,1], [0,1], "b--o", label="Random Model")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Recursive Operation Characteristic(ROC) curve - Ensember model")
plt.show()


# In[50]:


models = [rf_model, lr_model, ada_model, knn_model, nb_model]
models_name = [str(i) for i in models]

models_accuracy = []
for model in models:
  y_pred = model.predict(tr_x_test)
  test_accuracy = accuracy_score(y_test, y_pred) * 100
  print("Accuracy of Model: {} is {} %".format(str(model), test_accuracy))
  models_accuracy.append(test_accuracy)


# In[51]:


plt.figure(figsize=(8,4), dpi=85)
plt.bar(models_name, models_accuracy, color=list("rgbyc"))
plt.ylim(70,95)
plt.xlabel("Model Name")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models")
for index, data in enumerate(models_accuracy):
  plt.text(x=index, y=data+0.4, s=f"{data.round(2)}%", ha="center", fontsize=14)
plt.xticks(rotation=60)
plt.show()


# # Export model to a pickle file for reusing the model for prediction
# 
Based on the Accuracy Score and did hyperparameter tuning of models. Below is the best accuracy score that we can achieve for models:

   * RandomForestClassifier: 92.60%
   * AdaBoostClassifier: 90.58%
    
    
Exporting the RandomForestClassifier, for predicting the output.
# In[52]:


#Extract the model to a Pickle file

import pickle
pickle_out = open("rf_model.pkl", "wb")
pickle.dump(rf_hyper_model, pickle_out)
pickle_out.close

