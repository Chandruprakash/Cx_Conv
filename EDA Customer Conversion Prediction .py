#!/usr/bin/env python
# coding: utf-8

# # **CUSTOMER CONVERSION PREDICTION**
# 
# **We are performing Expoloratory data analysis for better understanding of data and draw insights from it.**

# In[5]:


import pandas as pd
df = pd.read_excel('Customer Conversion Prediction.xlsx')
df


# In[6]:


df.head()


# In[7]:


df.shape


# In[10]:


df.describe()


# In[13]:


df.isnull().any()


# In[14]:


df.dtypes


# **EDA - Exploratory Data Analysis**

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='call_type',ax=axes[0],data=df)
sns.countplot(x='call_type', hue='y',ax=axes[1], data=df)
plt.show()


# **Observation:
# From above stats, we conclude that cellular type conversation is contributing a lot towards customers opting for insurance plan.**

# In[16]:


fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='job',ax=axes[0],data=df)
sns.countplot(x='job', hue='y',ax=axes[1], data=df)
plt.xticks(rotation=45)
plt.xticks(rotation=45)


# **Below is above observation based on Job status Analysis:**
# 
# **The most targeted customers: Blue-collar job, but comparing with conversion rate, we end up in losing lot of money.(Better to avoid targetting Blue-collar customers a lot)
# Convertion rate is maximum: Management.
# Based on Power law distribution, ~80% of people option for insurance is from (Management and Technician)**

# In[17]:


fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='marital',ax=axes[0],data=df)
sns.countplot(x='marital', hue='y',ax=axes[1], data=df)
plt.xticks(rotation=45)


# In[18]:


fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='mon',ax=axes[0],data=df)
sns.countplot(x='mon', hue='y',ax=axes[1], data=df)


# In[19]:


fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='prev_outcome',ax=axes[0],data=df)
sns.countplot(x='prev_outcome', hue='y',ax=axes[1], data=df)


# Observation:
# 
# Since the unknown values are 90% we are droping this columns, as it is not adding any values to the final outcome. --> Perform CHI square test and conclude based on result
# 
# Chi_square_test = (obs_value-expected_val)^2/expected_val

# In[20]:


df_chi = pd.DataFrame()
df_chi["prev_outcome"] = df["prev_outcome"]
df_chi["mon"] = df["mon"]
ct = pd.crosstab(df_chi["prev_outcome"],df_chi["mon"])
ct


# In[21]:


from scipy import stats
stats.chi2_contingency(ct.values)
     

fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='education_qual',ax=axes[0],data=df)
sns.countplot(x='education_qual', hue='y',ax=axes[1], data=df)


# **Numerical Data Analysis**

# In[22]:


fig, axes=plt.subplots(nrows=2,figsize=(20,5))
sns.countplot(x='age',ax=axes[0],data=df)
sns.countplot(x='age', hue='y',ax=axes[1], data=df)
plt.xticks(rotation=45)
plt.show()


# Observation:
# 
# 1.People between 25 and 60 age are opting for insurance.
# 2.Senior citizens are not opting for insurance.
# 3.Also people below 18 years are not opting for insurance(Please keep age > 18 filter)

# In[23]:


fig, axes=plt.subplots(ncols=2,figsize=(20,5))
sns.countplot(x='day',ax=axes[0],data=df)
sns.countplot(x='day', hue='y',ax=axes[1], data=df)
plt.xticks(rotation=45)


# In[24]:


fig, axes=plt.subplots(nrows=2,figsize=(20,5))
sns.countplot(x='dur',ax=axes[0],data=df)
sns.countplot(x='dur', hue='y',ax=axes[1], data=df)
plt.xticks(rotation=45)


# In[28]:


fig, axes=plt.subplots(nrows=2,figsize=(20,5))
sns.countplot(x='num_calls',ax=axes[0],data=df)
sns.countplot(x='num_calls', hue='y',ax=axes[1], data=df)
plt.show()


# 1.When contacted over phone, mostly customers opt for insurance in max 3 attempts.
# 2.As s cost effective approach, avoid reaching customer over phone after 3 attempts
