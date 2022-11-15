#!/usr/bin/env python
# coding: utf-8

# # Surprise-Housing

# Surprise Housing is a US based company wants to enter the Australian market. The company uses the data analytics to buy properties at low prices and sell at high prices. So company has collected the data and want to know the important or influential variables of the data, so that they can build strategies on them.

# In[1]:


# importing the requisite libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ## 1.1 Simple Linear Regression

# In[2]:


# Reading the dataset

real=pd.read_csv(r"SurpriseHousing.csv")


# In[3]:


real.info()


# In[4]:


real.shape


# In[5]:


real.describe()


# In[6]:


real = real.rename(columns={'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF'})


# In[7]:


real.columns


# In[8]:


real_num=real.select_dtypes(exclude=['object'])

cols=real_num.columns
real_num[cols]=real_num[cols].apply(pd.to_numeric, errors='coerce')
print(real_num[cols])


# In[9]:


real_num.dtypes


# In[10]:


real[cols]=real_num[cols]


# In[11]:


real.select_dtypes(exclude=['object']).columns


# In[12]:


display(real.isnull().sum())


# In[13]:


real_nan=real[real.columns[real.isna().any()]]


# In[14]:


real_nan


# In[15]:


real_nan.isnull().sum()/len(real_nan)*100


# In[16]:


real_nan.dtypes


# So the columns having null values more thsn 30% can be deleted and other columns are dealt with traditional method.

# In[17]:


# Remove those 'real _nan' dataframe columns from dataframe 'real' 
real.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis='columns',inplace=True)


# In[18]:


real


# In[19]:


real['LotFrontage']=real.LotFrontage.fillna(real.LotFrontage.mean())


# In[20]:


real.LotFrontage.isna().sum()


# In[21]:


real.MasVnrType.value_counts()


# In[22]:


real = real[real.columns[real.notna().any()]]


# In[23]:


real.isna().sum()


# In[24]:


real.columns


# In[25]:


real.head()


# In[26]:


real.isnull().sum().sum()


# In[27]:


len(real)


# In[28]:


perc=(609/1460)*100
print(perc)


# In[29]:


real.columns[real.isna().any()].tolist()


# In[30]:


# for each column, get value counts in decreasing order and take the index (value) of most common class
for i in real.columns:
    if real[i].dtype ==object:
        real = real.fillna(real[i].value_counts().index[0])        


# In[31]:


real.isnull().sum().sum()


# In[32]:


real=real.dropna()


# ### Correlation

# In[33]:


# correlation matrix
cor_real = real.corr()
cor_real


# In[34]:


# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(30,30))

# heatmap
sns.heatmap(cor_real, cmap="YlGnBu", annot=True)
plt.show()


# In[35]:


real_non_object = real.select_dtypes(exclude=['object'])


# In[36]:


real_non_object.info()


# In[37]:


real_non_object = real_non_object.loc[:, real_non_object.columns != 'Id']


# In[38]:


real_non_object.head()


# ### Dealing with outliers using quantiles

# In[39]:


low = .05
high = .95
quant_real_non = real_non_object.quantile([low, high])
print(quant_real_non)


# In[40]:


real_non_object = real_non_object.apply(lambda x: x[(x>quant_real_non.loc[low,x.name]) | 
                                    (x<quant_real_non.loc[high,x.name])], axis=0)


# In[41]:


real_non_object = pd.concat([real.loc[:,'Id'], real_non_object], axis=1)


# In[42]:


real_non_object.columns


# In[43]:


real_non_object.isnull().sum()/len(real_non_object)*100


# As we can see thare are more than 95% of null values in the columns, so we decide to remove those columns.

# In[44]:


real_non_object.drop(['LowQualFinSF','KitchenAbvGr','3SsnPorch','PoolArea','MiscVal'],axis='columns',inplace=True)


# In[45]:


real_non_object.isnull().sum()/len(real_non_object)*100


# In[46]:


real_non_object.head()


# In[48]:


real_non_object.head(10)


# In[49]:


real_non_object.iloc[:,1]


# In[50]:


real_non_object.dtypes


# In[51]:


len(real_non_object.columns)


# In[52]:


real_non_object.shape


# In[53]:


real_non_object=real_non_object.astype(np.int64)


# In[54]:


real_non_object.dtypes


# In[56]:


real_object = real.select_dtypes(include=['object'])
real_object.head()


# In[57]:


real_object.isna().sum()


# Looks like there are no null values in these categorical variables.

# In[58]:


# we add a column from a real_non_object dataframe and later we use this column as a key to merge both dataframes.

real_object = real_object.join(real_non_object['Id'])


# In[59]:


real_object.head()


# In[60]:


real_merge= pd.merge(real_object, real_non_object, on='Id')


# In[61]:


real_merge.isnull().sum()/len(real_merge)*100


# There are no null values exist in the new dataframe "real_merge"

# In[62]:


real_col=real_merge.columns.to_list()


# In[ ]:





# Let us see the variables that are useful in meeting our business objectives

# ## 3. Data Preparation

# In[63]:


# split into X and y
X = real_merge.loc[:, real_col[:-1]] # predictors in variable X

y = real_merge['SalePrice'] # response variable in Y


# In[64]:


# creating dummy variables for categorical variables

# subset all categorical variables
real_categorical = X.select_dtypes(include=['object'])
real_categorical.head()


# In[65]:


len(real_categorical.columns)


# In[66]:


lise=list(real_categorical)


# In[67]:


lise


# In[68]:


samp_1=real_categorical[lise[0:14]]
samp_2=real_categorical[lise[15:31]]
samp_3=real_categorical[lise[32:40]]


# In[69]:


samp_1.head()


# In[70]:


samp_2.head()


# In[71]:


samp_3.head()


# In[72]:


samp_3.dtypes


# In[73]:


X['MasVnrArea'].value_counts()


# In[74]:


X['MasVnrArea']=X['MasVnrArea'].replace({'RL':0})


# In[75]:


X['MasVnrArea'].value_counts()


# In[76]:


X['GarageYrBlt']=X['GarageYrBlt'].replace({'RL':0})


# In[77]:


#real=real['MasVnrArea','GarageYrBlt'].astype(np.int64)
X['MasVnrArea']=pd.to_numeric(X['MasVnrArea'])

X['GarageYrBlt']=pd.to_numeric(X['GarageYrBlt'])


# In[78]:


print(X.select_dtypes(exclude=['object']))


# In[79]:


X['MasVnrArea']=X['MasVnrArea'].astype(np.int64)
X['GarageYrBlt']=X['GarageYrBlt'].astype(np.int64)


# In[80]:


X['GarageYrBlt'].value_counts()


# In[81]:


import datetime

today = datetime.date.today()

year = today.year

print(year)


# In[82]:


X['Garage_Blt_age']=year-X['GarageYrBlt']


# In[83]:


X.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[84]:


real_categorical.dtypes


# In[85]:


real_categorical.drop(['MasVnrArea','GarageYrBlt'],axis=1,inplace=True)


# In[86]:


real_categorical.dtypes


# ### Feature Extraction

# In[87]:


# convert into dummies - one hot encoding
real_dummies = pd.get_dummies(real_categorical, drop_first=True)
real_dummies.head()


# In[88]:


# drop categorical variables 
X = X.drop(list(real_categorical.columns), axis=1)


# In[89]:


X.head()


# In[90]:


# concat dummy variables with X
X = pd.concat([X, real_dummies], axis=1)


# In[91]:


X.head()


# In[92]:


X['House_age_in_years']=year-X['YearBuilt']
X['House_remodel_age_in_years']=year-X['YearRemodAdd']


# In[93]:


X.drop(['YearBuilt'],axis=1,inplace=True)
X.drop(['YearRemodAdd'],axis=1,inplace=True)


# In[94]:


X = X.loc[:, X.columns != 'Id']


# In[95]:


X.head()


# In[96]:


# scaling the features - necessary before using Ridge or Lasso
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)

cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[97]:


#y=y.values.reshape(-1,)


# In[98]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ## 2.1 Feature Selection

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Init the transformer
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)

# Fit to the training data
rfe.fit(X_train, y_train)


# In[ ]:


X_train_r=X_train.iloc[:,rfe.support_]


# In[ ]:


X_train_r.shape


# In[ ]:


y_train_r=y_train


# In[ ]:


y_train_r.shape


# In[ ]:


X_test_r=X_test.iloc[:,rfe.support_]


# In[ ]:


y_test_r=y_test


# ## 3. Model Building and Evaluation

# ### Linear Regression

# In[ ]:


# Instantiate
lm = LinearRegression()

# Fit a line
lm.fit(X_train_r, y_train_r)


# In[ ]:


# Print the coefficients and intercept
print(lm.intercept_)
print(lm.coef_)


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


y_pred_train = lm.predict(X_train_r)
y_pred_test = lm.predict(X_test_r)

metric = []
r2_train_lr = r2_score(y_train_r, y_pred_train)
print(r2_train_lr)
metric.append(r2_train_lr)

r2_test_lr = r2_score(y_test_r, y_pred_test)
print(r2_test_lr)
metric.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train_r - y_pred_train))
print(rss1_lr)
metric.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test_r - y_pred_test))
print(rss2_lr)
metric.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train_r, y_pred_train)
print(mse_train_lr)
metric.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test_r, y_pred_test)
print(mse_test_lr)
metric.append(mse_test_lr**0.5)


# ## Ridge and Lasso Regression

# ### Ridge Regression

# In[ ]:


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 70, 80, 100, 500, 1000 ]}


# In[ ]:


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error',  
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train_r, y_train_r) 


# In[ ]:


# Printing the best hyperparameter alpha
print(model_cv.best_params_)


# In[ ]:


#Fitting Ridge model for alpha = 100 and printing coefficients which have been penalised
alpha = 100
ridge = Ridge(alpha=alpha)

ridge.fit(X_train_r, y_train_r)
print(ridge.coef_)


# In[ ]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = ridge.predict(X_train_r)
y_pred_test = ridge.predict(X_test_r)

metric2 = []
r2_train_lr = r2_score(y_train_r, y_pred_train)
print(r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test_r, y_pred_test)
print(r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train_r - y_pred_train))
print(rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test_r - y_pred_test))
print(rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train_r, y_pred_train)
print(mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric2.append(mse_test_lr**0.5)


# ## Lasso

# In[ ]:


lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train_r, y_train_r) 


# In[ ]:


d = model_cv.best_params_


# In[ ]:


# Printing the best hyperparameter alpha
print(d)


# In[ ]:


#Fitting Ridge model for alpha = 500 and printing coefficients which have been penalised

alpha =d['alpha']

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train_r, y_train_r) 


# In[ ]:


lasso.coef_


# In[ ]:


X_train_r.iloc[:,lasso.coef_!=0]


# In[ ]:


X_lasso=X_train_r.iloc[:,lasso.coef_!=0]
X_lasso.columns


# In[ ]:


# Lets calculate some metrics such as R2 score, RSS and RMSE

y_pred_train = lasso.predict(X_train_r)
y_pred_test = lasso.predict(X_test_r)

metric3 = []
r2_train_lr = r2_score(y_train_r, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test_r, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train_r - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test_r - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train_r, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test_r, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5)


# In[ ]:


# Creating a table which contain all the metrics

lr_table = {'Metric': ['R2 Score (Train)','R2 Score (Test)','RSS (Train)','RSS (Test)',
                       'MSE (Train)','MSE (Test)'], 
        'Linear Regression': metric
        }

lr_metric = pd.DataFrame(lr_table ,columns = ['Metric', 'Linear Regression'] )

rg_metric = pd.Series(metric2, name = 'Ridge Regression')
ls_metric = pd.Series(metric3, name = 'Lasso Regression')

final_metric = pd.concat([lr_metric, rg_metric, ls_metric], axis = 1)

final_metric


# ### Lets observe the changes in the coefficients after regularization

# In[ ]:


betas = pd.DataFrame(index=X_train_r.columns)


# In[ ]:


betas.rows = X_train_r.columns


# In[ ]:


betas['Linear'] = lm.coef_
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_


# In[ ]:


pd.set_option('display.max_rows', None)
betas.head(68)


# ### Cross Validation

# In[ ]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = ridge, X = X_train_r, y = y_train_r, cv = 10)
print("Cross validation of Ridge model = ",cross_validation)
print("Cross validation of Ridge model (in mean) = ",cross_validation.mean())


# In[ ]:


cross_validation = cross_val_score(estimator = lasso, X = X_train_r, y = y_train_r, cv = 10)
print("Cross validation of Lasso model = ",cross_validation)
print("Cross validation of Lasso model (in mean) = ",cross_validation.mean())


# In[ ]:


cross_validation = cross_val_score(estimator = lm, X = X_train_r, y = y_train_r, cv = 10)
print("Cross validation of Linear model = ",cross_validation)
print("Cross validation of Linear model (in mean) = ",cross_validation.mean())


# ### Saving the Model

# In[ ]:


import pickle
from sklearn.metrics import accuracy_score

#dump information to that file
pickle.dump(ridge,open('model.pkl','wb'))

#load a model
pickle.load(open('model.pkl','rb'))


# In[ ]:




