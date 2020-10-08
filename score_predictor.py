#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import librairies
import pandas as pd
import pickle


# load the dataset
df = pd.read_csv('ipl.csv')


# In[9]:


df.head()


# In[10]:


# Data Cleaning
# removing unwanted columns
columns_to_remove = ['mid', 'batsman','bowler','striker', 'non-striker']
df.drop(labels = columns_to_remove, axis =1, inplace = True)


# In[11]:


df['bat_team'].unique()


# In[12]:


df['venue'].unique()


# In[14]:


# Keeping the consistant teams
consistant_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils',
        'Sunrisers Hyderabad']


# In[15]:


df = df[(df['bat_team']).isin(consistant_teams) & (df['bowl_team']).isin(consistant_teams)]


# In[16]:


df.head()


# In[20]:


# Removing first 5 overs in every match
# we need atleast 5 overs of data for prediction
df= df[df['overs'] >= 5]


# In[21]:


df.head()


# In[22]:


print(df['bat_team'].unique())
print(df['bowl_team'].unique())


# In[23]:


# converting column 'date' string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[33]:


# Data preprocessing
# Converting the categorical features using Onehotencoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team', 'venue'])


# In[25]:


encoded_df.head()


# In[28]:


encoded_df.columns


# In[34]:


encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
                        'bat_team_Kings XI Punjab','bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad', 'venue_Barabati Stadium',
       'venue_Brabourne Stadium', 'venue_Buffalo Park',
       'venue_De Beers Diamond Oval', 'venue_Dr DY Patil Sports Academy',
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
       'venue_Feroz Shah Kotla',
       'venue_Himachal Pradesh Cricket Association Stadium',
       'venue_Holkar Cricket Stadium',
       'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
       'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium',
       'venue_New Wanderers Stadium', 'venue_Newlands',
       'venue_OUTsurance Oval',
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
       'venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
       "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
       'venue_SuperSport Park', 'venue_Wankhede Stadium', 
        'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]


# In[36]:


# Splitting the data into train and test dataset
# time series data set is split into following method
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <=2016]
X_test = encoded_df.drop(labels='total', axis =1)[encoded_df['date'].dt.year >= 2017]


# In[39]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[41]:


X_train.drop(labels='date', axis =True, inplace= True)
X_test.drop(labels='date', axis=True, inplace=True)


# LINEAR REGRESSION

# In[43]:


# ############################################# #
# ### Model building ###
# ### Linear Regression
from sklearn.linear_model import LinearRegression as LR
regressor = LR()
regressor.fit(X_train, y_train)


# In[44]:


# Creating a pickle file for the classifier
filename = 'first-innings_score_predictor_lr_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# RIDGE REGRESSION

# In[46]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[54]:


ridge= Ridge()
parameters = { 'alpha' : [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)


# In[55]:


# Creating a pickle file for the classifier
filename = 'first-innings_score_predictor_ridge_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# LASSO REGRESSION

# In[56]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[58]:


lasso= Lasso()
parameters = { 'alpha' : [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters, scoring='neg_mean_squared_error', cv=10)
lasso_regressor.fit(X_train, y_train)


# In[61]:


# Creating a pickle file for the classifier
filename = 'first-innings_score_predictor_lasso_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# XG BOOST

# In[60]:


pip install xgboost


# In[62]:


import xgboost

xg_model = XGBRegressor()


# In[63]:


##### for ridge regressor 

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[64]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[65]:


prediction = ridge_regressor.predict(X_test)


# In[67]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[68]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test,prediction))
print('MSE:', metrics.mean_squared_error(y_test,prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[69]:


prediction = lasso_regressor.predict(X_test)


# In[70]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[71]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test,prediction))
print('MSE:', metrics.mean_squared_error(y_test,prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:




