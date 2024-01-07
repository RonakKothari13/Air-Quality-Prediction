#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd 


# In[62]:


df=pd.read_csv(r"C:\Users\RONAK KOTHARI\Desktop\AQI and Lat Long of Countries.csv")


# In[63]:


df


# In[64]:


df.columns


# In[65]:


df.shape


# In[66]:


df.isnull().sum()


# In[67]:


df.info()


# In[68]:


df['Country'].nunique()


# In[69]:


cat=[var for var in df.columns if df[var].dtype=='O']


# In[70]:


num=[var for var in df.columns if df[var].dtype!='O']


# In[71]:


for i in df.columns:
    print(f'{i}:{df[i].unique()}','\n')


# In[72]:


df.describe()


# In[73]:


df.describe(include='all').T


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[75]:


df.head()


# In[76]:


plt.figure(figsize=(15,8))
sns.countplot(x='AQI Category',data=df)


# In[77]:


df['AQI Category']=df['AQI Category'].transform(lambda x: 0 if x in ['Moderate','Good'] else 1 )


# In[78]:


df


# In[79]:


df.corr()


# In[80]:


sns.heatmap(df.corr(),annot=True)


# In[82]:


plt.figure(figsize=(15,8))
sns.countplot(x='AQI Category',data=df)


# In[83]:


sns.distplot(df['AQI Value'])


# In[84]:


df=df.drop(['Country','City'],axis=1)


# In[85]:


df


# In[86]:


colm=df.columns


# In[87]:


from sklearn.preprocessing import RobustScaler
model=RobustScaler()


# In[88]:


model.fit(df[num])


# In[89]:


df[num]=model.transform(df[num])


# In[90]:


df=pd.DataFrame(df,columns=colm)


# In[91]:


df


# In[92]:


sns.distplot(df['AQI Value'])


# In[93]:


df=pd.get_dummies(df,drop_first=True)


# In[94]:


df


# In[95]:


x=df.drop(['AQI Category','lat','lng'],axis=1)


# In[96]:


x


# In[97]:


y=df['AQI Category']


# In[98]:


y


# In[99]:


from imblearn.under_sampling import NearMiss


# In[100]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)


# In[101]:


from imblearn.combine import SMOTETomek
os=SMOTETomek(random_state=0)
columns=x_train.columns
os_data_x,os_data_y=os.fit_resample(x_train,y_train)
os_data_x=pd.DataFrame(os_data_x,columns=columns)
os_data_y=pd.DataFrame(os_data_y,columns=['AQI Category'])


# In[102]:


os_data_x


# In[103]:


os_data_y=os_data_y.values.ravel()


# In[104]:


# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 
rfc = RandomForestClassifier(n_estimators=10,max_features=5)
# fit the model
rfc.fit(os_data_x,os_data_y)
# Predict the Test set results
y_pred = rfc.predict(x_test)


# In[105]:


# Check accuracy score 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[106]:


# Print the Confusion Matrix and slice it into four pieces
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)


# In[107]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[108]:


grid_space={'criterion':['gini','entropy'],
    'max_depth':[2,3,None],
    'min_samples_split':[2,3],
    'min_samples_leaf':[1,2]}


# In[109]:


from sklearn.model_selection import GridSearchCV
rfc =RandomForestClassifier()
model = GridSearchCV(rfc,grid_space,cv=3,scoring='accuracy')
model.fit(os_data_x,os_data_y)


# In[110]:


model.best_params_


# In[111]:


model.best_score_


# In[112]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[113]:


log.fit(x_train,y_train)


# In[114]:


y_pred=log.predict(x_test)


# In[115]:


accuracy_score(y_test,y_pred)


# In[116]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print('Model accuracy: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))


# In[117]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[118]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Create a random forest classifier
rfc = RandomForestClassifier()

# Define the parameter distributions
param_dist = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(rfc, param_dist, n_iter=10, cv=5)

random_search.fit(x, y)

# Access the best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_


# In[119]:


best_params


# In[120]:


best_score


# In[ ]:




