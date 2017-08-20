import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
df1=pd.read_csv('test.csv')

target=df['Purchase']
train=df.drop('Purchase',axis=1)
test=df1
combined=train.append(test)
q=df[df['Gender']=='F']['Purchase']
p=df[df['Gender']=='M']['Purchase']
plt.plot(df[df['Gender']=='M']['Purchase'])
#plt.show()
plt.plot(df[df['Gender']=='F']['Purchase'])
#plt.show()
##check for outliers and missing values by visualization or numerically
gender_dummies=pd.get_dummies(combined['Gender'],prefix='Gender')
combined=pd.concat((combined,gender_dummies),axis=1)
combined.drop('Gender',axis=1,inplace=True)
#to see how age affect purchasing 
w=df.groupby('Age')['Purchase'].mean()
s=np.array(w)
plt.plot(s)
#plt.show()
age_dummies=pd.get_dummies(combined['Age'],prefix='Age')
combined=pd.concat((combined,age_dummies),axis=1)
combined.drop('Age',axis=1,inplace=True)
#approx same for all age groups

t=df.groupby('Occupation')['Purchase'].mean()

occupation_dummies=pd.get_dummies(combined['Occupation'],prefix='Occupation')
combined=pd.concat((combined,occupation_dummies),axis=1)
combined.drop('Occupation',axis=1,inplace=True)

df.groupby('City_Category')['Purchase'].mean()#give same approx
city_dummies=pd.get_dummies(combined['City_Category'],prefix='City_Caegory')
combined=pd.concat((combined,city_dummies),axis=1)
combined.drop('City_Category',axis=1,inplace=True)

years_dummies=pd.get_dummies(combined['Stay_In_Current_City_Years'],prefix='Stay_In_Current_City_Years')
combined=pd.concat((combined,years_dummies),axis=1)
combined.drop('Stay_In_Current_City_Years',axis=1,inplace=True)


martial_dummies=pd.get_dummies(combined['Marital_Status'],prefix='Marital_Status')
combined=pd.concat((combined,martial_dummies),axis=1)
combined.drop('Marital_Status',axis=1,inplace=True)
train['Product_Category_2'].fillna(train['Product_Category_2'].value_counts().index[0], inplace=True)
train['Product_Category_3'].fillna(train['Product_Category_3'].value_counts().index[0], inplace=True)
# Test set
test['Product_Category_2'].fillna(test['Product_Category_2'].value_counts().index[0], inplace=True)
test['Product_Category_3'].fillna(test['Product_Category_3'].value_counts().index[0], inplace=True)
p=train['Product_Category_2']
p=p.append(test['Product_Category_2'])
q=train['Product_Category_3']
q=q.append(test['Product_Category_3'])
combined['Product_Category_2']=p
combined['Product_Category_3']=q

combined.drop('User_ID',axis=1,inplace=True)
combined.drop('Product_ID',axis=1,inplace=True)
train1=combined.head(550068)
test1=combined.iloc[550068:]
target=np.array(target)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train1,target,test_size=0.2)
train3=x_train.head(40000)
ytrain=y_train[:40000]

from sklearn.svm import SVR
regressor=SVR(kernel='poly')
regressor.fit(train3,ytrain)




