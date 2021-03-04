import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data=pd.read_csv("breast_cancer.csv")
data.tail()
data.isna().sum()

print(len(data))
#as last column is useless so lets remove it
data=data.dropna(axis=1)
# last useless column is removed

display(data.describe())

data["diagnosis"].value_counts()

data["diagnosis"]=np.where(data["diagnosis"]=="M",1,0)
# changing the diagnosis value 1 represent Malignant and 0 represent benign

#visualising the diagonisis part
sns.countplot(data["diagnosis"])
data.groupby(["diagnosis"]).mean()

#eda 
#id column is useless lets remove it
a=data.columns.values.tolist()
data_sel=[x for x in a if x !="id"]
# and now final set
y=["diagnosis"]
x=[i for i in data_sel if i not in y]
pd.crosstab(data["diagnosis"],data["radius_mean"]).plot(kind="bar")


#evry part is in numeric form and is quiet usefull so lets go forward to featur selection 
# and feature selection can be done by RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
lgrg=LogisticRegression()
rfe=RFE(lgrg,33)
rfe.fit(data[x],data[y])
print(rfe.support_)
print(rfe.ranking_)

#taking those feature
feature=list(np.where(rfe.ranking_==1))
cols=list(np.asarray(x)[feature])
X=data[cols]
Y=data["diagnosis"]

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.metrics import accuracy_score
model=LogisticRegression(n_jobs=-1)
model.fit(x_train,y_train)
print(f"score is {model.score(x_test,y_test)}")
# by checking it is found that taking all the feature increases the accuracy a lot

# using k fold cross validation to check and randomize the dataset
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
kfold=model_selection.KFold(n_splits=10,random_state=7)
result=cross_val_score(LogisticRegression(),x_train,y_train,cv=kfold,scoring="accuracy")
result.mean()

#visualiziing the result
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,model.predict(x_test))
