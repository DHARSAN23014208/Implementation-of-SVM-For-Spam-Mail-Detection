# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: DHARSAN KUMAR R
RegisterNumber: 212223240028

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data Head:

![image](https://github.com/DHARSAN23014208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365413/01a05b47-6be1-4934-8edd-1058d3f6a784)

## Data Info:

![image](https://github.com/DHARSAN23014208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365413/78160d4f-35b8-4ff4-9253-fe74307e4eef)

## Data isnull():

![image](https://github.com/DHARSAN23014208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365413/52939c42-37a8-477c-b395-913de0eded23)

## y_pred:

![image](https://github.com/DHARSAN23014208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365413/d2ff78fd-f669-4a0a-b240-0cb45bd243c3)


## Accuracy:

![image](https://github.com/DHARSAN23014208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365413/a2267aa4-a9cb-43fb-9086-f940c2467288)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
