import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import pickle 


data = pd.read_csv(r"D:\Download\logit classification.csv") 

x = data.iloc[:,[2,3]].values 
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression() 
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test) 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print(cm) 

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report 
cr = classification_report(y_test, y_pred)
print(cr)  

bias = classifier.score(x_train,y_train)
print(bias) 

variance = classifier.score(x_test,y_test)
print(variance) 
 

with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)



#-----------Future Prediction-----
data1 = pd.read_csv(r"C:\Users\Sriya v\OneDrive\Desktop\final1.csv")

d2 = data1.copy()


le = LabelEncoder()
data1['Gender'] = le.fit_transform(data1['Gender'])

data1 = data1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
M = sc.fit_transform(data1)

y_pred1 = pd.DataFrame()
M.to_csv('trial.csv')
d2['y_pred1'] = classifier.predict(M)
 
d2.to_csv('pred_model.csv')

import os 
os.getcwd()

