import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('sensor.csv')
dataX = dataset.iloc[:, 2:54].values
datay = dataset.iloc[:, 54].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
datay = labelencoder_y.fit_transform(datay)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataX)
dataX = imputer.transform(dataX)

complete_data = np.column_stack([dataX,datay])

X=[]
y=[]

count = 0
for i in range(len(complete_data)):
    if complete_data[i,51] == 1 and count <15000 :
        X.append(complete_data[i,:-1])
        y.append(complete_data[i,51]) 
        count += 1
        
count1 = 0
for j in range(len(complete_data)):
    if complete_data[j,51] == 2 and count1 <15000 :
#        print(complete_data[j,51])
        X.append(complete_data[j,:-1])
        y.append(complete_data[j,51])
        count1 += 1
#        if len(y) >= 15000:
#            break

X = np.asarray(X)
y = np.asarray(y)

fig, ax = plt.subplots()
ax.hist(y)
        

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test,y_pred)
