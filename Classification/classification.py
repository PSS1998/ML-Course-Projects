#imports for data analysis and wrangling
import pandas as pd
import numpy as np

#Train_test_split
from sklearn.model_selection import train_test_split

#scaling
from sklearn.preprocessing import StandardScaler

#models
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#accuracy score
from sklearn.metrics import f1_score


#Importing data
heart = pd.read_csv('heart.csv')

heart = heart.drop(heart[heart['caa'] == 4].index)
heart = heart.drop(heart[heart['thall'] == 0].index)

#Extracting the features
X,y = heart.iloc[:,0:-1] , heart.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
knn2 =  KNeighborsClassifier(n_neighbors=2)
knn3 =  KNeighborsClassifier(n_neighbors=3)
knn4 =  KNeighborsClassifier(n_neighbors=4)
knn5 =  KNeighborsClassifier(n_neighbors=5)
knn6 =  KNeighborsClassifier(n_neighbors=6)
knn7 =  KNeighborsClassifier(n_neighbors=7)
knn8 =  KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=200)
svm = SVC()

#list of classifiers
models =     [("Random Forest",rf),
              ("SVM",svm),
              ("Logistic Regression",lr),
              ("2 Nearest Neighbors",knn2),
              ("3 Nearest Neighbors",knn3),
              ("4 Nearest Neighbors",knn4),
              ("5 Nearest Neighbors",knn5),
              ("6 Nearest Neighbors",knn6),
              ("7 Nearest Neighbors",knn7),
              ("8 Nearest Neighbors",knn8),
              ("Classification Tree", dt)]

for model_name, model in models:
    #fit clf to the training set
    model.fit(X_train, y_train)
    #predict the labels of the test set
    y_pred = model.predict(X_test)
    #Evaluate accuracy of the clf on the test set
    print('The F1 score of {:s} is : {:.3f}'.format(model_name, f1_score(y_test,y_pred)))

# best model
logistic_regression = models[2][1]

# x = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]
def predict_heart_attack(x):
    x2 = scaler.transform([x])
    return logistic_regression.predict(x2)

x2 = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
y2 = predict_heart_attack(x2)
print(y2)


