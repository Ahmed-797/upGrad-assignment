import pandas as pd #importing,handling data
import matplotlib.pyplot as plt #plotting

data = pd.read_csv('bank-additional-full.csv', sep = ';')

#Dependent variable
y = data['y']
y = y.replace(['yes','no'],(1,0))

from sklearn import preprocessing

def encode_features(df_train):
    features = ['job','marital','default','education','housing','loan','contact','month','day_of_week','poutcome']
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_train[feature])
        df_train[feature] = le.transform(df_train[feature])
    return df_train

#Encoding non-numerical values
data = encode_features(data)

#Independent variables
X = data.drop(['y'], axis = 1)

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

#Using RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, max_depth=10, random_state=100,criterion='gini')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Calculating accuracy of the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

#plotting importance of all factors
importances=pd.Series(clf.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(14,8))
