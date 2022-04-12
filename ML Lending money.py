import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn.metrics import mean_squared_error, r2_score


Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')
Nodig = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Education', 'Self_Employed', 'Married','Gender']
Intressant = ['ApplicantIncome', 'CoapplicantIncome', 'Education', 'Self_Employed', 'Married','Gender']

Testdata = Testcsv[Nodig].dropna()
data = Data[Nodig].dropna()

data.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
data.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
data.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

Testdata.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
Testdata.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
Testdata.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

X_train, Y_train, X_test, Y_test = data[Intressant], data['LoanAmount']\
    , Testdata[Intressant], Testdata['LoanAmount']

def LinReg():
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    return lr.score(X_test,Y_test)

def DecTree():
    clf = tree.DecisionTreeRegressor(max_depth=2).fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    mse = mean_squared_error(Y_test, prediction)
    r2 = r2_score(Y_test, prediction)

    sklearn.tree.plot_tree(clf)
    return "The R2 score is", r2,"The root mean squared error is", mse**(1/2)

plt.show()