import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics


Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')

Interessant = ['Self_Employed','Education','Married','ApplicantIncome','CoapplicantIncome','LoanAmount']
No_nan_data = Data[Interessant].dropna()
X_Interessant = ['ApplicantIncome','CoapplicantIncome']
test_data = Data[Interessant].dropna()

X = No_nan_data[X_Interessant]
Y = No_nan_data['LoanAmount']
X_test = test_data[X_Interessant]
Y_test = test_data['LoanAmount']

clf = DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction = clf.predict(X_test)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=X_Interessant,
                   filled=True)
plt.show()

