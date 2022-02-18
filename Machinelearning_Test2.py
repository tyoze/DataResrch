import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn import tree
from sklearn.metrics import accuracy_score

# import dataset and read it into pandasdataformat
Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')

#choosing categories
Interessant = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Education', 'Self_Employed', 'Married','Gender']
No_nan_data = Data[Interessant].dropna()
X_Interessant = ['ApplicantIncome', 'CoapplicantIncome','Education', 'Self_Employed', 'Married','Gender']
test_data = Testcsv[Interessant].dropna()

No_nan_data.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
No_nan_data.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
No_nan_data.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

test_data.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
test_data.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
test_data.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

#deploying chosen catagories on dataset
X = No_nan_data[X_Interessant]
Y = No_nan_data['LoanAmount']
X_test = test_data[X_Interessant]
Y_test = test_data['LoanAmount']

#decision tree model
clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)
prediction = clf.predict(X_test)
print(accuracy_score(prediction,Y_test))

#visual representation of the tree
sklearn.tree.plot_tree(clf)
plt.show()
