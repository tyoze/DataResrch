import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score

# import dataset and read it into pandasdataformat
Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')

#choosing categories
Interessant = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Education', 'Self_Employed', 'Married', 'Gender', 'Credit_History']
No_nan_data = Data[Interessant].dropna()
X_Interessant = ['ApplicantIncome', 'CoapplicantIncome', 'Education', 'Self_Employed', 'Married', 'Gender', 'Credit_History']
test_data = Testcsv[Interessant].dropna()

No_nan_data.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
No_nan_data.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
No_nan_data.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

test_data.replace(to_replace=['Male', 'Female'], value=[1, 0], inplace=True)
test_data.replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
test_data.replace(to_replace=['Graduate', 'Not Graduate'], value=[1, 0], inplace=True)

#deploying chosen catagories on dataset
X = No_nan_data[X_Interessant].to_numpy()
Y = No_nan_data['LoanAmount'].to_numpy()
X_test = test_data[X_Interessant].to_numpy()
Y_test = test_data['LoanAmount'].to_numpy()

#decision tree model
clf = tree.DecisionTreeRegressor(max_depth=2).fit(X,Y)
prediction = clf.predict(X_test)
mse = mean_squared_error(Y_test, prediction)
r2 = r2_score(Y_test,prediction)
print(mse,r2)

#visual representation of the tree
sklearn.tree.plot_tree(clf)
plt.show()
