import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# import dataset and read it into pandasdataformat
Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')

#choosing categories
Interessant = ['Self_Employed','Education','Married','ApplicantIncome','CoapplicantIncome','LoanAmount']
No_nan_data = Data[Interessant].dropna()
X_Interessant = ['ApplicantIncome']
test_data = Data[Interessant].dropna()

#deploying chosen catagories on dataset
X = No_nan_data[X_Interessant]
Y = No_nan_data['LoanAmount']
X_test = test_data[X_Interessant]
Y_test = test_data['LoanAmount']

#decision tree model
clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)

#visual representation of the tree
clf.max_depth()
