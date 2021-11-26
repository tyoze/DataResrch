import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')
Interessant = ['ApplicantIncome', 'LoanAmount']

Testdata = Testcsv[Interessant].dropna()
data = Data[Interessant].dropna()

X_train, Y_train = data['ApplicantIncome'], data['LoanAmount']
X_test, Y_test = Testdata['ApplicantIncome'], Testdata['LoanAmount']

LR = LinearRegression()
LR.fit(X_train.values.reshape(-1, 1), Y_train.values)

prediction = LR.predict(X_test.values.reshape(-1, 1))

plt.plot(X_test, prediction)
plt.scatter(X_test,Y_test)
plt.show()