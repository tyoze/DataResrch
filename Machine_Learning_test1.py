import pandas as pd
from sklearn.linear_model import LinearRegression

Testcsv = pd.read_csv(r'test_Y3wMUE5_7gLdaTN.csv')
Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')
Interessant = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount']

Testdata = Testcsv[Interessant].dropna()
data = Data[Interessant].dropna()

X_train, Y_train, X_test, Y_test = data[['ApplicantIncome','CoapplicantIncome']], data['LoanAmount']\
    , Testdata[['ApplicantIncome','CoapplicantIncome']], Testdata['LoanAmount']

lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)

print(lr.score(X_test,Y_test))