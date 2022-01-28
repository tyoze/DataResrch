import pandas as pd
from sklearn.linear_model import LinearRegression

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

lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)

print(lr.score(X_test,Y_test))