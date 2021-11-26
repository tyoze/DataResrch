import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv(r'train_u6lujuX_CVtuZ9i.csv')

def VergelijkenMetLoanAmount(Var1):
    Interessant = [Var1, 'LoanAmount']
    data = Data[Interessant].dropna()

    plt.scatter(data['LoanAmount'],data[Var1], alpha= 0.5)





VergelijkenMetLoanAmount('ApplicantIncome')
VergelijkenMetLoanAmount('CoapplicantIncome')

plt.show()