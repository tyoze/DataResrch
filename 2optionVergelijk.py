import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv(r'C:\Users\admin\OneDrive\Documenten\DataSets\train_u6lujuX_CVtuZ9i.csv')

def boxplot(list):
    return (np.median(list),np.percentile(list, 25),np.percentile(list, 75))

class MijnDataset:
    Gender
    LoanAmount
    Married


def VergelijkenMetLoanAmount(Var1, trueword, falseword):
    Interessant = [Var1, 'LoanAmount']
    data = Data[Interessant].dropna()


    winkel =list(data[Var1]),list(data['LoanAmount'])
    plotVar = list(data[Var1])
    loanamount = list(data['LoanAmount'])

    ListPos = []
    ListNeg = []

    for i in range(len(plotVar)):
        K = [plotVar[i], loanamount[i]]
        if K[0] == trueword:
            K[0] = 0
            ListPos.append(loanamount[i])
        elif K[0] == falseword:
            del (K[0])
            ListNeg.append(loanamount[i])

    graph = [sorted(ListPos), sorted(ListNeg)]
    fig, ax1 = plt.subplots()

    #ax1.boxplot(graph, showfliers=False)
    ax1.set_title(label=Var1)

    return ([item.get_ydata() for item in ax1.boxplot(graph, showfliers=False)['whiskers']],boxplot(ListPos),boxplot(ListNeg))


print(VergelijkenMetLoanAmount('Married', 'Yes', 'No'))
plt.show()