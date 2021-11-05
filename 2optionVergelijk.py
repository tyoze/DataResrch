import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv(r'C:\Users\admin\OneDrive\Documenten\DataSets\train_u6lujuX_CVtuZ9i.csv')

def boxplot(list):
    return (np.median(list),np.percentile(list, 25),np.percentile(list, 75))


def quadreg(Verg1, Verg2,title,Uppersubtitle,lowersubtitle):
    model1 = np.poly1d(np.polyfit([(i * 100) / (len(Verg1)) for i in range(len(sorted(Verg1)))], sorted(Verg1), 3))
    model2 = np.poly1d(np.polyfit([(i * 100) / (len(Verg2)) for i in range(len(sorted(Verg2)))], sorted(Verg2), 3))
    polyline = np.linspace(1, 100, 50)

    fig, (ax1, ax2) = plt.subplots(2,constrained_layout=True)
    fig.suptitle(title, fontsize=18)
    ax1.scatter([(i * 100) / (len(Verg1)) for i in range(len(sorted(Verg1)))], sorted(Verg1), alpha=0.1)
    ax1.plot(polyline, model1(polyline), color='b')
    ax1.set_title(Uppersubtitle)

    ax2.scatter([(i * 100) / (len(Verg2)) for i in range(len(sorted(Verg2)))], sorted(Verg2), alpha=0.1)
    ax2.plot(polyline, model2(polyline), color='r')
    ax2.set_title(lowersubtitle)
    print(model1, model2,sep='\n')
    plt.show()

class MijnDataset:
    pass
    #Gender
    #LoanAmount
    #Married


def VergelijkenMetLoanAmount(Var1, trueword, falseword):
    Interessant = [Var1, 'LoanAmount']
    data = Data[Interessant].dropna()


    #winkel =list(data[Var1]),list(data['LoanAmount'])
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

    quadreg(ListPos,ListNeg,Var1,trueword,falseword)

    #graph = [sorted(ListPos), sorted(ListNeg)]
   # fig, ax1 = plt.subplots()

    #ax1.boxplot(graph, showfliers=False)
    #ax1.set_title(label=Var1)

    #return ([item.get_ydata() for item in ax1.boxplot(graph, showfliers=False)['whiskers']],boxplot(ListPos),boxplot(ListNeg))


print(VergelijkenMetLoanAmount('Gender', 'Male', 'Female'),VergelijkenMetLoanAmount('Married', 'Yes', 'No')
      ,VergelijkenMetLoanAmount('Education', 'Graduate', 'Not Graduate'),VergelijkenMetLoanAmount('Self_Employed', 'Yes', 'No')
      ,VergelijkenMetLoanAmount('Credit_History', 1,0), sep='\n')

