import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv(r'C:\Users\admin\OneDrive\Documenten\DataSets\train_u6lujuX_CVtuZ9i.csv').dropna(axis=0)

gender = list(Data['Gender'])
loanamount = list(Data['LoanAmount'])

MaleLoan = []
FemaleLoan = []
for i in range(len(gender)):
        K = [gender[i], loanamount[i]]
        if K[0] == 'Male':
            K[0] = 0
            MaleLoan.append(loanamount[i])
        elif K[0] == 'Female':
            del(K[0])
            FemaleLoan.append(loanamount[i])
def boxplot(list):
    return (np.median(list), np.percentile(list, 25), np.percentile(list, 75))


print(boxplot(MaleLoan),boxplot(FemaleLoan))
def quadreg(Verg1, Verg2):
    model1 = np.poly1d(np.polyfit([(i*100)/(len(Verg1)) for i in range(len(sorted(Verg1)))],sorted(Verg1), 3))
    model2 = np.poly1d(np.polyfit([(i*100)/(len(Verg2)) for i in range(len(sorted(Verg2)))],sorted(Verg2), 3))
    polyline = np.linspace(1, 100, 50)

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.scatter([(i*100)/(len(Verg1)) for i in range(len(sorted(Verg1)))],sorted(Verg1), alpha=0.1)
    ax1.plot(polyline,model1(polyline), color='b')
    ax2.scatter([(i*100)/(len(Verg2)) for i in range(len(sorted(Verg2)))],sorted(Verg2),alpha=0.1)
    ax2.plot(polyline,model2(polyline),color='r')
    print(model1,model2)


#print([item.get_ydata() for item in ax1.boxplot(sorted(MaleLoan))['whiskers']], [item.get_ydata() for item in ax2.boxplot(sorted(FemaleLoan))['whiskers']])
#print([item.get_ydata() for item in ax1.boxplot(graph)['whiskers']])

plt.show()