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

graph = [MaleLoan,FemaleLoan]
fig, ax1 = plt.subplots()

ax1.boxplot(graph, showfliers=False)


#print([item.get_ydata() for item in ax1.boxplot(sorted(MaleLoan))['whiskers']], [item.get_ydata() for item in ax2.boxplot(sorted(FemaleLoan))['whiskers']])
print([item.get_ydata() for item in ax1.boxplot(graph)['whiskers']])

plt.show()