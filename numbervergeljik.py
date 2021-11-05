import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv(r'C:\Users\admin\OneDrive\Documenten\DataSets\train_u6lujuX_CVtuZ9i.csv').dropna(axis=0)
print(Data.columns)

loanamount = list(Data['LoanAmount'])
variabelenum = list(Data['Dependents'])

