import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# Skipping the 1st row, starting to read data from columns headers:
loan_stats_data = pd.read_csv(r'C:\Users\hila\ml_final_ex\LoanStats3a.csv',low_memory=False,skiprows=1)

# Deleting columns that contain only NaNs:
loan_stats_data=loan_stats_data.dropna(axis=1,how='all',thresh=30000)

# Create a boolean column for loan_status:
d = {'Charged Off':0, 'Fully Paid':1}
loan_stats_data['loan_status_bool'] = pd.Series(loan_stats_data['loan_status'].map(d))

# remove rows with NaN in loan_status_bool column:
loan_stats_data=loan_stats_data.dropna(axis=0, how='any', subset=['loan_status_bool'])

# Removing all columns that have only one unique value:
loan_stats_data=loan_stats_data[[c for c in list(loan_stats_data) if loan_stats_data[c].nunique(dropna=True)>1]]

# Find numeric variables:
numeric_data = loan_stats_data._get_numeric_data()



# change string variables with % to numeric:




# Find correlations between each column to the loan_status_bool:
r = numeric_data.corr(method="pearson")

plt.barh(r.columns[:-1],r.loan_status_bool[:-1])
plt.show()

r = r.loan_status_bool[:-1]
corr_params = r.values
is_corr = np.where(np.abs(corr_params)>0.1)

numeric_data[r.index[is_corr]].describe()

fig = plt.figure(figsize=(20,10))
plt.imshow(numeric_data[r.index[is_corr]].corr().as_matrix())
plt.colorbar()
plt.xticks(np.arange(0,len(is_corr[0])),list(r.index[is_corr[0]]))
plt.show()



d = {'Charged Off':0, 'Fully Paid':1}
loan_stats_data['loan_status_bool'] = pd.Series(loan_stats_data['loan_status'].map(d))



r = numeric_data.corr(method="pearson")
r = r.loan_status_bool[:-1]
corr_params = r.values
is_corr = np.where(np.abs(corr_params)>0.1)
selected_numeric_params = numeric_data[r.index[is_corr]]
selected_data = selected_numeric_params
loan_status = loan_stats_data['loan_status_bool']
x_train,x_test,y_train,y_test = train_test_split(selected_data,loan_status,test_size=0.25,random_state=0)