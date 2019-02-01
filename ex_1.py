import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score



# Skipping the 1st row, starting to read data from columns headers:
loan_stats_data = pd.read_csv(r'LoanStats3a.csv',low_memory=False,skiprows=1)

# Deleting columns that contain only NaNs:
loan_stats_data=loan_stats_data.dropna(axis=1,how='all',thresh=30000)

# Create a boolean column for loan_status:
d = {'Charged Off':0, 'Fully Paid':1}
loan_stats_data['loan_status_bool'] = pd.Series(loan_stats_data['loan_status'].map(d))

# remove rows with NaN in loan_status_bool column:
loan_stats_data=loan_stats_data.dropna(axis=0, how='any', subset=['loan_status_bool'])

# Removing all columns that have only one unique value:
loan_stats_data=loan_stats_data[[c for c in list(loan_stats_data) if loan_stats_data[c].nunique(dropna=True)>1]]

# Get the names of all non-numeric columns:
loan_stats_data.select_dtypes(exclude='number').columns

# Explore the possible values for some of the non-numeric variables:
print('addr_state:')
print(loan_stats_data.addr_state.unique())
print("We will examine the 'addr_state' as a categorical predictor.")

print('emp_length:')
print(loan_stats_data.emp_length.unique())
print("We will examine 'emp_length' as a ordinal variable")

print('grade:')
print(loan_stats_data.grade.unique())
print("We will examine 'grade' as a ordinal variable")

print('sub_grade:')
print(loan_stats_data.sub_grade.unique())
print("We will examine the 'grade' variable because it has less categories.")

print('home_ownership:')
print(loan_stats_data.home_ownership.unique())
print("We will examine the 'home_ownership' as a categorical predictor.")

print('int_rate:')
print(loan_stats_data.int_rate.unique())
# Seems like a a numeric variable so we will change it to float fractions:
loan_stats_data['int_rate']=loan_stats_data['int_rate'].str.rstrip('%').astype('float')/100

print('issue_d:')
print(loan_stats_data.issue_d.unique())
# We will examine the month as a categorical predictor:
loan_stats_data['issue_d'] = loan_stats_data['issue_d'].str[0:3]

print('revol_util:')
print(loan_stats_data.revol_util.unique())
# Seems like a a numeric variable so we will change it to float fractions:
loan_stats_data['revol_util']=loan_stats_data['revol_util'].str.rstrip('%').astype('float')/100

print('zip_code:')
print(loan_stats_data.zip_code.unique())
print("Too many categories. We won't use this variable in further analysis.")

# Find numeric variables:
numeric_data = loan_stats_data._get_numeric_data()

# Find correlations between each column to the loan_status_bool:
r = numeric_data.corr(method="pearson")

plt.figure(figsize=(20,10))
plt.barh(r.columns[:-1],r.loan_status_bool[:-1])
plt.axvline(color='black')
plt.xlim(-0.4, 0.4)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Parameter',fontsize=25)
plt.xlabel('Correlation coefficient (r)', fontsize=25)
plt.title('Correlations between all numerical\nparameters and loan status',fontsize=30)
plt.show()

# Choose only numeric parametirs with an absolute correlation greater than 0.1:
r = r.loan_status_bool[:-1]
corr_params = r.values
is_corr = np.where(np.abs(corr_params)>0.1)
selected_numeric_params = numeric_data[r.index[is_corr]]
# Leave only correlated parameters and display descriptive statistics:

selected_numeric_params.describe()

fig = plt.figure(figsize=(20,10))
plt.imshow(selected_numeric_params.corr().as_matrix())
plt.clim(-1,1)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Correlation coefficient (r)',fontsize=25)
plt.xticks(np.arange(0,len(is_corr[0])),list(r.index[is_corr[0]]), rotation='vertical',fontsize=20)
plt.yticks(np.arange(0,len(is_corr[0])),list(r.index[is_corr[0]]),fontsize=20)
plt.ylabel('Parameter',fontsize=25)
plt.xlabel('Parameter', fontsize=25)
plt.title('All pairewise correllations between\nthe parameters selected for further analysis',fontsize=30)

plt.show()

# Examining ordinal variables ('grade' & 'emp_length'):
ordinal_data=pd.DataFrame()
d={'10+ years':10, '< 1 year':0, '1 year': 1, '2 years': 2,'3 years': 3,'4 years': 4,'5 years': 5,
   '6 years': 6,'7 years': 7,'8 years': 8,'9 years': 9}
ordinal_data['emp_length'] = pd.Series(loan_stats_data['emp_length'].map(d))
d = {'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1}
ordinal_data['grade'] = pd.Series(loan_stats_data['grade'].map(d))
ordinal_data['loan_status_bool'] = loan_stats_data['loan_status_bool']
s = ordinal_data.corr(method="spearman")

plt.figure(figsize=(3,7))
plt.bar(s.columns[:-1],s.loan_status_bool[:-1],0.4)
plt.axhline(color='black')
plt.ylim(-0.1, 0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Parameter',fontsize=15)
plt.ylabel('Spearman correlation coefficient', fontsize=15)
plt.title('Correlations between ordinal\nparameters and loan status',fontsize=18)
plt.show()

# Examining categorical variables ('issue_d','home_ownership' & 'addr_state'):
grouped_home_status=loan_stats_data.groupby(["home_ownership","loan_status_bool"]).size()

def pctfunc(pct,allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:d}\n({:.1f}%)".format(absolute,pct)

plt.figure(figsize=(15,8))
plt.suptitle('Loan status for different home ownership status', fontsize=20)
plt.subplot(2,2,1)
plt.pie(grouped_home_status['MORTGAGE'],labels=['Charged off', 'Fully paid'], colors=['red','green'],
       autopct=lambda pct: pctfunc(pct,grouped_home_status['MORTGAGE']), textprops={'fontsize':12})
plt.title('MORTGAGE', fontsize=15)

plt.subplot(2,2,2)
plt.pie(grouped_home_status['OWN'],labels=['Charged off', 'Fully paid'], colors=['red','green'],
       autopct=lambda pct: pctfunc(pct,grouped_home_status['OWN']), textprops={'fontsize':12})
plt.title('OWN', fontsize=15)

plt.subplot(2,2,3)
plt.pie(grouped_home_status['OTHER'],labels=['Charged off', 'Fully paid'], colors=['red','green'],
       autopct=lambda pct: pctfunc(pct,grouped_home_status['OTHER']), textprops={'fontsize':12})
plt.title('OTHER', fontsize=15)

plt.subplot(2,2,4)
plt.pie(grouped_home_status['NONE'],labels=['Fully paid'], colors=['green'],
       autopct=lambda pct: pctfunc(pct,grouped_home_status['NONE']), textprops={'fontsize':12})
plt.title('NONE', fontsize=15)
plt.show()

loan_stats_data['month_num']=pd.to_datetime(loan_stats_data.issue_d, format="%b", errors="coerce").dt.month
grouped_issued_status=loan_stats_data.groupby(["loan_status_bool","month_num"]).size()

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.bar(np.arange(1,13), grouped_issued_status[1.0].values,color='green')
plt.bar(np.arange(1,13), grouped_issued_status[0.0].values, bottom=grouped_issued_status[1.0].values, color='red')
plt.xticks(np.arange(12,0,-1),loan_stats_data.issue_d.unique(), fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel('Month',fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Monthly frequency histogram',fontsize=20)
plt.legend(('Fully Paid','Charged Off'),fontsize=15)

plt.subplot(1,2,2)
ppo = 100* grouped_issued_status[1.0].values/(grouped_issued_status[1.0].values+grouped_issued_status[0.0].values)
plt.bar(np.arange(1,13), ppo, color='green')
plt.xticks(np.arange(12,0,-1),loan_stats_data.issue_d.unique(), fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel('Month',fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.title('Fully paid %',fontsize=20)
plt.ylim(0,100)

plt.show()

grouped_state_status=loan_stats_data.groupby(["loan_status_bool","addr_state"])
g=grouped_state_status.loan_status_bool.agg(['count'])
idx = pd.MultiIndex.from_product([g.index.levels[0],g.index.levels[1]])
g = g.reindex(idx,fill_value=0)
state_num = len(loan_stats_data.addr_state.unique())
plt.figure(figsize=(25,20))
plt.subplot(2,1,1)
plt.bar(np.arange(1,state_num+1), g.loc[1.0].values.ravel(),color='green')
plt.bar(np.arange(1,state_num+1), g.loc[0.0].values.ravel(), bottom=g.loc[1.0].values.ravel(), color='red')
plt.xticks(np.arange(1,state_num+1),g.index.get_level_values(1), fontsize=15)
plt.yticks(fontsize=12)
plt.xlabel('State',fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('State frequency histogram',fontsize=30)
plt.legend(('Fully Paid','Charged Off'),fontsize=20)

plt.subplot(2,1,2)
ppo = 100* g.loc[1.0].values.ravel()/(g.loc[1.0].values.ravel()+g.loc[0.0].values.ravel())
plt.bar(np.arange(1,state_num+1), ppo, color='green')
plt.xticks(np.arange(1,state_num+1),g.index.get_level_values(1), fontsize=15)
plt.yticks(fontsize=12)
plt.xlabel('State',fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.title('Fully paid %',fontsize=30)
plt.ylim(0,100)
plt.show()

# Split data to train, test & validate data sets:
selected_data = selected_numeric_params
loan_status = loan_stats_data['loan_status_bool']

x_train,x_test_val,y_train,y_test_val = train_test_split(selected_data,loan_status,test_size=0.4,random_state=0)
x_test,x_val,y_test,y_val = train_test_split(x_test_val,y_test_val,test_size=0.5,random_state=0)

lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
print('Accuracy of logistic regression model: {:.3f}'.format(lr_model.score(x_test,y_test)))

# Re-encoding ordinal variable as multiple boolean parameters:
enc = OneHotEncoder(handle_unknown = "ignore")
enc.fit(np.arange(1,7).reshape(-1,1))
x=enc.transform(ordinal_data.grade.reshape(-1,1)).toarray()
selected_data = selected_data.join(pd.DataFrame(x,columns=['A','B','C','D','E','F']))

scaler = StandardScaler().fit(x_train)
lr_model = LogisticRegression(penalty='l1')
x_train,x_test_val,y_train,y_test_val = train_test_split(selected_data,loan_status,test_size=0.4,random_state=0)
x_test,x_val,y_test,y_val = train_test_split(x_test_val,y_test_val,test_size=0.5,random_state=0)
scaler = StandardScaler().fit(x_train)
lr_model.fit(scaler.transform(x_train),y_train)
print('Accuracy of logistic regression model: {:.3f}'.format(lr_model.score(scaler.transform(x_test),y_test)))
############
 #vars selection:
model = SelectFromModel(lr_model,prefit=True)
model.transform(scaler.transform(x_train))

############
# error estimation - cross validation:
scaled_data = StandardScaler().fit(selected_data)
scores = cross_val_score(lr_model,scaled_data.transform(selected_data),loan_status,cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))