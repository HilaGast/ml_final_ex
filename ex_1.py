import pandas as pd

vars_description = pd.read_excel(r'C:\Users\hila\PycharmProjects\ml_course_codes\ml_course_codes\LCDataDictionary.xlsx')
loan_stats_data = pd.read_csv(r'C:\Users\hila\PycharmProjects\ml_course_codes\ml_course_codes\LoanStats3a.csv',low_memory=False)
# 1st row is irrelevant, therefore:
loan_stats_data = pd.read_csv(r'C:\Users\hila\PycharmProjects\ml_course_codes\ml_course_codes\LoanStats3a.csv',low_memory=False,skiprows=1)

#print(loan_stats_data.columns)

# many cols contains only ympty values (nans) - let's delete them:

loan_stats_data_new = loan_stats_data

for col in loan_stats_data.columns:
    current_col = loan_stats_data[col]
    to_del = True
    for r in current_col:
        if isNaN(current_col[r]):
            continue
        else:
            to_del = False
    if to_del:
        del loan_stats_data_new[col]

print(loan_stats_data_new.columns)