import pandas as pd
import numpy as np
import seaborn as sns
# from datetime import time, timedelta

#this code imports all the data_set to the work environment
# price_data is the table holding the prices of each houses identified as ID
# test_data is the dataframe holding the variables affecting the house prices
# train_data is the dataframe holding the data which will be used to train the model
# https://www.kaggle.com/code/emrearslan123/house-price-prediction/notebook

price_data = pd.read_csv("C:\\Users\\USER\\Documents\\CLASSES\\ACETEL\\house-prices-advanced-regression-techniques\\sample_submission.csv")

test = pd.read_csv("C:\\Users\\USER\\Documents\\CLASSES\\ACETEL\\house-prices-advanced-regression-techniques\\test.csv")

pd.options.display.max_columns = None

print(test.head(10))
print(price_data.head(10))

'''======================================================================================================================================='''

# this function prints a list of all attributes of the data_set with respective number of missing data in table format

def check_msn_val(data_set):

    ''' This function takes in data_set as an argument, and returns a dataframe of missing values in each attribute'''
    
    msn_val_table = pd.DataFrame(index=data_set.columns, columns=['Missn_Val'])
    for attrib in data_set.columns:
        a = []
        for index in data_set.index[data_set[attrib].isna() == True]:
            a.append(index)
        msn_val_table['Missn_Val'][attrib] = len(a)
    
    return (msn_val_table)

# this function prints only a list of attributes with missing values 


def prnt_msn_val(data_set):
    
    a = data_set.columns
    dic = {}
    
    print('ATTRIBUTES WITH MISSING OBSERVATIONS')
    print('======================================')
    for attrib in a:
        dic[attrib] = data_set.shape[0] - data_set[attrib].count()

    for i in dic:
        if dic[i] > 0:
            print(i, dic[i], sep=' : ')
prnt_msn_val(test)

'''========================================================================================================================================''''

# this code replaces and fills all null values with either a modal or median value so as to not skew the data_set
# filling and replacing was done singly as data_set contains different types of data
# and must be handled different. Hence, automating this process may be difficult

test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)
test['LotFrontage'].replace(np.NaN, 0, inplace=True)
test['Alley'].replace(np.NaN, 'Not_Avail', inplace=True)
test['Utilities'].fillna(test['Utilities'].mode()[0], inplace=True)
test['Exterior1st'].fillna(test['Exterior1st'].mode()[0], inplace=True)
test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0], inplace=True)
test['BsmtQual'].fillna('Not_Avail', inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].median(), inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].median(), inplace=True)
test['KitchenQual'].fillna(test['KitchenQual'].mode(), inplace=True)
test['Functional'].fillna(test['Functional'].mode(), inplace=True)
test['FireplaceQu'].fillna('Not_Avail', inplace=True)
test['GarageYrBlt'].fillna('Not_Avail', inplace=True)
test['GarageFinish'].fillna('Not_Avail', inplace=True)
test['PoolQC'].fillna('Not_Avail', inplace=True)
test['Fence'].fillna('Not_Avail', inplace=True)
test['MiscFeature'].fillna('Not_Avail', inplace=True)
test['SaleType'].fillna(test['SaleType'].mode()[0], inplace=True)
test["BsmtCond"].fillna('Not_Avail', inplace=True)
test["BsmtExposure"].fillna('Not_Avail', inplace=True)
test["BsmtFinType1"].fillna('Not_Avail', inplace=True)
test["BsmtFinType2"].fillna('Not_Avail', inplace=True)
test['KitchenQual'].fillna(test['KitchenQual'].mode()[0], inplace=True)
test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)
test['BsmtFinSF1'].replace(np.NaN, 0, inplace=True)
test['BsmtFinSF2'].replace(np.NaN, 0, inplace=True)
test['BsmtUnfSF'].replace(np.NaN, 0, inplace=True)
test['TotalBsmtSF'].replace(np.NaN, 0, inplace=True)
test["GarageType"].fillna('Not_Avail', inplace=True)
test["GarageCars"].replace(np.nan, 0, inplace=True)
test["GarageQual"].fillna('Not_Avail', inplace=True)
test["GarageCond"].fillna('Not_Avail', inplace=True)
test["GarageArea"].replace(np.nan, test['GarageArea'].mode()[0], inplace=True)

# this code is repeated to verify if all missing data have been taken care of
print('\n')
print('SECOND CHECK: ')
prnt_msn_val(test)

# this code replaces all missing values with NA (string type)

test['MasVnrType'].fillna('NA', inplace=True)
test['MasVnrArea'].fillna('NA', inplace=True)

copy_test = test.copy()

for i in test.index[test['MasVnrType'] == 'NA']:
    if test.loc[i, 'MasVnrType'] == 'NA' and test.loc[i, 'MasVnrArea'] == 'NA':
        test.loc[i, 'MasVnrType'] = 'None'
        test.loc[i, 'MasVnrArea'] = test['MasVnrArea'].mode()[0]
    elif test.loc[i, 'MasVnrType'] == 'NA' and test.loc[i, 'MasVnrArea'] != 'NA':        
        test.loc[i, 'MasVnrType'] = test['MasVnrType'][test['MasVnrType'] != 'None'].mode()[0]   # this line of code filters out none cos we can't have an area covered if the covering type is none.
            

# final check for missing data
print('\n')
print('FINAL CHECK')
prnt_msn_val(test)

# this code will merge price_data and test on ID attribute

merged_data = pd.merge(test, price_data, on='Id')


#this line of code changes the data types of the selected attributes

test[['Id', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'YrSold', 'MSSubClass']] = test[['Id', 'YearBuilt', 'YearRemodAdd', 'OverallQual',\
                                                                                                        'OverallCond', 'YrSold', 'MSSubClass']]\
                                                                                                        .astype('object')
test[['GarageArea', 'GarageCars',  'MasVnrArea']] = test[['GarageArea', 'GarageCars',  'MasVnrArea']].astype('float')

# this code saves the cleaned dataset into a csv file

test.to_csv('Cleaned_Test_data.csv')



# CODE BY HUMING-BIRD
