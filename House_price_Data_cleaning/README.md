## Data cleaning 

This is the process of removing/modifying incorrect or inconsistent data types from a dataset.

# House Prices Dataset

This dataset was downloaded from https://www.kaggle.com/code/emrearslan123/house-price-prediction/notebook
data shape = 1459 x 80
About 7000 total missing/null values


# Cleaning Procedure

Data was imported into python environment
All necessary data analysis libraries were also imported
.info() method showed availability of null values
Python functions were written to print out all attributes with missing values and their respective number of missing entries
(prnt_msn_val(data_set) and check_msn_val(dataset))
All missing values were replaced accordingly with 'Not_Avail' for attributes which are not available for that observation and of categorical type,
while '0' was used to replace attributes of numerical data type where such observation was alaso absent.
Finally, wherever there was missing entry where a value was expected, the modal value of the attribute was used as a replacement for categorical data type
and median values for numerical data type.
