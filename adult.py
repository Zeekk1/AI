#we are allowed to use pandas and scikit-learn 
# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(columns=["fnlwgt"])  # drops the column
	return df


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]
	

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	#return df.columns[df.isnull()]
	return [col for col in df.columns if df[col].isnull().any()]


# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.

def bachelors_masters_percentage(df):
	perc = df['education'].isin(["Bachelors", "Masters"]).mean() * 100
	return round(perc, 1)

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	newcopy=df.dropna().copy()
	
	return newcopy

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
    features = df.drop(columns=["class"]).copy()
    one_hot_df = pd.get_dummies(features)
    return one_hot_df

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    class_series = df['class'].copy()

    # Return the Series
    return class_series


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 

def dt_predict(X_train,y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return pd.Series(y_pred, index=X_train.index)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    y_pred_aligned = y_pred.reindex(y_true.index)
    error_rate = (y_pred_aligned != y_true).mean()
    return error_rate

