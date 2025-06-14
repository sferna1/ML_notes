import pandas as pd
import sklearn.datasets import fetch_california_housing

#fetching raw data
raw_data = fetch_california_housing

#using dictionary keys to access pandas dataframe (df) and series
X = raw_data['data']
y = raw_data['target']

#concatenating into 1 dataframe, just because
df = pd.concat([X,y], axis=1) #axis = 1 tells them to join together the columns

#look at the head of dataframe, prints the first 5 rows of the dataset
print(df.head())

#EDA: exploration of data analysis

