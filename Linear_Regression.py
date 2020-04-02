import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000) # number of chars. default=80
pd.options.mode.chained_assignment = None # to eliminate the SettingWithCopyWarning
files = [file for file in os.listdir("C:\\Users\\eserh\\Desktop\\Data Files\\Exercise2")]
data = pd.DataFrame()
for file in files:
     df = pd.read_csv("C:\\Users\\eserh\\Desktop\\Data Files\\Exercise2\\" + file)
     data = pd.concat([data, df])
data.to_csv("data.csv", index=False)
df = pd.read_csv("data.csv")
print(df.isnull().any())
null_columns = df.columns[df.isnull().any()]
index_null_entry = df[df["y"].isnull()][null_columns].index
df["y"].fillna(df.loc[index_null_entry, "x"], inplace = True)
x = np.array(df['x']).reshape(-1,1)
y = np.array(df['y'])
clf = LinearRegression()
clf.fit(x, y)
accuracy = clf.score(x, y)
print(accuracy)
