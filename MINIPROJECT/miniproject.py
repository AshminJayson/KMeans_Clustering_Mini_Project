#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

pd.pandas.set_option('display.max_columns', None)

from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.cluster import KMeans

#loading in the dataset

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.shape
df.shape
df.describe(include='all')

df.dtypes.value_counts() #data types

#checking for null values
df.isnull().sum().sort_values(ascending=False).head()

#on visualizing we can confirm that there are no null values
msno.matrix(df)

sns.countplot(x='Gender', data=df, palette="Set3")

#lets see how gender affects to all other features.

sns.pairplot(df, hue='Gender', vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            palette='husl',markers=['o','D'])

#let's look at the correlation
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='plasma', linewidth=0.1)