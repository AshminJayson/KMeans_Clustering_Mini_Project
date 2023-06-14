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

df = pd.read_csv('MINIPROJECT/Mall_Customers.csv')
df.head()
df.shape
df.info()
df.describe(include='all')

df.dtypes.value_counts() #data types

#checking for null values
df.isnull().sum().sort_values(ascending=False).head()

#on visualizing we can confirm that there are no null values
#msno.matrix(df)
#the plot with male and female in black background
#figure 1
figure_1=sns.countplot(x='Gender', data=df)

#lets see how gender affects to all other features.

figure_2=sns.pairplot(df, hue='Gender', vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            palette='husl',markers=['o','D'])

#let's look at the correlation
#heatmap is ocerlapping with the pairplot
corr = df.corr(numeric_only=True)
#figure_3=sns.heatmap(corr, annot=True, cmap='plasma', linewidth=0.1)

plt.figure(figsize=(14,5),facecolor='#54C6C0')

#number of genders
plt.subplot(1,2,1)
figure_4=sns.barplot(x=['Female','Male'], y=df['Gender'].value_counts(), data=df)
plt.xlabel("Gender", size=14)
plt.ylabel("Number", size=14)
plt.title("Number of Genders\n", color="red", size='22')

#mean spending score
spending_score_male = 0
spending_score_female = 0

for i in range(len(df)):
    if df['Gender'][i] == 'Male':
        spending_score_male = spending_score_male + df['Spending Score (1-100)'][i]
    if df['Gender'][i] == 'Female':
        spending_score_female = spending_score_female + df['Spending Score (1-100)'][i]

list_genders_spending_score_mean = [int(spending_score_male/df['Gender'].value_counts()[1]),int(spending_score_female/df['Gender'].value_counts()[0])]
series_genders_spending_score_mean = pd.Series(data = list_genders_spending_score_mean)

plt.subplot(1,2,2)
figre_5=sns.barplot(x=['Male (1)','Female (0)'], y=series_genders_spending_score_mean, palette='hsv')
plt.xlabel("Gender", size=14)
plt.ylabel("Mean Spending Score", size=14)
plt.title("Gender & Mean Spending Score\n", color="red", size='22')

#between age and spending score

'''plt.figure(figsize=(12,8))
sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'])
plt.title('Age vs Spending Score')'''

#between income and spending score
#let's add gender amidst income and spending score
plt.figure(figsize=(12,8))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],hue=df['Gender'])
plt.title('Annual Income vs Spending Score')

#to check for equal variance

#levene's test centred at
#median
stats.levene(df['Annual Income (k$)'],df['Spending Score (1-100)'],df['Age'], center='median')

#mean
stats.levene(df['Annual Income (k$)'],df['Spending Score (1-100)'],df['Age'], center='mean')

#getting categorical and nnumerical feature

numerical_features = [col for col in df.columns if df[col].dtypes != 'O']
categorical_features = [col for col in df.columns if df[col].dtype == 'O']
numerical_features

df.head()

#encoding our categorical columns
#this updates the dataframe to what we actually need 
temp = pd.get_dummies(df[categorical_features], drop_first=True)
temp

df = pd.concat([df, temp], axis=1)
df.drop(['CustomerID','Gender'],axis =1,inplace=True)

df.head()

#standardizing to make the variance same for all variables, also fro PCA
#the code standardizes the numerical features in df, making them have zero 
# mean and unit variance. This is a common preprocessing step in machine 
# learning pipelines, as it helps to bring features to a similar scale and 
# prevent any particular feature from dominating the learning algorithm based 
# on its magnitude.


sc = StandardScaler()
X = sc.fit_transform(df)

#PCA
#here in this code it actually we remove the column customer_id

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2 = pca.fit_transform(X)

print(df.head())
print(X_2[0:5,:])

#Elbow Method
#Inertia:It is the sum of squared distances of samples to their closest cluster center.

inertia = []
range_val = range(1,15)
for i in range_val:
    kmean = KMeans(n_clusters=i,n_init=10)
    kmean.fit_predict(X_2)
    inertia.append(kmean.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range_val, inertia, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
figure_5=plt.title('The elbow method using inertia')
plt.show()

#K-Means Clustering

clf = KMeans(n_clusters=4)
clf.fit_predict(X_2)

labels = clf.labels_
# Displaying the coordinates of the centroids
print("Coordinates of the Centroids:")
centroids = clf.cluster_centers_
labels


#cluster visualization
plt.figure(1,figsize=(16,9))

plt.scatter(X_2[labels == 0,0], X_2[labels==0,1], s=80, c='green', label='Cluster-1')
plt.scatter(X_2[labels == 1,0], X_2[labels==1,1], s=80, c='orange', label='Cluster-2')
plt.scatter(X_2[labels == 2,0], X_2[labels==2,1], s=80, c='red', label='Cluster-3')
plt.scatter(X_2[labels == 3,0], X_2[labels==3,1], s=80, c='purple', label='Cluster-4')

plt.scatter(centroids[:,0], centroids[:,1], s=400, c='black', marker='*',label='Centroids')

plt.title('Customers Clusters')
plt.xlabel('PCA Variable-1')
plt.ylabel('PCA Variable-2')
plt.legend()
plt.show()