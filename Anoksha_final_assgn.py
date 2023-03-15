# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:06:03 2023

@author: anoks
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Loading the breast cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df.head()

# Split the dataset into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Train and evaluate the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_y_pred)

# Train and evaluate the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_y_pred)

# Train and evaluate the Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_y_pred)

# Comparing the results
data = {'Logistic Regression': [lr_acc], 'KNN': [knn_acc], 'Naive Bayes': [nb_acc]}
results_df = pd.DataFrame(data, index=['Accuracy'])
print(results_df)

#Printing the comparison
results_df.to_csv('model_comparison.csv', index=False)

df = pd.read_csv('model_comparison.csv')
print(df)

#2
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Fetch the web page using requests library
url = "https://fireecology.springeropen.com/articles/10.1186/s42408-019-0062-8"
res = requests.get(url)

# Using BeautifulSoup Parsing HTML content
soup = BeautifulSoup(res.content, "html.parser")

# Extractraction of the article 
a = ""
for par in soup.find_all("p"):
    a += par.text

# Perform sentiment analysis on the entire article using NLTK
s = SentimentIntensityAnalyzer()
senti = s.polarity_scores(a)

# Print the sentiment score
print("Sentiment:", senti)

# Word cloud based on the article text
word = WordCloud(width=800, height=800, background_color="white").generate(a)

# Creating a bar chart for the sentiment scores
lab = ["Positive", "Negative", "Neutral"]
val = [senti["pos"], senti["neg"], senti["neu"]]
plt.bar(lab, val)
plt.title("Sentiment Scores")
plt.xlabel("Sentiment")
plt.ylabel("Score")
plt.show()

# Create a pie chart for the sentiment scores
plt.pie(val, labels=lab, autopct="%1.1f%%")
plt.title("Sentiment Scores")
plt.show()

# Displaying the word cloud
plt.imshow(word, interpolation='bilinear')
plt.axis("off")
plt.show()

#3
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
%matplotlib inline

# load the dataset
data = pd.read_csv('C:/Users/anoks/OneDrive/Desktop/AIML Intership/CC GENERAL.csv')
print(data.head())

# preprocess the data
data.drop('CUST_ID', axis=1, inplace=True)
data.fillna(method='ffill', inplace=True)

# standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# train the K-Means algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_

# visualize the clusters
plt.scatter(data_scaled[labels==0, 0], data_scaled[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(data_scaled[labels==1, 0], data_scaled[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(data_scaled[labels==2, 0], data_scaled[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker='*', color='black')
plt.title('Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()