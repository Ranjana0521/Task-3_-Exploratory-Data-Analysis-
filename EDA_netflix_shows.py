import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

netflix = pd.read_csv("netflix_titles.csv")
netflix.head()

plt.figure(figsize=(7,5))
plt.hist(netflix['release_year'], bins=20)
plt.title("Distribution of Release Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

#2
sns.countplot(x=netflix['type'])
plt.title("Movies vs TV Shows")
plt.show()

sns.countplot(y=netflix['rating'],
              order=netflix['rating'].value_counts().index)
plt.title("Content Rating Distribution")
plt.show()

#3
movies = netflix[netflix['type'] == "Movie"].copy()
movies['duration'] = movies['duration'].str.replace(" min","").astype(float)

sns.boxplot(x=movies['duration'])
plt.title("Movie Duration Outliers")
plt.show()

#4
netflix = pd.read_csv("netflix_titles.csv")
movies = netflix[netflix['type'] == 'Movie'].copy()
movies['duration'] = movies['duration'].str.replace(' min', '', regex=False)
movies['duration'] = movies['duration'].astype(float)
print(movies[['duration','release_year']].dtypes)

netflix_corr = movies[['duration','release_year']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(netflix_corr, annot=True, cmap='coolwarm')
plt.title("Netflix Correlation Heatmap")
plt.show()
