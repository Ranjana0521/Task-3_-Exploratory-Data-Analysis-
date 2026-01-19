from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = iris_data.target

iris.head()
#1_numerical_features
iris.hist(figsize=(10,7), bins=20)
plt.tight_layout()
plt.show()

#2_categorical_features
sns.countplot(x=iris['species'])
plt.title("Species Distribution")
plt.show()

#3_Outlier_distribution
for col in iris.columns[:-1]:
    sns.boxplot(x=iris[col])
    plt.title(col)
    plt.show()

#4_correlation_heatmap
sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')
plt.title("Iris Correlation Heatmap")
plt.show()
