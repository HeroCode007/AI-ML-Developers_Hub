# Task 1: Iris Dataset Exploration

## Objective
The objective of this task is to **learn how to load, inspect, and visualize a dataset** using Python.  
We explore the basic **data structure**, **summary statistics**, and **visualizations** to understand patterns, distributions, and relationships between variables in the dataset.

---

## Dataset
**Iris Dataset** (from UCI Machine Learning Repository) contains 150 samples of iris flowers with the following features:

- `sepal_length`  
- `sepal_width`  
- `petal_length`  
- `petal_width`  
- `species` (Setosa, Versicolor, Virginica)

The dataset is loaded using **pandas**.

---

## Steps Completed

### 1. Load the dataset
- Loaded using `pandas.read_csv()`  
- Assigned column names  
- Stored it in a DataFrame

### 2. Inspect the dataset
- `.head()` → Displayed the first 5 rows  
- `.shape` → Checked number of rows and columns  
- `.columns` → List of column names  
- `.info()` → Data types and null values  
- `.describe()` → Summary statistics (mean, std, min, max, quartiles)

### 3. Visualize the dataset
- **Scatter Plot** → Relationships between features (`sepal_length` vs `sepal_width`)  
- **Histograms** → Distribution of each feature  
- **Box Plots** → Detect outliers or unusual values  

---

## Libraries Used
- **pandas** → Data loading and manipulation  
- **seaborn** → Visualizations with style  
- **matplotlib** → Basic plotting

---

## Key Findings
- Species can be separated based on petal length and width.  
- Sepal length and width vary among species but overlap for some classes.  
- Some features have slight outliers visible in box plots.  
- Histograms show distributions are mostly normal with minor skew in some features.

---

## Code Snippet

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    names=col_names
)

print(iris.head())
print(iris.info())
print(iris.describe())

sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')
plt.show()
iris.hist(figsize=(10, 8))
plt.show()
sns.boxplot(data=iris, orient='h')
plt.show()
