Task 1: Iris Dataset Exploration
Objective

The objective of this task is to learn how to load, inspect, and visualize a dataset using Python.
We explore the basic data structure, summary statistics, and visualizations to understand patterns, distributions, and relationships between variables in the dataset.

Dataset

Iris Dataset (from UCI Machine Learning Repository) contains 150 samples of iris flowers with the following features:

sepal_length

sepal_width

petal_length

petal_width

species (Setosa, Versicolor, Virginica)

The dataset is loaded using pandas.

Steps Completed
1. Load the dataset

Loaded using pandas.read_csv()

Assigned column names

Stored it in a DataFrame

2. Inspect the dataset

.head() → Displayed the first 5 rows

.shape → Checked number of rows and columns

.columns → List of column names

.info() → Data types and null values

.describe() → Summary statistics (mean, std, min, max, quartiles)

3. Visualize the dataset

Scatter Plot → Relationships between features (sepal_length vs sepal_width)

Histograms → Distribution of each feature

Box Plots → Detect outliers or unusual values

Libraries Used

pandas → Data loading and manipulation

seaborn → Visualizations with style

matplotlib → Basic plotting

Key Findings

Species can be separated based on petal length and width.

Sepal length and width vary among species but overlap for some classes.

Some features have slight outliers visible in box plots.

Histograms show distributions are mostly normal with minor skew in some features.

Code Snippet
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

Task 2: House Price Prediction
Objective

The goal of this task is to build a regression model that predicts house prices based on features such as:

Square footage (GrLivArea)

Number of bedrooms (BedroomAbvGr)

Number of bathrooms

Location (Neighborhood, MSZoning, encoded numerically)

The task involves preprocessing the dataset, training regression models, evaluating them using MAE and RMSE, and visualizing actual vs predicted prices.

Dataset

The dataset contains housing information with features related to size, structure, and location.

Training dataset: train.csv

Testing dataset: test.csv

Key Features Used:

GrLivArea → Above ground living area (in square feet)

BedroomAbvGr → Number of bedrooms above ground

Neighborhood → Neighborhood of the house (categorical)

MSZoning → General zoning classification (categorical)

Target Variable:

SalePrice → Price of the house

Steps Completed
1. Load and Inspect Dataset

Loaded training and testing data using pandas

Displayed first few rows with .head()

Selected relevant features and dropped rows with missing values

2. Preprocessing

Numerical Features → StandardScaler to normalize values

Categorical Features → OneHotEncoder for encoding

num_features = ["GrLivArea", "BedroomAbvGr"]
cat_features = ["Neighborhood", "MSZoning"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

3. Train-Test Split

Split the data into training and test sets: 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

4. Model Building

Built Machine Learning Pipelines with preprocessing included:

Linear Regression

model_lr = Pipeline(steps=[("preprocess", preprocessor), ("model", LinearRegression())])


Gradient Boosting Regressor

model_gb = Pipeline(steps=[("preprocess", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])

5. Model Training

Trained both models using .fit()

model_lr.fit(X_train, y_train)
model_gb.fit(X_train, y_train)

6. Prediction and Evaluation

Predicted house prices on the test set

Evaluated models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)

def evaluate(pred, name):
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")


Key Results:

Gradient Boosting performed better than Linear Regression in capturing non-linear relationships

7. Visualization

Plotted Actual vs Predicted Prices for Gradient Boosting

plt.scatter(y_test, pred_gb)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Gradient Boosting)")
plt.show()

Libraries Used

pandas → Data handling

numpy → Numerical computations

matplotlib → Visualization

scikit-learn → Machine learning, preprocessing, metrics

Key Findings

Location and size significantly impact house price prediction

Gradient Boosting captures non-linearities better than Linear Regression

Preprocessing (scaling and encoding) improves model performance

Future Work

Hyperparameter tuning of Gradient Boosting for better accuracy

Try other regression models like XGBoost or Random Forest

Include additional features like year built, lot area, and condition

Deploy model with a web interface for live predictions
