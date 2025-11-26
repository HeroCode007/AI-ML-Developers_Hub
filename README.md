# AI/ML Internship Tasks

---

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
```
# TASK-2 House Price Prediction – Regression Model

A regression-based house price prediction project using size and location features from the housing dataset to predict house `SalePrice` with Linear Regression and Gradient Boosting models.

## Project Overview

This project builds supervised regression models to predict house prices using above-ground living area, number of bedrooms, and location-related attributes. It includes data preprocessing, model training, evaluation with MAE and RMSE metrics, and visualization of actual vs predicted prices.

## Dataset

- Training file: `train.csv` (with `SalePrice` target variable)
- Testing file: `test.csv`
- Features:
  - Numerical: `GrLivArea` (living area sqft), `BedroomAbvGr` (bedrooms)
  - Categorical: `Neighborhood`, `MSZoning`
- Target: `SalePrice`

## Methodology

1. Load and inspect dataset; handle missing values
2. Preprocess:
   - Normalize numerical features using `StandardScaler`
   - Encode categorical features using `OneHotEncoder` with `handle_unknown="ignore"`
3. Train-test split (80% train, 20% test) with fixed random seed for reproducibility
4. Build pipelines:
   - Linear Regression
   - Gradient Boosting Regressor
5. Train models with `.fit()`
6. Evaluate using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
7. Visualize actual vs predicted prices for Gradient Boosting model

## Models

## Models

- Linear Regression pipeline:
- Gradient Boosting pipeline:


## Evaluation

Evaluation metric example function:
Gradient Boosting outperforms Linear Regression with lower MAE and RMSE.

## Visualization

Example visualization code:

## Libraries Used

- pandas — Data handling
- numpy — Numerical computations
- matplotlib — Visualization
- scikit-learn — Preprocessing, model building, evaluation

## Key Findings

- Location and house size strongly influence price predictions.
- Gradient Boosting better captures nonlinear relationships than Linear Regression.
- Proper scaling and encoding of features improve model performance.

## Future Work

- Hyperparameter tuning of Gradient Boosting
- Experiment with XGBoost, Random Forest, or other models
- Add features like year built, lot area, and house condition
- Deploy the model with a web interface for live predictions

---

## How to Run

1. Clone the repository.  
2. Install dependencies via `pip install -r requirements.txt`.  
3. Place `train.csv` and `test.csv` in the project directory.  
4. Run the main notebook or script to train and evaluate models.  

---

# 🩺 TASK-3 Health Assistant – WEB App

A simple, safe, and friendly AI-powered Health Information Assistant built using **Streamlit**, **Hugging Face Router**, and **OpenAI-compatible API**. 

This assistant provides general health information, follows strict safety rules, and features a custom dark chat UI.

## 🚀 Features

- **✅ AI Health Assistant**  
  - Provides simple, safe, general health guidance.
  - Uses the `moonshotai/Kimi-K2-Instruct-0905` model via Hugging Face Router.
  - System prompt enforces strict health safety policies.

- **🔐 Safety Guardrails**  
  Prevents unsafe responses involving:
  - Diagnosis
  - Medication dosage
  - Emergency instructions
  - Sensitive or harmful topics

- **🎨 Custom Dark UI**  
  - Modern chat-style layout.
  - Distinct message bubble styles (User/Assistant).
  - Scrollable chat window.
  - "Clear Chat" button.

- **⚡ Real-Time Chat**  
  - Powered by `st.chat_input()`.
  - Smooth message updates with placeholders.

---

## 🧩 Tech Stack

| Component | Description |
| :--- | :--- |
| **Streamlit** | UI Framework |
| **HuggingFace Router** | API Gateway for LLM |
| **OpenAI Python SDK** | Client for chat completions |
| **Custom CSS** | Dark theme and chat styling |

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository


### 2️⃣ Create Virtual Environment (Optional)

### 3️⃣ Install Dependencies
## 🧠 How It Works

1.  **Query Safety Check** 🔍  
    Unsafe keywords trigger an automatic safe reply, bypassing the LLM to prevent harmful advice.

2.  **LLM Request** 💬  
    Safe messages are passed to the model with a system prompt:
    ```
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages,
        temperature=0.9,
        max_tokens=300,
    )
    ```

3.  **UI Chat Rendering** 🎨  
    Custom CSS renders messages as chat bubbles:
    - 🔵 **Blue:** User
    - 🔘 **Light Blue:** Assistant

---

## 📌 Future Improvements

- [ ] User authentication
- [ ] Chat export option
- [ ] Voice input integration (Whisper)
- [ ] More advanced health categories
- [ ] Emoji mood-based UI

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

MIT License © 2025





*This README is structured for clarity and ease of understanding for users and collaborators on GitHub.*


