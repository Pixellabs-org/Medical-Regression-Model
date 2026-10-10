# Regression Analysis Project: Predicting Medical Insurance Charges

## 1. Introduction

This project is about building a machine learning model to predict how much a person may pay for medical insurance. The target variable is the cost of insurance, which is a numerical value. Because the outcome is continuous, we use a regression model instead of a classification model.

The dataset used is the insurance dataset. It contains information such as age, sex, body mass index (BMI), number of children, smoking status, region, and insurance charges.

The main goal is to find which features influence insurance charges and to build a model that can predict future charges with reasonable accuracy.

---

## 2. Why Regression?

Regression is used when the dependent variable is numeric. In this case, we want to predict charges, which are measured in dollars. A commonly used technique is multiple linear regression.

The general form of a linear regression model is:

$$
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon
$$

Where:
- $Y$ is the target variable (insurance charges)
- $X_1, X_2, ..., X_n$ are the predictor variables
- $\beta_0$ is the intercept
- $\beta_1, \beta_2, ...$ are coefficients
- $\epsilon$ is the error term

---

## 3. Data Collection

The data is loaded from an online source using pandas. Pandas is a Python library used to work with tables of data.

The dataset has rows and columns. Each row represents a person and each column represents a feature.

Example columns include:
- age
- sex
- bmi
- children
- smoker
- region
- charges

After loading the data, we check its shape to know how many rows and columns are present.

### Example:

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(df.shape)
```

This tells us the size of the dataset.

---

## 4. Exploratory Data Analysis (EDA)

Before building a model, we need to understand the data.

EDA helps us:
- identify missing values
- detect duplicate records
- understand variable distributions
- find patterns between features and the target

### 4.1 Checking for Missing Values

Missing values can affect model performance. We check if any values are absent.

```python
df.isna().sum()
```

If the output shows zero for all columns, then the dataset is complete.

### 4.2 Checking for Duplicate Rows

Duplicate entries can skew the model. We remove or investigate them if found.

```python
df.duplicated().sum()
```

### 4.3 Viewing Summary Statistics

Summary statistics help us understand the central tendency and spread of the data.

```python
df["charges"].describe()
```

This gives information such as:
- mean
- standard deviation
- minimum
- maximum
- quartiles

The charge distribution often shows that the data is positively skewed, meaning a few people have very high insurance charges.

---

## 5. Visualizing the Data

Visual representations are useful in data science because they make patterns easier to understand.

### 5.1 Plotting Charge Distribution

We use a histogram and boxplot to examine the distribution of charges.

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
sns.histplot(df["charges"], kde=True)
plt.title("Medical Cost Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(df["charges"])
plt.title("Medical Cost Spread")
plt.show()
```

Interpretation:
- A right-skewed distribution suggests that insurance costs are not evenly spread.
- A few people pay much more than the majority.

### 5.2 Boxplots by Category

We compare charges across categories such as smoking status and region.

```python
for col in ["sex", "smoker", "region"]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], y=df["charges"])
    plt.title(f"{col} vs Charges")
    plt.show()
```

This shows whether certain groups tend to have higher charges than others.

---

## 6. Encoding Categorical Variables

Machine learning models work with numbers, not text categories. Therefore, categorical variables such as sex, smoker, and region must be converted into numerical form.

This is done using one-hot encoding or dummy variables.

### Example

```python
def dummies(x, frame):
    temp = pd.get_dummies(frame[x], drop_first=True).astype(int)
    frame = pd.concat([frame, temp], axis=1)
    frame.drop([x], axis=1, inplace=True)
    return frame
```

This function does the following:
- creates dummy variables for each category
- drops the first category to avoid multicollinearity
- converts the dummy variables to integers

For example:
- smoker: yes/no becomes a binary feature
- region: southeast, southwest, northwest becomes separate numeric columns

These encoded features are then used in the regression model.

---

## 7. Train-Test Split

To evaluate the model fairly, we split the data into two parts:
- training set: used to learn the patterns
- test set: used to measure how well the model performs on unseen data

```python
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(df, train_size=0.75, test_size=0.25, random_state=100)
```

A common split is 75% training and 25% testing.

---

## 8. Feature Scaling

Some numeric variables may have different ranges. For example:
- age may range from 18 to 64
- bmi may range from 15 to 53
- charges may be larger in value

Scaling makes these variables comparable and helps in model stability.

```python
from sklearn.preprocessing import MinMaxScaler

numerical_list = [x for x in df.columns if df[x].dtype in ("int64", "float64") and x != "charges"]
scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])
df_test[numerical_list] = scaler.transform(df_test[numerical_list])
```

This normalizes the features to a range between 0 and 1.

---

## 9. Defining Features and Target

The target variable is the variable we want to predict.

```python
y_train = df_train.pop("charges")
X_train = df_train

y_test = df_test.pop("charges")
X_test = df_test
```

So:
- `y_train` and `y_test` are the actual insurance charges
- `X_train` and `X_test` are the feature sets used to predict charges

---

## 10. Recursive Feature Elimination (RFE)

Not all features are equally useful. Some may add noise rather than improve the model. To select the most important predictors, we use Recursive Feature Elimination (RFE).

RFE works by repeatedly removing the least useful feature until the desired number of features remains.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

rfe = RFE(estimator=LinearRegression(), n_features_to_select=8)
rfe = rfe.fit(X_train, y_train)
```

This selects the best subset of variables for prediction.

---

## 11. Building the Regression Model

After selecting the important features, we build the linear regression model using the `statsmodels` library.

```python
import statsmodels.api as sm

X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_new = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train, X_train_new).fit()
print(lm.summary())
```

This model estimates the coefficients for each feature and shows whether each variable is statistically significant.

### Understanding the model summary

The summary provides important information such as:
- R-squared
- Adjusted R-squared
- p-values
- coefficients
- F-statistic

### Why this matters
- R-squared tells us how much of the variation in charges is explained by the model.
- p-values tell us whether a feature is statistically significant.
- If a p-value is greater than 0.05, the feature may not be useful in the model.

---

## 12. Checking Significance and Multicollinearity

### 12.1 p-values

We remove variables that do not significantly affect the target.

Example:

```python
X_train_new = X_train_new.drop(["male"], axis=1, errors="ignore")
```

This step removes a feature that is not contributing meaningfully.

### 12.2 VIF (Variance Inflation Factor)

VIF is used to detect multicollinearity, which happens when independent variables are highly correlated with each other.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["Features"] = X_train_new.columns
vif["VIF"] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
```

A high VIF indicates redundancy among features. We try to minimize this because excessive correlation can weaken the model.

---

## 13. Residual Analysis

Residuals represent the difference between the actual and predicted values.

```python
y_train_pred = lm.predict(X_train_new)

plt.figure(figsize=(8, 5))
sns.histplot(y_train - y_train_pred, bins=20, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()
```

If residuals are roughly centered around zero and follow a normal-like pattern, the model is acceptable.

This helps us evaluate whether the assumptions of linear regression are reasonably satisfied.

---

## 14. Predicting on the Test Set

Once the model is built and validated, we apply it to unseen test data.

```python
X_train_new = X_train_new.drop("const", axis=1)
X_test_new = X_test[X_train_new.columns]
X_test_new = sm.add_constant(X_test_new)

y_pred = lm.predict(X_test_new)
```

We then compare the predicted values to the actual values.

---

## 15. Model Evaluation

To measure the model performance, we use the R-squared score.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)
```

### Interpretation
- If the R-squared is close to 1, the model explains a large portion of the variance in the target variable.
- For this project, the value is relatively strong, showing that the model captures important patterns in the data.

We also plot actual values against predicted values.

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Charges")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

A good model should show points clustered around a straight line.

---

## 16. Final Interpretation

The final model suggests that insurance charges are influenced mainly by:
- smoking status
- age
- BMI
- some region-related factors

This means that people who smoke, are older, and have higher BMI tend to have higher insurance costs.

The model is useful because it can estimate insurance charges based on a person's profile.

---

## 17. Conclusion

This project demonstrates the step-by-step process of building a regression model:

1. load the dataset
2. explore the data
3. clean and encode features
4. split the data
5. select important variables
6. build the model
7. test assumptions
8. evaluate predictions

This is a standard workflow used in real-world data science projects. It helps analysts understand patterns in data and build prediction models that can support decision-making.

---

## 18. Summary

This regression model is a practical example of using machine learning to predict a continuous target variable. It follows a clear and structured process similar to a typical college data science assignment. The model helps explain how different factors affect medical insurance charges and can be used to estimate expected costs.
