import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------------------------------------------
# 2. EDA
# -------------------------------------------------------------------
print("\nDuplicate rows:", df.duplicated().sum())
print("\nMissing values:\n", df.isna().sum())

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title("Medical Cost Distribution")
sns.histplot(df["charges"], kde=True)

plt.subplot(1, 2, 2)
sns.boxplot(df["charges"])
plt.title("Medical Cost Spread")
plt.show()

print(df["charges"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 1]))

categorical_list = [x for x in df.columns if df[x].dtype == "object"]
print("\nCategorical columns:", categorical_list)

for col in categorical_list:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], y=df["charges"])
    plt.title(f"{col} vs Charges")
    plt.xticks(rotation=20)
    plt.show()

# -------------------------------------------------------------------
# 3. Dummy variable encoding
# -------------------------------------------------------------------
def dummies(x, frame):
    temp = pd.get_dummies(frame[x], drop_first=True).astype(int)
    frame = pd.concat([frame, temp], axis=1)
    frame.drop([x], axis=1, inplace=True)
    return frame

for col in ["sex", "smoker", "region"]:
    df = dummies(col, df)

print("\nAfter encoding:")
print(df.head())
print("Shape:", df.shape)

# -------------------------------------------------------------------
# 4. Train/test split
# -------------------------------------------------------------------
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size=0.75, test_size=0.25, random_state=100)

# -------------------------------------------------------------------
# 5. Scale numerical variables
# -------------------------------------------------------------------
numerical_list = [x for x in df.columns if df[x].dtype in ("int64", "float64") and x != "charges"]
scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])
df_test[numerical_list] = scaler.transform(df_test[numerical_list])

# -------------------------------------------------------------------
# 6. Split into X and y
# -------------------------------------------------------------------
y_train = df_train.pop("charges")
X_train = df_train

y_test = df_test.pop("charges")
X_test = df_test

# -------------------------------------------------------------------
# 7. RFE and model building
# -------------------------------------------------------------------
# RFE
rfe = RFE(estimator=LinearRegression(), n_features_to_select=8)
rfe = rfe.fit(X_train, y_train)
print("\nSelected features:")
print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))
print("Chosen columns:", X_train.columns[rfe.support_].tolist())

X_train_rfe = X_train[X_train.columns[rfe.support_]]


def build_model(X, y):
    X = sm.add_constant(X)
    lm = sm.OLS(y, X).fit()
    print(lm.summary())
    return X


def check_vif(X):
    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["VIF"] = round(vif["VIF"], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif


X_train_new = build_model(X_train_rfe, y_train)
print("\nVIF before dropping:\n", check_vif(X_train_new))

# Drop columns with weak significance or high VIF (to mimic the original workflow)
X_train_new = X_train_new.drop(["male"], axis=1, errors="ignore")
X_train_new = build_model(X_train_new, y_train)
print("\nVIF after dropping male:\n", check_vif(X_train_new))

# Final model
lm = sm.OLS(y_train, X_train_new).fit()
y_train_pred = lm.predict(X_train_new)

# Residual distribution
plt.figure(figsize=(8, 5))
sns.histplot(y_train - y_train_pred, bins=20, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()

# -------------------------------------------------------------------
# 8. Predict on test set
# -------------------------------------------------------------------
X_train_new = X_train_new.drop("const", axis=1)
X_test_new = X_test[X_train_new.columns]
X_test_new = sm.add_constant(X_test_new)
y_pred = lm.predict(X_test_new)

r2 = r2_score(y_test, y_pred)
print("\nR2 Score:", round(r2, 4))

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Charges")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

print(lm.summary())
print("\nInference:")
print("This regression model explains a meaningful portion of the variation in medical insurance costs.")
print("The strongest predictors include smoking status, age, BMI, and region-related variables.")
