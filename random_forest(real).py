import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


real_df = pd.read_csv('traffic_weather_full2020.csv')

#Check the quality of the data imported 
missing_percentage = real_df.isnull().sum() / len(real_df) * 100
duplicates = real_df.duplicated().sum()

print(missing_percentage)
print(f"Number of exact duplicate rows: {duplicates}")

real_df["5 Minutes"] = pd.to_datetime(real_df["5 Minutes"], errors="coerce")
real_df["Hour_Bin"] = real_df["5 Minutes"].dt.floor("H")



def clean_numeric(col):
    return (
        col.astype(str)
           .str.extract(r"([-+]?\d*\.?\d+)")     # extract numeric portion
           .astype(float)
    )

cols_to_clean = ["(mph)", "Wind Speed", "Wind Gust", "Flow", "Humidity", "Pressure", "Precip."]

for col in cols_to_clean:
    real_df[col] = clean_numeric(real_df[col])
hourly = real_df.groupby("Hour_Bin").agg({
    "Flow": "mean",
    "(mph)": "mean",
    "Humidity": "mean",
    "Pressure": "mean",
    "Wind Speed" : "mean",
    "Wind Gust" : "mean",
    "Pressure" : "mean",
    "Precip." : "mean",
})


target_col = "Flow"
X = real_df.drop(columns=[target_col])
y = real_df[target_col]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

#Set Train/Test/Validation split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

#build the regression pipeline
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

#train the model
model.fit(X_train, y_train)

#Evaluate on test and validation splits
valid_preds = model.predict(X_valid)
test_preds = model.predict(X_test)

print("Validation MAE:", mean_absolute_error(y_valid, valid_preds))
print("Validation RMSE:", np.sqrt(mean_squared_error(y_valid, valid_preds)))

print("Test MAE:", mean_absolute_error(y_test, test_preds))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_preds)))

#Importance Extraction
importances = model.named_steps["rf"].feature_importances_
feature_names = model.named_steps["preprocess"].get_feature_names_out()

feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp.sort_values(by="Importance", ascending=False, inplace=True)

print(feat_imp.head(20))

#graphs 

# Compute predictions on test set
test_preds = model.predict(X_test)
residuals = y_test - test_preds

#HISTOGRAM OF RESIDUALS
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, edgecolor='black')
plt.title("Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

#RESIDUALS VS PREDICTED
plt.figure(figsize=(8,5))
plt.scatter(test_preds, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(alpha=0.3)
plt.show()

# Extract feature importances
rf = model.named_steps["rf"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()
importances = rf.feature_importances_

# Build DataFrame
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

top_n = 20
top_features = feat_imp.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis() 
plt.title("Top Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

