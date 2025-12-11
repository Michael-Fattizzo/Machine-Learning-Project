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

#Import the data from the csv file
futuristic_df = pd.read_csv('futuristic_city_traffic.csv')
real_df = pd.read_csv('traffic_weather_full2020.csv')

#Check the quality of the data imported 
missing_percentage = futuristic_df.isnull().sum() / len(futuristic_df) * 100
duplicates = futuristic_df.duplicated().sum()

print(missing_percentage)
print(f"Number of exact duplicate rows: {duplicates}")

missing_percentage = real_df.isnull().sum() / len(real_df) * 100
duplicates = real_df.duplicated().sum()

print(missing_percentage)
print(f"Number of exact duplicate rows: {duplicates}")


#bin data by 2 hour intervals
bins = list(range(0, 25, 2))  # 0,2,4,...,24
labels = [f"{i}-{i+1}" for i in range(0, 24, 2)]

futuristic_df["Hour_Bin"] = pd.cut(
    futuristic_df["Hour Of Day"],
    bins=bins,
    labels=labels,
    right=False
)

futuristic_df.drop(columns=["Hour Of Day"], inplace=True)

#set target
target_col = "Traffic Density"

X = futuristic_df.drop(columns=[target_col])
y = futuristic_df[target_col]

#determin colums types
categorical_cols = [
    "City",
    "Vehicle Type",
    "Weather",
    "Economic Condition",
    "Day Of Week",
    "Random Event Occurred",
    "Hour_Bin"
]

numeric_cols = [
    "Speed",
    "Energy Consumption",
    "Is Peak"
]


#Proproces for cat data
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
        n_estimators=300,
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
