# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)


# Import the combined data
data = pd.read_csv(r"combined.csv")

# Convert date column to a date format
data["date"] = pd.to_datetime(data["date"])

# Sort date in ascending order
data = data.sort_values("date").reset_index(drop=True)

# ML 1: Neural Network

# Create lag variables for previous gage height, divided into 4 time intervals (15 min)
data["gage_lag1"] = data["gage height (ft)"].shift(1)
data["gage_lag2"] = data["gage height (ft)"].shift(2)
data["gage_lag3"] = data["gage height (ft)"].shift(3)
data["gage_lag4"] = data["gage height (ft)"].shift(4)

# Create lag variables for previous discharge
data["discharge_lag1"] = data["discharge (ft^3/s)"].shift(1)
data["discharge_lag2"] = data["discharge (ft^3/s)"].shift(2)
data["discharge_lag3"] = data["discharge (ft^3/s)"].shift(3)
data["discharge_lag4"] = data["discharge (ft^3/s)"].shift(4)

# create lag variables for previous precipitation
data["precip_lag1"] = data["precipitation (in)"].shift(1)
data["precip_lag2"] = data["precipitation (in)"].shift(2)
data["precip_lag3"] = data["precipitation (in)"].shift(3)
data["precip_lag4"] = data["precipitation (in)"].shift(4)

# Create rainfall accumulation features to evaluate rainfall across different time intervals
# 1-hour accumulated rainfall
data["rain_1hr"] = (
    data["precipitation (in)"]
    .rolling(window=4)
    .sum()
)

# 3-hour accumulated rainfall
data["rain_3hr"] = (
    data["precipitation (in)"]
    .rolling(window=12)
    .sum()
)

# 6-hour accumulated rainfall
data["rain_6hr"] = (
    data["precipitation (in)"]
    .rolling(window=24)
    .sum()
)

# 12-hour accumulated rainfall
data["rain_12hr"] = (
    data["precipitation (in)"]
    .rolling(window=48)
    .sum()
)

# 24-hour accumulated rainfall
data["rain_24hr"] = (
    data["precipitation (in)"]
    .rolling(window=96)
    .sum()
)

# Create target
data["target"] = (
    data["gage height (ft)"].shift(-4)
)

# Set features
features = [
    # Gage height lags
    "gage_lag1",
    "gage_lag2",
    "gage_lag3",
    "gage_lag4",

    # Discharge lags
    "discharge_lag1",
    "discharge_lag2",
    "discharge_lag3",
    "discharge_lag4",

    # Precipitation lags
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "precip_lag4",

    # Rainfall accumulation
    "rain_1hr",
    "rain_3hr",
    "rain_6hr",
    "rain_12hr",
    "rain_24hr"
]

# Remove missing values
data = data.dropna().reset_index(drop=True)

# Create X and y
X = data[features]

y = data["target"]

print("Number of observations:", len(data))
print("Number of input features:", len(features))

# Test/train split
split = int(len(data) * 0.80) #first 80% of readings used for training

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("Training period:")
print(data["date"].iloc[0])
print("to")
print(data["date"].iloc[split - 1])

print("Testing period:")
print(data["date"].iloc[split])
print("to")
print(data["date"].iloc[-1])

# Neural Network
model_nn = Pipeline([

    # Standardize the input variables
    ("scaler", StandardScaler()),

    # Neural Network
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(20,),
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    ))
])

# Train Neural Network
model_nn.fit(X_train, y_train)

# Make predictions
nn_prediction = model_nn.predict(X_test)

# ML 2: Random Forest

model_rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train random forest
model_rf.fit(X_train, y_train)

# Make predictions
rf_prediction = model_rf.predict(X_test)

# Create a baseline prediction
baseline_prediction = data["gage height (ft)"].iloc[split:]

# Evalute functions
def evaluate_model(name, actual, prediction):

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    r2 = r2_score(
        actual,
        prediction
    )

    print("------------------------------")
    print(name)
    print("------------------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

    return mae, rmse, r2

# Evalute the two models
nn_mae, nn_rmse, nn_r2 = evaluate_model(
    "Neural Network",
    y_test,
    nn_prediction
)

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    "Random Forest",
    y_test,
    rf_prediction
)

baseline_mae, baseline_rmse, baseline_r2 = evaluate_model(
    "Persistence Baseline",
    y_test,
    baseline_prediction
)

# Compare models
results = pd.DataFrame({
    "Model": [
        "Persistence Baseline",
        "Neural Network",
        "Random Forest"
    ],

    "MAE": [
        baseline_mae,
        nn_mae,
        rf_mae
    ],

    "RMSE": [
        baseline_rmse,
        nn_rmse,
        rf_rmse
    ],

    "R2": [
        baseline_r2,
        nn_r2,
        rf_r2
    ]
})

print("------------------------------")
print("MODEL COMPARISON")
print("------------------------------")

print(results)

# Comparing models visually
plt.figure(figsize=(12, 5))

plt.plot(
    data["date"].iloc[split:],
    y_test.values,
    label="Actual",
    linewidth=1
)

plt.plot(
    data["date"].iloc[split:],
    nn_prediction,
    label="Neural Network",
    linewidth=1
)

plt.plot(
    data["date"].iloc[split:],
    rf_prediction,
    label="Random Forest",
    linewidth=1
)

plt.xlabel("Date")
plt.ylabel("Gage Height (ft)")

plt.title(
    "Actual vs Predicted Gage Height: "
    "Neural Network vs Random Forest"
)

plt.legend()

plt.tight_layout()

plt.show()

# Calculate the threshold for high water level to identify floods anomalies
gage_train = data["gage height (ft)"].iloc[:split]

# Set a threshold - use 95th percentile as a threshold
high_water_threshold = gage_train.quantile(0.95) #<- no test value was above the 0.99% quantile, so I chose the 95% since there is no threshold published

print("------------------------------")
print(" Statistical high water level threshold")
print("------------------------------")

print("95th percentile threshold:",
      high_water_threshold, "ft")

# Identify high water ocuurances
actual_high_water = (
    y_test.values >= high_water_threshold
)

# Neural network predictions
nn_high_water_prediction = (
    nn_prediction >= high_water_threshold
)

# Random forest predictions
rf_high_water_prediction = (
    rf_prediction >= high_water_threshold
)

# Count high water occurances
print("Actual high-water observations:",
      actual_high_water.sum())

print("NN predicted high-water observations:",
      nn_high_water_prediction.sum())

print("RF predicted high-water observations:",
      rf_high_water_prediction.sum())

# Confusion matrix
cm_nn = confusion_matrix(
    actual_high_water,
    nn_high_water_prediction
)

print("------------------------------")
print("NEURAL NETWORK")
print("------------------------------")

print("Confusion Matrix:")
print(cm_nn)

print("Classification Report:")

print(
    classification_report(
        actual_high_water,
        nn_high_water_prediction,
        target_names=[
            "Normal",
            "High Water"
        ],
        zero_division=0
    )
)

cm_rf = confusion_matrix(
    actual_high_water,
    rf_high_water_prediction
)

print("------------------------------")
print("RANDOM FOREST")
print("------------------------------")

print("Confusion Matrix:")
print(cm_rf)

print("Classification Report:")

print(
    classification_report(
        actual_high_water,
        rf_high_water_prediction,
        target_names=[
            "Normal",
            "High Water"
        ],
        zero_division=0
    )
)

# Visulise threshold
plt.figure(figsize=(12, 5))

plt.plot(
    data["date"].iloc[split:],
    y_test.values,
    label="Actual",
    color="blue",
    linewidth=1
)

plt.plot(
    data["date"].iloc[split:],
    nn_prediction,
    label="Neural Network",
    color="orange",
    linewidth=1
)

plt.plot(
    data["date"].iloc[split:],
    rf_prediction,
    label="Random Forest",
    color="green",
    linewidth=1
)

plt.axhline(
    y=high_water_threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"95th Percentile Threshold "
          f"({high_water_threshold:.2f} ft)"
)

plt.xlabel("Date")
plt.ylabel("Gage Height (ft)")

plt.title(
    "Gage Height Predictions and "
    "Statistical High-Water Threshold"
)

plt.legend()
plt.tight_layout()
plt.show()
