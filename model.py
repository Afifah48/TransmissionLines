import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib


# 1. Load dataset

df = pd.read_csv("txline_dataset.csv")

# Inputs (X) and Outputs (y)
X = df[["Rp", "Lp", "Gp", "Cp", "freq", "ZL_real", "ZL_imag", "length"]].values
y = df.drop(columns=["Rp", "Lp", "Gp", "Cp", "freq", "ZL_real", "ZL_imag", "length"]).values


# 2. Train/validation split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Scale features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# 4. Define & train model

base_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    tree_method="hist"
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)


# 5. Evaluate
y_pred = model.predict(X_val)

output_names = [
    "Z0_real", "Z0_imag",
    "gamma_real", "gamma_imag",
    "alpha", "beta", "vp",
    "refl_real", "refl_imag", "vswr"
]

print("\nModel Performance:")
for i, name in enumerate(output_names):
    r2 = r2_score(y_val[:, i], y_pred[:, i])
    mse = mean_squared_error(y_val[:, i], y_pred[:, i])
    print(f"{name:10s} | R²: {r2*100:6.2f}% | MSE: {mse:.4e}")


# 6. Scatter plots (true vs predicted)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
plot_vars = [0, 1, 4, 9]  # Z0_real, Z0_imag, alpha, vswr

for i, idx in enumerate(plot_vars):
    axes[i].scatter(y_val[:, idx], y_pred[:, idx], alpha=0.6, color="blue")
    axes[i].plot(
        [y_val[:, idx].min(), y_val[:, idx].max()],
        [y_val[:, idx].min(), y_val[:, idx].max()],
        'r--'
    )
    axes[i].set_xlabel(f"True {output_names[idx]}")
    axes[i].set_ylabel(f"Predicted {output_names[idx]}")
    axes[i].set_title(f"True vs Predicted: {output_names[idx]}")

plt.tight_layout()
plt.show()

# 7. Save model & scaler

joblib.dump(model, "txline_model.pkl")
joblib.dump(scaler, "txline_scaler.pkl")
print("\n✅ Model and scaler saved as 'txline_model.pkl' and 'txline_scaler.pkl'")
