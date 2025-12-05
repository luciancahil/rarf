
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from RaRFRegressor_shared_overlap_visual import RaRFRegressor
from utils import smi2morgan
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


file = open("first_pred_vs_real.txt")

lines = file.readlines()

X = np.array([float(n) for n in lines[0].split(",")])


y = np.array([float(n) for n in lines[1].split(",")])

r2 = r2 = r2_score(X, y)

m, b = np.polyfit(X, y, 1)

perfect = (np.array(range(1000)) * (np.max(y) - np.min(y)) / 1000) + np.min(y)

print(f"R^2 value: {r2:.3f}")
print(f"Line of best fit: y = {m:.2f}x + {b:.2f}, where x is pred and y is actual")

plt.title("Predictions of RARF vs Real")

plt.xlabel("Predicitons")

plt.ylabel("Actual")


plt.scatter(perfect, perfect, label="Perfect Prediction", c='red',s=0.2)
plt.scatter(X, y, label="Model Predictions")

plt.legend()
plt.savefig("Results.png")