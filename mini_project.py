import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(42)

#getting data

n = 500

#true relationship: price = 100*sqft + 20000*beds - 500*age + 50000 + noise

sqft = np.random.uniform(500,3000,n)
beds = np.random.randint(1,6,n)
age = np.random.uniform(0,50,n)

price = 100*sqft + 20000*beds - 500*age + 50000 + np.random.normal(0,30000,n)

X = pd.DataFrame({"sqft": sqft, "beds": beds, "age": age})
y = price

print(f"\nDataset : {n} house, No of features {X.shape[1]}")

#now we split data

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(f"RMSE: Rs.{rmse:,.0f}")
print(f"R²:   {r2:.4f} ({r2*100:.1f}% variance explained)")

true_coeff = [100,20000,-500]
mdl = pipe.named_steps["model"]
print("=== COEFFICIENT COMPARISON ===")
print("   Feature    Learned     True  Close?")
print("-" * 42)

for name, learned, true in zip(["sqft","beds","age"], mdl.coef_,true_coeff):
    ok = "✓" if abs(learned - true) / abs(true) < 0.15 else "~"
    print(f"{name:>10} {learned:>+10,.1f} {true:>+8,d} {ok:>8}")
lbl = "intercept"
print(f"{lbl:>10} {mdl.intercept_:>+10,.0f}   +50,000")

#the data is coming so weird bcz it is scaled and to get in their og units i'll ve to go outside of the pipepline
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, s=20)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Price (Rs.)")
ax.set_ylabel("Predicted Price (Rs.)")
ax.set_title(f"House Price Predictor — R²={r2:.3f}", fontweight="bold")
plt.tight_layout(); plt.show()

print("\n✓ End-to-end ML pipeline complete!")
print("✓ Load → Explore → Preprocess → Split → Fit → Predict → Evaluate")

import joblib

# Save the entire pipeline (scaler + model together)
joblib.dump(pipe, "house_price_pipeline.pkl")

print("Model saved as house_price_pipeline.pkl")