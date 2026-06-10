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

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns = diabetes.feature_names ) #10 features 
y = diabetes.target  #this is the target column which we are trying to predict      
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

#or I could have done scaler.fit() and scaler.transform() separately, but its just a shortcut to do it in one line
print("=== BEFORE SCALING ===")
print(X.describe().loc[["mean","std"]].round(4))
print("\n=== AFTER SCALING ===")
print(X_scaled.describe().loc[["mean","std"]].round(4))
print("\nAll features now have mean≈0 and std≈1!")

#20% data pr test krege and 80% data pr train krege
X_train , X_test , y_train , y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)


#my first ML model

#step 1 create
model = LinearRegression()

#step 2 train
model.fit(X_train,y_train)
print(f"\nModel trained! Learned {len(model.coef_)} coefficients")

#step 3 predict
y_pred = model.predict(X_test)
print(f"{len(y_pred)} predictions generated")

# Show predictions vs actual
print("  Actual  Predicted    Error")
print("-" * 30)
for a, p in zip(y_test[:8], y_pred[:8]):
    print(f"{a:>8.0f} {p:>10.1f} {a-p:>+8.1f}")

#models coefficient and intercepts:-

print("=== LEARNED PARAMETERS ===")
print("(These are the weights gradient descent found)\n")
for name, coef in zip(X.columns, model.coef_):
    direction = "↑ increases" if coef > 0 else "↓ decreases"
    print(f"  {name:>5}: {coef:+8.2f}  ({direction} disease progression)")
label = "intercept"
print(f"  {label:>5}: {model.intercept_:+8.2f}")

#Lets find RMSE and R^2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
r2_train = model.score(X_train, y_train)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 (Test): {r2:.2f}")
print(f"R^2 (Train): {r2_train:.2f}")   
print(f"Gap is {(r2_train-r2):.4f} which is less than 0.1 nice training no overfitting")
gap = abs(r2_train-r2) #The abs() function calculates the absolute value.
        #It ensures that the gap is always a positive number (or zero), regardless of which score is higher.
status = "ok" if gap < 0.1 else "OVERFITTING!!"
print(f"Gap {gap} -- {status}")

'''R² of 0.45 means our model explains 45% of the variance in disease progression. 
   RMSE of 53.9 means predictions are off by ~54 units on average.
   The residuals are roughly centered at 0 — no systematic bias. 
   The train/test gap is small — no overfitting.'''


#building a pipeline to  automate the scaling and modeling steps

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

#this is training the pipeline, it will first scale the data and then fit the model on the scaled data
pipe.fit(X_train, y_train)

#this is predicting using the pipeline, it will first scale the test data and then make predictions using the model
pipe.predict(X_test)
r2_pipe = pipe.score(X_test, y_test)

#pipeline ensures scaler fits only on training data

