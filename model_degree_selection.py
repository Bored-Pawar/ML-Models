from sklearn.model_selection import train_test_split
import csv
import numpy as np

# intialize empty list to hold data from csv
features = []
targets = []

with open(r"C:\Users\Aditya Pramod Pawar\Machine Learning\multiple_variable_house_data.csv") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        features.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        targets.append(float(row[4]))

# first split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size = 0.4, random_state = 42)

# second split: 30% & 10% of main data ehich is 75:25
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.75, random_state = 42)

# import libraries for testing model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error 

best_cv_mse = float('inf')
best_degree = None
results = []
ALPHA = 0.1

for degree in range(1, 11):
    # build pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree = degree, include_bias = False)),
        ('scalar', StandardScaler()),
        ('ridge', Ridge(alpha = ALPHA))
    ])

    # Train on training set
    model.fit(X_train, y_train)
    
    # evaluate on cross validation data
    y_cv_pred = model.predict(X_cv)
    cv_mse = mean_squared_error(y_cv, y_cv_pred)

    results.append(cv_mse)

    if cv_mse < best_cv_mse:
        best_cv_mse = cv_mse
        best_degree = degree
    
    print(f"Degree {degree}: CV MSE = {cv_mse:.4f}")

# train best model on best degree
final_model = Pipeline([
    ('poly', PolynomialFeatures(best_degree, include_bias = False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha = ALPHA))
])

# Combine X_cv and y_cv with X_train and y_train as u have already done the validation
X_final = np.concatenate([X_train, X_cv])
y_final = np.concatenate([y_train, y_cv])

# Then train
final_model.fit(X_final, y_final)

y_test_pred = final_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"best degree: {best_degree}")
print(f"test MSE for best degree: {test_mse}")
