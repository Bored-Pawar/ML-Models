import csv
import matplotlib.pyplot as plt

# Initialize empty list
x, y = [], []

# Read CSV file
try:
    with open(r'C:\Users\Aditya Pramod Pawar\Machine Learning\Linear_regression\one_variable_house_data.csv') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    
    # print(x)
    # print(y)

    # Normalize x (square footage)
    x_mean = sum(x) / len(x)
    x_std = (sum((xi - x_mean) ** 2 for xi in x) / len(x)) ** 0.5
    x_norm = [(xi - x_mean) / x_std for xi in x]

except FileNotFoundError:
    print("Enter Data manually:\n")
    x = list(map(float, input("Enter space seperated square footage: \n").split()))
    y = list(map(float, input("Enter space seperated house prices: \n").split()))
    

# Initialize parameters
w = 0.0
b = 0.0

# Define Cost function(MSE) to calculate MSE >> mean square equation
def compute_cost_function(x, y, w, b):
    m = len(x)
    total_error = 0
    for i in range(m):
        total_error += (abs(x[i] * w + b - y[i])) ** 2
    total_error /= 2 * m
    return total_error

# define compute_gradients to calculate the derivatives
def compute_gradients(X, y, w, b):
    m = len(X)
    dj_dw = 0.0  
    dj_db = 0.0  
    for i in range(m):
        error = (w * X[i] + b) - y[i]  
        dj_dw += error * X[i]  
        dj_db += error
    dj_dw, dj_db = dj_dw / m, dj_db / m  
    return dj_dw, dj_db

#define gradient descent to update value of w and b
def gradient_descent(x_norm, y, w, b, alpha, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradients(x_norm, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Print cost every 100 iterations
        if i % 100 == 0:
            cost = compute_cost_function(x_norm, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.4f}")
            
    return w, b

#initialize alpha, iterations 
alpha = 0.01
iterations = 1000
# Runing gradient descent
w, b = gradient_descent(x_norm, y, w, b, alpha, iterations)

# After training (w and b are learned)
print("Enter square footages to predict price (enter -1 to stop):")
user_inputs = []
predictions = []

while True:
    x_input = float(input("sqft: "))
    if x_input == -1:
        break  
    try:
        x_input_norm = (x_input - x_mean) / x_std  # Normalize input
        price = w * x_input_norm + b  
        user_inputs.append(x_input)
        predictions.append(price)
        print(f"Predicted price: ${price:,.2f}\n")
    except ValueError:
        print("Invalid input! Enter a number or -1 to quit.")

# Plot original dataset
plt.scatter(x, y, c = 'blue', label = 'Actual Prices')

# Plot regression line
x_line = [min(x), max(x)]
x_line_norm = [(xi - x_mean) / x_std for xi in x_line]
y_line = [w * xi + b for xi in x_line_norm]
plt.plot(x_line, y_line, 'r-', label = f'Regression: ${w:.2f}*x + ${b:.2f}')
# Plot user predictions (if any)
if user_inputs:
    plt.scatter(user_inputs, predictions, c = 'green', marker = 's', s = 100, label = 'Your Predictions')

# Formatting
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
