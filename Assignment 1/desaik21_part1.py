import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

# Filter for 2018 and drop unused columns
df = data[data['Year'] == 2018].drop(columns=["World regions according to OWID", "Code"])

# Remove rows with missing values for core columns
df = df[df['Cantril ladder score'].notna() & df['GDP per capita, PPP (constant 2021 international $)'].notna()]

# Keep only rows where happiness score > 4.5
df = df[df['Cantril ladder score'] > 4.5]

# Extract arrays
gdp = df['GDP per capita, PPP (constant 2021 international $)'].values
happiness = df['Cantril ladder score'].values

def preprocess(x, y):
    # Get GDP and Happiness means/STD's
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    # Scale GDP and Happiness
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    X = np.column_stack((np.ones(len(x_scaled)), x_scaled))
    Y = y_scaled.reshape(-1, 1)
    return X, Y

def train_gradient_descent(X, Y, alpha, iterations):
    # get n (samples), features of X
    n, feat = X.shape
    # create column vector of random initial beta
    beta = np.random.randn(feat, 1)
    for iteration in range(iterations):
        gradients = 2/n * (X.T).dot(X.dot(beta) - Y)
        beta = beta - alpha * gradients
    return beta

def train_OLS(X, Y):
    #compute and return beta
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def predict_gradient_descent(X, beta):
    Y_hat = X*beta.T
    return np.sum(Y_hat,axis=1)

def predict_OLS(X, beta):
    #predict using beta
    Y_hat = X*beta.T
    return np.sum(Y_hat,axis=1)

# Preprocess for scatterplot
X, Y = preprocess(gdp, happiness)
x_normalized = X[:, 1].ravel() # convert to 1D array

# Plot raw data
fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(x_normalized, Y, label="Actual Data")

# Storage for results
results = []

# 5 alphas and 5 iterations to experiment (25 total)
alphas = [0.1, 0.01, 0.005, 0.0005, 0.00005]
iterations = [50, 100, 500, 1000, 2000]

# choose 7 different combinations to plot
plot_pairs = {(0.1, 50), (0.01, 100), (0.01, 1000), (0.005, 500), (0.0005, 500), (0.0005, 1000), (0.00005, 1000)}

### Output 1: Gradient Descent ###
for alpha in alphas:
    for iteration in iterations:
        # preprocess data
        X, Y = preprocess(gdp, happiness)
        # find beta values
        beta = train_gradient_descent(X, Y, alpha, iteration)
        # create matrix for plotting using beta values
        Y_pred = predict_gradient_descent(X, beta)
        # calculate MSE
        current_mse = np.mean((Y.ravel() - Y_pred) ** 2)
        # plot
        if (alpha, iteration) in plot_pairs:
            ax.plot(x_normalized, Y_pred, label=f"α={alpha:.5f}, iters={iteration}, MSE={current_mse:.5f}")
        # store results
        results.append((alpha, iteration, beta.flatten(), current_mse))

# Print result values in terminal
for alpha, iteration, beta_vals, mse in results:
    print(f"alpha={alpha:.5f}, iterations={iteration}, beta={beta_vals}, MSE={mse:.9f}")

# Plot 1: Different gradient descent regression lines against data
ax.set_xlabel("GDP per capita (normalized)")
ax.set_ylabel("Happiness (normalized)")
ax.set_title("Effect of Learning Rate and Epochs on Gradient Descent Regression")
ax.legend()
ax.set_ylim(top=3)


### Output 2: OLS vs Gradient Descent ###

# preprocess the inputs
X2,Y2 = preprocess(gdp, happiness)
#compute beta
beta2 = train_OLS(X2,Y2)
# use the computed beta for prediction
Y_pred2 = predict_OLS(X2,beta2)
# find lowest MSE from 25 experiments gradient descent
best_index = 0
best_mse = results[0][3]

for i in range(1, len(results)):
    if results[i][3] < best_mse:
        best_mse = results[i][3]
        best_index = i

best_alpha, best_iteration, best_beta, best_mse = results[best_index]
Y_best_gd = predict_gradient_descent(X, best_beta)

# print best gradient descent values
print(f"Best Results (Lowest MSE): alpha={best_alpha:.5f}, iterations={best_iteration}, beta={best_beta}, MSE={best_mse:.9f}")

# Plot raw data
fig2,ax2 = plt.subplots(figsize=(15, 8))
ax2.scatter(x_normalized, Y, label="Actual Data")

# Plot 2: OLS line and the lowest MSE Gradient Descent line against data
ax2.plot(x_normalized, Y_pred2, color='r', label='OLS')
ax2.plot(x_normalized, Y_best_gd, color='g', label=f"Best GD: α={best_alpha:.5f}, iters={best_iteration}, MSE={best_mse:.5f}")
ax2.set_xlabel("GDP per capita (normalized)")
ax2.set_ylabel("Happiness (normalized)")
ax2.set_title("OLS vs Gradient Descent on Cantril Ladder Score vs GDP per capita of countries (2018)")
ax2.legend()
plt.show()

