import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

# import data - modify data_testing with your filepath to test data
data_training = pd.read_csv("datasets/training_data.csv")
data_testing = pd.read_csv("datasets/training_data.csv") 

# function to normalize X,Y
def preprocess(x, y):
    # Get GDP and Happiness means/STD's
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    # Scale GDP and Happiness
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    X = np.column_stack((np.ones(len(x_scaled)), x_scaled))
    Y = y_scaled.reshape(-1, 1)
    return X, Y, x_mean, x_std, y_mean, y_std

# train using OLS model
def train_OLS(X, Y):
    #compute and return beta
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

# predict Y values using beta
def predict_OLS(X, beta):
    #predict using beta
    Y_hat = X*beta.T
    return np.sum(Y_hat,axis=1)
    

# Store both feature names and values
feature_names = ["Length", "Diameter", "Height", "Whole_weight",
                 "Shucked_weight", "Viscera_weight", "Shell_weight"]

features = [data_training[name].values for name in feature_names]

# split files into training vs testing age and compute age from rings
age_train = data_training["Rings"].values + 1.5
age_test = data_testing["Rings"].values + 1.5
num_features = len(features)

# split data (training/testing)
n = len(age_train)
# split by 75% training, 25% testing
split = int(0.75 * n)
train_idx = np.arange(split)
# if two files are the same, use 25% of data for testing; else, use the second file for testing
if data_training.equals(data_testing):
    test_idx = np.arange(split, n)
else:
    test_idx = np.arange(len(data_testing))
age_train = age_train[train_idx]
age_test = age_test[test_idx]

# plot for Y-values against Testing Data
fig, axs = plt.subplots(3, 3, constrained_layout=True)
axs = axs.ravel() 
# plot for Y-values against Training Data
fig, axs2 = plt.subplots(3, 3, constrained_layout=True)
axs2 = axs2.ravel()

# loop through features, find beta values for each and plot
for i in range(num_features):
    # get current feature being tested
    feature_column = features[i]
    feat_train = feature_column[train_idx]
    feat_test = feature_column[test_idx]

    # preprocess training and testing data
    X_train, Y_train, x_mean, x_std, y_mean, y_std = preprocess(feat_train, age_train)
    X_test_scaled = (feat_test - x_mean) / x_std
    Y_test_scaled = (age_test - y_mean) / y_std
    X_test = np.column_stack((np.ones(len(X_test_scaled)), X_test_scaled))
    Y_test = Y_test_scaled.reshape(-1,1)
    
    # linear regression using OLS on trained set to obtain beta values
    beta_train = train_OLS(X_train, Y_train)

    # predict Y values on test set using trained values from training set
    Y_pred_train = predict_OLS(X_train, beta_train)
    Y_pred_test = predict_OLS(X_test, beta_train)
    
    # compute error using MSE
    mse = np.mean((Y_test.ravel() - Y_pred_test) ** 2)
    
    # plot against test data
    ax = axs[i]
    ax.scatter(X_test[:,1], Y_test, label="Normalized Testing Data")
    ax.plot(X_test[:,1], Y_pred_test, color='r', label='OLS')
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel("Abalone Age")
    ax.set_title(f"Abalone Age vs {feature_names[i]} using Testing Data")
    ax.legend()

    # plot against training data
    ax2 = axs2[i]
    ax2.scatter(X_train[:,1], Y_train, label="Normalized Training Data")
    ax2.plot(X_train[:,1], Y_pred_train, color='r', label='OLS')
    ax2.set_xlabel(feature_names[i])
    ax2.set_ylabel("Abalone Age")
    ax2.set_title(f"Abalone Age vs {feature_names[i]} using Training Data")
    ax2.legend()

    # print beta and mse values for Y values vs test data 
    print("beta=", beta_train, "MSE=", mse)

plt.show()
