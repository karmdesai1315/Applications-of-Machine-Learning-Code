Author: Karm Desai
Submitted: October 21, 2025
Purpose: This README serves as information for the submission of the Jupyter file for Assignment 2 for Compsi 4AL3
Python Version: anaconda 3.12.11

#--------------- Instructions: ---------------#

The purpose of this assignment was to create a Support Vector Machine (SVM) using NASA data which distinguishes between a positive or negative solar flare event. This code runs the functions feature_experiment() and data_experiment() on datasets containing the solar flare information (as listed in the Assignment 2 pdf). The my_svm() class contains functions which are used within feature_experiment() and data_experiment() described below.

feature_experiment() trains and evaluates an SVM classifier using all possible combinations of the four feature sets (FS-I to FS- IV) from the 2010–2015 solar flare dataset. For each feature combination, it: Loads and combines positive and negative feature data -> Preprocesses the features(removes missing values and normalizes them) -> Performs 5-fold cross-validation, training the SVM on each split -> Calculates TSS (True Skill Statistic) scores for each fold and computes the average -> Computes a total confusion matrix (TP, FP, TN, FN) -> Displays a bar chart showing TSS scores across folds -> Prints performance metrics for each feature combination. The function outputs the TSS results and confusion matrices for all 15 feature combinations, allowing identification for the best-performing feature set.

data_experiment() extends the analysis performed in feature_experiment() to two datasets (2010–2015 and 2020–2024 solar flare datasets) to compare model performance over time (essentially running feature_experiment() on both sets). For each dataset, it: Runs all 15 feature set combinations (FS-I to FS-IV) -> Preprocesses the data (normalization and NaN removal) -> Performs 5-fold cross-validation for each feature combination -> Computes mean and standard deviation of TSS scores across folds -> Generates
confusion matrices (TP, FP, TN, FN) for each combination -> Plots bar charts of per-fold TSS scores and confusion matrices in a 3×5 grid -> Prints summary statistics and identifies the best-performing feature set for each dataset. The function ensures both datasets are evaluated using identical steps, producing a full performance comparison between solar flare prediction models trained on different time periods.

**NOTE: I structured data experiment in this way to best conform to the methods endorsed by TA's (from Teams channel) and instructions from the assignment itself.

In the training() function within my_svm(), you may tune the hyperparameter for kernel, C and gamma in SVC. These were experimented with various times and the current implementation is what yielded one of the higher average TSS score values across both datasets.

#--------------- Sources/References: ---------------#

1. scikitlearn website for how to use various functions in the library
   https://www.geeksforgeeks.org/machine-learning/classifying-data-using-support-vector-machinessvms-in-python/
2. PDF file contained information for confusion matrices plotting
3. Linear_Regression.py from Assignment 1 as reference for some code snippets
4. LLM Usage - ChatGPT (in no particular order):
    a. How do .npy files look?
    b. How does StandardScalar and SVC work?
    c. How can I create a map for columns amongst different files?
    c. How does fit_transform, KFold work (sklearn functions)
    d. What is allow_pickle? (Context: was getting errors in loading npy files)
    e. What does enumerate do?
    f. How can I plot 2x2 matrices using subplots?
    g. How do I plot and format a bar chart in Python?
    Total: 4.32g * 8 queries = 34.56g