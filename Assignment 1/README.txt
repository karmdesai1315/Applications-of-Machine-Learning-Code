Author: Karm Desai
Submitted: September 24, 2025
Purpose: This README serves as information for the submission of 2 files of Assignment 1 for Compsi 4AL3
Python Version: anaconda 3.12.11

*Note: Both questions can be run by simply hitting run

#--------------- Instructions for Question 1 ---------------#

*Part a*
This code outputs a plot with 8 different regression lines and has a legend to describe each line. The legend
displays the alpha value, iteration count (epochs) and the MSE (mean-squared error) value for a particular line.

There are 5 alpha and 5 iteration values chosen to test, and there is a double for loop which loops through
them all (25 total experiments) on lines 67-68. Line 71 contains the plot pairs you wish to visualize on the 
scatter plot (currently 7 different lines). You may change these values as you wish.

*Part b*
After line 101, the code will run OLS on the data. The second plot will print this OLS value against the 
lowest MSE value from the gradient descent in part a. 

When running, the code will output both plots, one after the other. The terminal window will print out the alpha, 
iteration, beta and MSE values for each of the 25 lines from the gradient descent model. It will also print the 
Best Results (lowest MSE) from the experiments. *It will only print in terminal when the two plots are closed*.

#--------------- Instructions for Question 2 ---------------#

The code requires 2 inputs, training_data and testing_data. The training_data is the data we were given and 
shouldn't be modified. The testing_data (Line 10) can be changed with your filepath to the data which needs to 
be tested. The code splits the training_data (what we were given) by 75% to use as training values (Line 50).

This code uses the OLS model for linear regression to determine the relationship between abalone's age against
various features (Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight). It first
splits the training data so roughly 2000 points are used as training. If the 2 input files are the same, it will
use the remaining training data points as testing data. Otherwise, it will use the testing_data file as testing
data. The code computes beta values using the training data first. It then uses those beta values on the unseen
testing values to find a predicted Y (Age) line. Finally, it plots using the Y line.

The code outputs 2 plots. One plot is Age vs Features using Training data as the actual data. The other plots 
Age vs Features using Testing data as the actual data. It will then print the trained beta values and error 
using the MSE approach in the terminal window.

#--------------- Sources/References: ---------------#

1. 4AL3 Week 2 Lecture 5: Regression & Gradient Descent - Slides 60-61
2. 4AL3 Week 2 Lecture 6: Regression & Gradient Descent - Slide 21
3. Linear_Regression.py from Assignment folder for multiple functions and code snippets
4. LLM Usage - ChatGPT (in no particular order):
    a. How can I modify my plotting code so that it all fits on one figure and doesn't overlap other graphs?
    b. Can you explain to me how gradient descent works exactly using a simple example?
    c. How does the .ravel() function work in python?
    d. How does the preprocess code in this file (Linear_Regression.py) work and why does the code do this in 
       regards to modelling?
    Total: 4.32g * 4 queries = 17.28g