###############################################################################
# FILENAME: NeuralNetwork.py
# AUTHOR: Matthew Hartigan
# DATE: 23-June-2021
# DESCRIPTION: A python script that implements a neural network.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from neupy.layers import *
from neupy import algorithms


class NeuralNetwork:

    # FUNCTIONS
    # Initialize class instance
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(self.input_file)

    def clean(self):
        print('\nCheck number of original entries:')
        print(self.df.apply(lambda x: [len(x)]))
        print('\nCheck number of unique entries:')
        print(self.df.apply(lambda x: [len(x.unique())]))
        print('\nCheck for duplicate entries, then remove:')
        print(len(self.df) - len(self.df.drop_duplicates()))
        self.df = self.df.drop_duplicates()
        print('\nShow data frame with duplicates removed')
        print(self.df)

    # Read in excel file data, print stats to terminal
    def summarize(self):
        print('Check shape:')
        print(self.df.shape)  # get dimensions of data frames
        print('\nCheck variable types:')
        print(self.df.info())
        print('\nCheck head of data set:')
        print(self.df.head(5))
        print('\Summary results:')
        print(self.df.describe(include='all'))
        print('\Check for missing values:')
        print(self.df.isnull().sum())
        print()

    # Linear regression model to compare neural network results to
    def linear_regression(self):
        X = pd.DataFrame(np.c_[self.df['fixme1'], self.df['fixme2']])    # independent variables
        Y = self.df['fixme_target']    # dependent variables

        # Create training and test data sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=333)

        # Build model on training data, then predict test values
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        Y_train_pred = lr.predict(X_train)
        Y_test_pred = lr.predict(X_test)

        # Calculate mean squared error for the test data
        MSE = metrics.mean_squared_error(Y_test, Y_test_pred)
        print('Mean squared error for linear model (test data) = ', '%.5f' %MSE)

    def neural_network(self):
        # Prepare data (must be normalized... i.e. scaled with min/max method with interval [0,1])
        x = pd.DataFrame(np.c_[self.df['fixme1'], self.df['fixme2']])  # independent variables
        y = self.df['fixme_target']  # dependent variables
        data_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()
        data = data_scaler.fit_transform(x)
        target = target_scaler.fit_transform(y.values.reshape(-1, 1))

        # Create training and test data sets
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=333)

        # Set parameters for hidden layers, node counts, etc.
        nodes1 = 5    # fixme
        nodes2 = 4    # fixme
        nodes3 = 3    # fixme
        n_inputs = 2    # fixme
        n_outputs = 1    # fixme

        # Apply neural network model
        network = join(Input(n_inputs), Sigmoid(nodes1), Sigmoid(nodes2), Sigmoid(nodes3), Linear(n_outputs))
        optimizer = algorithms.gd.rprop.RPROP(network, verbose=True, show_epoch=5)
        optimizer.train(x_train, y_train, x_test, y_test, epochs=100)

        # Visualize th neural network using graphyiz
        # Link: https:/graphyiz.gitlab.io/_pages/Download/Download_windows.html
        # os.environ['PATH'] += os.pathsep + 'fixme'
        # network.show()

        # Calculate mean squared error for the test data
        y_predict = optimizer.predict(x_test).round(1)
        NN_mse = np.mean((target_scaler.inverse_transform(y_test) -
                          target_scaler.inverse_transform(y_predict)) ** 2)
        print('Mean squared error for neural network (test data) = ', '%.5f' %NN_mse)


if __name__ == '__main__':
    assignment = NeuralNetwork('inputfile.xlsx')
    assignment.clean()
    assignment.summarize()
    assignment.linear_regression()
    assignment.neural_network()
