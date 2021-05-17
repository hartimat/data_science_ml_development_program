###############################################################################
# FILENAME: DataCleaning.py
# AUTHOR: Matthew Hartigan
# DATE: 19-April-2021
# DESCRIPTION: A python script to import and clean excel file data.
###############################################################################

# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt


class DataCleaning:

    # FUNCTIONS
    # Initialize instance of class
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.DataFrame

    # Read in excel file data, print stats to terminal
    def read(self):
        self.df = pd.read_excel(self.input_file)
        print('Check shape:')
        print(self.df.shape)    # get dimensions of data frames
        print('\nCheck variable types:')
        print(self.df.info())
        print('\nCheck head of data set:')
        print(self.df.head(5))

    # Compare number of entries to the number of unique values for each column variable
    def duplicates(self):
        print('\nCheck number of original entries:')
        print(self.df.apply(lambda x: [len(x)]))
        print('\nCheck number of unique entries:')
        print(self.df.apply(lambda x: [len(x.unique())]))
        print('\nCheck for duplicate entries:')
        print(len(self.df) - len(self.df.drop_duplicates()))
        self.df = self.df.drop_duplicates()
        print('\nShow data frame with duplicates removed')
        print(self.df)

    # Summarize data by displaying basic statistical values for each variable
    # Includes: mean, std dev, min, max, 25%, 50%, 75% values
    def summarize(self):
        # Statistical summary
        print('\nPrint summary of data set:')
        print(self.df.describe(include='all'))

        # Graphical summary
        print('\nHistograms')
            # FIXME: add as needed

        print('\nScatter Plots')
            # FIXME: add as needed

        print('\nBox Plots')
            # FIXME: add as needed

        print('\nCross Tables')
            # FIXME: add as needed


if __name__ == '__main__':
    assignment = DataCleaning('test.xlsx')
    assignment.read()   # read in input excel file and visually inspect
    assignment.duplicates()     # remove duplicate entries
    assignment.summarize()  # print mean, std, and 5 number summary for vars
