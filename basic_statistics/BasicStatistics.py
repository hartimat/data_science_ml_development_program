###############################################################################
# FILENAME: BasicStatistics.py
# AUTHOR: Matthew Hartigan
# DATE: 24-April-2021
# DESCRIPTION: A python script that runs basic statistical tests (e.g. Chi^2,
#              t-test) on an input excel data file.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, shapiro, levene

class BasicStatistics:

    # FUNCTIONS
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(self.input_file)

    def clean(self):
        print('\nCheck number of original entries:')
        print(self.df.apply(lambda x: [len(x)]))
        print('\nCheck number of unique entries:')
        print(self.df.apply(lambda x: [len(x.unique())]))
        print('\nCheck for duplicate entries:')
        print(len(self.df) - len(self.df.drop_duplicates()))
        self.df = self.df.drop_duplicates()

    def summarize(self):
        print('\nCheck shape:')
        print(self.df.shape)    # get shape of data frame
        print('\nCheck variable types:')
        print(self.df.info())
        print('\nCheck head of data set:')
        print(self.df.head(5))
        print()

    def chi_squared_test(self):
        # Make qualitative variables categorical for legibility
        df1 = self.df
        df1['Male'] = df1['Male'].map({0: 'Female', 1: 'Male'})
        df1['Male'] = df1['Male'].astype('category')
        df1.rename(columns={'Male': 'Gender'}, inplace=True)
        df1['Sport'] = df1['Sport'].map({0: 'Sedentary',
                                         1: 'Light Activity',
                                         2: 'Non-competitve Sports',
                                         3: 'Competitive Sports'})
        df1['Sport'] = df1['Sport'].astype('category')

        # Check cross tables for these two variables
        print(pd.crosstab(df1['Gender'], df1['Sport']))
        print(pd.crosstab(df1['Gender'], df1['Sport']), normalize=index)

        # Compute p-value
        ct1 = pd.crosstab(df1['Gender'], df1['Sport'])
        ct2 = chi2_contingency(ct1, correction=False)
        print('\nObserved Counts')
        print(ct1)
        print('\nExpected Counts')
        print(ct2[3])
        print('\nChi-Square')
        print(ct2[0])
        print('\nP-Value')
        print(ct2[1])

    def independent_t_test(self):
        # Make qualitative variables categorical for legibility
        df2 = self.df
        df2['BMIB'].replace('.', np.nan, inplace=True)  # replace '.' values with nan so mean can be computed
        df2.dropna(subset=['BMIB'], inplace=True)
        df2['Smoke'] = df2['Smoke'].map({0: 'Non-smoking',
                                         1: 'Smoking',
                                         2: 'Smoking',
                                         3: 'Smoking'})
        df2['Smoke'] = df2['Smoke'].astype('category')

        # Summarize stats for each category of smoker
        print(df2.groupby('Smoke')['BMIB'].describe())

        # Compute p-value
        subset0 = df2.loc[df2['Smoke'] == 'Non-Smoking']
        subset0_1 = subset0['BMIB']
        print(subset0)
        print(subset0_1)
        subset1 = df2.loc[df2['Smoke'] == 'Smoking']
        subset1_1 = subset1['BMIB']
        ttest, pval = ttest_ind(subset0_1, subset1_1, equal_var=False)
        print('\nt-test statistic = ', ttest)
        print('\np-value = ', pval)
        print('\nNon-smoking mean BMI:', subset0['BMIB'].mean())
        print('\nSmoking mean BMI:', subset1['BMIB'].mean())

        # Confirm normality with Shapiro test
        stat, p = shapiro(subset0['BMIB'])
        print('Shapiro statistic=%.3f, p=%.3f' % (stat, p))
        stat, p = shapiro(subset1['BMIB'])
        print('Shapiro statistic=%.3f, p=%.3f' % (stat, p))

        # Confirm homogeneity of variance with Levene test
        stat, p = levene(subset0['BMIB'], subset1['BMIB'])
        print('Levene statistic=%.3f, p=%.3f' % (stat, p))


if __name__ == '__main__':
    assignment = BasicStatistics('input_excel_file.xlsx')
    assignment.clean()
    assignment.summarize()
    assignment.chi_squared_test()
    assignment.independent_t_test()
