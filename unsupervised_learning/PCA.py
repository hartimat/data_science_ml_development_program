###############################################################################
# FILENAME: PCA.py
# AUTHOR: Matthew Hartigan
# DATE: 4-June-2021
# DESCRIPTION: A python script to apply PCA to data.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class PCA:

    # FUNCTIONS
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(input_file)
        self.features = None

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
        print('\nSummary results:')
        print(self.df.describe(include='all'))
        print('\nCheck for missing values: ')
        print(self.df.isnull().sum())
        print()

    def pca(self):
        # Define which columns in the data frame are features
        self.features = self.df.columns.values[1:]

        # Separate out features
        x = self.df.loc[:, self.features].values
        # Separate out target
        y = self.df.lc[:, ['target']].values
        # Standardize the features
        x = StandardScaler().fit_transform()

        pca = PCA(n_components=len(self.features), svd_solver='full')
        principal_components = pca.fit_transform(x)

        print(np.sqrt(pca.explained_variance_))
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_.cumsum())

        # Screeplot of PCAs
        plt.rcParams['figure.figsize'] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, len(self.features) + 1, step=1)
        y = np.sqrt(pca.explained_variance_)

        plt.ylim(0.0, 5)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Number of Components')
        plt.xticks(np.arange(0, len(self.features) + 2, step=1))
        plt.ylabel('Variance')
        plt.title('The number of components needed to explain variance')
        plt.axhline(y=1, color='r', linestyle='-')
        plt.text(0.5, 0.85, 'Eigenvalue=1', color='red', fontsize=10)
        ax.grid(axis='x')
        plt.show()

        # Cumulative variance explained
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

        # Investigate loading
        matrix_loads = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'and so on'])
        print(matrix_loads)

        # Add the PCA scores to the data set to use in modeling an dview
        pca_scores = pd.DataFrame(data=principal_components,
                                  columns=['PC1', 'PC2', 'and so on'])
        pca_all = pd.concat([pca_scores, self.df[['target']]], axis=1)

        # Run Random Forest models for most critical components (partial) and full components case
        partial_model = RandomForestClassifier(n_estimators=100,
                                               random_state=1234556,
                                               max_features='sqrt',
                                               n_jobs=-1,
                                               verbose=1,
                                               oob_score=True)
        full_model = RandomForestClassifier(n_estimators=100,
                                               random_state=1234556,
                                               max_features='sqrt',
                                               n_jobs=-1,
                                               verbose=1,
                                               oob_score=True)
        target = np.array(self.df.pop('target'))
        partial_model.fit(pca_scores['PC1', 'PC2'], target)
        full_model.fit(pca_scores, target)
        partial_model_predictions = partial_model.predict(pca_scores[['PC1', 'PC2']])
        full_model_predictions = full_model.predict(pca_scores)

        # Check corrolation
        raw_values = self.df.corr()
        seaborn.heatmap(raw_values)
        plt.show()

        pca_values = pca_all.corr()
        seaborn.heatmap(pca_values)
        plt.show()

        # Check out of bag errors
        print('\nPartial RF Model Score:')
        print(partial_model.oob_score_)
        print('\nFull RF Model Score:')
        print(full_model.oob_score_)

        # Check confusion matrix
        print('\nPartial RF Confusion Matrix:')
        partial_cm = confusion_matrix(target, partial_model_predictions)
        print(partial_cm)
        print('\nFull RF Confusion Matrix:')
        full_cm = confusion_matrix(target, full_model_predictions)
        print(full_cm)


if __name__ == '__main__':
    assignment = PCA('input_file_goes_here.xlsx')
    assignment.clean()
    assignment.summarize()
    assignment.pca()
