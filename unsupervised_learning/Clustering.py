###############################################################################
# FILENAME: Clustering.py
# AUTHOR: Matthew Hartigan
# DATE: 4-June-2021
# DESCRIPTION: A python script to cluster data.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


class Clustering:

    # FUNCTIONS
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(input_file)
        self.data_continuous_std = None
        self.cluster_count = 3
        self.data_clustered = None
        self.num_pca_components = 2
        self.centroids = None

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

    def prep_for_clustering(self):
        self.data_continuous_std = self.df  # default behavior
        # All data for k-means clustering must be continuous, so remove categorical variables as needed
        self.data_continuous_std = self.df.drop(['name_of_column_to_drop_goes_here'], axis=1)
        mms = MinMaxScaler()
        mms.fit(self.data_continuous_std)
        self.data_continuous_std = mms.transform(self.data_continuous_std)

    # Run elbow method
    def elbow(self):
        model = KMeans(random_state=1234556)
        visualizer = KElbowVisualizer(model, k=(2, 10), timings=False)
        visualizer.fit(self.data_continuous_std)
        visualizer.show()

    # Run silhouette method
    def silhouette(self):
        model = KMeans(random_state=1234556)
        visualizer = KElbowVisualizer(model, metric='silhouette', timings=False)
        visualizer.fit(self.data_continuous_std)
        visualizer.show()

    def cluster(self):
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=1234556)
        kmeans.fit(self.data_continuous_std)

        self.data_clustered = self.df
        string_count = str(self.cluster_count)
        self.data_clustered['cluster' + string_count] = kmeans.predict(self.data_continuous_std)
        self.centroids = kmeans.cluster_centers_

    def pca(self):
        data_pca = PCA(n_components=self.num_pca_components).fit_transform(self.data_continuous_std)
        pca_results = pd.DataFrame(data_pca, columns=['pca1', 'pca2'])
        string_count = str(self.cluster_count)

        palette = sns.color_palette('bright', self.cluster_count)
        sns.scatterplot(x='pca1', y='pca2', legend='full', palette=palette,
                        hue=self.data_clustered['cluster' + string_count], data=pca_results)
        plt.title('K-means Clustering with ' + str(self.num_pca_components) + 'dimensions')
        plt.show()

        # Show how many items are in each cluster
        string_count = str(self.cluster_count)
        print(self.data_clustered['cluster' + string_count].value_counts())
        print()

        # Show where the mean values are for each cluster
        for i in range(len(self.centroids)):
            print(tuple(zip(self.df.columns.values[1:], self.centroids)))
            print()


if __name__ == '__main__':
    assignment = Clustering('input_file_goes_here.xlsx')
    assignment.clean()
    assignment.summarize()
    assignment.prep_for_clustering()
    assignment.elbow()
    assignment.silhouette()
    assignment.cluster()
    assignment.pca()
