###############################################################################
# FILENAME: Trees.py
# AUTHOR: Matthew Hartigan
# DATE: 6-May-2021
# DESCRIPTION: A python script to run decision tree and random forest analysis.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn import tree
from sklearn.processing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class Trees:

    # FUNCTIONS
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_csv(self.input_file)

    # Implements a five fold crosvalidation of a multivariate regression model when given a training dataset
    def five_fold_crossvalidation(self):

        # Create the data set
        X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
        print(X.shape, y.shape)

        # Setup the crossvalidation procedure
        cv = KFold(nsplits=5, random_state=1, shuffle=True)

        # Create model
        model = LogisticRegression()

        # Evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=1)

        # Report performance
        print(scores)
        print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    def routine_1(self):
        # Read in data
        titanic_train = pd.read_csv('titanic_train.csv')
        titanic_train.head()

        # Subset data by columns that we want to keep
        titanic_keep = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        titanic_for_trees = titanic_train[titanic_keep]

        # Clean data
        titanic_for_trees = titanic_for_trees.dropna()
        titanic_for_trees = titanic_for_trees.reset_index(drop=True)

        # Split up training (X) and testing (y) matrices
        X = titanic_for_trees.drop(columns='Survived')
        y = titanic_for_trees['Survived']

        # One hot encode categorical variables
        categorical_variables = ['Pclass', 'Sex', 'Embarked']
        continuous_variables = ['Age', 'SibSp', 'Parch']
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(X[categorical_variables])
        X_ohed = one_hot_encoder.transform(X[categorical_variables]).toarray()
        X_ohed = pd.DataFrame(X_ohed, columns=one_hot_encoder.get_feature_names(categorical_variables))

        # Merge one hot encoded categorical variables with continuous variables
        X = pd.concat([X[continuous_variables], X_ohed], axis=1)
        print(X.head())

        # Create classifier with default tree pruning hyper parameters and fit to data
        classifier_def = tree.DecisionTreeClassifier()
        classifier_def.fit(X, y)

        # Takes a second to generate the tree...
        # Note the difference in default cp value in sklearn (0.0) resulting in full tree
        tree_plot_def = tree.plot_tree(classifier_def)

        # Modulate the complexity parameter to result in a far smaller tree
        # Experiment with different tree parameters using the difinitions here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        classifier_cp = tree.DecisionTreeClassifier(ccp_alpha=0.01)
        classifier_cp.fit(X, y)
        tree_plot_cp = tree.plot_tree(classifier_cp)

        # Evaluation function calculating a number of helpful error metrics
        def evaluate(model, test_features, test_labels):
            predictions = model.predict(test_features)
            precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
            accuracy = accuracy_score(y_pred=predictions, y_true=test_labels)
            mcc = matthews_corrcoef(y_pred=predictions, y_true=test_labels)

            print('Model Performance')
            print('Precision = {:0.2f}%'.format(precision))
            print('Recall = {:0.2f}%'.format(recall))
            print('F1-Score = {:0.2f}%'.format(f1_score))
            print('Accuracy = {:0.2f}%'.format(accuracy))
            print('MCC = {:0.2f}%'.format(mcc))

            return [precision, recall, f1_score, accuracy, mcc]

        # Split data into
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

        # Use a grid search over a specified range of possible hyperparameter values
        parameters = {'max_depth': range(3, 15),
                      'ccp_alpha': np.arange(0, .2, .01),
                      'max_features': range(3, 16)}

        # Run a grid search over hyper parameter ranges
        classifier = GridSearchCV(tree.DecisionTreeClassifier(),
                                  parameters,
                                  cv=10,
                                  n_jobs=-1)
        classifier.fit(X=X_train, y=y_train)

        # Extract best CV model and print its accuracy measure
        cv_best_model = classifier.best_estimator_
        print('Best CV accuracy: {}, with parameters: {}'.format(classifier.best_score_, classifier.best_params_))

        # Evaluate best CV model on holdout test set
        precision, recall, f1_score, accuracy, mcc = evaluate(cv_best_model, X_test, y_test)

        # Plot confusion matrix and best decision tree
        plot_confusion_matrix(cv_best_model, X_test, y_test)
        cv_best_model = tree.plot_tree(cv_best_model)

    def routine_2(self):
        def evaluate(model, test_features, test_labels):
            predictions = model.predict(test_features)
            r2 = r2_score(y_pred=predictions, y_true=test_labels)

            print('Model Performance')
            print('R2 = {:0.2f}.'.format(r2))

            return [r2]

        pima = pd.read_csv('PimaIndianDiabetes.csv')
        X = pima.drop(columns=['diabetes'])
        y = pima['diabetes']
        y = y.map({'neg': 0, 'pos': 1})

        # Split data into a 70/30 train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        rf = RandomForestRegressor()

        # Create the parameter grid based on the results of random search
        # This section can be modified to look into possible parameters
        param_grid = {
            'max_depth': [80, 90], # , 100, 200, None]
            'max_features': ['auto', 'sqrt', 'log2', None],
            'n_estimators': [10, 50, 100] # , 200, 300, 500, 1000]
        }

        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=param_grid,
                                   cv=10,
                                   n_jobs=-1,
                                   verbose=2)

        # Perform CV search over grid from hyper parameters
        grid_searh.fit(X_train, y_train)
        print('Best CV accuracy: {}, with parameters: {}'.format(
            grid_search.best_score_, grid_search.best_params_))

        best_model = grid_search.best_estimator_
        r2 = evaluate(best_model, X_test, y_test)

        # Now that we know the best hyperparameter, we refit our RF model
        # using them and the full data set at our disposal
        best_params = grid_search.best_params_
        full_model = RandomForestRegressor(**best_params)
        full_fit = full_model.fit(X, y)


if __name__ == '__main__':
    assignment = Trees('PimaIndiansDiabetes.csv')
    assignment.five_fold_crossvalidation()
    assignment.routine_1()
    assignment.routine_2()
