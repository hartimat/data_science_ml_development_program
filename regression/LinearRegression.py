###############################################################################
# FILENAME: LinearRegression.py
# AUTHOR: Matthew Hartigan
# DATE: 29-April-2021
# DESCRIPTION: A python script that runs single variable and multiple variable
#              linear regression models.
###############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pypot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import rcParams
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import shapiro
from scipy.stats import pearsonr
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import  variance_inflation_factor
from patsy import dmatrices
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import  anova_lm


class LinearRegression:

    # FUNCTIONS
    # Initialize class instance
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(self.input_file)

    # Compare number of entries to the number of unique values for each column variable
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
        print()

    # Perform regression (and multivariate regression) on given data set
    def regression(self):

        # Assign variables
        B = self.df['CommentCount24']
        # B = self.df['PostLength']
        # C = self.df['PostShareCount']
        # D = self.df['PostSaturday']
        E = self.df['NumberLikes24']

        # Create linear model
        PostLikeModel = sm.OLS(E, B).fit()
        PostLikeModel.summary()

        # Calculate RSE value
        np.sqrt(PostLikeModel.scale)

        # Plot the regression model
        sns.set(color_code=True)
        ax = sns.regplot(x='CommentCount24',
                         y='NumberLikes24',
                         data=self.df,
                         scatter_kws={"color": "blue"},
                         line_kws={"color": "red"})

        # Calculate predicted values on current data set
        self.df['DV_PostLikeModel'] = PostLikeModel.predict(B)

        # Assess prediction accuracy, create train / test data set
        B_train, B_test, E_train, E_test = train_test_split(B, E, test_size=0.20, random_state=123)

        # Build model on training data
        lr = LinearRegression()
        B_train = B_train.values.reshape(-1, 1)
        E_train = E_train.values.reshape(-1, 1)
        lr.fit(B_train, E_train)
        E_train_pred = lr.predict(B_train)

        # Predict earnings on test data using model from training data
        B_test = B_test.values.reshape(-1, 1)
        E_test = E_test.vallues.reshape(-1, 1)
        E_test_pred = lr.predict(B_test)

        # Compare the RMSE fr both test and train
        print('RMSE training data = ', np.sqrt(metrics.mean_squared_error(E_train, E_train_pred)))
        print('RMSE testing data = ', np.sqrt(metrics.mean_squared_error(E_test, E_test_pred)))

        # Observe correlation between actual / predicted in test data
        stat, p = pearsonr(E_test.ravel(), E_test_pred.ravel())
        print('\nCorrelation')
        print(stat)
        print()
        plt.scatter(x=E_test, y=E_test_pred)

        # Run diagnostics on linear regression model
        PostLikeModel = sm.OLS(E, B).fit()
        print(PostLikeModel.summary())

        # Plot residuals vs fitted
        residuals = PostLikeModel.resid
        fitted = PostLikeModel.fittedvalues
        smoothed = lowess(residuals, fitted)
        top3 = abs(residuals).sort_values(ascending=False)[:3]

        plt.rcParams.update({'font.size': 16})
        plt.rcParams['figure.figsize'] = (8, 7)
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, edgecolors='k', facecolor='none')
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
        ax.set_ylabel('Residuals')
        ax.set_xlabel('FittedValues')
        ax.set_title('Residuals vs Fitted')
        ax.plot([min(fitted), max(fitted)], [0, 0], color='k', linestyle=':', alpha=0.3)
        for i in top3.index:
            ax.annotate(i, xy=(fitted[i], residuals[i]))
        plt.show()

        # Normal Q-Q plot
        sorted_residuals = pd.Series(PostLikeModel.get_influence().resid_studentized_internal)
        sorted_residuals.index = PostLikeModel.resid.index
        sorted_residuals = sorted_residuals.sort_values(ascending=True)
        df = pd.DataFrame(sorted_residuals)
        df.columns = ['sorted_residuals']
        df['theoretical_quantiles'] = stats.probplot(df['sorted_residuals'], dist='none', fit=False)[0]
        rankings = abs(df['sorted_residuals']).sort_values(ascending=False)
        top3 = rankings[:3]

        fig, ax = plt.subplots()
        x = df['theoretical_quantiles']
        y = df['sorted_residuals']
        ax.scatter(x, y, edgecolor='k', facecolor='none')
        ax.set_title('Normal Q-Q')
        ax.set_ylabel('Standardized Residuals')
        ax.set_xlabel('Theoretical Quantiles')
        ax.plot([np.min([x, y])], np.max([x, y]), np.max([x, y]), color='r', ls='--')
        for val in top3.index:
            ax.annotate(val, xy=(df['theoretical_quantiles'].loc[val], df['sorted_residuals'].loc[val]))
        plt.show()

        # Scale - Location plot
        like_residuals = PostLikeModel.get_influence().resid_studetized_internal
        sqort_like_residuals = pd.Series(np.sqrt(np.abs(like_residuals)))
        sqort_like_residuals.index = PostLikeModel.resid.index
        smoothed = lowess(sqort_like_residuals, fitted)
        top3 = abs(sqort_like_residuals).sort_values(ascending=False)[:3]

        fig, ax = plt.subplots()
        ax.scatter(fitted, sqort_like_residuals, edgecolors='k', facecolors='none')
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
        ax.set_ylabel('$\sqrt{|Studentized \  Residuals|}$')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Scale-Location Plot')
        ax.set_ylim(0, max(sqort_like_residuals) + 0.1)
        for i in top3.index
            ax.annotate(i, xy=(fitted[i], sqort_like_residuals[i]))
        plt.show()

        # Residuals vs Leverage
        like_residuals = pd.Series(PostLikeModel.get_influence().resid_studentized_interanl)
        like_residuals.index = PostLikeModel.resid.index
        df = pd.DataFrame(like_residuals)
        df.columns = ['like_residuals']
        df['leverage'] = PostLikeModel.get_influence().hat_matrix_diag
        smoothed = lowess(df['like_residuals'], df['leverage'])
        sorted_like_residuals = abs(df['like_residuals']).sort_values(ascending=False)
        top3 = sorted_like_residuals[:3]

        fig, ax = plt.subplots
        x=df['leverage']
        y=df['like_residuals']
        xpos = max(x) + max(x) * 0.01
        ax.scatter(x, y, edgecolors='k', facecolor='none')
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
        ax.set_ylabel('Studentized residuals')
        ax.set_xlabel('Leverage')
        ax.set_title('Residuals vs Leverage')
        ax.set_ylim(min(y) - min(y) * 0.15, max(y) + max(y) * 0.15)
        ax.set_xlim(-0.01, max(x) + max(x) * 0.05)
        plt.tight_layout()
        for val in top3.index:
            ax.annotate(val, sy=(x.loc[val], y.loc[val]))

        cooksx = np.linspace(min(x), xpos, 50)
        p = lne(PostLikeModel.params)
        poscooks1y = np.sqrt((p * (1 - cooksx)) / cooksx)
        poscooks05y = np.sqrt(0.5 * (p * (1 - cooksx)) / cooksx)
        negcooks1y = -np.sqrt((p * (1 - cooksx)) / cooksx)
        negcooks05y = -np.sqrt(0.5 * (p * (1 - cooksx)) / cooksx)

        ax.plot(cooksx, poscooks1y, label='Cooks Distance', ls=':', color='r')
        ax.plot(cooksx, poscooks05y, ls=':', color='r')
        ax.plot(cooksx, negcooks1y, ls=':', color='r')
        ax.plot(cooksx, negcooks05y, ls=':', color='r')
        ax.plt([0, 0], ax.get_ylim(), ls=':', alpha=0.3, color='k')
        ax.plt(ax.get_xlim(), [0, 0], ls=':', alpha=0.3, color='k')
        ax.annotate('1.0', xy=(xpos, poscoos1y[-1]), color='r')
        ax.annotate('0.5', xy=(xpos, poscoos05y[-1]), color='r')
        ax.annotate('1.0', xy=(xpos, negcoos1y[-1]), color='r')
        ax.annotate('0.5', xy=(xpos, negcoos05y[-1]), color='r')
        ax.legend()
        plt.show()

        # Check statistical assumptions for linear regression hold true
        # Linearity of dependent and independent variable
        plt.scatter(x=self.df['CommentCount24'], y=self.df['NumberLikes24'])

        # Normality for the independent variable
        stat, p = shapiro(self.df['CommentCount24'])
        print('\nIV Shapiro Statistic=$.3f, p=%.3f' % (stat, p))

        self.df['CommentCount24'].plot.hist(alpha=0.5,
                                            bins=50,
                                            grid=True,
                                            legend=None,
                                            density=True,
                                            color='gray',
                                            edgecolor='black')
        mu = self.df['CommentCount24'].mean()
        variance = self.df['CommentCount24'].var()
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color='r')
        plt.show()

        qplot(self.df['CommentCount24'], line='q')
        plt.show()

        # Normality for the residuals
        stat, p = shapiro(residuals)
        print('\nResiduals Shapiro Statistic=%.3f, p=%.3f', % (stat, p))
        print()
        residuals.plot.hist(alpha=0.5,
                            bins=50,
                            grid=True,
                            legend=None,
                            density=True,
                            color='gray',
                            edgecolor='black')
        mu = residuals.mean()
        variance = residuals.var()
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color='r')
        plt.show()
        qplot(residuals, line='q')
        plt.show()

        # Multicollinearity
        df_sub = pd.DataFrame(self.df, columns=['NumberLikes24',
                                                'CommentCount24',
                                                'PostLength',
                                                'PostShareCount',
                                                'PostSaturday'])
        corrmatrix = df_sub.corr()
        print(corrmatrix)

        # VIF for independent variable
        y, X = dmatrices(formula_like='NumberLikes24 ~ CommentCount24 + PostLength + PostShareCount + PostSaturday',
                         data=self.df,
                         return_type='dataframe')
        vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print('\nCommentCount24 VIF =', vif[1])
        print('PostLength VIF', vif[2])
        print('PostShareCount VIF =', vif[3])
        print('PostSaturday VIF =', vif[4])

        # Autocorrelation
        print('\nDurbin-Watson =', durbin_watson(residuals))

        # Multivariate regression
        MultiReg = smf.ols('NumberLikes24 ~ CommentCount24 + PostLength + PostShareCount + PostSaturday',
                           data=self.df).fit()
        print(MultReg.summary())

        # Model Selection
        # Stepwise Regression
        DataSubset = pd.DataFrame(self.df, columns=['NumberLikes24', 'CommentCount24', 'PostLength', 'PostShareCount', 'PostSaturday'])

        # Linear model designed by forward regression
        def forward_selected(data, response):
            remaining = set(data.columns)
            remaining.remove(response)
            selected = []
            current_score, best_new_score = 0.0, 0.0
            while remaining and current_score == best_new_score:
                scores_with_candidates = []
                for candidate in remaining:
                    formula = '{} ~ {} + 1'.format(response, ' + '.join(selected + [candidate]))
                    score = smf.ols(formula, data).fit().rsquared_adj
                    scores_with_candidates.append((score, candidate))
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates.pop()
                if current_score > best_new_score:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
            formula = '{} ~ {} + 1'.format(response, ' + '.join(seleted))
            model = smf.ols(formula, data).fit()
            return model

        model = forward_selected(DataSubset, 'NumberLikes24')
        print('Model Selected')
        print(model.model.formula)
        print('R2=', model.rsquared_adj)


if __name__ == "__main__":
    assignment = LinearRegression('somefile.txt')
    assignment.clean()
    assignment.summarize()
    assignment.regression()
