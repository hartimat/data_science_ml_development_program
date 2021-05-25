###############################################################################
# FILENAME: TimeSeries.py
# AUTHOR: Matthew Hartigan
# DATE: 20-May-2021
# DESCRIPTION: A python script to run time series analysis.
###############################################################################

# IMPORTS
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.tsa.api import VAR
import datetime
import dateutil
import pmdarima as pm

class TimeSeries:

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
        print()

    def visualize(self):
        df = pd.read_csv('auscafe.csv')
        fig = px.line(df, x='Date', y='auscafe', title='Test Title')
        fig.update_xaxes(nticks=40)
        fig.show()

    def deocmpose(self):
        dataset = pd.read_csv('Raotbl6.csv', index_col=0)
        num_vars = dataset.shape[1]
        fig, axes = pyplot.subplots(num_vars, 1, figsize=(16, 12))
        for i in range(num_vars):
            col = dataset.columns[i]
            axes[i].plot(dataset[col])
            axes[i].set_xticks([], [])
            axes[i].set_title(col)

        # Apply VAR model to data set, summarize
        model = VAR(dataset)
        var_selected = model.select_order(maxlags=10)
        print(var_selected.summary())

        # Select model with highest AIC value
        model_fitted = model.fit(1)
        print(model_fitted.summary())
        lag_order = model_fitted.k_ar
        forecast_input = dataset.values[-lag_order:]
        quarters_to_predict = 24
        predicted_values = model_fitted.forecast(forecast_input, quarters_to_predict)
        dataset_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dataset.index]
        last_date = dataset_dates[-1]
        predicted_dates = []
        for _ in range(quarters_to_predict):
            next_date = last_date + dateutil.relativedelta.relativedelta(months=3)
            predicted_dates.append(next_date)
            last_date = next_date
        predicted_index = [date.strftime('%Y-%m-%d') for date in predicted_dates]

        fig, axes = pyplot.subplots(num_vars, 1, figsize=(16, 12))
        for i in range(num_vars):
            col = dataset.columns[i]
            axes[i].plot(dataset[col], color='blue')
            axes[i].plot(predicted_index, predicted_values[:, i], color='green')
            axes[i].set_xticks([], [])
            axes[i].set_title(col)
        pyplot.show()

        fit = pm.auto_arima(dataset['gdfce'], seasonal=True, stepwise=True, error_action='ignore', m=12, max_order=6)
        print(fit.summary())


if __name__ == '__main__':
    assignment = TimeSeries('Raotbl6.csv')
    assignment.clean()
    assignment.summarize()
    assignment.visualize()
    assignment.deocmpose()
    assignment.var()
