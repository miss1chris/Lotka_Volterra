######## Regression Trees###########################################
import joblib
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error  # RMSE
from sklearn.metrics import r2_score  # R-squared (R^2)
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


# Frame a time series as a supervised learning dataset.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=100, max_features=3)
    #   random_state=0, n_estimators=100, max_depth=None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False
    model.fit(trainX, trainy)
    # make a one-step prediction

    filename = 'finalized_model.sav'
    joblib.dump(model, filename)
    yhat = model.predict([testX])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>original=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error (MAE)
    error = mean_absolute_error(test[:, -1], predictions)
    # caculator RMSE
    mse = mean_squared_error(test[:, -1], predictions)
    rmse = sqrt(mse)
    # caculator R-squared (R^2)
    global r2_score
    r2_score = r2_score(test[:, -1], predictions)
    return error, rmse, r2_score, test[:, -1], predictions


# load the dataset  parse_dates=True, index_col="Date"
series = read_csv('./X.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
print(series.head())
values = series.values
# values = values.astype('float32')
# transform the time series data into supervised learning
data = series_to_supervised(values, 1, 1)
data.drop(data.columns[[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], axis=1, inplace=True)
print(data.head())
print(data.shape)
data = data.values
print(data)
# print(type(data))
# data=data.to_numpy()
print("---------------")

# evaluate
mae, rmse, r2_score, y, yhat = walk_forward_validation(data, 169)
int(len(data) * 0.3)
print('MAE: %.3f' % mae)
print('Test RMSE: %.3f' % rmse)
print('Test R^2: %.3f' % r2_score)
# plot expected vs predicted
pyplot.plot(y, label='Original')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.xlabel('Test Size')
pyplot.ylabel('Rainfall (mm)')
pyplot.show()
