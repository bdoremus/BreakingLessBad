import requests, pickle
import pandas as pd
import numpy as np
import datetime
from math import floor
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

# Libraries for pickled model
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

def get_weather_forecast():
    forecast = pd.DataFrame(columns=['date', 'businessUnit', 'maxTemp', 'minTemp', 'precip'])

    bu_url = {'anadarko': 'https://api.weather.gov/gridpoints/AMA/100,59',
              'arkoma': 'https://api.weather.gov/gridpoints/TSA/65,50',
              'durango': 'https://api.weather.gov/gridpoints/GJT/118,12',
              'easttexas': 'https://api.weather.gov/gridpoints/SHV/50,61',
              'farmington': 'https://api.weather.gov/gridpoints/ABQ/63,198',
              'wamsutter': 'https://api.weather.gov/gridpoints/RIW/127,43'}

    for bu, url in bu_url.items():
        print("Downloading forecast for:",bu)
        f_json = requests.get(url).json()

        # Check for errors in getting the info from the url
        if 'status' in f_json:
            print("API Load Error")
            print("Business Unit: ",bu)
            print("Error: ",forecast_list['status'])
            print("       ",forecast_list['detail'])
            continue

        # list of dicts with 'validTime' and 'value'
        # [{"validTime": "2017-10-02T09:00:00+00:00/PT2H", "value": 19.444444444445}]
        maxTemperatures = {}
        maxT_list = f_json['properties']['maxTemperature']['values']
        for i, entry in enumerate(maxT_list):
            # Take first 10 digits of date to get only 'year-mo-dy'
            date = datetime.datetime.strptime(entry['validTime'][:10], "%Y-%m-%d").date()

            # Convert the temp from Celcius to Fahrenheit
            temp = round(entry['value']*9/5+32)

            maxTemperatures[date] = temp


        # Take first 10 digits of date to get only 'year-mo-dy'
        # Convert the temp from Celcius to Fahrenheit
        # Make a dictionary of {date: minTemp}
        minT_list = f_json['properties']['minTemperature']['values']
        minTemperatures = {datetime.datetime.strptime(entry['validTime'][:10], "%Y-%m-%d").date(): round(entry['value']*9/5+32) for entry in minT_list}

        # minTemperatures = {}
        # for i, entry in enumerate(minT_list):
        #     # Take first 10 digits of date to get only 'year-mo-dy'
        #     date = datetime.datetime.strptime(entry['validTime'][:10], "%Y-%m-%d").date()
        #
        #     # Convert the temp from Celcius to Fahrenheit
        #     temp = round(entry['value']*9/5+32)
        #
        #     minTemperatures[date] = temp

        precipitations = {}
        precip_list = f_json['properties']['quantitativePrecipitation']['values']
        for i, entry in enumerate(precip_list):
            # Take first 10 digits of date to get only 'year-mo-dy'
            date = datetime.datetime.strptime(entry['validTime'][:10], "%Y-%m-%d").date()

            # Convert the amount from mm to inches
            amount = round(entry['value']/25.4, 2)

            # sum together all precip on a day
            if date in precipitations:
                precipitations[date] = round(precipitations[date]+amount, 2)
            else:
                precipitations[date] = amount

        # Add the forecast for this business unit
        dates = set(list(maxTemperatures.keys()) + list(minTemperatures.keys()) + list(precipitations.keys()))
        for date in dates:
            maxT = maxTemperatures[date] if date in maxTemperatures else np.nan
            minT = minTemperatures[date] if date in minTemperatures else np.nan
            precip = precipitations[date] if date in precipitations else 0
            forecast = forecast.append([{'date': date,
                                         'businessUnit': bu,
                                         'maxTemp': maxT,
                                         'minTemp': minT,
                                         'precip': precip}])
    print("Forecasted dates from {} to {}".format(forecast.date.min(), forecast.date.max()))
    return forecast


def forecast_failures(weather_forecast, model, day_group, sc, plot=False):
    # Create X
    X = create_X(weather_forecast, day_group)

    # If it's a gaussian process, scale the data (built in to LinearRegression pipeline)
    if sc is not None:
        X = pd.DataFrame(sc.transform(X), columns=X.columns, index=X.index)


    # Predict Y
    if model.__class__.__name__ == "Pipeline":
        y_pred = model.predict(X)
        sigma = None
    else:
        # Gausian Process - plot with confidence bounds
        y_pred, sigma = model.predict(X, return_std=True)

    # If predicted less than 0, set to 0
    y_pred[y_pred < 0] = 0

    if plot:
        modelname = "Gaussian Process" if sc is not None else "Linear Regression with polynomial degree {}".format(model.steps[1][1].degree)
        graph_model(X, y_pred, sigma, modelname)

    return X, y_pred, sigma


def create_X(weather_forecast, day_group):
    X = weather_forecast.copy()

    X = pd.get_dummies(X, columns=['businessUnit'], prefix='', prefix_sep='', drop_first=False)
    X.drop('anadarko', axis=1, inplace=True) # drop one of the dummy columns

    # Fill nans
    X.precip.fillna(0, inplace=True)
    X.fillna(method='pad', inplace=True)
    X.fillna(method='bfill', inplace=True) # in case the first date is empty
    X['avgTemp'] = (X.maxTemp + X.minTemp)/2

    # Find the start/end dates of the forecast
    # Not every business unit forecast is the same length
    minDate = datetime.date.min
    maxDate = datetime.date.max
    bu_masks, ignore = get_bu_masks(X)
    for bu_mask in bu_masks:
        bu_maxDate = X.date[bu_mask].max()
        bu_minDate = X.date[bu_mask].min()

        if maxDate > bu_maxDate:
            maxDate = bu_maxDate
        if minDate < bu_minDate:
            minDate = bu_minDate

    # Limit the forecast window to be the same in all business units
    X = X[(X.date >= minDate) & (X.date <= maxDate)]

    # How many day groups exist in the forecast?
    numGroups = floor(((maxDate - minDate).days + 1) / day_group)
    # Ensure there is at least one group
    numGroups = max(numGroups, 1)

    # Create groups of size: day_group
    data = pd.DataFrame(columns=['date', 'maxTemp', 'avgTemp', 'minTemp', 'precip', 'arkoma', 'durango', 'easttexas', 'farmington', 'wamsutter'])
    bu_masks, businessUnits = get_bu_masks(X)
    for i in range(numGroups):
        date_start = minDate + datetime.timedelta(days=(i*day_group))
        date_end = minDate + datetime.timedelta(days=((i+1)*day_group))

        # For the given date range, create an entry for each business unit separately
        for bu_mask, bu in zip(bu_masks, businessUnits):
            w = X[(bu_mask) & (X.date >= date_start) & (X.date < date_end)]

            data = data.append([{'date': date_start,
                                 'maxTemp': w.maxTemp.max(),
                                 'avgTemp': w.avgTemp.mean(),
                                 'minTemp': w.minTemp.min(),
                                 'precip': w.precip.sum(),
                                 'arkoma': 1 if bu == 'arkoma' else 0,
                                 'durango': 1 if bu == 'durango' else 0,
                                 'easttexas': 1 if bu == 'easttexas' else 0,
                                 'farmington': 1 if bu == 'farmington' else 0,
                                 'wamsutter': 1 if bu == 'wamsutter' else 0}])

    # change index to the date
    data.set_index(pd.DatetimeIndex(data.date), inplace=True)
    data.drop('date', axis=1, inplace=True)

    return data


'''
'''
def graph_model(X, y, sigma, modelname):
    fig = plt.figure()

    # Create list of masks to only show one business unit at a time
    bu_masks, businessUnits = get_bu_masks(X)
    for axisNum, (bu_mask, bu) in enumerate(zip(bu_masks, businessUnits)):
        ax = fig.add_subplot(3, 2, axisNum+1)
        y_vals = y[bu_mask]

        x_vals = X.index[bu_mask]

        if sigma is not None:
            # if GP model has sigma
            err = sigma[bu_mask]
            ax.bar(x_vals, y_vals, label='Predictions +/-1 std', yerr=err)
            ax.set_ylim(0, max((y_vals+err).max()*1.1, 0.01))
            ax.legend(loc='upper right')
        else:
            # Linear Regression model with no sigma
            ax.bar(x_vals, y_vals, label='Predictions')
            ax.set_ylim(0, max(y.max()*1.25, 0.01))

        # Format x axis to be yyyy-mm-dd
        yearsFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        ax.set_ylabel('Proprortion of all compressors that will fail')
        ax.set_title("Predicted compressor failure proportions in "+bu.upper())

    plt.suptitle('Predicted number of failures in each busines unit using a {} model'.format(modelname))
    plt.show()

    return None



def get_bu_masks(X):
    businessUnits = ['arkoma', 'durango', 'easttexas', 'farmington', 'wamsutter'] # anadarko was dropped as a dummy
    bu_masks = [(X[bu] > 0).values for bu in businessUnits]
    # add a mask for anadarko, which was dropped as a dummy
    bu_masks.append((X['arkoma'] <= 0) &\
              (X['durango'] <= 0) &\
              (X['easttexas'] <= 0) &\
              (X['farmington'] <= 0) &\
              (X['wamsutter'] <= 0))

    # Re-add anadarko to the list of business units
    businessUnits.append('anadarko')

    return bu_masks, businessUnits


if __name__ == '__main__':
    # Choices in how to run the file
    new_forecast = False
    model_types = ['GaussianProcessRegressor', 'n^1 LinearRegression', 'n^2 LinearRegression']
    model_choice_index = 2
    day_group_size = 7
    plot_results = True

    # Load the forecast
    if new_forecast:
        weather_forecast = get_weather_forecast()
        with open('./data/weather_forecast.pkl', 'wb') as f:
            pickle.dump(weather_forecast, f)
            print("Weather Forecast saved as './data/weather_forecast.pkl'")
    else:
        with open('./data/weather_forecast.pkl', 'rb') as f:
            weather_forecast = pickle.load(f)
    print("\nWeather Forecast:\n",weather_forecast.sort_values(['businessUnit', 'date']))

    # Load the model.  If it's Gaussian Process, load a scaler as well
    model = None
    sc = None
    # pick the right model file to load
    if model_choice_index == 0:
        # Gaussian Process
        print("\nModel used for predictions: Gaussian Process")
        model_file = './model_gp.pkl'
        with open('./scaler_gp.pkl', 'rb') as f_scaler:
            sc = pickle.load(f_scaler)
    elif (model_choice_index == 1) or (model_choice_index == 2):
        # n^1 or n^2 Linear Regression
        print("Model used for predictions: n^{} Linear Regression".format(model_choice_index))
        model_file = './model_n{}.pkl'.format(model_choice_index)


    with open(model_file, 'rb') as f_model:
        model = pickle.load(f_model)


    X, y_pred, sigma = forecast_failures(weather_forecast, model, day_group_size, sc, plot_results)
    bu_masks, businessUnits = get_bu_masks(X)
    print("\nPredicted proprortion of failures for the next {} day{}:".format(day_group_size, ("s" if day_group_size > 1 else "")))
    for bu, failureRate in zip(businessUnits, y_pred):
        print("\t{:.3f}: {}".format(failureRate, bu))
