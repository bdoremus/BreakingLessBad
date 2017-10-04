import numpy as np
import pandas as pd
import datetime, pickle
from math import floor
pd.options.mode.chained_assignment = None  # Turn off warnings for setting a slice (default='warn')


def load_create_fromDefaultLocations(print_summary=False):
    sfa_dirty = load_SurfaceFailureActions('./data/BPDataSurfaceFailureActions.csv', print_summary)
    bu = load_BusinessUnits('./data/Asset_to_Business_Unit_Translator.csv')
    sfa = check_businessUnitLatLon(sfa_dirty, bu)

    nf = load_NotFailures('./data/compressors_by_business_unit.csv')
    weather = load_weather('./data/Weather/', nf.businessUnit.unique())

    if print_summary:
        print("Surface Failures")
        print(sfa.head(1).T)
        print("-----------------------")
        print("\nBusiness Units\n")
        print(bu.head(1).T)
        print("-----------------------")
        print("\nNot Failures\n")
        print(nf.head(1).T)
        print("-----------------------")
        print("\nSurface Failures Checked\n")
        print(sfa.head(1).T)

    return sfa, nf, weather

################################
#### Surface Failure Action ####
def load_SurfaceFailureActions(fileName, print_summary=False):
    df = pd.read_csv(fileName, low_memory=False)

    # Only look at Compressor Failures
    df = df.loc[(df.surfaceFailureComponent.str.lower().str.contains('compressor')) | \
            (df.surfaceFailureComponentOther.str.lower().str.contains('compressor')) | \
            (df.surfaceFailureSubComponent.str.lower().str.contains('compressor')) | \
            (df.surfaceFailureSubComponentOther.str.lower().str.contains('compressor'))]

    # Set assetIds and assetName to all lowercase
    df.assetId = df.assetId.str.lower()
    df.assetName = df.assetName.str.lower()

    # Extract usable columns from assetId
    df = sfa_interpret_assetId(df)

    # Reduce replicateGroups to standard businessUnits
    df = sfa_fixReplicateGroups(df)

    # Rename columns to universal standards
    column_translator = {\
        'latitude': 'lat',
        'longitude': 'lon',
        'createdDate': 'dateWorkOrder',
        'surfaceFailureDate': 'dateFailure',
        'surfaceFailureType': 'failureType',
        'replicateGroup': 'businessUnit'}
    df.rename(columns=column_translator, inplace=True)

    # Convert date columns from strings to dates
    df.dateFailure = pd.to_datetime(df.dateFailure).dt.date

    # Choose the output columns we want
    identification_cols = ['assetId', 'api', 'wellFlac', 'businessUnit']
    geolocation_cols = ['lat', 'lon']
    feature_cols = []
    date_cols = ['dateFailure']
    output_columns = identification_cols + geolocation_cols + feature_cols + date_cols

    return df[output_columns]


def sfa_interpret_assetId(df):
    df.assetId.fillna("", inplace=True)

    # Remove the 'asset-' prefix, attempt to convert to integer
    df['assetId_r'] = ""
    df.loc[:, 'assetId_r'] = df.assetId.str.replace('asset-','', case=False)
    df['assetIdInt'] = pd.to_numeric(df.assetId_r, errors='coerce').fillna(0).astype(np.int64)

    # The assetId is a wellFlac if it is a seven digit number
    df['wellFlac'] = np.nan
    df.loc[(df.assetId_r.str.len() == 6) & (df.assetIdInt != 0), 'wellFlac'] = df.assetIdInt

    # The assetId is a small API number if it is a ten digit number
    df['api'] = np.nan
    df.loc[(df.assetIdInt != 0) & (df.assetId_r.str.len() >= 10) & (df.assetId_r.str.len() <= 14), 'api'] = df.assetIdInt

    # The assetId is uncategorized if it was none of the above
    df['uncategorized'] = np.nan
    unresolved = (df.wellFlac.isnull() & df.api.isnull())
    df.loc[unresolved, 'uncategorized'] = df.assetId_r

    return df


def sfa_fixReplicateGroups(df):
    # Fill empty columns with blank strings
    df.replicateGroup.fillna("", inplace=True)

    # Create groups by Business Unit
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('farmington')] = 'farmington'
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('easttexas')] = 'easttexas'
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('anadarko')] = 'anadarko'
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('wamsutter')] = 'wamsutter'
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('durango')] = 'durango'
    df.replicateGroup[df.replicateGroup.str.lower().str.contains('arkoma')] = 'arkoma'
    df.replicateGroup[df.replicateGroup.str.lower() == 'bp'] = 'arkoma'

    # Group anything else into 'other'
    df.replicateGroup[(df.replicateGroup != 'farmington') &\
         (df.replicateGroup != 'easttexas') &\
         (df.replicateGroup != 'anadarko') &\
         (df.replicateGroup != 'wamsutter') &\
         (df.replicateGroup != 'durango') &\
         (df.replicateGroup != 'arkoma')]\
         = 'other'

    return df
#### Surface Failure Action ####
################################

################################
#### Compressors Not Failed ####
def load_NotFailures(fileName):
    df = pd.read_csv(fileName)

    # lowercase column names
    df.columns = df.columns.str.lower()

    # Only keep those with status '99=OPERATING', 'OPERATING', 'Assumed-Operating', '99-NOCNT', or 'NOCNT'
    df = df.loc[df.status.str.lower().str.contains('operating') | df.status.str.lower().str.contains("nocnt")]

    # Change string entries to all lowercase
    df.businessunit = df.businessunit.str.lower()

    # strip whitespace from names ('east texas' -> 'easttexas')
    df.businessunit = df.businessunit.str.replace(' ', '')

    # Rename columns to universal standards
    column_translator = {\
        'pra api': 'api',
        'latitude': 'lat',
        'longitude': 'lon',
        'businessunit': 'businessUnit',
        'eqpmnt_tot_hrs': 'hrsOn'}
    df.rename(columns=column_translator, inplace=True)

    # Choose the output columns we want
    identification_cols = ['location', 'flac', 'api', 'businessUnit']
    feature_cols = ['hrsOn'] # 'manufacturer', 'model', 'hp', 'engineHP', 'wellType',

    df = df[identification_cols + feature_cols]

    return df


#### Compressors Not Failed ####
################################

################################
####     Business Units     ####
def load_BusinessUnits(fileName):
    # These numbers were taken from 2016 CDM (Clean Development Mechanism) environmental reports
    # They were separate files laden with GUI elements, and were manually combined into one csv
    # Most pruning was done by hand
    df = pd.read_csv(fileName)

    # Change text columns to lowercase
    df._id = df._id.str.lower()
    df.businessUnit = df.businessUnit.str.lower()

    # Remove spaces from businessUnits
    df.businessUnit = df.businessUnit.str.replace(' ', '')

    # Rename columns to universal standards
    df.columns = df.columns.str.lower()
    df.rename(index=str, columns={'_id': 'assetId', 'latitude': 'lat', 'longitude': 'lon', 'businessunit': 'businessUnit'}, inplace=True)

    # Choose the output columns we want
    identification_cols = ['assetId', 'businessUnit', 'lat', 'lon']
    df = df[identification_cols]

    return df

'''
Fix sfa replicateGroup to be the real businessUnit
'''
def check_businessUnitLatLon(sfa, bu):
    # Based on assetId, get the official Business Unit and lat/lon
    # left outer join of sfa onto bu.
    # Seems to fail on assetIds that are well names.  **********************************
    combined = pd.merge(sfa, bu, on='assetId', how='left', sort=False, suffixes=('', '_bu'), indicator=False)

    # If data existed in both locations, take the info from the bu dataframe
    # Overwrite the replicateGroup with the true business unit of the asset
    combined.loc[combined.businessUnit_bu.notnull(), 'businessUnit'] = combined.businessUnit_bu
    combined.drop('businessUnit_bu', axis=1, inplace=True)

    # Overwrite potentially erroneous lat/longs with the asset lat/lon
    combined[['lat', 'lon']][combined.lat_bu.notnull() & combined.lon_bu.notnull()] = combined[['lat_bu', 'lon_bu']]
    combined.drop('lat_bu', axis=1, inplace=True)
    combined.drop('lon_bu', axis=1, inplace=True)

    # Drop any rows that do not have a businessUnit
    combined = combined[combined.businessUnit != 'other']

    return combined
####     Business Units     ####
################################

def load_weather(folder, businessUnits):
    # Create a dataframe entry for each business unit's day
    weather = pd.DataFrame(columns=['DATE', 'businessUnit', 'TMAX', 'TMIN', 'PRCP', 'ELEVATION'])
    for bu in businessUnits:
        file_location = folder + bu + '.csv'
        df = pd.read_csv(file_location)

        # Convert date strings to datetime
        df.DATE = pd.to_datetime(df.DATE).dt.date

        # Fill precip Nans with 0
        df.PRCP.fillna(0, inplace=True)

        # Fill Temperature and Elevation nans with the last good date
        df.fillna(method='pad', inplace=True)

        # Create a column for the avg Temperature
        df['TAVG'] = (df.TMAX + df.TMIN)/2

        # Create a column for the businessUnit
        df['businessUnit'] = bu

        # Add the data for this businessUnit to the other weather info
        weather = pd.concat([weather, df], ignore_index=True)

    return weather


################################
####      Create X, y       ####

def create_Xy_fromDefaultLocations():
    combined, nf, weather = load_create_fromDefaultLocations()
    return create_percentFailedByDate(combined, nf, weather)


def create_percentFailedByDate(sfa, nf, weather, grp_size=1):
    # Take only select columns to predict on
    identification_cols = ['assetId', 'businessUnit']
    date_cols = ['dateFailure']
    sfa = sfa[identification_cols + date_cols]

    # Because each business unit started recording this information at a different time,
    # Find the largest min date to begin at and the smallest max date to end at
    minDate = datetime.date.min
    maxDate = datetime.date.max
    for bu in sfa.businessUnit.unique():
        bu_maxDate = sfa.dateFailure[sfa.businessUnit == bu].max()
        bu_minDate = sfa.dateFailure[sfa.businessUnit == bu].min()

        if maxDate > bu_maxDate:
            maxDate = bu_maxDate
        if minDate < bu_minDate:
            minDate = bu_minDate

    # Limit the input by the dates found above
    sfa = sfa[(sfa.dateFailure >= minDate) & (sfa.dateFailure <= maxDate)]
    weather = weather[(weather.DATE >= minDate) & (weather.DATE <= maxDate)]

    # Create a dictionary of businessUnit: daily avg number of compressors in the businessUnit
    # 1) Find the total number of compressor-hours over the year, then
    # 2) Divide by the number of hours in the year (it was a leap year)
    numUnits = (nf[['hrsOn', 'businessUnit']].groupby('businessUnit').sum()/366/24).to_dict()['hrsOn']

    # Create the dataframe to hold all features and target
    data = pd.DataFrame(columns=['date', 'businessUnit', 'failureRate', 'maxTemp', 'avgTemp', 'minTemp', 'precip'])

    # Create a data point for each group of days
    numDays = (maxDate-minDate).days
    numGroups = floor(numDays/grp_size)
    for i in range(numGroups):
        date_start = minDate + datetime.timedelta(days=(i*grp_size))
        date_end = minDate + datetime.timedelta(days=((i+1)*grp_size))

        # For each group of days, create a separate data point for each business unit
        for bu in nf.businessUnit.unique():
            # Find how many failures in the business unit on this date
            mask_buAndDate = (sfa.businessUnit == bu)\
                             & (sfa.dateFailure >= date_start)\
                             & (sfa.dateFailure < date_end)
            numFailures = sfa[mask_buAndDate].shape[0]
            failureRate = numFailures/numUnits[bu]

            # Get the weather in the business unit on this date
            w = weather[(weather.businessUnit == bu)\
                        & (weather.DATE >= date_start)\
                        & (weather.DATE < date_end)]

            # Create one row of data
            data = data.append([{'date': date_start,
                                 'businessUnit':bu,
                                 'failureRate': failureRate,
                                 'maxTemp': w.TMAX.max(),
                                 'avgTemp': w.TAVG.mean(),
                                 'minTemp': w.TMIN.min(),
                                 'precip': w.PRCP.sum()}])

    data = pd.get_dummies(data, columns=['businessUnit'], prefix='', prefix_sep='', drop_first=False)
    data.drop('anadarko', axis=1, inplace=True) # drop one of the dummy columns

    # change index to the date
    data.set_index(pd.DatetimeIndex(data.date), inplace=True)
    data.drop('date', axis=1, inplace=True)

    return data


if __name__ == '__main__':
    save_result = True
    print_summary = True
    num_days_in_group = 7
    sfa, nf, weather = load_create_fromDefaultLocations(print_summary)
    data = create_percentFailedByDate(sfa, nf, weather, grp_size=num_days_in_group)

    if save_result:
        # Create and pickle the data
        filename = './data/data_{}day.pkl'.format(num_days_in_group)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            print("\nFile written as:",filename)
