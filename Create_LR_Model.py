import pickle, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from matplotlib import pyplot as plt

##########################

def main(max_polynomial=1, save_model=False, num_days_in_group=1, print_model=False):
    X_train, X_test, y_train, y_test, X, y = loadPickledData(num_days_in_group)
    # PolynomialFeatures(degree=2, include_bias=False),
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=max_polynomial, include_bias=False), LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1))
    model.fit(X_train, y_train)

    score_model(model, X_test, y_test)

    # Print the coefficients
    if print_model:
        coefs = model.steps[2][1].coef_
        x_int = model.steps[2][1].intercept_
        coef_total = sum(abs(coefs))+abs(x_int)
        poly_names = model.steps[1][1].get_feature_names(X_train.columns)
        print("Coefficients:")
        print("| X_intercept (anadarko) | {:.2f}% |  ".format(100*x_int/coef_total))
        for n, c in zip(poly_names, coefs):
            if True:#abs(c/coef_total) > 0.01:
                n = str.strip(n)
                n = n.replace(" ", " * ")
                print("| {} | {:.2f}% |  ".format(n, 100*c/coef_total))

    if save_model:
        filename = './model_n{}.pkl'.format(max_polynomial)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            print("Model saved as ",filename)

    if print_model:
        graph_model(X, y, model)
    return None


def loadPickledData(num_days_in_group):
    filename = './data/data_{}day.pkl'.format(num_days_in_group)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        y = X.pop('failureRate').values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42, stratify=(y>0))

    return X_train, X_test, y_train, y_test, X, y


def score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # If predictions are less than 0, set them to 0
    y_pred[y_pred<0] = 0

    print("Mean Absolute Error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))
    print("Root Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)**0.5))
    print("R^2 Score: {:.2f}".format(r2_score(y_test, y_pred))) # If predicts the mean, score=0.  If perfect, score=1.  If worse than mean, negative.
    return None
##########################

def crossval_trials():
    X_train, X_test, y_train, y_test, X, y = loadPickledData()

    # Scale X
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test - scaler.transform(X_test)

    # Models to test using cross validation
    linear = LinearRegression()
    enet = ElasticNet()
    poly2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept=True, normalize=False))
    poly2_e = make_pipeline(PolynomialFeatures(degree=2), ElasticNet(fit_intercept=True, normalize=False))
    poly3 = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=True, normalize=False))
    poly3_e = make_pipeline(PolynomialFeatures(degree=3), ElasticNet(fit_intercept=True, normalize=False))
    dt = DecisionTreeRegressor()
    ab = AdaBoostRegressor()
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()

    model_types = [linear, enet, poly2, poly2_e, poly3, poly3_e, dt, ab, rf, gb]

    # Cross val scores for each model
    print("| Type | r2 | explained variance | Mean Abs Err | RMSE |  ")
    print("| ---- | -- | ------------------ -------------- | ---- | ")
    for m in model_types:
        r2 = cross_val_score(m, X_train, y_train, cv=3, scoring='r2', n_jobs=-1).mean()
        mae = -1 * cross_val_score(m, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
        mse = cross_val_score(m, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1).mean()
        rmse = (-1 * mse)**0.5

        if m.__class__.__name__ == 'Pipeline':
            name = m.steps[1][1].__class__.__name__ + " n^" + str(m.steps[0][1].degree)
        else:
            name = m.__class__.__name__
        name.replace('Regressor', '')

        print("| {} | {:.2f} | {:.2f} | {:.2f} |  ".format(name, r2, mae, rmse))


def graph_model(X, y, model):
    y_pred = model.predict(X)
    # If predictions are less than 0, set them to 0
    y_pred[y_pred<0] = 0

    fig = plt.figure()
    bu_masks, businessUnits = get_bu_masks(X)
    for aNum, (bu_mask, bu) in enumerate(zip(bu_masks, businessUnits)):
        ax = fig.add_subplot(3, 2, aNum+1)
        y_vals = y[bu_mask]
        y_pred_vals = y_pred[bu_mask]
        x_vals = X.index.date[bu_mask]

        ax.plot(x_vals, y_vals, 'r', label='true')
        ax.plot(x_vals, y_pred_vals, 'b', label='Prediction')

        ax.set_ylabel('Proportion of all compressors that fail')
        ax.set_title("Compressor failures in "+bu.upper())
        ax.set_ylim(0, y.max()*1.1)
        ax.legend(loc='upper right')
    plt.suptitle("Proportion of compressors that fail in each business unit\nPredictions using LinearRegression with a polynomial degree of {}".format(model.steps[1][1].degree))
    plt.show()

    return None


def get_bu_masks(X):
    businessUnits = ['arkoma', 'durango', 'easttexas', 'farmington', 'wamsutter'] # anadarko was dropped as a dummy
    bu_masks = [(X[bu] > 0) for bu in businessUnits]
    # add a mask for anadarko
    bu_masks.append((X['arkoma'] <= 0) &\
              (X['durango'] <= 0) &\
              (X['easttexas'] <= 0) &\
              (X['farmington'] <= 0) &\
              (X['wamsutter'] <= 0))
    businessUnits.append('anadarko')
    return bu_masks, businessUnits

if __name__ == '__main__':
    # Options for running this file
    save_model=True
    print_model=False
    max_polynomial = 2
    num_days_in_group = 7

    # crossval_trials()
    main(max_polynomial, save_model, num_days_in_group, print_model)
