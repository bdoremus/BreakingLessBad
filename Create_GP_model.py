from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import pickle, pandas as pd
pd.options.mode.chained_assignment = None  # Turn off warnings for setting a slice (default='warn')


'''
'''
def loadPickledData(num_days_in_group):
    filename = './data/data_{}day_noe.pkl'.format(num_days_in_group)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        y = X.pop('failureRate').values

    return X, y


'''
'''
def create_data(num_days_in_group):
    X, y = loadPickledData(num_days_in_group)

    # Process X and split
    X_data = X.copy()
    # X_data.drop('date', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size = 0.2, shuffle = True, random_state = 42, stratify=(y>0))
    sc = StandardScaler()

    # Scale everything, including dummies
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test =  pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
    X_data =  pd.DataFrame(sc.transform(X_data), columns=X_data.columns, index=X_data.index)

    return X_train, X_test, y_train, y_test, X, y, X_data, sc


'''
'''
def create_GPR(X_train, X_test, y_train, y_test):
    # Weekly R^2 = 0.6126
    # Daily R^2 = 0.3641
    kernel = RationalQuadratic() + 0.1 * WhiteKernel()
    # kernel = RationalQuadratic(length_scale=1e1,
    #                            alpha=1.0,
    #                            length_scale_bounds=(1e0, 1e2),
    #                            alpha_bounds=(1e-02, 1e2))\
    #          + 0.1*WhiteKernel(noise_level=1e0,
    #                            noise_level_bounds=(1e0, 1e0))

    # Weekly R^2 = 0.5696
    # Daily R^2 = 0.3629
    # kernel = Matern(length_scale=1e1, length_scale_bounds=(1e-03, 1e2), nu=2.5)+ 0.1*WhiteKernel(noise_level=1e1, noise_level_bounds=(1e-1, 1e1))

    # Weekly R^2 = 0.5572
    # Daily R^2 0.3726
    # kernel = RBF(length_scale=1e1, length_scale_bounds=(1e0, 1e2)) + 0.1*WhiteKernel(noise_level=1e1, noise_level_bounds=(1e-1, 1e1))

    # Weekly R^2 = 0.5189
    # Daily R^2 = 0.3738
    # kernel = ExpSineSquared(length_scale=1e1, periodicity=365, length_scale_bounds=(1e0, 1e2), periodicity_bounds=(1e0, 1e3)) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-3, 8))

    # Weekly R^2 = 0.4371
    # Daily R^2 = 0.2897
    # kernel = DotProduct(sigma_0=1, sigma_0_bounds=(1e-02, 1e1)) + 0.1*WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-2, 1e0))

    # Weekly R^2 = 0.3866
    # Daily R^2 = 0.0300
    # kernel = ConstantKernel(constant_value=1e-1, constant_value_bounds=(1e-5, 1e-1)) * RBF(length_scale=1e1, length_scale_bounds=(1e0, 1e1))+ WhiteKernel(noise_level=1e1, noise_level_bounds=(1e-1, 1e0))

    # Create GP model
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("R2: {:.4f}".format(r2_score(y_test, y_pred)))
    print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred)))
    print("RMSE: {:.4f}".format(mean_squared_error(y_test, y_pred)**0.5))

    return model


'''
'''
def create_pred(X, model):
    return model.predict(X, return_std=True)

'''
'''
def plot_confidence_interval(X, y, y_pred, sigma):
    fig = plt.figure()
    businessUnits = ['arkoma', 'durango', 'easttexas', 'farmington', 'wamsutter'] # anadarko was dropped as a dummy
    bu_masks = [(X[bu] > 0) for bu in businessUnits]
    # add a mask for anadarko
    bu_masks.append((X['arkoma'] <= 0) &\
              (X['durango'] <= 0) &\
              (X['easttexas'] <= 0) &\
              (X['farmington'] <= 0) &\
              (X['wamsutter'] <= 0))
    businessUnits.append('anadarko')
    for aNum, (bu_mask, bu) in enumerate(zip(bu_masks, businessUnits)):
        ax = fig.add_subplot(3, 2, aNum+1)
        y_vals = y[bu_mask]
        y_pred_vals = y_pred[bu_mask]
        x_vals = X.index[bu_mask]

        ax.plot(x_vals, y_vals, 'r', label='true')
        ax.plot(x_vals, y_pred_vals, 'b', label='Prediction')
        ax.fill_between(x_vals,
                 y_pred[bu_mask] - 1.96 * sigma[bu_mask],
                 y_pred[bu_mask] + 1.96 * sigma[bu_mask],
                 facecolor='blue', alpha=.1, label='95% confidence interval')
        ax.fill_between(x_vals,
                 y_pred[bu_mask] - 0.67449 * sigma[bu_mask],
                 y_pred[bu_mask] + 0.67449 * sigma[bu_mask],
                 facecolor='yellow', alpha=.4, label='50% confidence interval')

        ax.set_ylim(0, y.max()*1.25)
        # ax.set_ylim(0, y[bu_mask].max()*1.25)
        ax.set_ylabel('Proporition of all compressors that fail')
        ax.set_title("Compresor Failures in " + bu.upper)
        ax.legend(loc='upper right')

    fig.tight_layout()
    plt.suptitle("Proportion of compressors that fail in each business unit\nPredictions using a Gassian Process with two kernels: Rational Quadratic and White Noise")
    plt.show()
    return None


'''
'''
if __name__ == '__main__':
    create_graph = False
    only_test = False
    save_model = False
    num_days_in_group = 7
    X_train, X_test, y_train, y_test, X, y, X_data, sc = create_data(num_days_in_group)

    # Create kernel and GPR mode
    model = create_GPR(X_train, X_test, y_train, y_test)
    if save_model:
        with open('./model_gp.pkl', 'wb') as f_model, open('./scaler_gp.pkl', 'wb') as f_scaler:
            pickle.dump(model, f_model)
            pickle.dump(sc, f_scaler)
            print("Model saved as 'model_gp.pkl'")

    if create_graph and only_test:
        # Create predictions for only X_test
        y_pred, sigma = create_pred(X_test, model)
        plot_confidence_interval(X_test, y_test, y_pred, sigma)

    elif create_graph:
        # Create predictions for all of X
        y_pred, sigma = create_pred(X_data, model)
        plot_confidence_interval(X_data, y, y_pred, sigma)
