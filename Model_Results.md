# Predict the weekly failure rate
Predictions represent the percentage of the total number of compressors that fail that week.  
There are 300 days of complete data for all 6 asset regions.

| Group size | number of entries |
| ---------- | ----------------- |
| Week: 7 days | 252 |
| Half week: 3 days | 600 |
| 1 day | 1800 |

**There are only 252 rows to predict on:** 42 weeks with 6 business units

If I take each day and make one entry per compressor (which the data doesn't permit), then there will likely be ample data.

## Model Scores
Weekly predictions.

|          | GaussianProcess | n^2 LinearRegression | LinearRegression |
| -------- | --------------- | -------------------- | ---------------- |
| **R^2**  | 0.61            | 0.51             | 0.45 |
| **MAE**  | 0.07            | 0.09             | 0.10 |
| **RMSE** | 0.12            | 0.13             | 0.14 |


### Gaussian Process Regressor
Data is sanitized and scaled prior to being fit by the GPR model.
```python
kernel = RationalQuadratic() + 0.1 * WhiteKernel()

model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
```
Rational Quadratic kernels are:
```
 "...a scale mixture of squared exponential kernels with different length scales.
 This gives variations with a range of time-scales, the distribution peaking around
λ but extending to significantly longer period (but remaining rather smooth)"  
(http://www.robots.ox.ac.uk/~sjrob/Pubs/philTransA_2012.pdf)
```
### Polynomial (n^2) Linear Regression
Coefficients for terms worth > 1% of all coefs  

| Coefficient | Value |
| ----------- | ----- |
| X_intercept (anadarko) | -14.57% |  
| durango | -3.53% |  
| farmington | -2.81% |  
| easttexas | -2.54% |  
| wamsutter | -1.58% |  
| elevation | -1.77% |  
| elevation^2 | -5.96% |  
| arkoma^2 | 4.11% |  
| durango^2 | 5.37% |  
| farmington^2 | 5.81% |  
| easttexas^2 | 6.25% |  
| wamsutter^2 | 6.50% |  
| elevation * farmington | -2.50% |  
| elevation * wamsutter | -2.21% |  
| elevation * arkoma | 1.35% |  
| elevation * durango | 2.90% |  
| elevation * easttexas | 4.78% |  
| arkoma * wamsutter | 1.07% |  
| durango * easttexas | 1.15% |  
| durango * farmington | 1.88% |  
| arkoma * farmington | 2.41% |  
| arkoma * durango | 3.34% |  
| farmington * wamsutter | 3.83% |  
| durango * wamsutter | 4.38% |  


## Linear Regression
Coefficients for all terms.

| Coefficient | Value |
| ----------- | ----- |
| X_intercept (anadarko) | 24.39% |  
| maxTemp | 16.40% |
| avgTemp | -18.52% |
| minTemp | -4.93% |
| precip | 1.33% |
| elevation | 0.33% |
| arkoma | 5.07% |
| durango | -6.81% |
| easttexas | 5.20% |
| farmington | -1.09% |
| wamsutter | 15.92% |


# What I *Should* be doing
Create this model in PyMC3.  This will allow for multilevel modeling (the model can change slightly for each business unit) and interpretability of features.
