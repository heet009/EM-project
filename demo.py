import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Create DataFrame and clean the data
data = pd.read_excel('G2.xlsx')


# 1A. Unit Root Test (ADF Test)
def check_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Test for {name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print('\n')

# Check stationarity for all variables
for column in ['M', 'Gr', 'Inf']:
    check_stationarity(data[column], column)

# 2. Regression Analysis
X = data[['Inf', 'Gr']]
y = data['M']

# First standardize the features
X_standardized = (X - X.mean()) / X.std()
y_standardized = (y - y.mean()) / y.std()

# Add constant term
X_standardized_with_const = sm.add_constant(X_standardized)

# Create both sklearn and statsmodels models
model_standardized = LinearRegression(fit_intercept=False)
model_standardized.fit(X_standardized_with_const, y_standardized)

# Add statsmodels OLS model
model_sm = sm.OLS(y_standardized, X_standardized_with_const).fit()

# 3. Standardized coefficients for comparing influence
X_standardized = (X - X.mean()) / X.std()
y_standardized = (y - y.mean()) / y.std()


model_standardized = LinearRegression(fit_intercept=False)
model_standardized.fit(X_standardized, y_standardized)  # Now using standardized y as well

# 5. Model without inflation
X_no_inf = data[['Gr']]
model_no_inf = LinearRegression(fit_intercept=False)
model_no_inf.fit(X_no_inf, y)
r2_no_inf = model_no_inf.score(X_no_inf, y)

# 6. Error term analysis
residuals = model_sm.resid
acf_values = acf(residuals)

# Plot correlogram
plt.figure(figsize=(10, 6))
plt.bar(range(len(acf_values)), acf_values)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
plt.title('Correlogram of Residuals')
plt.xlabel('Lag')
plt.ylabel('ACF')

# 6. Multicollinearity test
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 7. Plot actual vs predicted values
y_pred = model_sm.predict(X_standardized_with_const)
y_pred_original = y_pred * y.std() + y.mean()
plt.figure(figsize=(10, 6))
plt.plot(data.index, y_standardized, label='Actual')
plt.plot(data.index, y_pred_original, label='Predicted')
plt.title('Actual vs Predicted Money Supply Changes (Standardized)')
plt.xlabel('Time')
plt.ylabel('Money Supply Change (Standardized)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Print summary statistics
print("\nRegression Results:")
print(model_sm.summary())

print("\nVIF Results:")
print(calculate_vif(X))

print("\nR-squared without inflation:", r2_no_inf)