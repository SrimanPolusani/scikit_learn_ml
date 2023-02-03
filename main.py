# Import statements
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Converting txt file data into nd arrays
X_train = np.loadtxt('./houses.txt', usecols=(0, 1, 2, 3), delimiter=',')
y_train = np.loadtxt('./houses.txt', usecols=4, delimiter=',')
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
scaler = StandardScaler()  # Initialing scaler for feature scaling
sgdr = SGDRegressor()  # Initializing stochastic gradient descent regressor
X_norm = scaler.fit_transform(X_train)

# See the difference before and after feature normalization
print("peak to peak in X_train (Before): {}".format(np.ptp(X_train, axis=0)))
print("peak to peak in X_nor (After): {}".format(np.ptp(X_norm, axis=0)))

# Regression model
sgdr.fit(X_norm, y_train)
print(sgdr)

# Intercept and Co-efficient
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print("Model parameters: w = {} & b = {}".format(w_norm, b_norm))

# Make predictions and checking accuracy
y_pred_sgdr = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm
print("Is sgdr prediction and np.dot() prediction same?: {}".format((y_pred_sgdr == y_pred).all()))
print("Predicted values: {}".format(y_pred[:6]))
print("Actual values: {}".format(y_train[:6]))
print("Accuracy: {0:.2f}%".format(
    100 - np.mean(
        abs(
            100 - ((100 * y_pred) / y_train)
        )
    )
)
)

# Let's make our data visualization very colorful.
X_colors = [['#60100B', '#E3242B'], ['#354A21', '#3CB043'], ['#710193', '#365FCF'], ['#1F456E', '#63C5DA']]

# Plotting predicted values and actual values with matplotlib.
fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey=True, dpi=120)
for i in range(len(axes)):
    k = 0
    for j in range(len(axes)):
        if i == 1:
            k = 1
        # Logic behind adding k: Index of row in X_train is equal to sum of i and j. PLUS '1' additionally when i = 1.
        # (i,j) --> [(0,0), (0,1), (1,0), (1,1)] = [0, 1, 2, 3]  <--- index of rows
        index = i + j + k
        axes[i, j].scatter(X_train[:, index], y_train, color=X_colors[index][0], label='Actual')
        axes[i, j].set_xlabel(X_features[index])
        axes[i, j].set_ylabel('Price')
        axes[i, j].scatter(X_train[:, index], y_pred, color=X_colors[index][1], label='Predicted')
        axes[i, j].legend()

fig.suptitle('Accuracy of linear regression with zscore normalization')
plt.show()
