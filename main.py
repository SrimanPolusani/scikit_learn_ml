# Import statements
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ['size(sqft)', 'bedrooms', 'floors', 'age']
# Let's make our data visualization very colorful.
X_COLORS = [['#60100B', '#E3242B'], ['#354A21', '#3CB043'], ['#710193', '#365FCF'], ['#1F456E', '#63C5DA']]

# Logistic Regression data
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


# <---Multiple Linear Regression--->
class LinearRegressor:
    def __init__(self, txt_file_path):
        self.txt = txt_file_path
        self.X_train = None
        self.y_train = None
        self.X_norm = None
        self.sgdr = None

    # <---Converts txt file data into nd arrays--->
    def txt_ndarray_convertor(self):
        self.X_train = np.loadtxt(self.txt, usecols=(0, 1, 2, 3), delimiter=',')
        self.y_train = np.loadtxt(self.txt, usecols=4, delimiter=',')

    # <---Normalizes data--->
    def normalizer(self):
        # Initialing scaler for feature scaling
        scaler = StandardScaler()
        self.X_norm = scaler.fit_transform(self.X_train)

        # Difference in ptp values before and after feature normalization
        print("peak to peak in X_train (Before): {}".format(np.ptp(self.X_train, axis=0)))
        print("peak to peak in X_nor (After): {}".format(np.ptp(self.X_norm, axis=0)))

    # <---Main function--->
    def regressor(self):
        self.txt_ndarray_convertor()
        self.normalizer()

        # Initializing stochastic gradient descent regressor
        self.sgdr = SGDRegressor()
        self.sgdr.fit(self.X_norm, self.y_train)

        # Intercept and Co-efficient
        final_b = self.sgdr.intercept_
        final_w = self.sgdr.coef_
        print("Model parameters: w = {} & b = {}".format(final_w, final_b))
        return final_w, final_b

    # <---Makes predictions--->
    def make_predictions(self):
        w, b = self.regressor()

        y_pred_sgdr = self.sgdr.predict(self.X_norm)
        y_pred = np.dot(self.X_norm, w) + b
        print("Is sgdr prediction and np.dot() prediction same?: {}".format((y_pred_sgdr == y_pred).all()))
        print("Predicted values: {}".format(y_pred[:6]))
        print("Actual values: {}".format(self.y_train[:6]))

        # Checks accuracy and prints it
        print("Accuracy: {0:.2f}%".format(
            100 - np.mean(
                abs(
                    100 - ((100 * y_pred) / self.y_train)
                )
            )
        )
        )
        return y_pred_sgdr

    # <---Visualizes accuracy of the model--->
    def visualize_results(self):
        predictions = self.make_predictions()
        # Plots predicted values and actual values with matplotlib.
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey=True, dpi=90)
        for i in range(len(axes)):
            k = 0
            for j in range(len(axes)):
                if i == 1:
                    k = 1
                # Logic behind adding k: Index of row in X_train is equal to sum of i and j. PLUS '1' additionally when i = 1.
                # (i,j) --> [(0,0), (0,1), (1,0), (1,1)] = [0, 1, 2, 3]  <--- index of rows
                index = i + j + k
                axes[i, j].scatter(self.X_train[:, index], self.y_train, color=X_COLORS[index][0], label='Actual')
                axes[i, j].set_xlabel(FEATURE_NAMES[index])
                axes[i, j].set_ylabel('Price')
                axes[i, j].scatter(self.X_train[:, index], predictions, color=X_COLORS[index][1], label='Predicted')
                axes[i, j].legend()

        fig.suptitle('Accuracy of linear regression with zscore normalization')
        plt.show()


# <---Logistic Regression--->
class LogisticRegressor:
    def __init__(self, Xtrain, ytrain):
        self.X_train = Xtrain
        self.y_train = ytrain
        self.logistic_regressor = None
        self.no_of_ex = self.X_train.shape[0]

    def main_func(self):
        # Initializing stochastic gradient descent regressor
        self.logistic_regressor = LogisticRegression()
        self.logistic_regressor.fit(self.X_train, self.y_train)
        final_b = self.logistic_regressor.intercept_
        print('Final b: {}'.format(final_b))
        final_w = self.logistic_regressor.coef_
        print('Final w: {}'.format(final_w))

    def make_predictions(self):
        self.main_func()
        y_pred = self.logistic_regressor.predict(self.X_train)
        print('The scikit-learn regressor predicted values: {}'.format(y_pred))
        return y_pred

    def visualize_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for axis_num, axis in enumerate(axes):
            axis.scatter(self.X_train[0:, axis_num], self.y_train, color=X_COLORS[axis_num][0], s=100, label='Actual')
            axis.scatter(self.X_train[0:, axis_num], self.make_predictions(), color=X_COLORS[axis_num][1], s=25,
                         label='Predicted')
            axis.set_xlabel('$X_{}$'.format(axis_num))
            axis.set_ylabel('Y')
            axis.set_title('Predicted Vs Actual Values')
            axis.legend()
        fig.suptitle('Accuracy Visualization')
        plt.show()


# Instance of the linear regression object
linear_regression = LinearRegressor('./houses.txt')
linear_regression.visualize_results()

# Instance of the logistic regression object
logistic_regression = LogisticRegressor(X, y)
logistic_regression.visualize_results()
