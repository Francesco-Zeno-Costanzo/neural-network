"""
Code to have a visual representation of the
underfitting and overfitting of a neural network using sklearn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(0) 

#=============================================================
# Creation of dataset 
#=============================================================
X = np.linspace(0, 10, 50)#np.arange(0, 10, 0.2) # 1 feauture
n = len(X)
y = np.cos(X) + 2*np.random.random(n)   # target

X = np.expand_dims(X, axis=1) # sklearn accept only matrix

# split dataset in test and train 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# for plot
Xp = np.linspace(0, 10, 1000)
Xp = np.expand_dims(Xp, axis=1)

#=============================================================
# Creation of models 
#=============================================================

# good fit
modello_fit = MLPRegressor(hidden_layer_sizes=[50], max_iter=10000)

# underfitting
modello_under = MLPRegressor(hidden_layer_sizes=[1], max_iter=10000,)

# overfitting
modello_over = MLPRegressor(hidden_layer_sizes=[200, 200, 200, 200], max_iter=10000)

#=============================================================
# Train and prediction
#=============================================================

# train
modello_fit.fit(X_train, y_train)

modello_under.fit(X_train, y_train)

modello_over.fit(X_train, y_train)

# prediction
p_train_fit = modello_fit.predict(X_train)
p_test_fit = modello_fit.predict(X_test)
p_fit =  modello_fit.predict(Xp)

p_train_under = modello_under.predict(X_train)
p_test_under = modello_under.predict(X_test)
p_under =  modello_under.predict(Xp)

p_train_over = modello_over.predict(X_train)
p_test_over = modello_over.predict(X_test)
p_over =  modello_over.predict(Xp)

#=============================================================
# Error on train and test
#=============================================================

dp_train_fit = mean_absolute_error(y_train, p_train_fit)
dp_test_fit = mean_absolute_error(y_test, p_test_fit)

dp_train_under = mean_absolute_error(y_train, p_train_under)
dp_test_under = mean_absolute_error(y_test, p_test_under)

dp_train_over = mean_absolute_error(y_train, p_train_over)
dp_test_over = mean_absolute_error(y_test, p_test_over)

#=============================================================
# Print result
#=============================================================

print('error for good  fit')
print(f'err_train = {dp_train_fit},  err_test = {dp_test_fit}')

print('error for under fit')
print(f'err_train = {dp_train_under},  err_test = {dp_test_under}')

print('error for over  fit')
print(f'err_train = {dp_train_over},  err_test = {dp_test_over}')

#=============================================================
# Plot
#=============================================================

plt.figure(1)
plt.title('Network comparison', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.grid()
plt.errorbar(X_train[:,0], y_train, fmt='.', c='y', label='train data')
plt.errorbar(X_test[:,0], y_test, fmt='.', c='b', label='test data')
plt.plot(Xp[:,0], p_fit,'k', label='fit')
plt.plot(Xp[:,0], p_over, 'r', label='over fit')
plt.plot(Xp[:,0], p_under, 'g', label='under fit')
plt.legend(loc='best')
plt.show()
