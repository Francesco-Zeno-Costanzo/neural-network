import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from neural import NeuralNetworkRegressor

np.random.seed(0)

#=============================================================
# Creation of dataset
#=============================================================
X = np.linspace(0, 2, 50)#np.arange(0, 10, 0.2) # 1 feauture
n = len(X)
y = np.cos(2*np.pi*X) + 0.5*np.random.random(n)   # target

X = np.expand_dims(X, axis=1)


# split dataset in test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, y)
X_train, X_test = X_train.T, X_test.T


# for plot
Xp = np.linspace(0, 2, 1000)
Xp = np.expand_dims(Xp, axis=1)

#=============================================================
# Creation of models
#=============================================================
n_epoch = 3000 + 1
lr_rate = 0.01

# With tanh the last plot is more smooth
NN = NeuralNetworkRegressor([50, 50, 50], n_epoch, f_act='tanh')

result = NN.train(X_train, Y_train, alpha=lr_rate, verbose=True)

L_t = result['train_Loss']
L_v = result['valid_Loss']

#=============================================================
# Plot Loss
#=============================================================

plt.figure(1)
plt.plot(np.linspace(1, n_epoch, n_epoch), L_t, 'b', label='train Loss')
plt.plot(np.linspace(1, n_epoch, n_epoch), L_v, 'r', label='validation loss')
plt.title('Binay cross entropy', fontsize=15)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(loc='best')
#plt.savefig("Loss_fit.pdf")
plt.grid()

#=============================================================
# Plot
#=============================================================

p_fit = NN.predict(Xp.T)
p_fit = p_fit[0]

plt.figure(2)
plt.title('Network regression result', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.grid()
plt.errorbar(X_train.T[:,0], Y_train, fmt='.', c='y', label='train data')
plt.errorbar(X_test.T[:,0],  Y_test, fmt='.', c='b', label='test data')
plt.plot(Xp[:,0], p_fit,'k', label='fit')
plt.legend(loc='best')
plt.show()
