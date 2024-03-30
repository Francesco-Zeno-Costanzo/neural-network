import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural import NeuralNetworkClassifier

#dati che verrano utilizzati
iris_dataset = datasets.load_iris()
#print(iris_dataset["DESCR"])

#caratteristiche, dati in input
X = iris_dataset.data

#output, cioe' quello che il modello dovrebbe predire
Y = iris_dataset.target

#divido i dati, un parte li uso per addestrare, l'altra per
#testare se il modello ha imparato bene
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
X_train, X_test = X_train.T, X_test.T

#=============================================================
# Parameter of computation and train of the network
#=============================================================

N       = len(X_train[0,:])
n_epoch = 400 + 1
lr_rate = 0.01

NN     = NeuralNetworkClassifier([50, 50], n_epoch, f_act='relu')
result = NN.train(X_train, Y_train, alpha=lr_rate, verbose=True)
A      = NN.predict(X_train[:, :N-N//4 ])

NN.confmat(Y_train[:N-N//4], A, plot=True, title='Confusion matrix for train data', k=0)
#plt.savefig("conf_mat_fit_train.pdf")
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
# Test of the network
#============================================================= 

A = NN.predict(X_test)
M = NN.confmat(Y_test, A, plot=True, title='Confusion matrix for test data', k=2)
#plt.savefig("conf_mat_fit_test.pdf")
acc = NN.accuracy(A, Y_test)
print(f"Accuracy on test set = {acc:.5f}")

plt.tight_layout()
plt.show()
