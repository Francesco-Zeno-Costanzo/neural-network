"""
Code for number recognition of the mnist dataset
with a neural network written from scratch
"""
import numpy as np
import matplotlib.pyplot as plt

from neural import NeuralNetwork

#=============================================================
# load of dataset 
#=============================================================

# train, all 60000 data is too much
N = 3000
train_data = np.loadtxt('MNIST_data/mnist_train.csv', max_rows=N, delimiter=',')
# normalize input 
X_train, Y_train = train_data[:, 1:].T/255, train_data[:, 0]
Y_train = np.array([int(y) for y in Y_train])
# test
M = 1000
test_data = np.loadtxt('MNIST_data/mnist_test.csv', max_rows=M, delimiter=',')
X_test, Y_test = test_data[:, 1:].T/255, test_data[:, 0]
Y_test = np.array([int(y) for y in Y_test])

#=============================================================
# Parameter of computation and train of the network
#=============================================================

n_epoch = 3000 + 1
lr_rate = 0.05
NN = NeuralNetwork([50, 50], n_epoch, f_act='relu')
result = NN.train(X_train, Y_train, alpha=lr_rate, verbose=True)
A = NN.predict(X_train[:, :N-N//4 ])
NN.confmat(Y_train[:N-N//4], A, plot=True, k=0)
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
M = NN.confmat(Y_test, A, plot=True, k=3)
plt.savefig("conf_mat_fit_test.pdf")
#acc = NN.accuracy(A, Y_test)
print(f"Accuracy on test set = {acc:.5f}")

plt.figure(2, figsize=(16, 10))
for i in range(40):
    plt.subplot(5, 8, 1+i)
    image = X_test[:, i]    
    image = image.reshape((28, 28)) * 255
    plt.title(f'pred={A[i]} label={Y_test[i]}')
    plt.imshow(image)

plt.tight_layout()

#plt.savefig("MNIST.pdf")
plt.show()
