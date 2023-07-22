"""
Simple example code for creating a neural network
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy

np.random.seed(69420)

class NeuralNetwork:
    '''
    This class implement a multilayer perceptron neural network.
    Is possible to choose how many layers use and how many neurons
    are in each layer. 
    This network is for classifications and the activantion function
    implemented are only tanh and relu, sufficient for demonstration
    purposes. The opitimizer used is adam, Loss is binary cross entropy.
    In the case of classifications with multiple classes, one ot encoding
    is used, and the loss is calculated as the average on each output.
    '''
    
    def __init__(self, layers, n_epoch, f_act='tanh'):
        '''
        Initialize the neural network for a classification problem
        
        Parameters
        ----------
        layers : list
            list which must contain the number of neurons for each layer
            the number of layers is len(layers) and layers[i] is the 
            number of neurons on the i-th layer. Only hidden layers must
            be celaderd, input and output are read from data 
        n_epoch : int
            number of training epochs of the network
        f_act : {'tanh', 'relu'}, optional, default tanh
            activation function for hidden layers
        '''
        # hidden and output layers
        self.layers   = layers
        self.n_layers = len(self.layers) + 1 # plus one for output layer
        # number of training epochs
        self.n_epoch  = n_epoch
        # weights, bias and predictions for each layers
        self.W = []
        self.B = []
        self.A = []
        self.Z = []
        # momenta for update
        self.mw = []
        self.vw = []
        self.mb = []
        self.vb = []
        # some parameters of network
        self.f_act = f_act
        self.dfeat = 0 # dimension of features, number of neurons in the input  layer
        self.dtarg = 0 # dimension of targets,  number of neurons in the output layer

        
        
    def Loss(self, Yp, Y):
        '''
        loss function, binary crosss entropy
          
        Parameters
        ----------
        Yp : 1darray
            actual prediction
        Y : 1darray
            Target
          
        Returns
        -------
        float, binary crosss entropy
        '''
        if self.dtarg == 1:
            m = len(Y) # len of data
            return -np.sum(xlogy(Y, Yp) + xlogy(1 - Y, 1 - Yp)) / m 
        elif self.dtarg > 1:
            m = len(Y) # len of data
            Y = self.one_hot(Y)
            L = [-np.sum(xlogy(Y[i, :], Yp[i,:]) + xlogy(1 - Y[i,:], 1 - Yp[i,:])) / m for i in range(Y.shape[0])]
            return np.mean(L)
            
            
    def act(self, x):
        '''
        Activation function for hidden layers;
        Only tanh and relu are implemented
         
        Parameters
        ----------
        x : N x 1 matrix
            iterpediate step of a layer
          
        Returns
        -------
        tanh(x) or relu(x) 
        '''
        if self.f_act == 'tanh':
            return np.tanh(x)
        if self.f_act == 'relu': 
            return np.maximum(0, x)
    
    
    def d_act(self, x):
        '''
        Derivative of activation function for hidden layers;
        Only tanh and relu are implemented
         
        Parameters
        ----------
        x : N x 1 matrix
            iterpediate step of a layer
          
        Returns
        -------
        1/cosh(x)**2 or step function
        '''
        if self.f_act == 'tanh':
            return 1/np.cosh(x)**2
        if self.f_act == 'relu': 
            return x > 0    

    
    def sigmoid(self, x):
        '''
        Activation function for output layers;
        always sigmoid for a classification
         
        Parameters
        ----------
        x : N x 1 matrix
            iterpediate step of a layer
          
        Returns
        -------
        sigmoid
        '''
        return 1/(1+np.exp(-x)) 

    
    def initialize(self):
        '''
        Function for the initializzation of weights and bias.
        For all hidden layers we use he initializzation and
        for last layer we use xavier initalization.
        We also define the matrix of momenta for adam.
        '''
    
        l = [self.dfeat] + self.layers + [self.dtarg]

        # he initialization for hidden layers; good for relu
        for i in range(1, self.n_layers):
            s = np.sqrt(2/l[i])
            self.W.append(np.random.randn(l[i], l[i-1]) * s )
            self.B.append(np.random.randn(l[i], 1     ) * s )
                
        # random initialization, for output Xavier initialization
        M = np.sqrt( 6 / (l[-1] + l[-2]) )
        self.W.append( (2*np.random.rand(l[-1], l[-2]) - 1) * M)
        self.B.append( (2*np.random.rand(l[-1], 1)     - 1) * M)
        
        # initialization of momenta for minimizzation
        for i in range(self.n_layers):
            self.mw.append(np.zeros(self.W[i].shape))
            self.mb.append(np.zeros(self.B[i].shape))
            self.vw.append(np.zeros(self.W[i].shape))
            self.vb.append(np.zeros(self.B[i].shape))
        
    
    def predict(self, X, train_data=False, all_data=False):
        '''
        Function to made prediction; foward propagation
        
        Parameters
        ----------
        X : 2d array
            matrix of featurs
        train_data : bool, optional, default False
            If True the propagation is done on self.A and self.Z because
            for backpropagation we must use each iteration on network-
            It is convenient when the method is called outside the class
            so you must pass only data and get final prediction
        all_data : bool, optional, default False
            if True all output is returned with all values, while if False
            only the predicted class is returned.
            Usefull only for muticlass classifications
        
        Returns
        -------
        A : 2darray
            prediction of network
        '''
        if train_data:
            self.A.append(X) # to start the iteration
            for i in range(self.n_layers):
                self.Z.append(self.W[i] @ self.A[i] + self.B[i])
                
                if i == self.n_layers-1:
                    # activation function for last layer
                    self.A.append(self.sigmoid(self.Z[i]))
                else:
                    # activation function for all other layers
                    self.A.append(self.act(self.Z[i]))
                
        else :
            A = np.copy(X) # to start the iteration
            for i in range(self.n_layers):
                Z = self.W[i] @ A + self.B[i]
                
                if i == self.n_layers-1:
                    # activation function for last layer
                    A = self.sigmoid(Z)
                else:
                    # activation function for all other layers
                    A = self.act(Z)

            if self.dtarg > 1 :
                if not all_data:
                    return np.argmax(A, 0)
                else :
                    return A
            elif self.dtarg == 1 :
                return A
                    
            
    def one_hot(self, Y):
        '''
        Function for one hot encoding
        
        Parameter
        ---------
        Y : 1darray
            target
        
        Return
        ------
        one_hot : 2darray
            matrix that contain one on the index class
         
        Example
        -------
        >>> y = np.array([1, 5, 6])
        >>> one_hot(y)
        >>> array([[0., 0., 0.],
                   [1., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
        '''
        one_hot = np.zeros((Y.size, np.max(Y) + 1))
        one_hot[np.arange(Y.size), Y] = 1
        return one_hot.T
    
    
    def backpropagation(self, X, Y):
        '''
        Function for backward propagation.
        compute de derivative of the loss function
        respect to weights and bias.
        
        Parameters
        ----------
        X : 2darray
            matrix of features
        Y : 2darray
            matrix of target
        
        Returns
        -------
        dw : 2darray
            dloss/dw gradient for update of weights
        db : 2darray
            dloss/db gradient for update of bias
        '''
        
        m  = len(Y) # len of data
        db = [np.zeros(b.shape) for b in self.B]
        dw = [np.zeros(w.shape) for w in self.W]
        
        if self.dtarg > 1: Y = self.one_hot(Y)
        
        # output layer
        delta  = self.A[-1] - Y
        db[-1] = np.sum(delta, axis=1, keepdims=True) / m
        dw[-1] = delta @ self.A[-2].T / m
        
        # loop over hidden layers
        for l in range(2, self.n_layers):
            z = self.Z[-l]
            delta = (self.W[-l+1].T @ delta) * self.d_act(z)
            db[-l] = np.sum(delta, axis=1, keepdims=True) / m
            dw[-l] = delta @ self.A[-l-1].T / m 

        return dw, db
    
        
    def adam(self, epoch, dW, dB, alpha, b1, b2, eps):
        '''
        Implementation of Adam alghoritm, Adaptive Moment Estimation for
        update of weights and bias.
        
        Parameters
        ----------
        epoch : int
            cuttente iteration
        dW : 2darray
            dloss/dw gradient for update of weights
        dB : 2darray
            dloss/db gradient for update of bias
        alpha : float, optional default 0.01
            size of step to do, typical value is 0.001
        b1 : float, optional, default 0.9
            Decay factor for first momentum
        b2 : float, optional, default 0.999
            Decay factor for second momentum
        eps : float, optional, default 1e-8
            parameter of alghoritm, to avoid division by zero
        '''
        
        for i in range(1, self.n_layers):
            # udate weights
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * dW[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * dW[i]**2
            mw_hat = self.mw[i] / (1 - b1**(epoch + 1) )
            vw_hat = self.vw[i] / (1 - b2**(epoch + 1) )
            dw = alpha * mw_hat / (np.sqrt(vw_hat) + eps)
            self.W[i] -= alpha * dw
            # update bias
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * dB[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * dB[i]**2
            mb_hat = self.mb[i] / (1 - b1**(epoch + 1) )
            vb_hat = self.vb[i] / (1 - b2**(epoch + 1) )
            db = alpha * mb_hat / (np.sqrt(vb_hat) + eps)
            self.B[i] -= alpha * db

    
    def accuracy(self, Yp, Y):
        '''
        accuracy of prediction. We use for binary classifications:
        accuracy = 1 - | sum ( prediction - target )/target_size |
        While for more class, first we identify the predicted class
        (np.argmax(prediction, 0), in prediction function) and then
        compare the results with : sum(prediction == traget)/target_size
        the bollean value will cast into 0 or 1
         
        Parameters
        ----------
        Yp : 1darray
            actual prediction
        Y : 1darray
            Target
        
        Returns
        -------
        a : float
            accuracy
        '''
        if np.max(Y)==1:
            m = len(Y)
            a = 1 - abs(np.sum(Yp.ravel() - Y)/m)
        else:
            m = len(Y)
            a = np.sum(Yp == Y)/m
        return a
          
               
    def train(self, X, Y, alpha=0.01, b1=0.9, b2=0.999, eps=1e-8, verbose=False):
        '''
        Function for train the network, 
        the data are splitted to copute validation loss
        
        Parameters
        ----------
        X : 2darray
            matrix of features
        Y : 2darray
            matrix of target
        alpha : float, optional default 0.01
            size of step to do, typical value is 0.001
        b1 : float, optional, default 0.9
            Decay factor for first momentum
        b2 : float, optional, default 0.999
            Decay factor for second momentum
        eps : float, optional, default 1e-8
            parameter of adam alghoritm, to avoid division by zero
        verbose : bool
            if True print loss and accuracy each 100 epoch
        
        Returns
        -------
        result : dict
            params     --> weights and bias of network
            train_Loss --> loss on train data
            valid_Loss --> loss on validation data
        '''
        
        L_t = np.zeros(self.n_epoch) # training loss
        L_v = np.zeros(self.n_epoch) # validation loss
        
        K = X.shape[0]       # number of features
        N = X.shape[1]       # total number of data
        M = N//4             # nuber of data for validation
        
        # first and last layers
        self.dfeat = K
        self.dtarg = np.max(Y)
        if self.dtarg == 1:
            pass
        else:
            self.dtarg += 1 # there is 0 class

        # split dataset in validation and train 
        X_train, Y_train = X[:, :N-M ], Y[:N-M ] 
        X_valid, Y_valid = X[:,  N-M:], Y[ N-M:]
          
        self.initialize() # initialize weights and bias
        
        for e in range(self.n_epoch):
            # train
            self.predict(X_train, train_data=True)
            L_t[e] = self.Loss(self.A[-1], Y_train)
            # validation
            Yp = self.predict(X_valid, all_data=True)          
            L_v[e] = self.Loss(Yp, Y_valid)
            # update
            dW, dB = self.backpropagation(X_train, Y_train)
            self.adam(e, dW, dB, alpha=alpha, b1=b1, b2=b2, eps=eps)
            
            if not e % 100 and verbose:
                acc = self.accuracy(np.argmax(self.A[-1], 0), Y_train)
                print(f'Loss = {L_t[e]:.5f}, accuracy = {acc:.5f}, epoch = {e} \r', end='')
            
            self.A[:] = [] # I clean the lists otherwise it 
            self.Z[:] = [] # continues to add to the queue
        
        if verbose: print()
          
        result = {'params'     : (self.W, self.B),
                  'train_Loss' : L_t,
                  'valid_Loss' : L_v,
                 }
             
        return result
    
    def confmat(self, true_target, pred_target, plot=True, k=0):
        '''
        Function for creation and plot of confusion matrix
        
        Parameters
        ----------
        true_target : 1darray
            vaules that must be predict
        pred_target : 1darray
            values that the network has predict
        plot : bool, optional, default True
            if True the matix is plotted. 
        k : int, optional, default 0
            number of figure, necessary in order not to overlap figures
        
        Return
        ------
        mat : 2darray
            confusion matrix
        '''
        
        dat = np.unique(true_target)      # classes
        N   = len(dat)                    # Number of classes 
        mat = np.zeros((N, N), dtype=int) # confusion matrix
        
        # creation of confusion matrxi
        for i in range(len(true_target)):
            mat[true_target[i]][pred_target[i]] += 1
       
        if plot :
            fig = plt.figure(0, figsize=(7, 7))
            ax = fig.add_subplot()
            
            c = ax.imshow(mat, cmap=plt.cm.Blues) # plot matrix
            b = fig.colorbar(c, fraction=0.046, pad=0.04)
            # write on plot the value of predictions
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(x=j, y=i, s=mat[i, j],
                    va='center', ha='center')
            
            # Label
            ax.set_xticks(dat, dat)
            ax.set_yticks(dat, dat)
            ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

            plt.xlabel('Predict label', fontsize=15)
            plt.ylabel('True label', fontsize=15)
            plt.title('Confusion Matrix', fontsize=15)
            plt.tight_layout()
            
        return mat
        
    
if __name__ == '__main__':       

#=============================================================
#   Creation of dataset 
#=============================================================

    N = 5000                          # number of train points
    M = 1000                          # number of test  points
    X = np.random.random(size=(2, N)) # Two features
    Y = np.ones(N, dtype=int)         # one Target
        
    # Selection of some regions where the target value is different
    for x1, x2, i in zip(X[0, :], X[1, :], range(N)):
        if np.sqrt( (x1 - 0.3)**2 + (x2 - 0.3)**2 ) < 0.2:
            Y[i] = 0
        if np.sqrt( (x1 - 0.65)**2 + (x2 - 0.7)**2 ) < 0.2:
            Y[i] = 0
    
    # split dataset in test and train 
    X_train, Y_train = X[:, :N-M ], Y[:N-M ] 
    X_test,  Y_test  = X[:,  N-M:], Y[ N-M:]

#=============================================================
#   Parameter of computation and train of the network
#=============================================================

    n_epoch = 5000 + 1  # number of epoch
    lr_rate = 0.1       # learning rate

    NN = NeuralNetwork([20, 20], n_epoch)
    result = NN.train(X_train, Y_train, alpha=lr_rate, verbose=True)
    L_t = result['train_Loss']
    L_v = result['valid_Loss']

#=============================================================
#   Plot Loss
#============================================================= 

    plt.figure(1)
    plt.plot(np.linspace(1, n_epoch, n_epoch), L_t, 'b', label='train Loss')
    plt.plot(np.linspace(1, n_epoch, n_epoch), L_v, 'r', label='validation loss')
    plt.title('Binay cross entropy', fontsize=15)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()

#=============================================================
#   Test of the network
#============================================================= 
    
    A = NN.predict(X_test)
    acc  = NN.accuracy(A, Y_test)
    loss = NN.Loss(A, Y_test)
    
    print(f'Loss     on test  set = {loss:.5f}')
    print(f"Accuracy on test  set = {acc:.5f}")
    # bound of plot
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
        
    # data to predict to create the decision boundary
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = NN.predict(np.array([xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
        
    # Plot boundary as a contour plot
    fig = plt.figure(2, figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.contourf(xx, yy, Z, cmap='plasma')
    plt.scatter(X_test[0, :], X_test[1, :], c=Y_test, cmap='plasma', s=8)
    plt.title(f'Neural network', fontsize=15)
    plt.ylabel('x2', fontsize=15)
    plt.xlabel('x1', fontsize=15)

    plt.show()

