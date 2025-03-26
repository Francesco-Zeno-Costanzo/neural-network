"""
Simple example code for creating a neural network
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69420)

class NeuralNetworkClassifier:
    '''
    This class implement a multilayer perceptron neural network.
    Is possible to choose how many layers use and how many neurons
    are in each layer. 
    This network is for classifications and the activation function
    implemented are only tanh and relu, sufficient for demonstration
    purposes. The optimizer used is adam, Loss is cross entropy.
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
            be declared, input and output are read from data 
        n_epoch : int
            number of training epochs of the network
        f_act : {'tanh', 'relu'}, optional, default tanh
            activation function for hidden layers
        '''
        # Hidden and output layers
        self.layers   = layers
        self.n_layers = len(self.layers) + 1 # Plus one for output layer
        # Number of training epochs
        self.n_epoch  = n_epoch
        # Weights, bias and predictions for each layers
        self.W = []
        self.B = []
        self.A = []
        self.Z = []
        # Momenta for update
        self.mw = []
        self.vw = []
        self.mb = []
        self.vb = []
        # Some parameters of network
        self.f_act = f_act
        self.dfeat = 0 # Dimension of features, number of neurons in the input  layer
        self.dtarg = 0 # Dimension of targets,  number of neurons in the output layer
        # Size of all data and validation
        self.N = 0 # ALL
        self.M = 0 # Validation

        if f_act not in ['tanh', 'relu']:
            msg = 'Only tanh and relu are implemented'
            raise NotImplementedError(msg)
        
        
    def Loss(self, Yp, Y):
        '''
        loss function,

        Parameters
        ----------
        Yp : 1darray
            actual prediction
        Y : 1darray
            Target

        Returns
        -------
        float, loss
        '''
        Y_onehot = self.one_hot(Y)
        return -np.sum(Y_onehot * np.log(Yp + 1e-9)) / len(Y)
        

    def act(self, x):
        '''
        Activation function for hidden layers;
        Only tanh and relu are implemented
         
        Parameters
        ----------
        x : N x 1 matrix
            intermediate step of a layer
          
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
            intermediate step of a layer
          
        Returns
        -------
        1/cosh(x)**2 or step function
        '''
        if self.f_act == 'tanh':
            return 1/np.cosh(x)**2
        if self.f_act == 'relu': 
            return x > 0


    def softmax(self, x):
        '''
        Activation function for output layers;
        always sigmoid for a classification
         
        Parameters
        ----------
        x : N x 1 matrix
            intermediate step of a layer
          
        Returns
        -------
        sigmoid
        '''
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)


    def initialize(self):
        '''
        Function for the initializzation of weights and bias.
        For all hidden layers we use he initializzation and
        for last layer we use xavier initalization.
        We also define the matrix of momenta for adam.
        '''
    
        l = [self.dfeat] + self.layers + [self.dtarg]

        # He initialization for hidden layers; good for relu
        for i in range(1, self.n_layers):
            s = np.sqrt(2/l[i])
            self.W.append(np.random.randn(l[i], l[i-1]) * s )
            self.B.append(np.random.randn(l[i], 1     ) * s )
                
        # Random initialization, for output Xavier initialization
        M = np.sqrt( 6 / (l[-1] + l[-2]) )
        self.W.append( (2*np.random.rand(l[-1], l[-2]) - 1) * M)
        self.B.append( (2*np.random.rand(l[-1], 1)     - 1) * M)
        
        # Initialization of momenta for minimization
        for i in range(self.n_layers):
            self.mw.append(np.zeros(self.W[i].shape))
            self.mb.append(np.zeros(self.B[i].shape))
            self.vw.append(np.zeros(self.W[i].shape))
            self.vb.append(np.zeros(self.B[i].shape))


    def predict(self, X, train_data=False, all_data=False):
        '''
        Function to made prediction; feedfoward propagation
        If you want predict on train data you must consider
        the division beetwen train and validation

        Parameters
        ----------
        X : 2d array
            matrix of features
        train_data : bool, optional, default False
            If True the propagation is done on self.A and self.Z because
            for backpropagation we must use each iteration on network-
            It is convenient when the method is called outside the class
            so you must pass only data and get final prediction
        all_data : bool, optional, default False
            if True all output is returned with all values, while if False
            only the predicted class is returned.
            useful only for muticlass classifications

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
                    # Activation function for last layer
                    self.A.append(self.softmax(self.Z[i]))
                else:
                    # Activation function for all other layers
                    self.A.append(self.act(self.Z[i]))

        else :
            A = np.copy(X) # to start the iteration
            for i in range(self.n_layers):
                Z = self.W[i] @ A + self.B[i]
                
                if i == self.n_layers-1:
                    # Activation function for last layer
                    A = self.softmax(Z)

                else:
                    # Activation function for all other layers
                    A = self.act(Z)

            return np.argmax(A, 0) if not all_data else A  # Output finale


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

        # Output layer
        delta  = self.A[-1] - Y
        db[-1] = np.sum(delta, axis=1, keepdims=True) / m
        dw[-1] = delta @ self.A[-2].T / m

        # Loop over hidden layers
        for l in range(2, self.n_layers):
            z = self.Z[-l]
            delta = (self.W[-l+1].T @ delta) * self.d_act(z)
            db[-l] = np.sum(delta, axis=1, keepdims=True) / m
            dw[-l] = delta @ self.A[-l-1].T / m 

        return dw, db


    def adam(self, epoch, dW, dB, alpha, b1, b2, eps):
        '''
        Implementation of Adam algorithm, Adaptive Moment Estimation for
        update of weights and bias.

        Parameters
        ----------
        epoch : int
            current iteration
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
            parameter of algorithm, to avoid division by zero
        '''

        for i in range(1, self.n_layers):
            # Update weights
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * dW[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * dW[i]**2
            mw_hat = self.mw[i] / (1 - b1**(epoch + 1) )
            vw_hat = self.vw[i] / (1 - b2**(epoch + 1) )
            dw = alpha * mw_hat / (np.sqrt(vw_hat) + eps)
            self.W[i] -= alpha * dw
            # Update bias
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


    def train(self, X, Y, alpha=0.01, b1=0.9, b2=0.999, eps=1e-8, cut=4, verbose=False):
        '''
        Function for train the network, 
        the data are splitted to copute validation loss

        Parameters
        ----------
        X : 2darray
            matrix of features (features x number of data)
        Y : 2darray
            matrix of target
        alpha : float, optional default 0.01
            size of step to do, typical value is 0.001
        b1 : float, optional, default 0.9
            Decay factor for first momentum
        b2 : float, optional, default 0.999
            Decay factor for second momentum
        eps : float, optional, default 1e-8
            parameter of adam algorithm, to avoid division by zero
        cut : int, optional, default=4
            fraction of input data to use for validation.
            E.g. if N is the number of data, we use N/4 for validation and N-N/4 for train
        verbose : bool
            if True print loss and accuracy each 100 epoch

        Returns
        -------
        result : dict
            params     --> weights and bias of network
            train_Loss --> loss on train data
            valid_Loss --> loss on validation data
        '''
        
        L_t = np.zeros(self.n_epoch) # Training loss
        L_v = np.zeros(self.n_epoch) # Validation loss

        self.N = X.shape[1]       # Total number of data
        self.M = self.N//cut      # Number of data for validation

        # First and last layers
        self.dfeat = X.shape[0]   # Number of features
        self.dtarg = np.max(Y)
        if self.dtarg == 1:
            pass
        else:
            self.dtarg += 1       # There is 0 class

        # Split dataset in validation and train 
        X_train, Y_train = X[:, :self.N-self.M ], Y[:self.N-self.M ] 
        X_valid, Y_valid = X[:,  self.N-self.M:], Y[ self.N-self.M:]

        self.initialize() # Initialize weights and bias

        for e in range(self.n_epoch):
            # Train
            self.predict(X_train, train_data=True)
            L_t[e] = self.Loss(self.A[-1], Y_train)
            # Validation
            Yp = self.predict(X_valid, all_data=True)          
            L_v[e] = self.Loss(Yp, Y_valid)
            # Update
            dW, dB = self.backpropagation(X_train, Y_train)
            self.adam(e, dW, dB, alpha=alpha, b1=b1, b2=b2, eps=eps)

            if not e % 100 and verbose:
                acc = self.accuracy(np.argmax(self.A[-1], 0), Y_train)
                print(f'Loss = {L_t[e]:.5f}, Valid Loss = {L_v[e]:.5f}, accuracy = {acc:.5f}, epoch = {e} \r', end='')

            del self.A[:] # I clean the lists otherwise it 
            del self.Z[:] # continues to add to the queue

        if verbose: print()

        result = {'params'     : (self.W, self.B),
                  'train_Loss' : L_t,
                  'valid_Loss' : L_v,
                 }

        return result


    def confmat(self, true_target, pred_target, plot=True, title='', k=0):
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

        dat = np.unique(true_target)      # Classes
        N   = len(dat)                    # Number of classes 
        mat = np.zeros((N, N), dtype=int) # Confusion matrix

        # Creation of confusion matrix
        for i in range(len(true_target)):
            mat[true_target[i]][pred_target[i]] += 1

        if plot :
            fig = plt.figure(k, figsize=(7, 7))
            ax = fig.add_subplot()

            c = ax.imshow(mat, cmap=plt.cm.Blues) # plot matrix
            b = fig.colorbar(c, fraction=0.046, pad=0.04)
            # Write on plot the value of predictions
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
            plt.title(title, fontsize=15)
            plt.tight_layout()

        return mat

#========================================================================================
#========================================================================================
#========================================================================================

class NeuralNetworkRegressor:
    '''
    This class implement a multilayer perceptron neural network.
    Is possible to choose how many layers use and how many neurons
    are in each layer.
    This network is for regression and the activation function
    implemented are only tanh and relu, sufficient for demonstration
    purposes. The optimizer used is adam, Loss is mean square error.
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
            be declared, input and output are read from data
        n_epoch : int
            number of training epochs of the network
        f_act : {'tanh', 'relu'}, optional, default tanh
            activation function for hidden layers
        '''
        # Hidden and output layers
        self.layers   = layers
        self.n_layers = len(self.layers) + 1 # plus one for output layer
        # Number of training epochs
        self.n_epoch  = n_epoch
        # Weights, bias and predictions for each layers
        self.W = []
        self.B = []
        self.A = []
        self.Z = []
        # Momenta for update
        self.mw = []
        self.vw = []
        self.mb = []
        self.vb = []
        # Some parameters of network
        self.f_act = f_act
        self.dfeat = 0 # Dimension of features, number of neurons in the input  layer
        self.dtarg = 0 # Dimension of targets,  number of neurons in the output layer
        # Size of all data and validation
        self.N = 0 # ALL
        self.M = 0 # Validation

        if f_act not in ['tanh', 'relu']:
            msg = 'Only tanh and relu are implemented'
            raise NotImplementedError(msg)


    def Loss(self, Yp, Y):
        '''
        Loss function

        Parameters
        ----------
        Yp : 1darray
            actual prediction
        Y : 1darray
            Target

        Returns
        -------
        float, loss
        '''
        return np.mean((Yp - Y)**2)


    def act(self, x):
        '''
        Activation function for hidden layers;
        Only tanh and relu are implemented

        Parameters
        ----------
        x : N x 1 matrix
            intermediate step of a layer

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
            intermediate step of a layer

        Returns
        -------
        1/cosh(x)**2 or step function
        '''
        if self.f_act == 'tanh':
            return 1/np.cosh(x)**2
        if self.f_act == 'relu':
            return x > 0


    def linear(self, x):
        '''
        Activation function for output layers;
        always linear for a regressor

        Parameters
        ----------
        x : N x 1 matrix
            intermediate step of a layer

        Returns
        -------
        linear
        '''
        return x


    def initialize(self):
        '''
        Function for the initializzation of weights and bias.
        For all hidden layers we use he initializzation and
        for last layer we use xavier initalization.
        We also define the matrix of momenta for adam.
        '''

        l = [self.dfeat] + self.layers + [self.dtarg]

        # He initialization for hidden layers; good for relu
        for i in range(1, self.n_layers):
            s = np.sqrt(2/l[i])
            self.W.append(np.random.randn(l[i], l[i-1]) * s )
            self.B.append(np.random.randn(l[i], 1     ) * s )

        # Random initialization, for output Xavier initialization
        M = np.sqrt( 6 / (l[-1] + l[-2]) )
        self.W.append( (2*np.random.rand(l[-1], l[-2]) - 1) * M)
        self.B.append( (2*np.random.rand(l[-1], 1)     - 1) * M)

        # Initialization of momenta for minimization
        for i in range(self.n_layers):
            self.mw.append(np.zeros(self.W[i].shape))
            self.mb.append(np.zeros(self.B[i].shape))
            self.vw.append(np.zeros(self.W[i].shape))
            self.vb.append(np.zeros(self.B[i].shape))


    def predict(self, X, train_data=False):
        '''
        Function to made prediction; feedfoward propagation
        If you want predict on train data you must consider
        the division beetwen train and validation

        Parameters
        ----------
        X : 2d array
            matrix of features
        train_data : bool, optional, default False
            If True the propagation is done on self.A and self.Z because
            for backpropagation we must use each iteration on network-
            It is convenient when the method is called outside the class
            so you must pass only data and get final prediction

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
                    # Activation function for last layer
                    self.A.append(self.linear(self.Z[i]))
                else:
                    # Activation function for all other layers
                    self.A.append(self.act(self.Z[i]))

        else :
            A = np.copy(X) # To start the iteration
            for i in range(self.n_layers):
                Z = self.W[i] @ A + self.B[i]

                if i == self.n_layers-1:
                    # Activation function for last layer
                    A = self.linear(Z)

                else:
                    # Activation function for all other layers
                    A = self.act(Z)

            return A


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

        m  = len(Y) # Len of data
        db = [np.zeros(b.shape) for b in self.B]
        dw = [np.zeros(w.shape) for w in self.W]


        # Output layer
        delta  = self.A[-1] - Y
        db[-1] = np.sum(delta, axis=1, keepdims=True) / m
        dw[-1] = delta @ self.A[-2].T / m

        # Loop over hidden layers
        for l in range(2, self.n_layers):
            z = self.Z[-l]
            delta = (self.W[-l+1].T @ delta) * self.d_act(z)
            db[-l] = np.sum(delta, axis=1, keepdims=True) / m
            dw[-l] = delta @ self.A[-l-1].T / m

        return dw, db


    def adam(self, epoch, dW, dB, alpha, b1, b2, eps):
        '''
        Implementation of Adam algorithm, Adaptive Moment Estimation for
        update of weights and bias.

        Parameters
        ----------
        epoch : int
            current iteration
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
            parameter of algorithm, to avoid division by zero
        '''

        for i in range(1, self.n_layers):
            # Update weights
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * dW[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * dW[i]**2
            mw_hat = self.mw[i] / (1 - b1**(epoch + 1) )
            vw_hat = self.vw[i] / (1 - b2**(epoch + 1) )
            dw = alpha * mw_hat / (np.sqrt(vw_hat) + eps)
            self.W[i] -= alpha * dw
            # Update bias
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * dB[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * dB[i]**2
            mb_hat = self.mb[i] / (1 - b1**(epoch + 1) )
            vb_hat = self.vb[i] / (1 - b2**(epoch + 1) )
            db = alpha * mb_hat / (np.sqrt(vb_hat) + eps)
            self.B[i] -= alpha * db


    def train(self, X, Y, alpha=0.01, b1=0.9, b2=0.999, eps=1e-8, cut=4, verbose=False):
        '''
        Function for train the network,
        the data are splitted to copute validation loss

        Parameters
        ----------
        X : 2darray
            matrix of features (features x number of data)
        Y : 2darray
            matrix of target
        alpha : float, optional default 0.01
            size of step to do, typical value is 0.001
        b1 : float, optional, default 0.9
            Decay factor for first momentum
        b2 : float, optional, default 0.999
            Decay factor for second momentum
        eps : float, optional, default 1e-8
            parameter of adam algorithm, to avoid division by zero
        cut : int, optional, default=4
            fraction of input data to use for validation.
            E.g. if N is the number of data, we use N/4 for validation and N-N/4 for train
        verbose : bool
            if True print loss and accuracy each 100 epoch

        Returns
        -------
        result : dict
            params     --> weights and bias of network
            train_Loss --> loss on train data
            valid_Loss --> loss on validation data
        '''

        L_t = np.zeros(self.n_epoch) # Training loss
        L_v = np.zeros(self.n_epoch) # Validation loss

        self.N = X.shape[1]       # Total number of data
        self.M = self.N//cut      # Number of data for validation

        # First and last layers
        self.dfeat = X.shape[0]       # Number of features
        self.dtarg = Y.shape[0] if Y.ndim > 1 else 1


        # Split dataset in validation and train
        X_train, Y_train = X[:, :self.N-self.M ], Y[:self.N-self.M ]
        X_valid, Y_valid = X[:,  self.N-self.M:], Y[ self.N-self.M:]

        self.initialize() # Initialize weights and bias

        for e in range(self.n_epoch):
            # Train
            self.predict(X_train, train_data=True)
            L_t[e] = self.Loss(self.A[-1], Y_train)
            # Validation
            Yp = self.predict(X_valid)
            L_v[e] = self.Loss(Yp, Y_valid)
            # Update
            dW, dB = self.backpropagation(X_train, Y_train)
            self.adam(e, dW, dB, alpha=alpha, b1=b1, b2=b2, eps=eps)

            if not e % 100 and verbose:
                print(f'Loss = {L_t[e]:.5f}, Valid Loss = {L_v[e]:.5f}, epoch = {e} \r', end='')

            del self.A[:] # I clean the lists otherwise it
            del self.Z[:] # continues to add to the queue

        if verbose: print()

        result = {'params'     : (self.W, self.B),
                  'train_Loss' : L_t,
                  'valid_Loss' : L_v,
                 }

        return result


