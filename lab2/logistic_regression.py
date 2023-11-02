import numpy as np


eps = np.finfo(float).eps


def sigmoid(x):
    """
    Element-wise sigmoid function

    Parameters
    ----------
    x: np.array
        a numpy array of any shape

    Returns
    -------
    np.array
        an array having the same shape of x.
    """

    """
    Apply the sigmoid function on x.
    See https://en.wikipedia.org/wiki/Sigmoid_function
    """
    sigmoide=1/(1+ np.exp(-x) + eps)

    return sigmoide


def loss(y_true, y_pred):
    """
    The binary crossentropy loss.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)

    Returns
    -------
    float
        the value of the binary crossentropy.
    """

    """
    Compute the average binary cross entropy between
    y_true (targets) and y_pred (predictions).

    https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    
    """

    val = -(y_true * np.log(y_pred + eps) + (1 - y_true)*np.log(1-y_pred + eps))
    return np.mean(val)


def dloss_dw(y_true, y_pred, X):
    """
    Derivative of loss function w.r.t. weights.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)
    X: np.array
        predicted data. shape=(n_examples, n_features)

    Returns
    -------
    np.array
        derivative of loss function w.r.t weights.
        Has shape=(n_features,)
    """

    N = X.shape[0]
    val=-(np.transpose(X)@(y_true - y_pred ))/N
    """
    Compute the derivative of loss function w.r.t. weights.
    For its computation, please refer to the slide.
    """
    return val


class LogisticRegression:
    """ Models a logistic regression classifier. """

    def __init__(self):
        """ Constructor method """

        # weights placeholder
        self._w = None

    def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        n_epochs: int
            number of gradient updates.
        learning_rate: float
            step towards the descent.
        verbose: bool
            whether or not to print the value of cost function.
        """
        n_samples, n_features = X.shape

        # weight initialization
        self._w = np.random.randn(n_features) * 0.001

        for e in range(n_epochs):

            """
            # Compute predictions   
            
            """
            print(self._w)
            p=sigmoid(np.matmul(X,self._w))
            """
            # Print loss between Y and predictions p
            # -> L = ...
            """
            L=loss(Y,p)

            # Uncomment the following lines when the loss is implemented to print it
            if verbose and e % 500 == 0:
                print(f'Epoch {e:4d}: loss={L}')

            """
            # Update w based on the gradient descent rule
            # w(t+1) = w(t) - learning_rate * dL/dw(t)
            # -> self.w = ...
            """
            self._w=self._w - learning_rate * dloss_dw(Y,p,X)


    def predict(self, X):
        """
        Function that predicts.

        Parameters
        ----------
        X: np.array
            data to be predicted. shape=(n_test_examples, n_features)

        Returns
        -------
        prediction: np.array
            prediction in {0, 1}.
            Shape is (n_test_examples,)
        """

        """
        Compute predictions.
        a) compute the dot product between X and w
        b) apply the sigmoid function (this way, y in [0,1])
        c) discretize the output (this way, y in {0,1})
        """

        product=X @ self._w
        solution=sigmoid(product)
        return (solution >= 0.5).astype(int) #bello castato
        # remove random prections before coding the solution
