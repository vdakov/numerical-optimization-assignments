import numpy as np
import json
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(10)

class NeuralNetwork:
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in 
        self.n_hidden = n_hidden
        self.n_out = n_out 

        self.theta = self.init_params()

    def init_params(self):
        theta = {}
        theta['W0'] = np.array(0)
        theta['b0'] = np.array(0)
        theta['W1'] = np.array(0)
        theta['b1'] = np.array(0)
        return theta

    def export_model(self):
        with open(f'model.json', 'w') as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)

def task():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss and training accuracy
            - ax[1] Plot showing the confusion matrix on the test data (using matplotlib.pyplot.imshow)
            - ax[2] Scatter plot showing the labeled training data
            - ax[3] Plot showing the learned decision boundary weighted by the logits output (using matplotlib.pyplot.imshow)
    """
    with np.load('data.npz') as data_set:
        # get the training data
        x_train = data_set['x_train']
        y_train = data_set['y_train']

        # get the test data
        x_test = data_set['x_test']
        y_test = data_set['y_test']

    print(f'\nTraining set with {x_train.shape[0]} data samples')
    print(f'Test set with {x_test.shape[0]} data samples')
    

    extent = [x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()]
    cl_colors = ['blue', 'orange', 'purple', 'red', 'green']
    cmap = colors.ListedColormap(cl_colors)

    fig, ax = plt.subplots(1,4,figsize=(18,4))
    
    ## ax[0] Plot showing the training loss and training accuracy
    #ax[0].set_title('Training loss')
    #
    ## ax[1] Plot showing the confusion matrix on the test data (using matplotlib.pyplot.imshow)
    #conf_mat = np.eye(len(np.unique(y_train)))
    #conf = ax[1].imshow(conf_mat), ax[1].set_title('Confusion matrix (test data)')
    #fig.colorbar(conf[0], ax=ax[1],shrink=0.5)
    #ax[1].set_xticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_xlabel('predicted label')
    #ax[1].set_yticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_ylabel('actual label')
#
    ## ax[2] Scatter plot showing the labeled training data
    #for idx, cl in enumerate(['class 1', 'class 2', 'class 3', 'class 4', 'class 5']):
    #    ax[2].scatter(x_train[:,0][y_train==idx],x_train[:,1][y_train==idx],label=cl,c=cl_colors[idx])
    #ax[2].set_title('Training data')
    #ax[2].legend() 
    #ax[2].set_xlabel(r'$x_1$'), ax[2].set_ylabel(r'$x_2$')
#
    ## ax[3] Plot showing the learned decision boundary weighted by the logits output (using matplotlib.pyplot.imshow)
    #N = 500
    #ax[3].imshow(np.ones((N,N)), alpha=np.random.rand(N,N), origin='lower', extent=extent, cmap=cmap, interpolation="nearest")
    #ax[3].set_title('Learned decision boundary')
    #ax[3].set_xlabel(r'$x_1$'), ax[3].set_xlabel(r'$x_1$')

    """
    Start your code here

    """
    def forward(x, y_oh):
        z1 = network.theta['W0']@x + network.theta['b0']
        a1 = softplus(z1)
        z2 = network.theta['W1']@a1 + network.theta['b1']
        a2 = softmax(z2)
        l = cross_entropy(a2, y_oh)
        return l, a2, z2, a1, z1
    
    def predict_class(x):
        z1 = network.theta['W0']@x + network.theta['b0']
        a1 = softplus(z1)
        z2 = network.theta['W1']@a1 + network.theta['b1']
        a2 = softmax(z2)
        return np.argmax(a2)
    
    def backward(xs, a2, z2, a1, z1, target):
        d_a2_z2 = np.diag(a2) @ (np.diag(np.ones(len(a2))) - a2.T)
        y_oh = one_hot(target)
        d_L_a2 = -np.linalg.inv(np.diag(a2)) @ y_oh 
        d_z2_h2 = np.diag(np.ones(len(z2)))
        d_h2_a1 = network.theta['W1']
        d_a1_z1 = np.diag(np.exp(z1) / (np.exp(z1) + 1))
        d_z2_b1 = np.eye(len(z2), len(network.theta['b1']))
        d_z1_b0 = np.eye(len(z1), len(network.theta['b0']))
        d_h2_W1 = a1.T.reshape(len(a1.T), 1)
        d_h1_W0 = xs.T.reshape(len(xs.T), 1)
        d_z1_h1 =  np.eye(len(z1), len(z1))

        delta1 = d_L_a2 @ d_a2_z2
        delta2 = (delta1.T @ d_z2_h2 @  d_h2_a1 @ d_a1_z1).T
        
        delta1= delta1.reshape(len(delta1), 1)
        delta2= delta2.reshape(len(delta2), 1)
 

        grad_W0 = (delta2.T @ d_z1_h1).T @ d_h1_W0.T
        grad_b0 = (delta2.T @ d_z1_b0).T
        grad_W1 = (delta1.T @ d_z2_h2).T @ d_h2_W1.T
        grad_b1 = (delta1.T @ d_z2_b1).T

        return {'W0': grad_W0, 'W1': grad_W1, 'b0': grad_b0, 'b1': grad_b1}
    
    def one_hot(y):
        onehot = np.zeros(n_out)
        onehot[y] = 1
        return onehot
    
    def softplus(z1): 
        return np.log(1.0 + np.power(np.e, z1))
    
    def softmax(z2): 
        return np.power(np.e, z2) / np.sum(np.power(np.e, z2))
    
    def cross_entropy(a, y):
        return -np.sum(y*np.log(a))
    
    def flat_forward(theta_flat, x, y):
        W0 = theta_flat[:network.n_hidden*network.n_in].reshape((n_hidden,n_in))
        b0 = theta_flat[network.n_hidden*network.n_in:network.n_hidden*network.n_in + network.n_hidden]
        W1 = theta_flat[network.n_hidden*network.n_in + network.n_hidden:network.n_hidden*network.n_in + network.n_hidden + network.n_hidden*network.n_out].reshape((network.n_out, network.n_hidden))
        b1 = theta_flat[-network.n_out:]

        a = softmax(W1 @ softplus(W0 @ x  + b0) + b1)
        y_OH = one_hot(y)
        return cross_entropy(a, y_OH)
    
    def flatten(p):
        return np.concatenate((p['W0'].flatten(), p['b0'].flatten(), p['W1'].flatten(), p['b1'].flatten()))

    def check_gradient_point(x, y):
        l, a2, z2, a1, z1 = forward(x, one_hot(y))
        gradients = backward(x, a2, z2, a1, z1, y)
        analytic_grad = flatten(gradients)
        theta_flat = flatten(network.theta)
        numeric_grad = approx_fprime(theta_flat, flat_forward, 1e-3, x, y)

        return np.allclose(numeric_grad,analytic_grad, rtol=0, atol=1e-2)

    def check_gradient(x_train, y_train):
        ok =True
        for xs, ys in zip(x_train, y_train):
            ok = ok and check_gradient_point(xs, ys)

        print(f'Gradient check for all points: {ok}')


    
    n_in = 2
    n_hidden = 12
    n_out = 5
    epochs = 1000
    lr = 0.1

    network = NeuralNetwork(n_in, n_hidden, n_out)
    confustion_matrix = np.zeros((n_out, n_out))
    history = []

    network.theta['W0'] = np.random.uniform(low=-1.0/n_in, high=1.0/n_in, size=(n_hidden, n_in))
    network.theta['W1'] = np.random.uniform(low=-1.0/n_hidden, high=1.0/n_hidden, size=(n_out, n_hidden))
    network.theta['b0'] = np.random.uniform(low=-1.0/n_in, high=1.0/n_in, size=n_hidden)
    network.theta['b1'] = np.random.uniform(low=-1.0/n_hidden, high=1.0/n_hidden, size=n_out)

    check_gradient(x_train, y_train)
    
    # Training
    S = len(x_train)
    for it in range(epochs):

        loss = 0
        g_w0 = 0
        g_w1 = 0
        g_b0 = 0
        g_b1 = 0
        for xs, target in zip(x_train, y_train):
            y_one_hot = one_hot(target)
            l, a2, z2, a1, z1 = forward(xs, y_one_hot)
            loss += l

            gradients = backward(xs, a2, z2, a1, z1, target)
            g_w0 += gradients['W0']
            g_w1 += gradients['W1']
            g_b0 += gradients['b0'].flatten()
            g_b1 += gradients['b1'].flatten()

        g_w0 /= S
        g_w1 /= S
        g_b0 /= S
        g_b1 /= S

        network.theta['W0'] -= lr * g_w0
        network.theta['W1'] -= lr * g_w1
        network.theta['b0'] -= lr * g_b0
        network.theta['b1'] -= lr * g_b1
           
        loss = loss/S
        history.append(loss)

        
    y_pred_train = []
    for x, truth in zip(x_train, y_train):
        predicted = predict_class(x)
        y_pred_train.append(predicted)
    
    
    y_pred_train = np.array(y_pred_train)
    train_accuracy = len(y_train[y_train == y_pred_train]) / len(y_train)
    print(f'Train Accuracy: {train_accuracy}')
    network.export_model()

    # Testing
    y_pred_test = []
    test_loss = 0
    for x, truth in zip(x_test, y_test):
        l, a, _, _, _ = forward(x, one_hot(truth))
        predicted = np.argmax(a)
        test_loss += l
        y_pred_test.append(predicted)
        confustion_matrix[truth][predicted] += 1
    y_pred_test = np.array(y_pred_test)
    test_accuracy = len(y_test[y_test == y_pred_test]) / len(y_test)
    test_loss = test_loss/(x_test.shape[0])
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test loss: {test_loss}')


    #Plotting
    ax[0].semilogy(np.asarray(history))

    # Confusion matrix
    conf = ax[1].imshow(confustion_matrix), ax[1].set_title('Confusion matrix (test data)')
    fig.colorbar(conf[0], ax=ax[1],shrink=0.5)
    ax[1].set_xticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_xlabel('predicted label')
    ax[1].set_yticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_ylabel('actual label')

    # Scatter plots
    for idx, cl in enumerate(['class 1', 'class 2', 'class 3', 'class 4', 'class 5']):
        ax[2].scatter(x_train[:,0][y_train==idx],x_train[:,1][y_train==idx],label=cl,c=cl_colors[idx])
    ax[2].set_title('Training data')
    ax[2].legend() 
    ax[2].set_xlabel(r'$x_1$'), ax[2].set_ylabel(r'$x_2$')

    # Descion boundiries
    N = 500
    x1_values = np.linspace(extent[0], extent[1], N)
    x2_values = np.linspace(extent[2], extent[3], N)
    xx1, xx2 = np.meshgrid(x1_values, x2_values)
    meshgrid_samples = np.c_[xx1.ravel(), xx2.ravel()]
    predictions = []
    confidence = []
    for x in meshgrid_samples:
        p = forward(x, one_hot(0))[1]
        predictions.append(np.argmax(p))
        confidence.append(np.max(p))

    predictions, confidence = np.array(predictions), np.array(confidence)
    predictions, confidence = predictions.reshape(xx1.shape), confidence.reshape(xx1.shape)
    ax[3].imshow(predictions, alpha=confidence, origin='lower', extent=extent, cmap=cmap, interpolation="nearest")
    ax[3].set_title('Learned decision boundary')
    ax[3].set_xlabel(r'$x_1$')


    return fig

if __name__ == '__main__':
    pdf = PdfPages("figures.pdf")
    fig = task()
    pdf.savefig(fig)
    pdf.close()
