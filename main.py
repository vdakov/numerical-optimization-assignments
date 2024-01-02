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
    

    extent = (x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max())
    cl_colors = ['blue', 'orange', 'purple', 'red', 'green']
    cmap = colors.ListedColormap(cl_colors)

    fig, ax = plt.subplots(1,4,figsize=(18,4))
    # ax[0] Plot showing the training loss and training accuracy
    ax[0].set_title('Training loss')
    
    # ax[1] Plot showing the confusion matrix on the test data (using matplotlib.pyplot.imshow)
    conf_mat = np.eye(len(np.unique(y_train)))
    conf = ax[1].imshow(conf_mat), ax[1].set_title('Confusion matrix (test data)')
    fig.colorbar(conf[0], ax=ax[1],shrink=0.5)
    ax[1].set_xticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_xlabel('predicted label')
    ax[1].set_yticks(list(np.arange(len(np.unique(y_train))))), ax[1].set_ylabel('actual label')

    # ax[2] Scatter plot showing the labeled training data
    for idx, cl in enumerate(['class 1', 'class 2', 'class 3', 'class 4', 'class 5']):
        ax[2].scatter(x_train[:,0][y_train==idx],x_train[:,1][y_train==idx],label=cl,c=cl_colors[idx])
    ax[2].set_title('Training data')
    ax[2].legend() 
    ax[2].set_xlabel(r'$x_1$'), ax[2].set_ylabel(r'$x_2$')

    # ax[3] Plot showing the learned decision boundary weighted by the logits output (using matplotlib.pyplot.imshow)
    N = 500
    ax[3].imshow(np.ones((N,N)), alpha=np.random.rand(N,N), origin='lower', extent=extent, cmap=cmap, interpolation="nearest")
    ax[3].set_title('Learned decision boundary')
    ax[3].set_xlabel(r'$x_1$'), ax[3].set_xlabel(r'$x_1$')
    

    """
    Start your code here

    """
    def forward(x):
        z1 = network.theta['W0']@x + network.theta['b0']
        a1 = softplus(z1)
        z2 = network.theta['W1']@a1 + network.theta['b1']
        a2 = softmax(z2)
        return a2, z2, a1, z1
    
    def softplus(z1): 
        return np.log(1.0 + np.power(np.e, z1))
    
    def softmax(z2): 
        return np.power(np.e, z2) / np.sum(np.power(np.e, z2))
    
    def backward(xs, a2, z2, a1, z1, target):
        
        grad_W0 = 1
        grad_b0 = 1
        grad_W1 = 1
        grad_b1 = 1
        return {'W0': grad_W0, 'W1': grad_W1, 'b0': grad_b0, 'b1': grad_b1}
    
    def one_hot(y):
        onehot = np.zeros(n_out)
        onehot[y-1] = 1
        return onehot
    
    def cross_entropy(a, y):
        return -np.sum(y*np.log(a))

    n_in = 2
    n_hidden = 12
    n_out = 5
    epoch = 200
    lr = 0.01

    network = NeuralNetwork(n_in, n_hidden, n_out)
    confustion_matrix = np.zeros((n_out, n_out))
    history = []

    network.theta['W0'] = np.random.uniform(low=-1.0/n_in, high=1.0/n_in, size=(n_hidden, n_in))
    network.theta['W1'] = np.random.uniform(low=-1.0/n_hidden, high=1.0/n_hidden, size=(n_out, n_hidden))
    network.theta['b0'] = np.random.uniform(low=-1.0/n_in, high=1.0/n_in, size=n_hidden)
    network.theta['b1'] = np.random.uniform(low=-1.0/n_hidden, high=1.0/n_hidden, size=n_out)

    # Training
    for _ in range(epoch):
        loss = 0
        for xs, target in zip(x_train, y_train):
            y_one_hot = one_hot(target)
            a2, z2, a1, z1 = forward(xs)
            loss += cross_entropy(a2, y_one_hot)

            gradients = backward(xs, a2, z2, a1, z1, target)
            network.theta['W0'] -= lr*gradients['W0']
            network.theta['W1'] -= lr*gradients['W1']
            network.theta['b0'] -= lr*gradients['b0']
            network.theta['b1'] -= lr*gradients['b1']
        
        loss = loss/x_train.shape[0]
        history.append(loss)


    
    # Testing
    for x, truth in zip(x_test, y_test):
        predicted = 1#np.argmax(forward(x))
        confustion_matrix[truth][predicted] += 1

    # Write down everything
    # Just copy the above code and add the 
    #network.export_model()

    return fig

if __name__ == '__main__':
    pdf = PdfPages("figures.pdf")
    fig = task()
    pdf.savefig(fig)
    pdf.close()
