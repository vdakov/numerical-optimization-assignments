import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def decompose_image_to_blocks(img, window_size):
    """ Rearrange img of (N,N) into non-overlapping blocks of (N_blocks,window_size**2).
        Make sure to determine N_blocks from the image size. 
    """
    N = img.shape[0]
    N_blocks = N // window_size
    blocks = img.reshape(N_blocks, window_size, N_blocks, window_size).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(N_blocks, N_blocks, -1).flatten()
    return blocks

def rearrange_image_from_blocks(blocks, img_size):
    """ Function to rearrange non-overlapping blocks of (N_blocks,window_size**2) into img (N,N). """
    N_blocks = blocks.shape[0]
    window_size = int(np.sqrt(blocks.shape[2]))
    return blocks.reshape(N_blocks, N_blocks, window_size, window_size).transpose(0, 2, 1, 3).reshape(img_size, img_size)


def DCT2_2D(d, nB):
    """ Function to get 2D DCT basis functions of size (d, d, nB, nB).
        d represents the dimensions of the DCT basis image 
        nB is the size of the non-overlapping blocks per dimension
        Reshape to (d**2, nB**2) to conveniently work with this. 
    """
    pass

def DCT2_1D(d, n):
    """ Function to get 1D DCT basis functions of size (d, n)
        n: signal dimension, d: basis functions 
    """
    dct_matrix = np.zeros((d, n))

    for k in range(d):
        for i in range(n):
            dct_matrix[k, i] = np.sqrt(2 / n) * np.cos((np.pi * k * (2 * i + 1)) / (2 * n))

    return dct_matrix


def task1(signal):
    """ Signal Denoising

        Requirements for the plots:
            -ax[0,0] - Results for low noise and K=15
            -ax[0,1] - Results for high noise and K=15
            -ax[1,0] - Results for low noise and K=100
            -ax[1,1] - Results for low noise and K=5

    """

    fig, ax = plt.subplots(2,2, figsize=(16,8))
    fig.suptitle('Task 1 - Signal denoising task', fontsize=16)

    ax[0,0].set_title('a)')
    ax[0,1].set_title('b)')
    ax[1,0].set_title('c)')
    ax[1,1].set_title('d)')

    
    var = (0.01**2, 0.03**2, 0.01**2, 0.01**2)
    d = (15, 15, 100, 5)

    """ Start of your code
    """
    K = (15,15,100,5) # 

    def frank_wolfe( x0, K, solution):

        xk = x0
        for k in range(K):
            pk = solution(xk)
            tk = 2.0/(k+1)
            xk = (1.0-tk)*xk + tk*pk
        
        return xk 

    for idx, delta2 in enumerate(var):
        noised_signal = signal + delta2*np.random.normal(size=signal.shape)
        
        A = DCT2_1D(d[idx], signal.shape[0])
        
        denoised = frank_wolfe(noised_signal, K[idx], )
        ax[idx//2, idx%2].plot(denoised)       

        



    """ End of your code
    """
    return fig

def task2(img):
    """ Image Compression

        Requirements for fig1:
            - ax[0] the groundtruth grayscale image 
            - ax[1] the compressed image from the Frank Wolfe algorithm

        Requirements for fig2:
            - ax[0] the groundtruth grayscale image 
            - ax[1] the compressed image using LASSO and \lambda=0.01
            - ax[2] the compressed image using LASSO and \lambda=0.1
            - ax[3] the compressed image using LASSO and \lambda=1.0

    """

    fig1, ax1 = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    fig1.suptitle('Task 2 - Image compression', fontsize=16)
    ax1[0].set_title('GT')
    ax1[1].set_title('Cond. GD')
    ax1[0].imshow(img,'gray')
    for ax_ in ax1:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    # lasso
    lamb_arr = np.array([0.01,0.1,1.])

    fig2, ax2 = plt.subplots(1,len(lamb_arr)+1,sharex=True,sharey=True,figsize=(10,4))
    plt.suptitle('Task 2 - LASSO', fontsize=16)
    ax2[0].set_title('GT')
    ax2[0].imshow(img,'gray')
    for ax_ in ax2:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    for l_idx, l in enumerate(lamb_arr):
        ax2[l_idx+1].set_title(r'$\lambda$=%.2f' %l)

    """ Start of your code
    """
   


    
    """ End of your code
    """
    return fig1, fig2


if __name__ == "__main__":
    # load 1D signal and 2D image
    with np.load('data.npz') as data:
        signal = data['sig']
        img = data['img']

    pdf = PdfPages('figures.pdf')
    fig_signal = task1(signal)
    figures_img = task2(img)

    pdf.savefig(fig_signal)
    for fig in figures_img:
        pdf.savefig(fig)
    pdf.close()
