import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def decompose_image_to_blocks(img, window_size):
    """ Rearrange img of (N,N) into non-overlapping blocks of (N_blocks,window_size**2).
        Make sure to determine N_blocks from the image size. 
    """
    h, w = img.shape
    n_b = window_size
    num_blocks = (h * w) / np.square(n_b)
    blocks = []
    
    for i in range(h // n_b):
        for j in range(w // n_b):
            block = img[i*n_b:(i+1)*n_b, j*n_b:(j+1)*n_b]
            block = block.flatten()
            blocks.append(block)

    blocks = np.array(blocks)

    return blocks

def rearrange_image_from_blocks(blocks, img_size):
    """ Function to rearrange non-overlapping blocks of (N_blocks,window_size**2) into img (N,N). """
    N_blocks = int(np.sqrt(blocks.shape[0]))  
    window_size = int(np.sqrt(blocks.shape[1]))
    return blocks.reshape(N_blocks, N_blocks, window_size, window_size).transpose(0, 2, 1, 3).reshape(img_size, img_size)


def DCT2_2D(d, nB):
    """ Function to get 2D DCT basis functions of size (d, d, nB, nB).
        d represents the dimensions of the DCT basis image 
        nB is the size of the non-overlapping blocks per dimension
        Reshape to (d**2, nB**2) to conveniently work with this. 
    """
    dct_matrix = np.zeros((d, d, nB, nB))

    n = nB

    
    for l in range(d):
        alpha_l =  np.sqrt(1 / n) if l == 0 else np.sqrt(2 / n)
        for m in range(d):
            alpha_m =  np.sqrt(1 / n) if m == 0 else np.sqrt(2 / n)
            for i in range(nB):
                for j in range(nB):
                    a = alpha_l * alpha_m
                    dct_matrix[l, m, i, j] = a * np.cos((np.pi / n) * l * (i + 0.5)) * np.cos((np.pi / n) * m * (j + 0.5))

    return dct_matrix.reshape(d**2, nB**2) 
                    

def compute_gradient_task_1(A, x, b):
    return A @ (A.T @ x - b)


def conditional_gradinent_norm(x, p_k, grad):
    return (x - p_k).T @ grad



def DCT2_1D(d, n):
    """ Function to get 1D DCT basis functions of size (d, n)
        n: signal dimension, d: basis functions 
    """
    dct_matrix = np.zeros((d, n))


    for j in range(d):
        for i in range(n):
            if j != 0:
                alpha = np.sqrt(2 / n) 
                dct_matrix[j, i] = alpha * np.cos((np.pi / n) * j * (i + 0.5))
            else:
                alpha = 1.0 / np.sqrt(n)
                dct_matrix[j, i] = alpha * np.cos((np.pi / n) * j * (i + 0.5))
            

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

    K = 250
    

    def frank_wolfe(x0, K, b, A):

        xk = x0
        
        
        for k in range(1, K):
            pk = np.zeros(x0.shape[0])
            grad = compute_gradient_task_1(A, xk, b)
            pk_index = np.argmin(grad, keepdims=True)[0]
            pk[pk_index] = 1
    
            if(np.isclose(conditional_gradinent_norm(xk, pk, grad), 1e-3, atol=1e-3)):
                print(k)
                break
            tk = 2.0/(k+1)
            xk = (1.0-tk)*xk + tk*pk
            

        return A.T @ xk 
    

    n = signal.shape[0] 
    for idx, sigma2 in enumerate(var):
        noised_signal = signal + np.random.normal(0, np.sqrt(sigma2), size=n)

        A = DCT2_1D(d[idx], n)
        x0 = np.zeros(d[idx])
        x0[0] = 1 #can be anything, as long it is withing convex set


        denoised = frank_wolfe(x0, K, noised_signal, A)
        ax[idx//2, idx%2].plot(signal, color='black', label='Original Signal')
        ax[idx//2, idx%2].plot(noised_signal, color='red', label='Noised Signal')   
        ax[idx//2, idx%2].plot(denoised, color='green', label='Denoised Signal')
        ax[idx//2, idx%2].legend()

        



    """ End of your code
    """
    return fig

def compute_gradient_task_2(A, x, b):
    return  A.T @ (A @ x - b) # as b in R^n^2 

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




    def frank_wolfe_img_compression(K, b, A, t):

        x0 = np.random.rand(b.shape[0])
        xk = (x0 / np.sum(x0)) * 0.01

        for k in range(1, K):
            gradient = compute_gradient_task_2(A, xk, b)
            gradient[0] = 0
            grad_index = np.argmax(np.abs(gradient))

            sign = np.sign(gradient[grad_index])
            e_i = np.zeros(x0.shape)
            e_i[grad_index] = 1 
            pk = -t * sign * e_i
            tk = 2.0 / (k + 2)

            xk = (1.0 - tk) * xk + tk * pk


        xk[0] = b[0]  
        
   
        return A.T @ xk # should be like this since A.T here is our A in the formulation?
    
    n_b = 8
    blocks = decompose_image_to_blocks(img, n_b)
    d = 8
    t = 0.01
    A = DCT2_2D(d, n_b)
    K = 100
    compressed = []



    for b_s in blocks:
        x_s = frank_wolfe_img_compression(K, b_s, A, t)
        compressed.append(x_s)
    
    compressed = np.array(compressed)
    rearranged = rearrange_image_from_blocks(compressed, 256)
    
    ax1[1].imshow(rearranged,'gray')


    ##LASSO


    for l_idx, l in enumerate(lamb_arr):
        lasso_compressed = []

        for b_s in blocks:
            A_t_b_s = A.T @ b_s 
            x_s =  np.abs(np.abs(A_t_b_s) - l)* np.sign(A_t_b_s) # is not quite the same formula? 
            lasso_compressed.append(A.T @ x_s) # same thing as before?

        lasso_compressed = np.array(lasso_compressed)
        lasso_rearranged = rearrange_image_from_blocks(lasso_compressed, 256)

        ax2[l_idx+1].imshow(lasso_rearranged, 'gray')
            

   

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
