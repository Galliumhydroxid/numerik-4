import datetime

import numpy as np
from scipy import misc
from matplotlib import pyplot as plt

COMPRESSION_RATIOS = []
COMPRESSION_ERRORS = []

def get_size(matrices):
    size = 0
    for matrix in matrices:
        prod = 1
        for dimension in matrix.shape:
            prod = prod * dimension
        size = size + prod
    return size

def frobenius(A):
    norm = 0
    for i in range(A.shape[0]):
        sum = 0
        for j in range(A.shape[1]):
            sum = sum + pow(A[i, j], 2)
        norm = norm + sum
    return np.sqrt(norm)


def set_title():
    plt.suptitle(f"Peter Albrecht (2411389), Fabian Lübbe (2421736)\n"
                 f"Time: {datetime.datetime.now()}")

def compress(k):
    image = misc.ascent()
    u, s, vh = np.linalg.svd(image)
    u_out = u[:,:k]
    s_out = s[:k]
    vh_out = vh[:k,:]
    output = np.dot(np.dot(u_out, np.diag(s_out)), vh_out)

    # compression ratio
    uncompressed_size = get_size([u, s, vh])
    compressed_size = get_size([u_out, s_out, vh_out])
    ratio = compressed_size / uncompressed_size
    COMPRESSION_RATIOS.append(ratio)

    # compression error
    COMPRESSION_ERRORS.append(frobenius(image - output)/frobenius(image))
    return output

K_VALUES = [5, 20, 75]

if __name__ == "__main__":
    plt.gray()
    fig, axisses = plt.subplots(1, len(K_VALUES))
    #fig.tight_layout(pad=5)
    for i, k in enumerate(K_VALUES):
        matrix = compress(k)
        axisses[i].imshow(matrix)
        axisses[i].set_xticklabels([])
        axisses[i].set_yticklabels([])
        axisses[i].set_title(f"K={k},\n"
                             f"Compression = {round(COMPRESSION_RATIOS[i], 2)},\n"
                             f"Error = {round(COMPRESSION_ERRORS[i], 2)}",
                             fontsize=10)
    set_title()
    plt.savefig('images.png')
    plt.show()
    # plot of errors
    COMPRESSION_ERRORS = []
    end = 512
    for i in range(1, end+1):
        print(f"{i}/{end} - {round(100*(i/(end)))}%")
        compress(i)
    t = np.arange(1, end+1, 1)
    plt.plot(t, COMPRESSION_ERRORS)
    #plt.x_label("Matrix size")
    set_title()

    plt.savefig('error.png')
