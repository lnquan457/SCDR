import os

import numpy as np
from inc_pca import IncPCA
from scipy.stats import ortho_group
from procrustes import orthogonal
import matplotlib.pyplot as plt


def plot_one(data, title):
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.savefig(os.path.join("logs", "{}.jpg".format(title)))
    plt.show()


if __name__ == '__main__':
    # random input 10x7 matrix A
    a_big = np.random.rand(300, 2)
    a = a_big[:100]

    # random orthogonal 2x2 matrix T (acting as Q)
    t = ortho_group.rvs(2)

    # target matrix B (which is a shifted AT)
    b = np.dot(a, t) + np.random.rand(1, 2)

    # orthogonal Procrustes analysis with translation
    result = orthogonal(a, b, scale=False, translate=True)

    # compute transformed matrix A (i.e., A x Q)
    aq = np.dot(result.new_a, result.t)
    aq_big = np.dot(a_big, result.t)
    aqqq = IncPCA.geom_trans(b, a)

    # display Procrustes results
    print("Procrustes Error = ", result.error)
    print("\nDoes the obtained transformation match variable t? ", np.allclose(t, result.t))
    print("Does AQ and B matrices match?", np.allclose(aq, result.new_b))

    plot_one(a, "Origin A")
    plot_one(b, "Origin B")
    plot_one(aqqq, "siPCA A")

    print("Transformation Matrix T = ")
    print(result.t)
    print("")

    # print("Matrix A (after translation and scaling) = ")
    # print(result.new_a)
    # print("")
    plot_one(result.new_a, "A after translation and scaling")

    # print("Matrix AQ = ")
    # print(aq)
    # print("")
    plot_one(aq, "Matrix AQ")
    plot_one(aq_big, "Matrix AQ Big")

    # print("Matrix B (after translation and scaling) = ")
    # print(result.new_b)
    plot_one(result.new_b, "New B")