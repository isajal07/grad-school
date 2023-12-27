import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
    return np. dot(Z.T, Z)

def find_pcs(cov):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvectors, eigenvalues

def project_data(Z, PCS, L):
    i = np.argmax(L)
    Z_star = np.dot(Z, PCS[i])
    return Z_star

def show_plot(Z, Z_star) :
    x= Z[:, 0]
    y = Z[:, 1]
    plt.scatter(x, y, color='green')
    X = Z_star
    y = np.zeros (Z_star.size)
    plt.scatter(x, y)
    plt.show()