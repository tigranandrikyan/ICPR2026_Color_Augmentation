import numpy as np
import constants

# Fancy PCA

class FancyPCA: 
    
    def fancy_pca(self, data):
        mean_free_data = data - np.mean(data, axis=(0,1))  # Calculate the mean along the first two axes and subtract it to center the data
        data_covarianz = np.cov(mean_free_data, rowvar=False)  # Compute the covariance matrix of the centered data; rowvar=False means each column represents a variable
        eig_values, eig_vecs = np.linalg.eigh(data_covarianz)  # Compute the eigenvalues and eigenvectors of the covariance matrix
        # Randomly scale eigenvalues using a normal distribution based on mean and standard deviation
        alpha = constants.FANCY_PCA_STANDARD_DEVIATION * np.random.randn(3) + constants.FANCY_PCA_MEAN
        eig_values *= alpha
        p_mat = np.array(eig_vecs)  # Convert eigenvectors into a matrix
        add_vec = np.dot(p_mat, eig_values)  # Compute the modification vector by multiplying eigenvector matrix with eigenvalue vector
        data += add_vec  # Add the modification vector to the original data
        data = np.clip(data, 0, 1)  # Clip the data values to the range [0, 1] to avoid invalid values
        return data  # Return the transformed data
