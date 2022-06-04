import numpy as np
def PCA(n_components,data):
    # removing the last 4 columns 
    data =data.iloc[:, :-4]
    # covariance matrix
    cov_matrix=np.cov([data['column 1'],data['column 2'],data['column 3'],data['column 4'],data['column 5']],bias=True)
    # eigenvalues and eigenvectors 
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    # sort the eigenvectors with the same order
    sorted_eigenvectors = eigenvectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    return np.dot(eigenvector_subset.transpose(),data.transpose()).transpose()
