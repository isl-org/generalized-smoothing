import numpy as np

def gram_schmidt(matrix, choice):

    """ Make the choice (row or column) orthogonal using Gram-Schmidt """

    if choice == 'row':
        nrow = matrix.shape[0]
        orthos = np.copy(matrix) # orthogonal vectors
        normals = np.zeros(matrix.shape) # normalized orthos
        for k in range(1, nrow):
            normals[(k-1),:] = orthos[(k-1),:]/np.linalg.norm(orthos[(k-1),:]) 
            orthos[k,:] -= np.dot(np.transpose(normals[0:k,:]), np.dot(normals[0:k,:], matrix[k,:]))
        normals[(nrow-1),:] = orthos[(nrow-1),:]/np.linalg.norm(orthos[(nrow-1),:])
    else:
        raise NotImplementedError

    return orthos, normals
