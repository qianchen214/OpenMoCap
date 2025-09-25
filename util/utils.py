import numpy as np

def get_transfer_matrix(vecs_1, vecs_2):
    '''
    Attention: the three points should be non-collinear
    input: list 1 and list 2 ([[x1,y1,z1],[x2,y2,z2],[....]])
    return: R and T
    '''
    _vecs_1 = np.array(vecs_1)
    _vecs_2 = np.array(vecs_2)
    _mean_vec1 = np.mean(_vecs_1, axis=0)
    _mean_vec2 = np.mean(_vecs_2, axis=0)
    _shift1 = _vecs_1 - _mean_vec1.reshape(1, 3)  # np.tile(_mean_vec1, (N, 1))
    _shift2 = _vecs_2 - _mean_vec2.reshape(1, 3)  # np.tile(_mean_vec2, (N, 1))
    X = _shift1.T
    Y = _shift2.T
    _t = X.dot(Y.T)
    U, Sigma, Vt = np.linalg.svd(_t)
    _reflection = np.identity(3)
    _reflection[2, 2] = np.linalg.det(Vt.T.dot(U.T))
    R = Vt.T.dot(_reflection)
    R = R.dot(U.T)
    # print (np.shape(R),np.shape(_mean_vec1))
    T = - R.dot(_mean_vec1) + _mean_vec2
    T = T.reshape(-1, 1)
    return R, T


import torch

def get_transfer_matrices_gpu(vecs_1, vecs_2, device='cuda'):
    '''
    Attention: the three points in each sample should be non-collinear
    input: tensors vecs_1 and vecs_2 with shape (N, L, 3)
    return: batch of R and T matrices for each sample
    '''
    # Ensure input is on the specified device
    vecs_1 = vecs_1.to(device)
    vecs_2 = vecs_2.to(device)
    
    # Compute means across the second dimension (L)
    mean_vec1 = torch.mean(vecs_1, dim=1, keepdim=True)
    mean_vec2 = torch.mean(vecs_2, dim=1, keepdim=True)
    
    # Subtract means
    shift1 = vecs_1 - mean_vec1
    shift2 = vecs_2 - mean_vec2
    
    # Batch matrix multiplication
    X = shift1.transpose(1, 2)  # Shape: (N, 3, L)
    Y = shift2.transpose(1, 2)  # Shape: (N, 3, L)
    T = torch.matmul(X, Y.transpose(1, 2))  # Shape: (N, 3, 3)
    
    # Batched SVD
    U, Sigma, Vt = torch.linalg.svd(T)
    
    # Handle possible reflections
    reflections = torch.eye(3, device=device).unsqueeze(0).repeat(T.shape[0], 1, 1)
    determinants = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    reflections[:, 2, 2] = determinants
    
    # Compute rotations
    R = torch.matmul(Vt.transpose(1, 2), torch.matmul(reflections, U.transpose(1, 2)))
    
    # Compute translations
    T = -torch.matmul(R, mean_vec1.transpose(1, 2)) + mean_vec2.transpose(1, 2)
    
    return R, T.squeeze(-1)
