from typing import Union

from base_mat import base_mat, mat_type
from lazy_mat import lazy_mat
from sparse_mat import sparse_mat
from dense_mat import dense_mat

import numpy as  np

def identity (nqbits):
    return sparse_mat(mat=np.eye(int(2**nqbits)))

def apply (matrix:base_mat, vector):
    '''
    Apply a matrix to a vector ( { MATRIX } · [ vector ] )
    
    matrix: any kind of matrix that properly implements the base_mat class (e.g. dense, sparse, lazy).
    
    vector: numpy 1d array.
    '''
    if type(matrix) == lazy_mat:
        matrix = matrix.eval ()
    return ( matrix.nparray() @ np.reshape(vector, (len(vector), 1)) ).flatten()
    
def mat_scalar_mul (scalar, matrix:base_mat, lazy = True):
    '''Scalar multiplication of matrix.'''
    if lazy:
        return lazy_mat(scalar, matrix, mat_scalar_mul)
    if matrix.type == mat_type.dense:
        return dense_mat(scalar * matrix.mat)
    if matrix.type == mat_type.sparse:
        return sparse_mat(csr=[scalar * np.array(matrix.csr_val), matrix.csr_col, matrix.csr_row])

def mat_mul (left:base_mat, right:base_mat, lazy = True):
    '''Matrix multiplication. ( { LEFT MATRIX }·{ RIGHT MATRIX } )'''
    if lazy:
        return lazy_mat(left, right, mat_mul)
    # Evaluate lazy matrix
    if left.type == mat_type.lazy:
        left = left.eval()
    if right.type == mat_type.lazy:
        right = right.eval()
    # Dot product
    if left.type == mat_type.dense:
        if right.type == mat_type.dense: # Dense * Dense
            return dense_mat(np.dot(left.mat, right.mat))
        if right.type == mat_type.sparse: # Dense * Sparse
            return dense_mat(right.left_mat_dot(left.nparray()))
    if left.type == mat_type.sparse:
        if right.type == mat_type.dense: # Sparse * Dense
            return dense_mat(left.right_mat_dot(right.nparray()))
        if right.type == mat_type.sparse: # Sparse * Sparse
            return sparse_mat.dot(left, right)

def mat_tensor (left:base_mat, right:base_mat, lazy = True):
    '''Matrix tensor product. ( { LEFT MATRIX }(x){ RIGHT MATRIX } )'''
    if lazy:
        return lazy_mat(left, right, mat_tensor)
    # Evaluate lazy matrix
    if left.type == mat_type.lazy:
        left = left.eval()
    if right.type == mat_type.lazy:
        right = right.eval()
    # Tensor product
    if left.type == mat_type.dense:
        if right.type == mat_type.dense: # Dense (x) Dense
            arr = np.zeros((left.m * right.m, left.n * right.n))
            for a in range(left.m):
                for b in range(left.n):
                    for c in range(right.m):
                        for d in range(right.n):
                            arr[a*right.m + c, b*right.n + d] = left.get(a, b) * right.get(c, d)
            return dense_mat(arr)
        if right.type == mat_type.sparse: # Dense (x) Sparse
            return right.left_mat_tensor(left.nparray())
    if left.type == mat_type.sparse:
        if right.type == mat_type.dense: # Sparse (x) Dense
            return left.right_mat_tensor(right.nparray())
        if right.type == mat_type.sparse: # Sparse (x) Sparse
            return sparse_mat.tensor(left, right)
            
def vec_scalar_mul (scalar, vector):
    return scalar * np.array(vector)

def vec_tensor (left, right): # tested
    '''Vector tensor product. ( [ left vector ](x)[ right vector ] )'''
    arr = np.zeros (len(left) * len(right), dtype=complex)
    idx = 0
    for i in left:
        for j in right:
            arr[idx] = i * j
            idx += 1
    return arr

def vec_tensor_all (arr):
    '''Tensor product of an list of vectors, from right to left'''
    if len(arr) == 1:
        return arr[0]
    return vec_tensor_all(arr[:-2] + [vec_tensor(arr[-2], arr[-1])])