class base_mat:
    
    def __init__(self):
        self.m = None # num row
        self.n = None # num col
        self.type = None
        
    def shape (self):
        '''Returns Shape of the matrix as a tuple. (row, col)'''
        return self.m, self.n
    
    def get (self, i, j):
        '''Returns the element at position i, j.'''
        return None
    
    def transpose (self):
        '''Returns the transpose of the matrix as a new matrix.'''
        return None
    
    def nparray (self):
        '''Returns a numpy 2d array repreesentation of the matrix.'''
        return None

class mat_type:
    dense = 0
    sparse = 1
    lazy = 2
    