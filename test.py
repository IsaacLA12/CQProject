
class A ():
    def __init__ (self):
        self.var1 = "class A"
        self.var2 = 233
        
    def fun1 (self):
        print ("[class A]", self.var2)
    
    def fun2 (self):
        print ("[fun 2]", self.var1)

class B (A):
    def __init__ (self):
        super().__init__ ()
        self.var1 = "class B"
    
    def fun1 (self):
        print ("[class B]", self.var2)
    
import mat_math_util
import numpy as np

from dense_mat import dense_mat, mat_type
from sparse_mat import sparse_mat
from lazy_mat import lazy_mat

from quantum_registor import quantum_registor

def test1 ():
    a = A()
    b = B()
    a.fun1()
    b.fun1()
    print ()
    a.fun2()
    b.fun2()
    
def test2 (): # vec tensor product
    a = np.array ([2,3,4])
    b = np.array ([1,2])
    print (mat_math_util.vec_tensor(a, b))
    print (np.tensordot(a, b, 0).flatten())

def test3 (): # sparse mat
    m = np.array([ [1,0,0,0], [0,0,1,0], [0,0,0,0], [0,1,0,0] ])
    print (m)
    a = sparse_mat(mat=m)
    print (a.type == mat_type.sparse)
    print (a.dok)
    print (a.shape())
    print (a.nparray())

def test4 (): # dense tensor product
    a = dense_mat (np.array ([[0,1],[1,0]]))
    b = dense_mat (np.array([ [1,0,0,0], [0,0,1,0], [0,0,0,0], [0,1,0,0] ]))
    print (mat_math_util.mat_tensor(a, b, False))

def test5 (): # lazy mat
    a = dense_mat (np.array ([[0,1],[1,0]]))
<<<<<<< HEAD
<<<<<<< HEAD
    b = sparse_mat (mat=[ [1, 0], [0, 1] ])
    # b = dense_mat (np.array([ [1, 0], [0, 1] ]))
    c = mat_math_util.mat_mul(b, a)
=======
=======
>>>>>>> parent of 0acbc55 (update)
    # b = sparse_mat (mat=[ [1, 0], [0, 1] ])
    b = dense_mat (np.array([ [1, 0], [0, 1] ]))
    c = mat_math_util.mat_mul(a, b)
>>>>>>> parent of 0acbc55 (update)
    d = mat_math_util.mat_mul(b, c)
    # print (d.nparray())
    # print ( b.nparray()@(b.nparray()@a.nparray()) )
    print (b.left_mat_dot( np.array ([[0,1],[1,0]]) ))
    # print (a.nparray()@np.array([ [1, 0], [0, 1] ]))
    
if __name__ == '__main__':
    # r = quantum_registor(3)
    # print (r.h_all())
    test5()
