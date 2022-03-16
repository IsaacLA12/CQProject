import numpy as np
import collections
import itertools
from operator import add
from functools import reduce

class SPA():
    def __init__(self,n):
        self.w  = [0 for _ in range(n)]
        self.b  = [False for _ in range(n)]
        self.ls = []
    
    def scatter(self,val,pos):
        if not self.b[pos]:
            self.w[pos] = val
            self.b[pos] = True
            self.ls.append(pos)
        else:
            self.w[pos] += val

    def gather(self,val,col,nzc):
        nzi = 0
        for x in self.ls:
            0
        return nzi

class sparse_mat():
    """
    The class implements sparse matrices using the csr (compressed rows) model.
    Stores the three csr vectors and also a dok (dictionary of {k:(row,col) v:(value)})
    representation of the dense matrix.
    Also stores the dimensions of the matrix.
    """
    def __init__(self,mat=None,mat_as_dict=None,csr=None,dim=None): 
        if mat_as_dict is not None:
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
            self.m = dim[0]
            self.n = dim[1]
        
        elif csr is not None:
            self.csr_val = csr[0]
            self.csr_col = csr[1]
            self.csr_row = csr[2]
            self.m = dim[0]
            self.n = dim[1]
        else:
            self.mat_to_dok(mat)
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
            self.m = len(mat)
            self.n = len(mat[0])

        # Num of rows and cols
        self.sm_as_dok = {}
        if mat_as_dict is not None:
            self.sm_as_dok = mat_as_dict
        else:
            self.csr_to_dok()
    
    def mat_to_dok(self,mat):
        """
        Converts the dense matrix to dok format and stores it as a class atribute.
        """
        for row in mat:
            for col,val in enumerate(row):
                # Store each element diferent from 0
                if val != 0:
                    self.sm_as_dok[(row,col)] = val

    def csr_to_dok(self):
        """
        Converts the csr matrix to dok format and stores it as a class atribute.
        """
        for row in range(self.m):
            cols = self.csr_col[self.csr_row[row]:self.csr_row[row+1]]
            vals = self.csr_val[self.csr_row[row]:self.csr_row[row+1]]
            for col,val in zip(cols,vals):
                # Store each element in dok format 
                self.sm_as_dok[(row,col)] = val

    def dok_to_csr(self):
        """
        Returns the dok sparse matrix to csr sparse matrix format.
        """
        # Sort dictionary by rows
        od = collections.OrderedDict(sorted(self.sm_as_dok.items()))
        # Prepare the three lists for csr
        csr_val = []
        csr_col = []
        n_rows,_ = list(od.items())[-1]
        csr_row = np.zeros(n_rows[0]+2)
        for k, v in od.items():
            csr_val.append(v)
            csr_col.append(k[1])
            csr_row[k[0]+1] += 1
        return np.array(csr_val),np.array(csr_col),np.cumsum(csr_row, dtype=int)
    
    def csr_to_mat(self):
        res = np.zeros((self.m,self.n))
        for row in range(self.m):
            cols = self.csr_col[self.csr_row[row]:self.csr_row[row+1]]
            vals = self.csr_val[self.csr_row[row]:self.csr_row[row+1]]
            for col,val in zip(cols,vals):
                # Store each element in dok format 
                res[row][col] = val
        return res
    
    def transpose(self):
        """
        Returns a the transpose of the sparse matrix as a new sparse matrix.
        The algorithm transforms internally from a csr format to a csc format, compresion of columns istead of rows.
        """
        nnz = len(self.csr_val)

        trans_val = [0 for _ in range(nnz)]
        trans_col = [0 for _ in range(nnz)]
        trans_row = [0 for _ in range(len(self.csr_row))]

        # Count elements in each column:
        cnt = [0 for _ in range(self.n)]
        for k in range(nnz):
            col = self.csr_col[k]
            cnt[col] += 1
        # Cumulative sum to set the column pointer of transposed matrix
        for i in range(1, self.n+1):
            trans_row[i] = trans_row[i-1] + cnt[i-1]

        for row in range(self.m):
            for j in range(self.csr_row[row], self.csr_row[row+1]):
                col = self.csr_col[j]
                dest = trans_row[col]

                trans_col[dest] = row;
                trans_val[dest] = self.csr_val[j];
                trans_row[col] += 1
        # now shift trans_row
        trans_row = [0] + trans_row
        trans_row.pop(-1)

        return sparse_mat(csr=[np.array(trans_val),np.array(trans_col),np.array(trans_row)],dim=[self.n,self.m])
    
    # Operations with dense vectors and dense matrix

    def left_vec_dot(self,arr):
        """

        """
        if len(arr[0]) != self.m:
            raise ValueError("Inconsistent shapes")
        res = [0 for _ in range(self.m)]
        trans = self.transpose()
        for i in range(self.m):
            for k in range(trans.csr_row[i], trans.csr_row[i+1]):
                j = trans.csr_col[k]
                res[i] += trans.csr_val[k] * arr[0][j]
        return np.reshape(np.array(res),(1,self.n))

    def right_vec_dot(self,arr):
        if len(arr) != self.n:
            raise ValueError("Inconsistent shapes")
        res = [0 for _ in range(self.n)]
        for i in range(self.n):
            for k in range(self.csr_row[i], self.csr_row[i+1]):
                j = self.csr_col[k]
                res[i] += self.csr_val[k] * arr[j][0]
        return np.reshape(np.array(res),(self.m,1))
    
    def left_mat_dot(self,dense_mat):
        n = len(dense_mat[0])
        if n != self.m:
            raise ValueError("Inconsistent shapes")
        m = len(dense_mat)
        res = np.ndarray((m,self.n))
        csr_row = 0
        for i in range(n):
            start, end = self.csr_row[i], self.csr_row[i + 1]
            for j in range(start, end):
                col, val = self.csr_col[j], self.csr_val[j]
                for k in range(n):
                    dense_value = dense_mat[k][csr_row]
                    res[k][col] += val * dense_value
            csr_row += 1
        return res

    def right_mat_dot(self,dense_mat):
        m = len(dense_mat)
        if self.n != m:
            raise ValueError("Inconsistent shapes")
        n = len(dense_mat[0])
        res = np.ndarray((self.m,n))
        for i in range(n):
            aux = self.right_vec_dot(dense_mat[:,i,None])
            res[:,i] = np.reshape(aux,(self.m))
        return res
    
    def left_mat_tensor(self,dense_mat):
        m = len(dense_mat)
        n = len(dense_mat[0])
        res_val = []
        res_col = []
        res_row = [0]
        num_elems_row = []
        for i in range(self.m):
            num_elems_row.append(self.csr_row[i+1]-self.csr_row[i])
        n_elems = len(num_elems_row)
        cum_elems_row = np.cumsum(num_elems_row)
        for row in dense_mat:
            for i in num_elems_row:
                res_row.append(res_row[-1]+i*n)
            for r in range(self.m):
                for col,elem in enumerate(row):
                    if elem != 0:
                        start, end = self.csr_row[r], self.csr_row[r + 1]
                        tmp_val = self.csr_val[start:end]
                        tmp_col = self.csr_col[start:end]
                        res_val.append(list(map(lambda x: x*elem,tmp_val)))
                        res_col.append(list(map(lambda x: x+col*n,tmp_col)))
            zero_count = row.count(0)
            for i in range(zero_count): 
                res_row[-n_elems:] = list(map(lambda x,y: x-y,res_row[-n_elems:],cum_elems_row))
        
        # Flatten list
        res_val = list(itertools.chain(*res_val))
        res_col = list(itertools.chain(*res_col))

        return sparse_mat(csr=[res_val,res_col,res_row],dim=[self.m*m,self.n*n])


    def right_mat_tensor(self,dense_mat):
        m = len(dense_mat)
        n = len(dense_mat[0])
        res_val = []
        res_col = []
        res_row = [0]
        for r in range(self.m):
            start, end = self.csr_row[r], self.csr_row[r + 1]
            tmp_val = self.csr_val[start:end]
            tmp_col = self.csr_col[start:end]
            for c,e in zip(tmp_col,tmp_val):
                for row in dense_mat:
                    zero_count = row.count(0)
                    res_row.append(res_row[-1]+(end-start)*(n-zero_count))
                    for col,elem in enumerate(row):
                        if elem != 0:
                            res_val.append(elem*e)
                            res_col.append(col+c*n)

        return sparse_mat(csr=[res_val,res_col,res_row],dim=[self.m*m,self.n*n])

    def show(self):
        print("Values:",self.csr_val)
        print("Cols:",self.csr_col)
        print("Compresed rows:",self.csr_row)
    
    def shape(self):
        return self.m,self.n

    # Static methods apply when operating with TWO sparse matrices

    @staticmethod
    def add(sm1,sm2):
        if sm1.m != sm2.m or sm1.n != sm2.n:
            raise ValueError("Inconsistent shapes")
        res_val = []
        res_col = []
        res_row = [0]
        for row in range(len(sm1.csr_row)-1):
            cols_row_sm1 = sm1.csr_col[sm1.csr_row[row]:sm1.csr_row[row+1]]
            cols_row_sm2 = sm2.csr_col[sm2.csr_row[row]:sm2.csr_row[row+1]]

            vals_row_sm1 = sm1.csr_val[sm1.csr_row[row]:sm1.csr_row[row+1]]
            vals_row_sm2 = sm2.csr_val[sm2.csr_row[row]:sm2.csr_row[row+1]]

            aux_dic_merge = {}
            for c,e in zip(cols_row_sm1,vals_row_sm1):
                if c in aux_dic_merge:
                    aux_dic_merge[c] += e
                else:
                    aux_dic_merge[c] = e
            
            for c,e in zip(cols_row_sm2,vals_row_sm2):
                if c in aux_dic_merge:
                    aux_dic_merge[c] += e
                else:
                    aux_dic_merge[c] = e
            
            od = collections.OrderedDict(sorted(aux_dic_merge.items()))
            for k, v in od.items():
                res_val.append(v)
                res_col.append(k)
            res_row.append(res_row[row]+len(aux_dic_merge))

        return sparse_mat(csr=[np.array(res_val),np.array(res_col),np.array(res_row)],dim=[sm1.m,sm1.n])

    @staticmethod
    def dot(sm1,sm2):
        return 0
    
    @staticmethod
    def tensor(sm1,sm2):
        res_val = []
        res_col = []
        res_row = [0]
        for r in range(sm1.m):
            start, end = sm1.csr_row[r], sm1.csr_row[r + 1]
            tmp_val = sm1.csr_val[start:end]
            tmp_col = sm1.csr_col[start:end]
            flag = False
            for r2 in range(sm2.m):
                start2, end2 = sm2.csr_row[r2], sm2.csr_row[r2 + 1]
                tmp_val2 = sm2.csr_val[start2:end2]
                tmp_col2 = sm2.csr_col[start2:end2]
                if not flag:
                    res_row.append(res_row[-1]+(end-start)*(end2-start2))
                print(tmp_val2)
                for col,elem in zip(tmp_col2,tmp_val2):
                    for c,e in zip(tmp_col,tmp_val):
                        res_val.append(elem*e)
                        res_col.append(col+c*sm2.n)
            flag = True
            print("end")
        return sparse_mat(csr=[res_val,res_col,res_row],dim=[sm1.m*sm2.m,sm1.n*sm2.n])
        



[[1., 0., 0., 0.],
 [5., 0., 0., 1.],
 [0., 0., 2., 0.],
 [0., 3., 0., 0.]]

[[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 5.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  0.,  0.],
 [25.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5., 0.,  0.,  1.],
 [ 0.,  0., 10.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  2.,  0.],
 [ 0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 3.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 10.,  0.,  0.,  2.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0., 15.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.,  0.,  9.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]]