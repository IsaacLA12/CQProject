import numpy as np
import collections

class sparse_mat():
    def __init__(self,mat,mat_as_dict=None):  
        if mat_as_dict is not None:
            self.sm_as_dok = mat_as_dict
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
        
        else:
            self.mat_to_dok(mat)
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
        # Num of rows and cols
        self.m = len(self.csr_row)-1
        self.n = max(self.csr_col)+1
        self.scalar = 1
    
    def mat_to_dok(self,mat):
        for row in mat:
            for col,val in enumerate(row):
                if val != 0:
                    self.sm_as_dok[(row,col)] = val

    def dok_to_csr(self):
        od = collections.OrderedDict(sorted(self.sm_as_dok.items()))
        csr_val = []
        csr_col = []
        n_rows,_ = list(od.items())[-1]
        csr_row = np.zeros(n_rows[0]+2)
        for k, v in od.items():
            csr_val.append(v)
            csr_col.append(k[1])
            csr_row[k[0]+1] += 1
        return np.array(csr_val),np.array(csr_col),np.cumsum(csr_row, dtype=int)
    
    def transpose(self):
        nnz = len(self.csr_val)

        trans_val = [0 for _ in range(nnz)]
        trans_col = [0 for _ in range(nnz)]
        trans_row = [0 for _ in range(len(self.csr_row))]

        # Count elements in each column:
        cnt = [0 for _ in range(self.n)]
        for k in range(nnz):
            col = self.csr_col[k]
            cnt[col] += 1
        # Cumulative sum to set the column pointer of matrix B
        for i in range(1, self.n+1):
            trans_row[i] = trans_row[i-1] + cnt[i-1]

        for row in range(self.m):
            for j in range(self.csr_row[row], self.csr_row[row+1]):
                col = self.csr_col[j]
                dest = trans_row[col]

                trans_col[dest] = row;
                trans_val[dest] = self.csr_val[j];
                trans_row[col] += 1
        # now shift Bp
        trans_row = [0] + trans_row
        trans_row.pop(-1)

        return sparse_mat(csr=[np.array(trans_val),np.array(trans_col),np.array(trans_row)])
    
    # Operations wit dense vectors and dense matrix

    def left_vec_dot(self,arr):
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
        return 0
    def right_mat_tensor(self,dense_mat):
        return 0

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

        return sparse_mat(csr=[np.array(res_val),np.array(res_col),np.array(res_row)])

    @staticmethod
    def dot(sm1,sm2):
        return 0
    
    @staticmethod
    def tensor(sm1,sm2):
        return 0

