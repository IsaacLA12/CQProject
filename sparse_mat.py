import numpy as np
import collections
import itertools


class sparse_mat():
    """
    The class implements sparse matrices using the csr (compressed rows) model.
    Stores the three csr vectors and also a dok (dictionary of {k:(row,col) v:(value)})
    representation of the dense matrix.
    Also stores the dimensions of the matrix.


    Compressed Sparse Row matrix
    This can be instantiated in several ways:
        sparse_mat(mat=M)
            with a dense matrix or 2-dim ndarray M
        sparse_mat(csr=S,dim=d)
            with another sparse matrix by just sending the triplet of csr array-like implementation 
            (values,columns,compressed rows), the dim parameter is a tuple representing (#rows,#cols)
        sparse_mat(dok=D,dim=d)
            with a dok representation of a sparse matrix, just need the dictionary and the dim as before
            
    Attributes
    ----------
    dok : dictionary / hash table
        Dok representation of the matrix
    csr_val : list
         CSR format data array of the non zero elements of the matrix
    csr_col : list
        CSR format data array of the column index of the non zero elements of the matrix
    csr_row : list
        CSR format data array of the compressed rows of the non zero elements of the matrix
    m : int
        First dimension of the matrix aka number of rows
    n : int
        Second dimension of the matrix aka number of columns
   
    Notes on CSR
    ------------
    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products
    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive i.e. adding elements (consider LIL or DOK)

    Methods
    -------
    See each method for detailed information, the following is just a list.
        mat_to_dok(mat)
        show()
        shape()
        csr_to_dok()
        dok_to_csr()
        csr_to_mat()
        transpose()
        left_vec_dot()
        right_vec_dot()
        left_mat_dot()
        right_mat_dot()
        left_mat_tensor()
        right_mat_tensor()
    
    Static methods
    --------------
    See each static method for detailed information, the following is just a list.
        add()
        dot()
        tensor()

    Notes on methods
    ----------------
    Recall that in Python, methods are applied to instances of the class and Static methods to the class itself.
    So for example:
        >>> m1 = sparse_mat(M)
        >>> m2 = m1.transpose()
        >>> m3 = sparse_mat.dot(m1,m2)
    """
    def __init__(self,mat=None,dok=None,csr=None,dim=None): 
        """
        Instantiates the sparse matrix by the given matrix in some format and its dimension.
        Dimension can be avoided in case of instantiating from a dense matrix, error halts if not provided.
        """
        # Matrix given in Dok format
        if dok is not None:
            self.dok = dok
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
            self.m = dim[0]
            self.n = dim[1]
            
        # Matrix given in CSR format
        elif csr is not None:
            self.csr_val = csr[0]
            self.csr_col = csr[1]
            self.csr_row = csr[2]
            self.m = dim[0]
            self.n = dim[1]
            self.dok = self.csr_to_dok()

        # Matrix given in Dense format
        elif mat is not None:
            self.dok = self.mat_to_dok(mat)
            res = self.dok_to_csr()
            self.csr_val = res[0]
            self.csr_col = res[1]
            self.csr_row = res[2]
            self.m = len(mat)
            self.n = len(mat[0])
        
        # Otherwise, error
        else:
            raise ValueError("Parameters needed for sparse matrix")
    
    def show(self):
        """
        Prints the three CSR arrays of the matrix
        """
        print("Values:",self.csr_val)
        print("Cols:",self.csr_col)
        print("Compresed rows:",self.csr_row)
    
    def shape(self):
        """
        Returns Shape of the matrix as a tuple
        """
        return self.m,self.n

    def mat_to_dok(self,mat):
        """
        Returns Dok format of a Dense matrix.
        """
        dok = {}
        for row_ind,row in enumerate(mat):
            for col_ind,val in enumerate(row):
                # Store each element diferent from 0
                if val != 0:
                    dok[(row_ind,col_ind)] = val
        return dok

    def csr_to_dok(self):
        """
        Returns Dok format of a CSR matrix, CSR matrix previously instantiated.
        (Should be called only when instantiated)
        """
        dok = {}
        for row_ind in range(self.m):
            # Getting row elements and columns
            cols = self.csr_col[self.csr_row[row_ind]:self.csr_row[row_ind+1]]
            vals = self.csr_val[self.csr_row[row_ind]:self.csr_row[row_ind+1]]
            for col_ind,val in zip(cols,vals):
                # Store each element in dok format 
                dok[(row_ind,col_ind)] = val
        return dok

    def dok_to_csr(self):
        """
        Returns CSR format of a Dok matrix, Dok matrix previously instantiated.
        (Should be called only when instantiated)
        """
        # Sort dictionary by rows
        od = collections.OrderedDict(sorted(self.sm_as_dok.items()))
        n_rows,_ = list(od.items())[-1]
        # Prepare the three lists for csr
        csr_val = []
        csr_col = []
        csr_row = np.zeros(n_rows[0]+2)

        # Add to the CSR arrays the correspondent values, col index and row compresion
        for k, v in od.items():
            csr_val.append(v)
            csr_col.append(k[1])
            csr_row[k[0]+1] += 1

        # Extra step needed for row compression (cumulative sum)
        return np.array(csr_val),np.array(csr_col),np.cumsum(csr_row, dtype=int)
    
    def csr_to_mat(self):
        """
        Returns Dense format of the CSR matrix, CSR matrix previously instantiated.
        """
        dense = np.zeros((self.m,self.n))
        # Iterate though rows
        for row in range(self.m):
            # Uncompress rows
            cols = self.csr_col[self.csr_row[row]:self.csr_row[row+1]]
            vals = self.csr_val[self.csr_row[row]:self.csr_row[row+1]]
            for col,val in zip(cols,vals):
                # Store each element in dense mat 
                dense[row][col] = val
        return dense
    
    def transpose(self):
        """
        Returns the transpose of the sparse matrix as a new sparse matrix.
        The algorithm transforms internally from a CSR format to a CSC format. 
        (Compresion of columns istead of rows)
        """
        # Initialize transposed matrix
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
    def right_vec_dot(self,arr):
        """
        Returns the column vector of application of the matrix to vector on the right: v -> Av
        """
        if len(arr) != self.n:
            raise ValueError("Inconsistent shapes")
        res = [0 for _ in range(self.n)]
        for i in range(self.n):
            for k in range(self.csr_row[i], self.csr_row[i+1]):
                j = self.csr_col[k]
                res[i] += self.csr_val[k] * arr[j][0]
        return np.reshape(np.array(res),(self.m,1))

    def left_vec_dot(self,arr):
        """
        Returns the row vector of application of the matrix to vector on the left: v -> vA
        """
        if len(arr[0]) != self.m:
            raise ValueError("Inconsistent shapes")
        res = [0 for _ in range(self.m)]
        # Same as right_vec_dot but the sparse matrix gets transposed
        trans = self.transpose()
        for i in range(self.m):
            for k in range(trans.csr_row[i], trans.csr_row[i+1]):
                j = trans.csr_col[k]
                res[i] += trans.csr_val[k] * arr[0][j]
        return np.reshape(np.array(res),(1,self.n))
    
    def left_mat_dot(self,dense_mat):
        """
        Returns the Dense matrix after application of the matrix to the Dense Matrix on the left: DM -> DM·SM
        """
        # Simple algorith for matrix multiplication, traversing left mat by rows and right mat by columns.
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
        """
        Returns the Dense matrix after application of the matrix to the Dense Matrix on the right: DM -> SM·DM
        """
        # Algorithm consist of application of right_vec_dot m times n times beeing n the number of columns of Dense mat
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
        """
        Compute the tensor product of the matrix and the Dense Matrix on the left: DM -> DM x SM
        Returns a Sparse matrix
        """
        # Algorithm consist of incremental building of a sparse matrix in CRS format by computing the resulting rows
        # Since there is no restriction on zero values in the Dense mat, we have to check elements and not compute
        # the multiplication but still remember the index and position in order to keep consistency
        # Can also be interpreted as the kroneker product of matrices
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
        """
        Compute the tensor product of the matrix and the Dense Matrix on the right: DM -> SM x DM
        Returns a Sparse matrix
        """
        # Algorithm is basically the same as left_mat_tensor but in this case the code is cleaner since
        # we are traversing now the sparse matrix by rows and 'copying chunks' of the dense mat so the
        # traverseof  the sparse mat by columns is much simpler and we dont need to take care of zeros
        # as those get eliminated while we are constucting the result matrix
        # Note that in this case the most inner loop can not be parallelized as we did in the mentioned method above
        # with the use of the high order function map
        # Can also be interpreted as the kroneker product of matrices
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


    # Static methods apply when operating with TWO sparse matrices so we dont have to deal with commutativity problems
    # as the user selects the applying order by the order of the parameters, equivalently those methods can be seen
    # as prefix operators
    @staticmethod
    def add(sm1,sm2):
        """
        Returns the addition of two Sparse matrices as a nes Sparse matrix
        """
        # TODO-Code can be much easier and optimized 
        if sm1.m != sm2.m or sm1.n != sm2.n:
            raise ValueError("Inconsistent shapes")
        res_val = []
        res_col = []
        res_row = [0]
        for row in range(sm1.m):
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
        """
        Returns the dot product of two Sparce matrices as anew Sparse Matrix
        """
        # Optimized algorithm exploiting CSR structure using a SPA (sparse accomulator) and using row-wise multiplication
        if sm1.n != sm2.m:
            raise ValueError("Inconsistent shapes")
        res_val = []
        res_col = []
        res_row = [0]
        for r in range(sm1.m):
            start, end = sm1.csr_row[r], sm1.csr_row[r + 1]
            tmp_val = sm1.csr_val[start:end]
            tmp_col = sm1.csr_col[start:end]
            spa = [0 for _ in range(sm1.n)]
            for i,e in zip(tmp_col,tmp_val):
                start2, end2 = sm2.csr_row[i], sm2.csr_row[i + 1]
                tmp_val2 = sm2.csr_val[start2:end2]
                tmp_col2 = sm2.csr_col[start2:end2]
                for c,v in zip(tmp_col2,tmp_val2):
                    spa[c] +=e*v
            n_elems = 0
            for k,x in enumerate(spa):
                if x != 0:
                    res_val.append(x)
                    res_col.append(k)
                    n_elems += 1
            res_row.append(res_row[-1]+ n_elems)
        
        return sparse_mat(csr=[res_val,res_col,res_row],dim=[sm1.m,sm2.n])

    
    @staticmethod
    def tensor(sm1,sm2):
        """
        Returns the tensor product of two Sparse matrices as a new Sparse matrix
        """
        # Follows the idea of combining the the two previous tensor product algorithms metioned before
        # Iterate through the rows of left mat, for each elem 'copy chunk' of each matrix in desired final position
        # We don't have the problem of 0 elemnts sice we ensure there are no zeros on our CSR implementations.
        # We can also exploit multiples properies by iterating just though existing element and can easily precompute 
        # the number of elements that we will add on each row of the final matrix, nevertheless a new problem arises
        # as we have to use a flag to avoid repeating calculations if a row has more than one element.
        # In other words we can precompute how many element we will add on a new row by knowing the rows of the two matrices   
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
