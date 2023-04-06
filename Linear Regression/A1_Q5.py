# %% [markdown]
# # CMPT 726/410 Assignment 1 Question 5
# 
# Import necessary Python packages.

# %%
import numpy as np
import scipy.io
path_prefix = ""

# %% [markdown]
# ## Part (b)

# %%
mdict = scipy.io.loadmat("{}a.mat".format(path_prefix))
#mdict = scipy.io.loadmat("/Users/davidqian/Desktop/CMPT 410/Assignment/Assiignment1/CMPT726-410_A1_Starter_Code/a.mat")
x = mdict['x'][0]
u = mdict['u'][0]

# Required: write code below that produces two variables, A and B, which
# are scalars of type numpy.float64 that represent the model parameters 
# A and B

### start 1 ###
W  = np.array([0,0])
Vx = np.array(x)
Y  = Vx[1:]
Vx = Vx[:-1]
Vu = np.array(u)
Vu = Vu[:-1]
X  = np.column_stack((Vx, Vu))
W = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)

A = W[0]
B = W[1]
#sizex = X.shape
#sizeY = Y.shape
#print(sizex)
#print(sizeY)
### end 1 ###

# Do not modify the lines below
assert(isinstance(A, np.float64))
assert(isinstance(B, np.float64))

print("A: {}, B: {}".format(A, B))

# %% [markdown]
# ## Part (d)

# %%
mdict = scipy.io.loadmat("{}b.mat".format(path_prefix))
#mdict = scipy.io.loadmat("/Users/davidqian/Desktop/CMPT 410/Assignment/Assiignment1/CMPT726-410_A1_Starter_Code/b.mat")
X_raw = mdict['x']
U_raw = mdict['u']

# Required: write code below that produces two variables, A and B, which
# are 3 x 3 float64 matrices of type numpy.ndarray that represent the 
# model parameters A and B

### start 2 ###
Vx = np.array(X_raw)
Vx = Vx[:,:,0]
Y  = Vx[1:]
Vx = Vx[:-1]
Vu = np.array(U_raw)
Vu = Vu[:,:,0]
Vu = Vu[:-1]
X  = np.column_stack((Vx, Vu))
W = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)
A = W[:3, :]
B = W[3:, :]
A = A.T
B = B.T
### end 2 ###

# Do not modify the lines below
assert(isinstance(A, np.ndarray))
assert(isinstance(B, np.ndarray))
assert(A.shape == (3,3))
assert(B.shape == (3,3))
assert(A.dtype == np.float64)
assert(B.dtype == np.float64)

print("A: \n{}, \nB: \n{}".format(A, B))

# %% [markdown]
# ## Part (f)
# 
# Solving for the OLS formulation of (b)

# %%
mdict = scipy.io.loadmat("{}b.mat".format(path_prefix))
#mdict = scipy.io.loadmat("/Users/davidqian/Desktop/CMPT 410/Assignment/Assiignment1/CMPT726-410_A1_Starter_Code/b.mat")

X_raw = mdict['x']
U_raw = mdict['u']

# Required: write code below that produces two variables, A and B, which
# are 3 x 3 float64 matrices of type numpy.ndarray that represent the 
# model parameters A and B

### start 3 ###
Vx = np.array(X_raw)
Vx = Vx[:,:,0]
Ty  = Vx[1:]
Y = Ty.reshape(3*Ty.shape[0], 1)

Vx = Vx[:-1]
Vu = np.array(U_raw)
Vu = Vu[:,:,0]
Vu = Vu[:-1]
x_t  = np.column_stack((Vx, Vu))
X = np.zeros((3*Ty.shape[0], 18))
j = 0 
for i in range(x_t.shape[0]):
    X[j,:6] = x_t[i,:]
    j = j+1
    X[j,6:12] = x_t[i,:]
    j = j+1
    X[j,12:] = x_t[i,:]
    j=j+1

W = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)
va1 = np.array(W[:3,0])
va2 = np.array(W[6:9,0])
va3 = np.array(W[12:15,0])
vb1 = np.array(W[3:6,0])
vb2 = np.array(W[9:12,0])
vb3 = np.array(W[15:,0])
A = np.vstack((va1, va2, va3))
B = np.vstack((vb1,vb2,vb3))
### end 3 ###

# Do not modify the lines below
assert(isinstance(A, np.ndarray))
assert(isinstance(B, np.ndarray))
assert(A.shape == (3,3))
assert(B.shape == (3,3))
assert(A.dtype == np.float64)
assert(B.dtype == np.float64)

print("A: \n{}, \nB: \n{}".format(A, B))


