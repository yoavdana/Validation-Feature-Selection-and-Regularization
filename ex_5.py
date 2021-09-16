import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model,datasets

#generate data
MAX_DEG=15
X_MAX=2.2
X_MIN=-3.2
NUM_SAMPLES=1500
SIGMA=2


##############################################################################
#generate data
##############################################################################
def epsilon(num_samples,sigma):
    return np.random.normal(0,sigma,num_samples)

def f_x(x):

    return (x + 3)*(x + 2)*(x + 1)*(x-1)*(x-2)

def gen_X(min,max,num_samples):
    return np.random.uniform(min,max,num_samples)


data_X=gen_X(X_MIN,X_MAX,NUM_SAMPLES)
data_y=f_x(data_X)+epsilon(NUM_SAMPLES,SIGMA)
D_x,D_y=data_X[:1000],data_y[:1000] #training +validation set
T_x,T_y=data_X[1000:],data_y[1000:] #test set
S_x = D_x[:500] #train set
S_y = D_y[:500] #train set
V_x = D_x[500:] #validation set
V_y = D_y[500:] #validation set

#plot data
plt.figure()
plt.plot(data_X,data_y,'*')
plt.show()
##############################################################################
#train polynomial fit model
##############################################################################
def x_to_polynom (x, deg):
    X = np.ones((deg+1, x.size))
    for d in range(deg):
        X[d+1, :] = X[d, :] * x
    return X

def train(x,y,deg):
    X=x_to_polynom(x,deg)
    return np.linalg.inv(X@X.T) @ X @ y

def validate(x,y,h_d,deg):
    X=x_to_polynom(x,deg)
    y_est=h_d @ X
    mse=np.mean((y_est - y)**2)
    return mse


##############################################################################
### apply 2-fold cross validation on polynomial fit
##############################################################################

##train
h_d=[train(S_x,S_y,d+1) for d in range(MAX_DEG) ]
##validation
losses=[validate(V_x, V_y, h_d[d], d+1) for d in range(MAX_DEG)]
##find optimum polynomial degree
best_deg = np.argmin(np.array(losses))
best_h = h_d[best_deg]
print('Best degree is:', best_deg+1)

##############################################################################
### apply 5-fold cross validation on polynomial fit
##############################################################################

def k_sets(x,y,k):
    k_y=np.split(y,k)
    k_x = np.split(x,k)
    return k_x,k_y

k_x,k_y=k_sets(D_x,D_y,5)
V_x,V_y=k_x[-1],k_y[-1]
h_d=0
losses_5=0
##############################################################################
###compare 2-fold with 5-fold
##############################################################################
for i in range(len(k_y)-1):
    h_d_curr=np.array([train(k_x[i], k_y[i], d + 1) for d in range(MAX_DEG)])
    losses_5 += np.array([validate(V_x, V_y, h_d_curr[d], d + 1) for d in
                          range(MAX_DEG)])
    h_d+=h_d_curr
h_d/=(len(k_y)-1)
losses_5/=(len(k_y)-1)
degs=np.arange(1,MAX_DEG+1)
plt.figure()
plt.plot(degs,losses)
plt.plot(degs,losses_5)
plt.legend(['2-fold','5-fold'])
plt.show()
best_deg = np.argmin(np.array(losses_5))
best_h = h_d[best_deg]
print('Best degree is:', best_deg+1)

h_best = train(D_x, D_y, best_deg)
# Validation step
loss_best = validate(T_x, T_y, h_best, best_deg)
print('The test loss is:', loss_best)

##############################################################################
#apply k-fold Regularization cross validation on polynomial fit
##############################################################################
#load data
X, y = datasets.load_diabetes(return_X_y=True)
num_evaluations=50
alpha_range = np.linspace(0.001, 2, num=num_evaluations)
train_size=60
X_train = X[:train_size, :]
y_train = y[:train_size]
X_test = X[train_size:, :]
y_test = y[train_size:]

##############################################################################
#different regulation definition
##############################################################################
def train_ridge(X, y, alpha):
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X, y)
    return clf

def train_lasso(X, y, alpha):
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X, y)
    return clf

def train_lr(X, y):
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    return clf

def validate(X, y, clf):
    return np.mean((clf.predict(X) - y)**2)
##############################################################################
kfold = 4
avg_train_err_ridge = np.zeros(num_evaluations)
avg_validation_err_ridge = np.zeros(num_evaluations)
groups = np.remainder(np.arange(y_train.size), kfold)
for k in range(kfold):
    S_k_x = X_train[groups != k]
    S_k_y = y_train[groups != k]
    V_k_x = X_train[groups == k]
    V_k_y = y_train[groups == k]
    h_d = [train_ridge(S_k_x, S_k_y, alpha) for alpha in alpha_range]
    loss_train_d = [validate(S_k_x, S_k_y, clf) for clf in h_d]
    loss_validation_d = [validate(V_k_x, V_k_y, clf) for clf in h_d]
    avg_train_err_ridge += np.array(loss_train_d) / kfold
    avg_validation_err_ridge += np.array(loss_validation_d) / kfold

# Lasso regression

avg_train_err_lasso = np.zeros(num_evaluations)
avg_validation_err_lasso = np.zeros(num_evaluations)
groups = np.remainder(np.arange(y_train.size), kfold)
for k in range(kfold):
    S_k_x = X_train[groups != k]
    S_k_y = y_train[groups != k]
    V_k_x = X_train[groups == k]
    V_k_y = y_train[groups == k]
    h_d = [train_lasso(S_k_x, S_k_y, alpha) for alpha in alpha_range]
    loss_train_d = [validate(S_k_x, S_k_y, clf) for clf in h_d]
    loss_validation_d = [validate(V_k_x, V_k_y, clf) for clf in h_d]
    avg_train_err_lasso += np.array(loss_train_d) / kfold
    avg_validation_err_lasso += np.array(loss_validation_d) / kfold



plt.plot(alpha_range, avg_train_err_ridge, label='train: ridge')
plt.plot(alpha_range, avg_validation_err_ridge, label='validation: ridge')
plt.plot(alpha_range, avg_train_err_lasso, label='train: lasso')
plt.plot(alpha_range, avg_validation_err_lasso, label='validation: lasso')
plt.legend()
plt.show()