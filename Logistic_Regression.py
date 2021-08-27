import numpy as np
import matplotlib.pyplot as plt


## Data Generation

mat_covs = np.array([[[1,0.9],[0.9,1]],[[1,0.9],[0.9,1]]])
# mat_covs = np.array([[[1,0.8],[0.8,1]],[[1,0.8],[0.8,1]]])

mus =  np.array([[4,3],[2,3]])
Ns = np.array([400,400])

X = np.zeros((0,mus.shape[1]))
Y = np.zeros(0)

cls = 0
for mu,mat_cov,N in zip(mus, mat_covs, Ns):
    X_ = np.random.multivariate_normal(mu, mat_cov, N)
    Y_ = np.ones(N)*cls
    X = np.vstack((X,X_))
    Y = np.hstack((Y,Y_))
    cls += 1
    
cls_unique = np.unique(Y)

def plot_data():
    legends = []
    for cls in cls_unique:
        idx = Y==cls
        plt.plot(X[idx,0],X[idx,1],'.')
        legends.append(cls.astype('int'))
        cls += 1

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(legends)
    plt.grid(True)
    
plot_data()
plt.show()

## Train Data
N_iter = 2000
N = X.shape[0]
alpha = 1
beta_0 = np.random.random()
beta_1 = np.random.random()
beta_2 = np.random.random()
x1 = X[:,0]
x2 = X[:,1]
y = Y
rng = np.random.default_rng(2021)

history = []
history.append([beta_0,beta_1,beta_2])
for i in range(N_iter):
    
    beta = beta_2 * x2 + beta_1 * x1 + beta_0
    sig_beta = 1/(1+np.exp(-beta))
    diff_sig_beta = np.exp(-beta)/(1+np.exp(-beta))**2
    
    diffbeta = x2
    beta_2_next = beta_2 - alpha/N*( (2*sig_beta*diff_sig_beta*diffbeta).sum() - 2* (y*diff_sig_beta*diffbeta).sum()  )
    
    diffbeta = x1
    beta_1_next = beta_1 - alpha/N*( (2*sig_beta*diff_sig_beta*diffbeta).sum() - 2* (y*diff_sig_beta*diffbeta).sum()  )
    
    diffbeta = 1
    beta_0_next = beta_0 - alpha/N*( (2*sig_beta*diff_sig_beta*diffbeta).sum() - 2* (y*diff_sig_beta*diffbeta).sum()  )
    
    beta_2, beta_1, beta_0 = beta_2_next, beta_1_next, beta_0_next
    history.append([beta_0,beta_1,beta_2])
history = np.array(history)


## Plot beta_0, beta_1, beta_2
plt.plot(history[:,0])
plt.plot(history[:,1])
plt.plot(history[:,2])

plt.show()


## Plot Trained Classifier
xt1 = np.linspace(-2, 10, 100).reshape(-1)
xt2 = np.linspace(1, 7, 100).reshape(-1)

xx1, xx2 = np.meshgrid(xt1, xt2)

(beta_0, beta_1, beta_2) = history[i]
x = xx1*beta_1 + xx2*beta_2 + beta_0
z = 1/(1+np.exp(-x))
plt.contourf(xt1,xt2,z)
plot_data()
plt.xlim([xt1.min(),xt1.max()])
plt.ylim([xt2.min(),xt2.max()])
plt.colorbar()
plt.clim(0,1)
plt.show()


## Plot by Time
xt1 = np.linspace(-2, 10, 100).reshape(-1)
xt2 = np.linspace(1, 7, 100).reshape(-1)

xx1, xx2 = np.meshgrid(xt1, xt2)

idxs = np.arange(len(history))[:5*10:5]
for i in idxs:
#     (beta_0, beta_1, beta_2) in history:
    (beta_0, beta_1, beta_2) = history[i]
    x = xx1*beta_1 + xx2*beta_2 + beta_0
    z = 1/(1+np.exp(-x))
    plt.contourf(xt1,xt2,z)
    plot_data()
    plt.xlim([xt1.min(),xt1.max()])
    plt.ylim([xt2.min(),xt2.max()])
    plt.colorbar()
    plt.clim(0,1)
    plt.show()
