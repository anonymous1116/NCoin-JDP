import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold



def gaussian_kernel(distances, bandwidth):
    """ Compute Gaussian kernel weights for given distances. """
    return np.exp(-0.5 * (distances / bandwidth) ** 2)

def local_linear_regression(X, y, X_query, bandwidth=0.5, k=20):
    """
    Perform local linear regression for multi-dimensional data.
    
    Parameters:
    - X: (n_samples, n_features) Training input data
    - y: (n_samples,) Target values
    - X_query: (m_samples, n_features) Points where predictions are made
    - bandwidth: Controls the locality of regression (smaller = more localized)
    - k: Number of nearest neighbors used for regression
    
    Returns:
    - y_pred: (m_samples,) Predictions at X_query points
    """
    y_pred = np.zeros(len(X_query))

    for i, x0 in enumerate(X_query):
        # Compute Euclidean distances
        distances = np.linalg.norm(X - x0, axis=1)
        
        # Select k nearest neighbors
        nearest_idx = np.argsort(distances)[:k]
        X_local = X[nearest_idx]
        y_local = y[nearest_idx]
        weights = gaussian_kernel(distances[nearest_idx], bandwidth)
        
        # Fit Weighted Least Squares (WLS) Regression
        model = LinearRegression()
        model.fit(X_local, y_local, sample_weight=weights)
        
        # Predict at query point
        y_pred[i] = model.predict(x0.reshape(1, -1))
    return y_pred, model



def learning_checking_kernel(Y_val, Y_val_pred, num =10000):
    Y_val = Y_val
    Y_val_pred = Y_val_pred
    _, p = Y_val.size()
    
    true_name = []
    esti_name = []
    
    for i in range(p):
        true_name.append(r'true $\theta_' + str(i) + '$')
        esti_name.append(r'$\hat{\theta}_' + str(i) + '$')
    
    indices = torch.tensor(np.random.randint(_, size=num)).to("cpu")
    Y_val = Y_val[indices,:]
    Y_val_pred = Y_val_pred[indices,:]
    
    ## Plot for model checking
    lim_left = torch.quantile(Y_val,.0001, 0).detach().cpu().numpy()
    lim_right = torch.quantile(Y_val,.9999, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, p, figsize=(20,3))
    fig.suptitle('Learning Checking', fontsize= 10)
    
    for i in range(p):
        lim0 = lim_left[i]
        lim1 = lim_right[i]

        axes[i].scatter(Y_val[:,i], Y_val_pred[:,i], marker='o', color='b', s= 1)
        axes[i].set_xlabel(true_name[i], fontsize=15)
        axes[i].set_ylabel(esti_name[i], fontsize=15)
        axes[i].plot(np.linspace(lim0, lim1, 1000), np.linspace(lim0, lim1, 1000), color = "red", linestyle='dashed', linewidth = 2.5)
        axes[i].set_axisbelow(True)
        axes[i].grid(color='gray', linestyle='dashed')
        axes[i].set_ylim([lim0, lim1])
        axes[i].set_xlim([lim0, lim1])