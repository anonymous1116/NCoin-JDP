import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from NCoinJDP import ABC_rej


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

def bandwidth_select(x0, X_val, Y_val, X_train, Y_train, device, tol = .05):
    k = int(tol * X_train.size(0))
    bandwidths = np.exp(np.linspace(-5,2,30))
    bandwidth_list = []
    for j in range(Y_val.size(1)):
        y_pred = []
        models = []
        
        X_val_loc, Y_val_loc = ABC_rej(x0, X_val, Y_val, tol = tol, device = device)
        
        for l in range(len(bandwidths)):
            try:
                y_pred_j_tmp, model_j = local_linear_regression(X_train, Y_train[:, j], X_val_loc, bandwidth=bandwidths[l], k=k)
                y_pred_j = torch.mean((torch.tensor(y_pred_j_tmp) - Y_val_loc[:,j]) ** 2) ** (1/2)
                print(y_pred_j)
            except Exception as e:
                print(f"Error encountered with bandwidth {bandwidths[l]}: {e}")
                y_pred_j, model_j = np.nan, None  # or another placeholder value

            y_pred.append(y_pred_j)
            models.append(model_j)
        print(f"{l}th bandwidth")
        # Convert y_pred to a numpy array for easier handling
        y_pred_array = np.array(y_pred, dtype=float)  # Ensure it's a float array to handle np.nan

        # Mask out nan values
        valid_indices = ~np.isnan(y_pred_array)  # Boolean mask: True where values are not nan

        # Check if there are any valid values
        if np.any(valid_indices):
            min_index = np.argmin(y_pred_array[valid_indices])  # Find min index in valid values
            min_index = np.where(valid_indices)[0][min_index]  # Map back to original index
            print(f"The index of the minimum valid y_pred value is: {min_index}")
        else:
            print("All values in y_pred are NaN, cannot determine the minimum index.")

        bandwidth_list.append(bandwidths[min_index])
    return bandwidth_list