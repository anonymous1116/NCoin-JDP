import torch
import torch.distributions as D
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def NCoinJDP_train(X, Y, net_str, device="cpu", p_train=0.7, N_EPOCHS=250, lr=1e-3, val_batch = 10_000, early_stop_patience = 20):
    torch.set_default_device(device)
    X = X.to(device)
    Y = Y.to(device)
    net = copy.deepcopy(net_str)
    net = net.to(device)  # ensure net is on the correct device

    L = Y.size(0)
    L_train = int(L * p_train)
    L_val = L - L_train
    
    indices = torch.randperm(L)

    # Divide Data
    X_train = X[indices[:L_train]]
    Y_train = Y[indices[:L_train]]

    X_val = X[indices[L_train:]]
    Y_val = Y[indices[L_train:]]

    del X, Y  # Free memory for the full dataset

    # Define the batch size
    BATCH_SIZE = 64

    # Use torch.utils.data to create a DataLoader
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))

    def weighted_mse_loss(input, target, weight):
        return (weight * (input - target) ** 2).mean()

    
    out_range = [
        torch.quantile(Y_train, .01, 0).detach().cpu().numpy(),
        torch.quantile(Y_train, .99, 0).detach().cpu().numpy()
    ]
    weight_1 = torch.tensor(1/(out_range[1] - out_range[0])**2)
    
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9)
    #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    weight_decay_scheduler = WeightDecayScheduler(optimizer, initial_weight_decay=1e-5, factor=0.5, patience=10)
    
    train_error_plt = []
    val_error_plt = []

    best_val_loss = float('inf')
    best_model_state = None

    # Create DataLoader for entire training set (for evaluation)
    eval_train_dataset = TensorDataset(X_train, Y_train)
    eval_train_dataloader = DataLoader(eval_train_dataset, batch_size=val_batch, shuffle=True, generator=torch.Generator(device=device))

    # Create DataLoader for validation set (for evaluation)
    eval_val_dataset = TensorDataset(X_val, Y_val)
    eval_val_dataloader = DataLoader(eval_val_dataset, batch_size=val_batch, shuffle=True, generator=torch.Generator(device=device))
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(N_EPOCHS):
        net.train()
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_batch_pred = net(x_batch)
            loss = weighted_mse_loss(y_batch_pred, y_batch, weight_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and id_batch % 300 == 0:
                loss_value = loss.item()
                current = (id_batch + 1)* len(x_batch)
                print(f"train_loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

        with torch.no_grad():
            net.eval()
            # Evaluate on validation set in batches
            val_loss_accum = 0.0
            for (x_batch, y_batch) in eval_val_dataloader:
                y_pred_batch = net(x_batch)
                batch_loss = weighted_mse_loss(y_pred_batch, y_batch, weight_1).item()
                val_loss_accum += batch_loss
            val_loss = val_loss_accum * val_batch/ L_val
            val_error_plt.append(torch.tensor(val_loss))
                
        if epoch % 10 == 0:
            with torch.no_grad():
                net.eval()
            train_loss_accum = 0.0
            for (x_batch, y_batch) in eval_train_dataloader:
                y_pred_batch = net(x_batch)
                batch_loss = weighted_mse_loss(y_pred_batch, y_batch, weight_1).item()
                train_loss_accum += batch_loss
            train_loss = train_loss_accum * val_batch/ L_train  # Normalize by total number of samples
            train_error_plt.append(torch.tensor(train_loss))  # Store as tensor for consistency
            
            print(f"Epoch {epoch + 1}\n-------------------------------")
            print(f"train_loss {train_loss:>7f} val_loss {val_loss:>7f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0  # Reset the counter for early stopping
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve}/{early_stop_patience} epochs.")

        # Early stopping condition
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered. Restoring best model...")
            print(f"It stops within {epoch}/{N_EPOCHS}")
            early_stop = True
            break

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        weight_decay_scheduler.step(epoch)
    
    torch.cuda.empty_cache()
    del net, X_train, Y_train, X_val, Y_val, dataloader, weight_1
    print(f"============= Best validation loss: {best_val_loss} =============")
    return best_model_state, best_val_loss

def compute_mad(X):
    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        X = X.to('cuda')

    # Compute the median for each column
    medians = torch.median(X, dim=0).values  # Shape: (num_columns,)

    # Compute the absolute deviations from the median
    abs_deviation = torch.abs(X - medians)  # Broadcasting over rows

    # Compute the MAD for each column
    mad = torch.median(abs_deviation, dim=0).values  # Shape: (num_columns,)

    # Return the result on the CPU
    return mad.cpu()

def cond_mad_train(X, resid, net_var, device = "cpu", p_train = 0.7, N_EPOCHS = 250, lr = 1e-3, val_batch = 10_000):
    torch.set_default_device(device)
    X = X.to(device)
    resid = resid.to(device)
    resid = torch.max(torch.abs(resid), torch.ones(1) * 1e-30).log()
    
    return NCoinJDP_train(X, resid, net_var, device, p_train, N_EPOCHS, lr, val_batch = val_batch)


def ABC_rej(x0, X_cal, Y_cal, tol, device):
    # Move all tensors to the target device at once
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    Y_cal = Y_cal.to(device)
    
    # Calculate the squared Euclidean distance
    mad = compute_mad(X_cal)
    mad = torch.reshape(mad, (1, X_cal.size(1))).to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2/(mad+1e-8) **2, 1))

    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    
    # Select points within tolerance and return to CPU if needed
    return X_cal[wt1].cpu(), Y_cal[wt1].cpu()


class WeightDecayScheduler:
    def __init__(self, optimizer, initial_weight_decay, factor, patience):
        """
        Custom scheduler to adjust weight decay.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust weight decay for.
            initial_weight_decay (float): Starting weight decay value.
            factor (float): Multiplicative factor to reduce weight decay.
            patience (int): Number of epochs to wait before reducing weight decay.
        """
        self.optimizer = optimizer
        self.initial_weight_decay = initial_weight_decay
        self.factor = factor
        self.patience = patience
        self.epochs_since_last_update = 0

    def step(self, epoch):
        if self.epochs_since_last_update >= self.patience:
            self.epochs_since_last_update = 0
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] *= self.factor
                print(f"Epoch {epoch}: Reduced weight decay to {param_group['weight_decay']:.6e}")
        else:
            self.epochs_since_last_update += 1


def learning_checking(X, Y, net, num = 10000, name = None):
    net = net.to("cpu")
    X = X.to("cpu")
    Y = Y.to("cpu")
    _, p = Y.size()
    true_name = []
    esti_name = []
    
    for i in range(p):
        true_name.append(r'true $\theta_' + str(i) + '$')
        esti_name.append(r'$\hat{\theta}_' + str(i) + '$')
    
    indices = torch.tensor(np.random.randint(_, size=num)).to("cpu")
    X_test = X[indices,:]
    Y_test = Y[indices,:]
    
    
    with torch.no_grad():
        net.eval()
        tmp = net(X_test)
        tmp = tmp.detach().cpu().numpy()

    ## Plot for model checking
    lim_left = torch.quantile(Y_test,.0001, 0).detach().cpu().numpy()
    lim_right = torch.quantile(Y_test,.9999, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(tmp[0]), figsize=(20,3))
    fig.suptitle('Learning Checking', fontsize= 10)

    for i in range(p):
        lim0 = lim_left[i]
        lim1 = lim_right[i]

        tmp1 = tmp[:, i]
        axes[i].scatter(Y_test[:,i], tmp1, marker='o', color='b', s= 1)
        axes[i].set_xlabel(true_name[i], fontsize=15)
        axes[i].set_ylabel(esti_name[i], fontsize=15)
        axes[i].plot(np.linspace(lim0, lim1, 1000), np.linspace(lim0, lim1, 1000), color = "red", linestyle='dashed', linewidth = 2.5)
        axes[i].set_axisbelow(True)
        axes[i].grid(color='gray', linestyle='dashed')
        axes[i].set_ylim([lim0, lim1])
        axes[i].set_xlim([lim0, lim1])
