

import torch

def resid_chunk_process(X, Y, net, chunk_size = 10_000,device = "cpu", bounds = None):
    # Adjust this based on your GPU memory
    X = X.to(device)
    Y = Y.to(device)
    net = net.to(device)
    

    num_chunks = X.size(0) // chunk_size
    resid_tmp = []

    with torch.no_grad():
        net.eval()
        for i in range(num_chunks + 1):
            start = i * chunk_size
            end = (i + 1) * chunk_size if (i + 1) * chunk_size < X.size(0) else X.size(0)

            X_chunk = X[start:end].to(device)
            Y_chunk = Y[start:end].to(device)
            Y_chunk_predict = net(X_chunk)
            if bounds is not None:
                bounds_tensor = torch.tensor(bounds).to(device)
                Y_chunk_predict = torch.clamp(Y_chunk_predict, bounds_tensor[:,0] ,bounds_tensor[:,1]) 
            resid_chunk = Y_chunk - Y_chunk_predict
            resid_tmp.append(resid_chunk.cpu())  # Move back to CPU to free GPU memory
    del X_chunk, Y_chunk, resid_chunk, Y_chunk_predict
    # Clear cached memory
    torch.cuda.empty_cache()
    return torch.cat(resid_tmp, dim=0)