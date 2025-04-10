import torch

def reduce_density(x, decrease_rate=0.95): 

    y = torch.zeros_like(x) # Use advanced indexing to assign the selected values 
    for i in range(x.size(0)):
        non_zero_indices = (x[i] != 0).nonzero(as_tuple=False) 
        num_elements = round(non_zero_indices.shape[0] * decrease_rate)
        selected_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_elements]] 

        y[i,0,selected_indices[:, 1], selected_indices[:, 2]] = x[i, 0, selected_indices[:, 1], selected_indices[:, 2]] 

    return y
