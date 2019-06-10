import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from etn import coordinates


def visualize_transformation(x, model, device='cpu', figsize=(4, 3), cmap=None):
    def tensor_to_numpy(x):
        x = x.squeeze().cpu()
        if x.dim() == 3:
            x = x.permute(1, 2, 0)
        x = x.mul(model.normalization_std)
        x = x.add(model.normalization_mean)
        return x.numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_numpy(x), cmap=cmap)
    plt.title('input')
    plt.show()
    
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    pred, tf_out = model.predict(x, tf_output=True, device=device)
    
    prev_transform = lambda x: x
    with torch.no_grad():
        for transform, param, heatmap, module in zip(
            tf_out['transform'],
            tf_out['params'],
            tf_out['maps'],
            model.model.transformer.transformers):

            print(type(module).__name__)
            param = [p.item() for p in param]
            print('predicted parameter(s) =', param)

            # transformer input (after coordinate transform)
            grid = module.coords(x.shape[-2:], ulim=module.ulim, vlim=module.vlim, device=x.device)
            grid = prev_transform(grid.unsqueeze(0))
            x_in = F.grid_sample(x, grid)

            # transformer output
            grid = transform(coordinates.identity_grid(x.shape[-2:]).unsqueeze(0))
            x_out = F.grid_sample(x, grid)

            if heatmap is None:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2 + 1, figsize[1]))
            elif type(heatmap) is tuple:
                # two parameters
                f, (ax1, ax2, hm1, hm2) = plt.subplots(1, 4, figsize=(figsize[0]*4 + 3, figsize[1]))
                hm1.plot(heatmap[0].squeeze().cpu().numpy())
                hm2.plot(heatmap[1].squeeze().cpu().numpy())
                hm1.set_title('transformer feature map 1')
                hm2.set_title('transformer feature map 2')
                hm1.grid(True)
                hm2.grid(True)
            else:
                # one parameter
                f, (ax1, ax2, hm1) = plt.subplots(1, 3, figsize=(figsize[0]*3 + 2, figsize[1]))
                hm1.plot(heatmap.squeeze().cpu().numpy())
                hm1.set_title('transformer feature map')
                hm1.grid(True)
                
            ax1.imshow(tensor_to_numpy(x_in), cmap=cmap)
            ax2.imshow(tensor_to_numpy(x_out), cmap=cmap)
            ax1.set_title('transformer input in canonical coordinates')
            ax2.set_title('transformer output')
            prev_transform = transform
            plt.show()
            
        grid = model.model.coords(x.shape[-2:], device=x.device)
        grid = prev_transform(grid.unsqueeze(0))
        x_in = F.grid_sample(x, grid)
        f = plt.figure(figsize=figsize)
        plt.imshow(tensor_to_numpy(x_in), cmap=cmap)
        plt.title('output after final coordinate transformation')
        plt.show()
        