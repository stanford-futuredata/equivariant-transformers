"""
Experiments on Projective MNIST dataset.
"""

import numpy as np
import torch
import fire
from experiments import Model, Dataset
from etn import coordinates, networks, transformers


class MNISTModel(Model):
    tf_default_opts = {
        'in_channels': 1, 
        'kernel_size': 3, 
        'nf': 32, 
        'strides': (2, 1),
    }
    
    net_default_opts = {
        'nf': 32, 
        'p_dropout': 0.3, 
        'pad_mode': (None, 'cyclic'), 
        'pool': (True, True, False),
    }
    
    normalization_mean = torch.Tensor([16.2884])
    normalization_std = torch.Tensor([56.2673])
    
    def __init__(self, 
                 tfs=[transformers.ShearX, 
                      transformers.HyperbolicRotation,
                      transformers.PerspectiveX,
                      transformers.PerspectiveY],
                 coords=coordinates.logpolar_grid, 
                 net=networks.BasicCNN,
                 equivariant=True,
                 tf_opts=tf_default_opts,
                 net_opts=net_default_opts,
                 seed=None,
                 load_path=None):
        """MNIST model"""
        super().__init__(tfs=tfs, coords=coords, net=net, 
                         equivariant=equivariant, tf_opts=tf_opts,
                         net_opts=net_opts, seed=seed, load_path=load_path)
        
    def __str__(self):
        return "Projective MNIST classification"
        
    def _load_dataset(self, path, num_examples=None):
        return Dataset(path=path, num_examples=num_examples, 
                       normalization=(self.normalization_mean, 
                                      self.normalization_std))

    def train(self, 
              num_epochs=300,
              batch_size=128, 
              optimizer_opts={'amsgrad': True, 'lr': 2e-3, 'weight_decay': 0.},
              lr_schedule={'step_size': 1, 'gamma': 0.99},
              **kwargs):
        super().train(
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            optimizer_opts=optimizer_opts, 
            lr_schedule=lr_schedule,
            **kwargs)


if __name__ == '__main__':
    fire.Fire(MNISTModel)
