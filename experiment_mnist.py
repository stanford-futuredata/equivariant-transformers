"""
Experiments on Projective MNIST dataset.
"""

import numpy as np
import torch
import fire
from experiments import Model, Dataset
from etn import coordinates, networks, transformers


class MNISTModel(Model):
    # transformer defaults
    tf_default_opts = {
        'in_channels': 1, 
        'kernel_size': 3, 
        'nf': 32, 
        'strides': (2, 1),
    }
    
    # classification network defaults
    net_default_opts = {
        'nf': 32, 
        'p_dropout': 0.3, 
        'pad_mode': (None, 'cyclic'), 
        'pool': (True, True, False),
    }
    
    # optimizer defaults
    optimizer_default_opts = {
        'amsgrad': True,
        'lr': 2e-3,
        'weight_decay': 0.,
    }
    
    # learning rate schedule defaults
    lr_default_schedule = {
        'step_size': 1,
        'gamma': 0.99,
    }
    
    # dataset mean and standard deviation
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
        tf_opts_copy = dict(self.tf_default_opts)
        tf_opts_copy.update(tf_opts)
            
        net_opts_copy = dict(self.net_default_opts)
        net_opts_copy.update(net_opts)
        
        super().__init__(tfs=tfs, coords=coords, net=net, 
                         equivariant=equivariant, tf_opts=tf_opts_copy,
                         net_opts=net_opts_copy, seed=seed, load_path=load_path)
        
    def __str__(self):
        return "Projective MNIST classification"
        
    def _load_dataset(self, path, num_examples=None):
        return Dataset(path=path, num_examples=num_examples, 
                       normalization=(self.normalization_mean, 
                                      self.normalization_std))

    def train(self, 
              num_epochs=300,
              batch_size=128, 
              optimizer_opts=optimizer_default_opts,
              lr_schedule=lr_default_schedule,
              **kwargs):
        optimizer_opts_copy = dict(self.optimizer_default_opts)
        optimizer_opts_copy.update(optimizer_opts)
        
        lr_schedule_copy = dict(self.lr_default_schedule)
        lr_schedule_copy.update(lr_schedule)
        
        super().train(
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            optimizer_opts=optimizer_opts_copy, 
            lr_schedule=lr_schedule_copy,
            **kwargs)


if __name__ == '__main__':
    fire.Fire(MNISTModel)
