"""
Base classes and functions for experiments.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import Output
from collections import defaultdict
from tqdm import tqdm
import logging
import warnings
import time
from etn import coordinates, networks, transformers


class Model(object):
    def __init__(self, 
                 tfs=[],
                 coords=coordinates.identity_grid, 
                 net=None,
                 equivariant=True,
                 downsample=1,
                 tf_opts={},
                 net_opts={},
                 seed=None,
                 load_path=None,
                 loglevel='INFO'):
        """
        Model base class.
        """
        # configure logging
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)
        
        logging.info(str(self))
        
        if load_path is not None:
            logging.info('Loading model from file: %s -- using saved model configuration' % load_path)
            spec = torch.load(load_path)
            tfs = spec['tfs']
            coords = spec['coords']
            net = spec['net']
            equivariant = spec['equivariant']
            downsample = spec['downsample']
            tf_opts = spec['tf_opts']
            net_opts = spec['net_opts']
            seed = spec['seed']      
        
        if net is None:
            raise ValueError('net parameter must be specified')
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)      
        
        # build transformer sequence
        if len(tfs) > 0:
            pose_module = networks.EquivariantPosePredictor if equivariant else networks.DirectPosePredictor
            tfs = [getattr(transformers, tf) if type(tf) is str else tf for tf in tfs]
            seq = transformers.TransformerSequence(*[tf(pose_module, **tf_opts) for tf in tfs])
            #seq = transformers.TransformerParallel(*[tf(pose_module, **tf_opts) for tf in tfs])
            logging.info('Transformers: %s' % ' -> '.join([tf.__name__ for tf in tfs]))
            logging.info('Pose module: %s' % pose_module.__name__)
        else:
            seq = None
        
        # get coordinate function if given as a string
        if type(coords) is str:
            if hasattr(coordinates, coords):
                coords = getattr(coordinates, coords)
            elif hasattr(coordinates, coords + '_grid'):
                coords = getattr(coordinates, coords + '_grid')
            else:
                raise ValueError('Invalid coordinate system: ' + coords)
        logging.info('Coordinate transformation before classification: %s' % coords.__name__)
                
        # define network
        if type(net) is str:
            net = getattr(networks, net)
        network = net(**net_opts)
        logging.info('Classifier architecture: %s' % net.__name__)
        
        self.tfs = tfs
        self.coords = coords
        self.downsample = downsample
        self.net = net
        self.equivariant = equivariant
        self.tf_opts = tf_opts
        self.net_opts = net_opts
        self.seed = seed
        self.model = self._build_model(net=network, transformer=seq, coords=coords, downsample=downsample)
        
        logging.info('Net opts: %s' % str(net_opts))
        logging.info('Transformer opts: %s' % str(tf_opts))
        if load_path is not None:
            self.model.load_state_dict(spec['state_dict'])     
    
    def _build_model(self, net, transformer, coords, downsample):
        return networks.TransformerCNN(
            net=net,
            transformer=transformer,
            coords=coords,
            downsample=downsample)
            
    def _save(self, path, **kwargs):
        spec = {
            'tfs': [tf.__name__ for tf in self.tfs],
            'coords': self.coords.__name__,
            'net': self.net.__name__,
            'equivariant': self.equivariant,
            'downsample': self.downsample,
            'tf_opts': self.tf_opts,
            'net_opts': self.net_opts,
            'seed': self.seed,
            'state_dict': self.model.state_dict(),
        }
        spec.update(kwargs)
        torch.save(spec, path)
    
    def _load_dataset(self, path, num_examples=None):
        # override in subclasses to handle custom preprocessing / different data formats
        return Dataset(path=path, num_examples=num_examples)
    
    def train(self, 
              num_epochs=300,
              num_examples=None,
              batch_size=128,
              valid_batch_size=100, 
              train_path=None,
              valid_path=None,
              train_dataset_opts={},
              valid_dataset_opts={},
              optimizer='Adam', 
              optimizer_opts={'amsgrad': True, 'lr': 2e-3, 'weight_decay': 0.},
              lr_schedule={'step_size': 1, 'gamma': 0.99},
              save_path=None,
              show_plot=False,
              device='cuda:0'):
        """Train the model."""
        if save_path is not None:
            logging.info('Saving model with lowest validation error to %s' % save_path)
        else:
            warnings.warn('save_path not specified: model will not be saved')
        
        # load training and validation data
        if train_path is None:
            raise ValueError('train_path must be specified')
        if valid_path is None:
            raise ValueError('valid_path must be specified')
            
        logging.info('Loading training data from %s' % train_path)
        train_loader = torch.utils.data.DataLoader(
            self._load_dataset(
                path=train_path, 
                num_examples=num_examples, 
                **train_dataset_opts),
            shuffle=True, 
            batch_size=batch_size, 
            drop_last=True)
        
        logging.info('Loading validation data from %s' % valid_path)
        valid_loader = torch.utils.data.DataLoader(
            self._load_dataset(
                valid_path, 
                **valid_dataset_opts), 
            shuffle=False,
            batch_size=valid_batch_size, 
            drop_last=False)
        
        self.model.to(device)
        optim = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_opts)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, **lr_schedule)
        
        if show_plot:
            plotter = Plotter(show_plot=True)
            plotter.show()

        best_err = float('inf')
        start_time = time.time()
        for i in range(num_epochs):
            # train for one epoch
            logging.info('Training epoch %d' % (i+1))
            train_losses = self._train(optim, scheduler, train_loader, device)
            
            # evaluate on validation set
            logging.info('Evaluating model on validation set')
            valid_loss, valid_err = self._test(valid_loader, device)
            logging.info('Validation loss = %.2e, validation error = %.4f' % (valid_loss, valid_err))
            
            # save model with lowest validation error seen so far
            if (save_path is not None) and (valid_err < best_err):
                logging.info('Saving model with better validation error: %.2e (previously %.2e)' % (valid_err, best_err))
                best_err = valid_err
                self._save(save_path, epoch=i+1, valid_err=valid_err)
                
            # update plot
            if show_plot:
                plotter.update(train_loss=train_losses, valid_loss=valid_loss, valid_err=valid_err)
        
        logging.info('Finished training in %.1f s' % (time.time() - start_time))
        return self
    
    def test(self, batch_size=100, test_path=None, test_dataset_opts={}, device='cuda:0'):
        """Test the model."""
        if test_path is None:
            raise ValueError('test_path must be specified')
        logging.info('loading test data from %s' % test_path)
        loader = torch.utils.data.DataLoader(
            self._load_dataset(
                test_path,
                **test_dataset_opts),
            shuffle=False,
            batch_size=batch_size,
            drop_last=False)
        
        self.model.to(device)
        loss, err_rate = self._test(loader, device)
        logging.info('Test loss = %.2e' % loss)
        logging.info('Test error = %.4f' % err_rate)
        return loss, err_rate
    
    def predict(self, input, device='cuda:0', tf_output=False):
        """Predict a distribution over labels for a single example."""
        self.model.eval()
        self.model.to(device)
        x = input.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.model(x, tf_output=tf_output)
            logits = out[0] if tf_output else out
            probs = F.softmax(logits.squeeze(0), dim=-1)
            if tf_output:
                return probs, out[1]
            else:
                return probs
    
    def _train(self, optim, scheduler, loader, device):
        self.model.train()
        losses = []
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        return losses
    
    def _test(self, loader, device): 
        self.model.eval()
        total_loss = 0
        total_err = 0
        count = 0
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            count += x.shape[0]
            with torch.no_grad():
                logits = self.model(x)
                yhat = torch.argmax(logits, dim=-1)
                total_err += (y != yhat).sum().item()
                total_loss += F.cross_entropy(logits, y, reduction='sum').item()
        loss = total_loss / count
        err_rate = total_err / count
        return loss, err_rate

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, num_examples=None, normalization=None):
        self.path = path
        self.normalization = normalization      
        self.data, self.targets = torch.load(self.path)
        if self.data.dim() == 3:
            self.data = self.data.unsqueeze(1)  # singleton channel dimension
        
        if num_examples is not None:
            self.data = self.data[:num_examples]
            self.targets = self.targets[:num_examples].type(torch.long)
            
        if normalization is not None:
            mean, std = normalization
            mean = torch.Tensor(mean).view(1, -1, 1, 1)
            std = torch.Tensor(std).view(1, -1, 1, 1)
            self.data.add_(-mean).div_(std)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)
    
    
class Plotter(object):
    def __init__(self, id_string='', width=12, height=2.5, show_plot=True):
        """A dynamic plotting widget for tracking training progress in notebooks."""
        self.id_string = id_string
        self.width = width
        self.height = height
        self.output = Output()
        self.metrics = defaultdict(list)
        self.show_plot = show_plot

    def update(self, **metrics):
        for k, v in metrics.items():
            if type(v) is list:
                self.metrics[k] += v
            else:
                self.metrics[k].append(v)
        
        self.output.clear_output(wait=True)
        with self.output:
            if self.show_plot:
                self.plot()
                plt.show()
            maxlen = max(map(len, self.metrics.keys()))
            print(self.id_string)
            for k, v in self.metrics.items():
                print(('%' + str(maxlen) + 's') % k,
                      '| current = %.2e' % v[-1], 
                      '| max = %.2e (iter %4d)' % (np.max(v), np.argmax(v)), 
                      '| min = %.2e (iter %4d)' % (np.min(v), np.argmin(v)))
    
    def show(self):
        display(self.output)
    
    def progress_string(self):
        s = self.id_string + '\n'
        maxlen = max(map(len, self.metrics.keys()))
        for k, v in self.metrics.items():
            s += ''.join([('%' + str(maxlen) + 's') % k,
                      '| current = %.2e' % v[-1], 
                      '| max = %.2e (iter %4d)' % (np.max(v), np.argmax(v)), 
                      '| min = %.2e (iter %4d)' % (np.min(v), np.argmin(v))])
            s += '\n'
        return s
    
    def plot(self):
        fig = plt.figure(figsize=(self.width, self.height * len(self.metrics)))
        axs = fig.subplots(len(self.metrics))
        fig.suptitle(self.id_string)
        if len(self.metrics) == 1:
            axs = [axs]
        for ax, (k, v) in zip(axs, self.metrics.items()):
            ax.plot(v)
            ax.grid()
            ax.set_title(k)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

        