"""
Dataset download and preprocessing.
"""

import numpy as np
import torch
import torchvision
import fire
import PIL
import h5py
import logging
import os
from tqdm import tqdm


class Preprocessor(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.train_path = os.path.join(self.data_dir, 'train.pt')
        self.valid_path = os.path.join(self.data_dir, 'valid.pt')
        self.test_path = os.path.join(self.data_dir, 'test.pt')
    
    def projective_mnist(self, seed=1, output_size=64, copies=8, num_train=10000, num_valid=5000):
        logging.info('Projective MNIST dataset')
        logging.info('seed = %d, output_size = %d, copies = %d, num_train = %d, num_valid = %d' 
                     % (seed, output_size, copies, num_train, num_valid))
        
        mnist_train = torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
        
        np.random.seed(seed)
        idxs = np.random.choice(len(mnist_train), size=(num_train + num_valid), replace=False)
        train_idxs = idxs[:num_train]
        valid_idxs = idxs[num_train:]

        logging.info('Generating pose parameters')
        train_params = [random_projective_transform() for _ in range(copies*len(train_idxs))]
        valid_params = [random_projective_transform() for _ in range(len(valid_idxs))]
        test_params = [random_projective_transform() for _ in range(copies*len(mnist_test))]

        train_data = torch.FloatTensor(copies*len(train_idxs), output_size, output_size)
        train_labels = torch.LongTensor(copies*len(train_idxs))
        
        logging.info('Transforming training examples')
        for i, idx in tqdm(enumerate(train_idxs)):
            img, y = mnist_train[idx]
            img = img.convert(mode='F')
            y = y.item()
            for j in range(copies):    
                params = train_params[j*len(train_idxs) + i]
                timg = projective(img, canvas=(output_size, output_size), **params)
                train_data[j*len(train_idxs) + i] = torch.FloatTensor(np.array(timg)).clamp_(0., 255.)
                train_labels[j*len(train_idxs) + i] = y
                
        torch.save((train_data, train_labels), self.train_path)
        logging.info('Saved training set to %s' % self.train_path)

        valid_data = torch.FloatTensor(len(valid_idxs), output_size, output_size)
        valid_labels = torch.LongTensor(len(valid_idxs))
        
        logging.info('Transforming validation examples')
        for i, idx in tqdm(enumerate(valid_idxs)):
            img, y = mnist_train[idx]
            img = img.convert(mode='F')
            timg = projective(img, canvas=(output_size, output_size), **valid_params[i])
            y = y.item()
            valid_data[i] = torch.FloatTensor(np.array(timg)).clamp_(0., 255.)
            valid_labels[i] = y
            
        torch.save((valid_data, valid_labels), self.valid_path)
        logging.info('Saved validation set to %s' % self.valid_path)
            
        test_data = torch.FloatTensor(copies*len(mnist_test), output_size, output_size)
        test_labels = torch.LongTensor(copies*len(mnist_test))

        logging.info('Transforming test examples')
        for idx in tqdm(range(len(mnist_test))):
            img, y = mnist_test[idx]
            img = img.convert(mode='F')
            y = y.item()
            for j in range(copies):
                params = test_params[j*len(mnist_test) + idx]
                timg = projective(img, canvas=(output_size, output_size), **params)
                test_data[j*len(mnist_test) + idx] = torch.FloatTensor(np.array(timg)).clamp_(0., 255.)
                test_labels[j*len(mnist_test) + idx] = y
   
        torch.save((test_data, test_labels), self.test_path)
        logging.info('Saved test set to %s' % self.test_path)

        logging.info('Saving pose parameters to %s' % self.data_dir)
        torch.save(train_params, os.path.join(self.data_dir, 'train_params.pt'))
        torch.save(valid_params, os.path.join(self.data_dir, 'valid_params.pt'))
        torch.save(test_params, os.path.join(self.data_dir, 'test_params.pt'))
        logging.info('Done')
    
    def svhn(self, seed=1, num_valid=5000):
        logging.info('SVHN dataset')
        logging.info('seed = %d, num_valid = %d' % (seed, num_valid))
        data = torchvision.datasets.SVHN(self.data_dir, 'train', download=True)
        test_data = torchvision.datasets.SVHN(self.data_dir, 'test', download=True)
        
        # split into training and validation sets
        np.random.seed(seed)
        valid_idxs = np.random.choice(len(data), size=num_valid, replace=False)
        train_idxs = np.delete(np.arange(len(data)), valid_idxs)
        np.random.shuffle(train_idxs)
        
        train_data = data.data[train_idxs]
        train_labels = data.labels[train_idxs]
        valid_data = data.data[valid_idxs]
        valid_labels = data.labels[valid_idxs]
        
        # save in serialized tensor format
        torch.save((torch.FloatTensor(train_data).div_(255), torch.LongTensor(train_labels)), self.train_path)
        logging.info('Saved training set to %s' % self.train_path)
        torch.save((torch.FloatTensor(valid_data).div_(255), torch.LongTensor(valid_labels)), self.valid_path)
        logging.info('Saved validation set to %s' % self.valid_path)
        torch.save((torch.FloatTensor(test_data.data).div_(255), torch.LongTensor(test_data.labels)), self.test_path)
        logging.info('Saved test set to %s' % self.test_path)
        logging.info('Done')


def random_projective_transform():
    pc = 0.8
    pa = np.random.uniform(-1, 1)
    pb = np.random.uniform(-1, 1) * (1 - np.abs(pa))
    perspective = (pc*pa, pc*pb)
    
    s = np.exp(np.random.uniform(0., np.log(2.)))
    aspect = np.exp(np.random.uniform(-np.log(1.5), np.log(1.5)))
    scale = (s*aspect, s/aspect)
    
    angle = np.random.uniform(-np.pi, np.pi)
    
    shear = np.random.uniform(-1.5, 1.5)
    return {
        'translation': (0., 0.),
        'angle': angle,
        'shear': shear,
        'perspective': perspective,
        'scale': scale,
    }
        
        
def projective(img, canvas=(64, 64), translation=(0., 0.), 
               angle=0., scale=(1., 1.), shear=0., 
               perspective=(0., 0.)):
    
    t = translation
    s = scale
    p = perspective
    ca, sa = np.cos(angle), np.sin(angle)
    
    f = canvas[0] / img.size[0]  # assume same aspect ratio
    p = (f*p[0], f*p[1])
    
    mat = np.array([
        s[0]*ca + t[0]*p[0],
        s[1]*(shear*ca - sa) + t[0]*p[1] ,
        s[0]*sa + t[1]*p[0],
        s[1]*(shear*sa + ca) + t[1]*p[1],
    ]).reshape([2, 2])
    
    pa = [
        ((-mat[0,0]+mat[0,1]+t[0])/(-p[0]+p[1]+1.), 
         (-mat[1,0]+mat[1,1]+t[1])/(-p[0]+p[1]+1.)), # (-1, +1)
        
        ((mat[0,0]+mat[0,1]+t[0])/(p[0]+p[1]+1.),
         (mat[1,0]+mat[1,1]+t[1])/(p[0]+p[1]+1.)), # (+1, +1)
        
        ((-mat[0,0]-mat[0,1]+t[0])/(-p[0]-p[1]+1.),
         (-mat[1,0]-mat[1,1]+t[1])/(-p[0]-p[1]+1.)), # (-1, -1)
        
        ((mat[0,0]-mat[0,1]+t[0])/(p[0]-p[1]+1.),
         (mat[1,0]-mat[1,1]+t[1])/(p[0]-p[1]+1.)), # (+1, -1)
    ]
    
    w, h = canvas
    img = img.transform(canvas, PIL.Image.PERSPECTIVE, 
                        data=(1, 0, -w//2+img.size[0]//2, 
                              0, 1, -h//2+img.size[1]//2,
                              0, 0), 
                        resample=PIL.Image.NEAREST)
    
    pa = [(w*(x+1)/2, h*(1-y)/2) for x, y in pa]
    pb = [(0, 0), (w, 0), (0, h), (w, h)] 
    params = _find_coeffs(pa, pb)
    img = img.transform(canvas, PIL.Image.PERSPECTIVE, data=params, resample=PIL.Image.BICUBIC)
    return img


def _find_coeffs(pa, pb):
    # pa: target coordinates, pb: source coordinates
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(Preprocessor)

