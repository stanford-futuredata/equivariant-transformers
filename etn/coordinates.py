"""
Sampling grids for 2D canonical coordinate systems.
Each coordinate system is defined by a pair of coordinates u, v. 
Each grid function maps from a grid in (u, v) coordinates to a collection points in Cartesian coordinates.
"""

import numpy as np
import torch


def identity_grid(output_size, ulim=(-1, 1), vlim=(-1, 1), out=None, device=None):
    """Cartesian coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = vs
    return torch.stack([xs, ys], 2, out=out)


def polar_grid(output_size, ulim=(0, np.sqrt(2.)), vlim=(-np.pi, np.pi), out=None, device=None):
    """Polar coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us * torch.cos(vs)
    ys = us * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def logpolar_grid(output_size, ulim=(None, np.log(2.)/2.), vlim=(-np.pi, np.pi), out=None, device=None):
    """Log-polar coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    rs = torch.exp(us)
    xs = rs * torch.cos(vs)
    ys = rs * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def shearx_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Horizontal shear coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian y-coordinate limits
        vlim: (float, float), x/y ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    ys = us
    xs = us * vs
    return torch.stack([xs, ys], 2, out=out)


def sheary_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Vertical shear coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), y/x ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = us * vs
    return torch.stack([xs, ys], 2, out=out)


def scalex_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Horizontal scale coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits 
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu/2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    
    xs = torch.exp(us)
    ys = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, -xs])
        ys = torch.cat([ys, ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), -xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), ys])
    return torch.stack([xs, ys], 2, out=out)


def scaley_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Vertical scale coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic y-coordinate limits
        vlim: (float, float), Cartesian x-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu/2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    
    ys = torch.exp(us)
    xs = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def hyperbolic_grid(output_size, ulim=(-np.sqrt(0.5), np.sqrt(0.5)), vlim=(-np.log(6.), np.log(6.)), out=None, device=None):
    """Hyperbolic coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), hyperbolic angular coordinate limits
        vlim: (float, float), hyperbolic log-radial coordinate limits 
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    
    rs = torch.exp(vs)
    xs = us * rs
    ys = us / rs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def perspectivex_grid(output_size, ulim=(1, 8), vlim=(-0.99*np.pi/2, 0.99*np.pi/2), out=None, device=None):
    """Horizontal perspective coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), x^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    
    xl = -1 / us.flip([1])
    xr =  1 / us
    yl = -xl * torch.tan(vs)
    yr =  xr * torch.tan(vs)

    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0]-1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0]-1, 1), yr])
    return torch.stack([xs, ys], 2, out=out)


def perspectivey_grid(output_size, ulim=(1, 8), vlim=(-0.99*np.pi/2, 0.99*np.pi/2), out=None, device=None):
    """Vertical perspective coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), y^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    
    yl = -1 / us.flip([1])
    yr =  1 / us
    xl = -yl * torch.tan(vs)
    xr =  yr * torch.tan(vs)

    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0]-1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0]-1, 1), yr])
    return torch.stack([xs, ys], 2, out=out)


def spherical_grid(output_size, ulim=(-np.pi/4, np.pi/4), vlim=(-np.pi/4, np.pi/4), out=None, device=None):
    """Spherical coordinate system.
    
    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), latitudinal coordinate limits
        vlim: (float, float), longitudinal coordinate limits 
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor
        
    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    su, cu = torch.sin(us), torch.cos(us)
    sv, cv = torch.sin(vs), torch.cos(vs)
    xs = cu * sv / (np.sqrt(2.) - cu * cv)
    ys = su / (np.sqrt(2.) - cu * cv)
    return torch.stack([xs, ys], 2, out=out)

    