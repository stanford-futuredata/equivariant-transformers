"""
Sampling grids for canonical coordinate systems.
"""

import numpy as np
import torch


def identity_grid(output_size, ulim=(-1, 1), vlim=(-1, 1), out=None, device=None):
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = vs
    return torch.stack([xs, ys], 2, out=out)


def polar_grid(output_size, ulim=(0, np.sqrt(2.)), vlim=(-np.pi, np.pi), out=None, device=None):
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us * torch.cos(vs)
    ys = us * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def logpolar_grid(output_size, ulim=(None, np.log(2.)/2.), vlim=(-np.pi, np.pi), out=None, device=None):
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
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    ys = us
    xs = us * vs
    return torch.stack([xs, ys], 2, out=out)


def sheary_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = us * vs
    return torch.stack([xs, ys], 2, out=out)


def scalex_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
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
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    su, cu = torch.sin(us), torch.cos(us)
    sv, cv = torch.sin(vs), torch.cos(vs)
    xs = cu * sv / (np.sqrt(2.) - cu * cv)
    ys = su / (np.sqrt(2.) - cu * cv)
    return torch.stack([xs, ys], 2, out=out)

    