# ---------------------------------------------------------------------------------------
# Original author: Robert van der Drift
# Source repo: precip_extremes_RH_RCE (van der Drift & O'Gorman 2025, arXiv:2412.16306),
#   precip/scripts/resampling.py
# Vendored into SAM_HEAT on 2026-10-01. Changes: imports changed from 'scripts.*' to package-relative ('.extremes', '.thermo_SAM'). Otherwise verbatim; np.trapz was removed in NumPy 2 (use np.trapezoid).
# ---------------------------------------------------------------------------------------
from .extremes import da_flatten
from .thermo_SAM import dq_dz
import numpy as np

# Functions in this file handle data resampling for the block bootstrapping method.

def gen_resampler(bsize,nt,nx,ny):
    '''
        Create a "resampler" lambda function that generates the same resampling
        provided any data array as an input.
    '''
    np.random.seed()
    nsamps = nt*nx*ny//bsize**2
    tsamps = np.random.randint(nt,size=nsamps)
    xsamps = np.random.randint(nx//bsize,size=nsamps)
    ysamps = np.random.randint(ny//bsize,size=nsamps)
    resamp = lambda da: np.r_[[da[t,slice(bsize*x,bsize*(x+1)),slice(bsize*y,bsize*(y+1))] for t,x,y in zip(tsamps,xsamps,ysamps)]]
    return resamp

def resample(prec_raw, dy_raw, th_raw, sidx, block_size, nt, nx, ny, nz, zs):
        '''
            Resampling for P, dy, th, and C

            prec_raw: Full, unsorted P output from SAM
            dy_raw: Full, unsorted and unintegrated rho*w
            th_raw: Full, unsorted and unintegrated (dqsat/dz)_ma
            sidx: Starting index to average resampled data above (equal to the number of samples times 0.01)
            
            block_size: How big to make the blocks (in x and y)
            nx: number of x samples (grid size: 128)
            ny: number of y samples (grid size: 128)
            nt: number of time samples (30 days, 3 hours apart: 241)
            nz: number of vertical levelz
            zs: heights
        '''
        nsamp = nx*ny*nt
        resampler = gen_resampler(block_size,nt,nx,ny)
        
        # print("Resampling")
        prec = resampler(prec_raw).flatten()
        prec_sort = np.argsort(prec)[sidx:]
        prec = prec[prec_sort]

        dy = resampler(dy_raw).reshape(nsamp,nz)
        th = resampler(th_raw).reshape(nsamp,nz)
        
        # print("Condensation")
        cond = np.trapz(dy*th, zs)
        cond_sort = np.argsort(cond)[sidx:]
        cond = cond[cond_sort]

        # print("(Thermo)dynamic")
        dy = dy[cond_sort]
        th = th[cond_sort]        

        ret = (prec.mean(), cond.mean(), dy.mean(axis=0), th.mean(axis=0))
        return ret

def resample_eff(prec_raw, dy_raw, th_raw, evap_raw, conv_raw, sidx, block_size, nt, nx, ny, nz, zs):
        '''
            Resampling for the P, dy, th, C, E, and A
        '''
        nsamp = nx*ny*nt
        resampler = gen_resampler(block_size,nt,nx,ny)
        
        # print("Resampling")
        prec = resampler(prec_raw).flatten()
        prec_sort = np.argsort(prec)[sidx:]
        prec = prec[prec_sort]

        dy = resampler(dy_raw).reshape(nsamp,nz)
        th = resampler(th_raw).reshape(nsamp,nz)
        
        # print("Condensation")
        cond = np.trapz(dy*th, zs)
        cond_sort = np.argsort(cond)[sidx:]
        cond = cond[cond_sort]

        # print("(Thermo)dynamic")
        dy = dy[cond_sort]
        th = th[cond_sort]        
        
        # print("Conversion/Evaporation")
        evap = resampler(evap_raw).reshape(nsamp,nz)
        conv = resampler(conv_raw).reshape(nsamp,nz)
        
        evap = np.trapz(evap, zs)[prec_sort]
        conv = np.sort(np.trapz(conv, zs))[sidx:]
        
        ret = (prec.mean(), cond.mean(), dy.mean(axis=0), th.mean(axis=0), evap.mean(), conv.mean())
        return ret
