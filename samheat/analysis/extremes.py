# ---------------------------------------------------------------------------------------
# Original author: Robert van der Drift
# Source repo: precip_extremes_RH_RCE (van der Drift & O'Gorman 2025, arXiv:2412.16306),
#   precip/scripts/extremes.py
# Vendored into SAM_HEAT on 2026-10-01. Changes: imports changed from 'scripts.*' to package-relative ('.thermo_SAM', '.lcl'). Otherwise verbatim, including upstream bugs that are NOT used by samheat.analysis.repro: load_3D uses an undefined R, save_qt_unint calls an undefined load_3D_daily, save_unint writes an undefined evap_sort; np.trapz was removed in NumPy 2 (use np.trapezoid).
# ---------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from .thermo_SAM import *
from .lcl import lcl
 
def get_lcl(sfc):
    '''
    Wrapper of Romps' LCL function.
    '''
    
    RHs = sfc['QV']/qsat(sfc['TABS'],sfc['p'])
    return lcl(sfc['p']*1e2, sfc['TABS'],rh=RHs)


def load_3D(fpath,load_last):
    '''

    Loads SAM 3D output files (plus precip from 2D files over
    the last "load_last" days.
    
    fpath: Path to SAM simulation. Should have OUT_3D/ inside. 
    load_last: number of days back to save 3D files from;
    
    '''

    zt = 16e3 # Upper bound for loading 3D data
    data = xr.open_mfdataset(fpath+'OUT_3D/*.nc')#[das]
    ndays = data['time'].values[-1]
    data = data.sel(time=slice(ndays-load_last,ndays))
    prec = xr.open_mfdataset(fpath+'OUT_2D/*.nc').sel(time=data['time']).compute()['Prec']
    
    data['rho'] = data['p'].isel(time=0)*100/R/data['TABS'].mean(('x','y','time'))
    return data.sel(z=slice(0,zt)), prec


# Some functions for manipulating xr DataArrays

def da_flatten(da,nsamp,zsize): 
    return da.swapaxes(1,-1).reshape((nsamp,zsize))

def da_sort(da):
    return np.argsort(da.compute().values.flatten())


####################
# COMPUTE AND SAVE #
#   EXTREME VALS   #
####################    

perc_thresh = 0.9

def save_qt_unint(rv,fpath,ndays):

    '''

    Generates "qvT" files of SAM 3D output files in columns
    of high condensation.
    
    '''

    global perc_thresh
    
    # Load in all necessary data
    print("Loading Data")
    rvstr = 'RV'+str(rv)
    
    data, _ = load_3D_daily(fpath+rvstr+'/',ndays)
    exts = xr.open_dataset(fpath+'extremes/condensation_'+rvstr+'.nc')

    zs = data['z']
    zsize = zs.size
    nsamp = data['TABS'].size//zsize
    sidx  = int(perc_thresh*nsamp) # sidx = "start index"

    prec_sort = exts['prec_sort'].values
    cond_sort = exts['cond_sort'].values

    # Sort by condensation
    qv   = da_flatten(data['QV'].values,nsamp,zsize)[cond_sort]
    qn   = da_flatten(data['QN'].values,nsamp,zsize)[cond_sort]
    qp   = da_flatten(data['QP'].values,nsamp,zsize)[cond_sort]
    tabs = da_flatten(data['TABS'].values,nsamp,zsize)[cond_sort]
    w    = da_flatten(data['W'].values,nsamp,zsize)[cond_sort]
    
    print("Saving Output")
    output = xr.Dataset(coords={'perc':np.linspace(0,1,nsamp)[sidx:], 'z':zs},
                       data_vars={'qv':(['perc','z'], qv),
                                  'qn':(['perc','z'], qn),
                                  'qp':(['perc','z'], qp),
                                  'tabs':(['perc','z'], tabs),
                                  'w':(['perc','z'], w)})
    
    output.to_netcdf(fpath+'extremes/qvT_prec_'+rvstr+'.nc')
    return

def save_unint(rv,fpath,ndays):
    '''

    Sort and save P and unintegrated C, dy, and th.

    '''
    
    global perc_thresh
    
    # Load in all necessary data
    print("Loading Data")
    rvstr = 'RV'+str(rv)
    data, prec = load_3D(fpath+rvstr+'/',ndays)
    data['dqdz'] = dq_dz(data['TABS'],data['p'])*1e3
    data['qsat'] = qsat(data['TABS'],data['p'])
    
    zsize = data['z'].size
    nsamp = prec.size
    sidx  = int(perc_thresh*nsamp)

    prec_sort = np.argsort(prec.values.swapaxes(1,-1).flatten())[sidx:]
    prec = prec.values.flatten()[prec_sort]

    # Calculate LCL
    zs = data['z']
    zLCLs = get_lcl(data.isel(z=0)) + zs[0]
        
    # Integrand Masks
    # lclmask = 1 # Full integral
    lclmask = (zLCLs <= zs) # Above LCL only

    # dymask = 1 # Full integral
    dymask = (data['W'] > 0) # Updrafts only
    
    print("Calculating Condensation")
    dy = da_flatten((data['W']*data['rho']*dymask).values,nsamp,zsize) # For Instantaneous
    # dy = da_flatten((data['WUP']*data['rho']).values,nsamp,zsize)      # For 3-hrly
    th = da_flatten(-(data['dqdz']*lclmask).values,nsamp,zsize)

    cond = dy*th
    cond_sort = np.argsort(np.trapz(cond, data['z'].values))[sidx:]
    cond = cond[cond_sort]
    
    print("Saving Output")
    output = xr.Dataset(coords={'perc':np.linspace(0,1,nsamp)[sidx:], 'z':zs},
                       data_vars={'prec':(['perc'], prec),
                                  'cond':(['perc','z'], cond),
                                  'dy':(['perc','z'], dy),
                                  'th':(['perc','z'], th),
                                  'cond_sort':(['perc'], cond_sort),
                                  'prec_sort':(['perc'], prec_sort),
                                  'evap_sort':(['perc'], evap_sort)})
    
    output.to_netcdf(fpath+'extremes/condensation_'+rvstr+'.nc')
    return


def save_eff_unint(rv,fpath,ndays):
    '''

    Sort and save high-percentile E and A.

    '''

    global perc_thresh
    
    # Load in all necessary data
    print("Loading Data")
    rvstr = 'RV'+str(rv)
    data, prec = load_3D(fpath+rvstr+'/',ndays)
    
    zs = data['z'].values
    nz = data['z'].size
    nsamp = int(data['TABS'].size/nz)
    sidx  = int(perc_thresh*nsamp)

    t0 = data['time'][0]
    t1 = data['time'][-1]

    prec_sort = np.argsort(prec.values.flatten())
    
    print("Calculating Evaporation")
    # Re-evaporation calculated in columns of extreme P
    evaps = da_flatten(-(data['rho']*data['EVAP']).values.swapaxes(0,1).swapaxes(2,3),nsamp,nz)[prec_sort[sidx:]]
    
    print("Calculating Auto/Accr")
    # Precipitation generation (conv = conversion) sorted independently
    convs = da_flatten((data['rho']*data['CONV']).values.swapaxes(0,1).swapaxes(2,3),nsamp,nz)
    convs_sort = np.argsort(np.trapz(convs,zs))[sidx:]
    convs = convs[convs_sort]
    
    print("Saving Output")
    output = xr.Dataset(coords={'perc':np.linspace(0,1,nsamp)[sidx:],
                                'z':data['z']},
                       data_vars={'evaps':(['perc','z'], evaps),
                                  'convs':(['perc','z'], convs)})

    output.to_netcdf(fpath+'/extremes/unint_effs_RV%d.nc'%rv)
