
# data processing
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
from msm_analysis import util
import lmfit as lm

def IQrotate(data):
    """
    Function from Gijs that rotates data towards max variance in x axis. 
    """
    I = np.real(data)
    Q = np.imag(data)
    Cov = np.cov(I,Q)
    A = scipy.linalg.eig(Cov)
    eigvecs = A[1]
    if A[0][1]>A[0][0]:
        eigvec1 = eigvecs[:,0]
    else:
        eigvec1 = eigvecs[:,1]
    theta = np.arctan(eigvec1[0]/eigvec1[1])
    return theta

def IQrotate_data(data):
    return data*np.exp(1.j*IQrotate(data))

def subtract_background(quantity, axis=1):
    """ Subtracts the background (median in vertical direction) of a two tone plot """
    median = np.nanmedian(quantity, axis=axis)
    if axis==1:
        median_rep = np.repeat(median[:, np.newaxis], np.shape(quantity)[axis], axis=axis)
    elif axis==0 :
        median_rep = np.repeat(median[np.newaxis,:], np.shape(quantity)[axis], axis=axis)
    return quantity - median_rep

def extract_ami_x(ds):
    """ Extracts the value for AMI Bx from a snapshot dict. 
    Input: xarray dataset
    Output: Bz (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['instruments']['AMI430_x']['parameters']['field']['value']

def extract_ami_z(ds):
    """ Extracts the value for AMI Bz from a snapshot dict. 
    Input: xarray dataset
    Output: Bz (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['instruments']['AMI430_z']['parameters']['field']['value']

def extract_yoko_x(ds):
    """ Extracts the value for yokogawa Bx from a snapshot dict. 
    Input: xarray dataset
    Output: Bz (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['instruments']['yoko_magnet']['parameters']['field']['value']

def extract_Bz(ds):
    """ Extracts the value for Bz from a snapshot dict. 
    Input: xarray dataset
    Output: Bz (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['instruments']['Rotated_Magnet']['parameters']['z']['value']

def extract_gamma(ds):
    """ Extracts the value for the alignment angle gamma from a snapshot dict. 
    Input: xarray dataset
    Output: gamma (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['instruments']['Rotated_Magnet']['parameters']['gamma']['value']

def extract_period(ds):
    """ Extracts the Bx value for a flux period from a snapshot dict. 
    Input: xarray dataset
    Output: field flux period Bx (float)"""
    snapshot = json.loads(ds.snapshot)
    return snapshot['station']['parameters']['Field_flux_period']['value']

def correct_flux(flux, Bz, gamma, period):
    """ Converts raw magnet flux value into rotated coordinate system flux
    Input: flux (array or float), Bz (float), gamma(float), period(float)
    Output: corrected flux (array or float)"""
    x_offset = np.sin(gamma)*Bz
    return flux - x_offset/period

def make_selection(freqs, phase, fitline, width):
    """Selects region of width 'width' around fitline. Rest of data is converted to np.nan
    INPUTS
    ------------------
    freqs: frequencies. (1d array-like)
    phase: phase. (2d array-like)
    fitline: array of frequency coordinates (1d array-like)
    width: int/float
    
    OUTPUTS
    phase array. Everything outside the selection is converted to np.nan"""
    
    fitline_ext = np.repeat(np.array(fitline)[:, np.newaxis], len(phase[0,:]), axis=1)
    freqs_ext = np.repeat(np.array(freqs)[np.newaxis, :], len(phase[:, 0]), axis=0)
    return np.where(abs(freqs_ext - fitline_ext) < width, phase, np.NaN)

def extract_datapoints(selection, perc, type_='max'):
    """ Returns indices of datapoints either above or below a percentage ('perc'), dependent on 'type_'
    INPUTS
    ------------------
    selection: selection around initial guess. (2d np array)
    perc: percentile (int/float)
    type_: {'max', 'min'}
    
    OUTPUTS
    ------------------
    Boolean 2d array, shape is the same as 'selection'"""
    
    threshold = np.nanpercentile(selection, perc)
    #threshold_ = np.repeat(threshold[:, np.newaxis], selection.shape[1], axis=1)
    if type_=='max':
        return np.where(selection > threshold), threshold
    if type_=='min':
        return np.where(selection < threshold), threshold

def sinus(x, amp, per, phase, off):
    return amp*np.cos(2*np.pi/per*(x-phase))+off

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def vertex_parabola(x, a, h, k):
    """Here, (h,k) are the coordinates of the vertex"""
    return a*(x-h)**2 + k

def guess_parabola(pmin, pmax):
    a = (pmax[1] - pmin[1])/(pmax[0] - pmin[0])**2
    b = -2*a*pmin[0]
    c = pmin[1] + a*pmin[0]**2

    h = -b/(2*a)
    k = -b**2/(4*a)+c
    return [a,h,k]

def fit_parabola(ds, fluxdata, freqdata, plot=True, guess=None, sub_ax = None, **kwargs):
    """Fits a parabola through extracted datapoints. 
       Datapoints are grouped per flux. We take the median as the datapoint (max 1 per flux value), and save the std.
       Then, we fit a parabola through the datapoints, with the std as sigma in the fit. 
       
        INPUTS:
        -------------------------
        ds: 2D xarray DataSet (phase, with coords: flux & drive frequency)
        fluxdata: x data of extracted points (np array)
        freqdata: y data of extracted points (np array)
        plot: plotting flag (Boolean)
        sub_ax: plotting axis
        **kwargs: for plotting
        
        OUTPUTS:
        ------------------------
        fluxfit: x values of fitted parabola (np array)
        freqfit: y values of fitted parabola (np array)
        popt: optimal parameters of parabola fit (0x3 np array)
        pcov: covariance matrix of parabola fit (3x3 np array)"""
    
    flux_unique = np.unique(fluxdata)
    freqs_unique = np.zeros(len(flux_unique))
    freqs_std = np.zeros(len(flux_unique))
    guess = np.array(guess)
    i = 0
    
    # for each flux point, we only want one datapoint. This prevents the fit from becoming unbalanced if there are 
    # big clusters. 
    
    for f in flux_unique:
        index = np.where(fluxdata==f)[0]
        freqs = freqdata[index]
        freqs_unique[i] = np.median(freqs)
        freqs_std[i] = np.std(freqs)
        i += 1

    #error due to the limited y-grid size
    normal_error = 1/2*np.abs(ds.drive_frequency[1] - ds.drive_frequency[0])
    sigma = np.sqrt(normal_error.values**2 + freqs_std**2)

    #plt.scatter(flux_unique, freqs_unique)
    fit_model = lm.Model(vertex_parabola)
    fit_params = fit_model.make_params()
    fit_params['a'].set(value=guess[0])
    fit_params['h'].set(value=guess[1])
    fit_params['k'].set(value=guess[2])
    #fit_params.add('Phi_min', expr='-b/(2*a)')
    #fit_params.add('f_min', expr='-b**2/(4*a)+c')

    out = fit_model.fit(freqs_unique, x=flux_unique, params=fit_params, scale_covar=True)
    
    print(out.fit_report())
    a_fit = out.params['a'].value
    h_fit = out.params['h'].value
    k_fit = out.params['k'].value

    fluxfit = np.linspace(np.min(flux_unique), np.max(flux_unique), 1000)  # finer grid for the fit line
    freqfit = vertex_parabola(fluxfit, a_fit, h_fit, k_fit)
    
    if plot==True:
        sub_ax.scatter(fluxdata, freqdata, alpha=0.3)
        sub_ax.errorbar(flux_unique, freqs_unique, yerr=sigma, color='red', marker='.', linestyle='None')
        sub_ax.plot(fluxfit, freqfit)
        sub_ax.set_xlabel(r'flux ($\Phi_0$)')

    return fluxfit, freqfit, out

def find_extrema(ds, guess, neighbor, bounds=(0.2, 0.8), perc=5, type_='max', method='parabola', plot=True, ax=None, **kwargs):
    """Finds the extremum (minimum or maximum) of a transition line based on an initial guess. 
    1) Based on the clicked guess (say: minimum) and a neighbor (maximum), construct a parabola (guessline)
    2) Around this parabola, select a region of height bounds[1] and width bounds[0]
    3) Inside the selection, extract either top (perc) percentile or bottom, dependent on 'type_'
    4) Datapoints are grouped per flux. We take the median as the datapoint (max 1 per flux value), and save the std
    5) Fit a parabola through the datapoints, with the std as sigma in the fit. 
    6) Determine the minimum or maximum of the fit. 
    
    INPUTS:
    -------------------------
    ds: xarray DataSet
    guess: tuple
    neighbor: tuple
    bounds: (horizontal, vertical) (tuple)
    perc: percentile (int)
    type_: {'max', 'min'}
    plot: plotting flag (Boolean)
    **kwargs: for plotting
    
    OUTPUTS:
    -------------------------
    fluxfit: x values of fitted parabola (np array)
    freqfit: y values of fitted parabola (np array)
    popt: optimal parameters of parabola fit (0x3 np array)
    pcov: covariance matrix of parabola fit (3x3 np array)
    extremum: (flux, freq) coordinates of minimum or maximum (tuple)
    """
    
    xrange = ds.flux.sel(flux=slice(guess[0]-bounds[0], guess[0]+bounds[0]))
    if method=='parabola':
        guess_pars = guess_parabola(guess, neighbor)
        guessline = vertex_parabola(xrange, *guess_pars)
    elif method=='square':
        guessline = np.repeat(guess[1].values, len(xrange))
    phase_cut = util.subtract_background(ds.phase).sel(flux=slice(guess[0]-bounds[0], guess[0]+bounds[0]))
    sel = make_selection(ds.drive_frequency, phase_cut, guessline, bounds[1])
    if type_ == 'max':
        data_index, threshold = extract_datapoints(sel, 100-perc, type_)
    if type_ == 'min':
        data_index, threshold = extract_datapoints(sel, perc, type_)
    fluxdata = xrange[data_index[0]]
    freqdata = ds.drive_frequency[data_index[1]]
    fluxfit, freqfit, out = fit_parabola(ds, fluxdata, freqdata, guess=guess_pars, plot=False)
        
    if plot==True:
        pcm = util.subtract_background(ds.phase).transpose().plot(ax=ax[0], cmap='YlGnBu_r',vmin=-1,vmax=1, 
                                                                          add_colorbar=False, robust=True)
        ax[0].set_title('')
        ax[0].plot(*guess, '*', color='black', alpha=0.9) 
        ax[0].plot(xrange, guessline, color='red')
        
        ax[1].pcolormesh(xrange, ds.drive_frequency, sel.T, cmap='YlGnBu_r', vmin=-1, vmax=1)
        ax[1].scatter(xrange[data_index[0]], ds.drive_frequency[data_index[1]], s=0.3, color='red')
        
        fluxfit, freqfit, out = fit_parabola(ds, xrange[data_index[0]], ds.drive_frequency[data_index[1]], guess=guess_pars, plot=True, sub_ax=ax[2])
        
        pcm = util.subtract_background(ds.phase).transpose().plot(ax=ax[3],cmap='YlGnBu_r',vmin=-1,vmax=1, 
                                                                                  add_colorbar=False, robust=True)
        plt.title('')
        ax[3].plot(fluxfit, freqfit, color='r', linewidth=0.5, linestyle='--')
        ax[3].axvline(out.params['h'].value, color='white', linestyle='--')

    return fluxdata, freqdata, fluxfit, freqfit, out, pcm

def coeff_of_variance(popt, pcov):
    cv = np.zeros(len(popt))
    for i in range(len(popt)):
        cv[i] = np.sqrt(pcov[i,i])/popt[i]
    return cv