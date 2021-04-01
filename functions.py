
# data processing
import numpy as np
import json

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
