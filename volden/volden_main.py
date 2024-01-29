#!/usr/bin/env python
# coding: utf-8




get_ipython().run_line_magic('matplotlib', 'inline')
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from radfil import radfil_class, styles
from astropy import units as u
from astropy.nddata.utils import Cutout2D
import numpy as np
from astropy.wcs import WCS
#from fil_finder import FilFinder2D
from scipy.optimize import least_squares
import scipy.optimize as optimize
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy.stats as stats
import math
import time
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import sys
import warnings





def mean_profiles(radius, profile, imgscale):

  """
  Returns: Mean radial profiles of the filament using 1-D linear interpolation

  Arguments:

  radius : numpy.ndarray
      radial distance array
  profile : numpy.ndarray
      profile array
  imgscale : float
      image scale of the image

  """
  # Storing the maximum radius values of rach cuts for the interpolation process
  max_radius = [max(radius[i]) for i in range(len(radius))]
  mean_radius = np.linspace(0, max(max_radius), num = round(max(max_radius)/imgscale))
  
  #Performing 1D linear interpolation
  interp = [np.interp(mean_radius, radius[i], profile[i]) for i in range(len(radius))]
  mean_profile = np.mean(interp, axis=0)

  return mean_radius, mean_profile





def volden_radfil(image, header, spine, sample_int, distance, cutdist, avg = True):

    """
    Building the radial profiles of the filament using RadFil package

    ----------
    The below arguments are taken from Zucker et al. 2018. More details on the 
    below arguments can be found at https://github.com/catherinezucker/radfil.git 
    Here we have used the essential arguments of radfil and also added some new 
    codes in the radfil source code to build the radial profiles using radial cuts 
    at an constant angle.

    Arguments:
  
    image : numpy.ndarray
        A 2D array of the image data to be analyzed.
    header : astropy.io.fits.Header
        The header corresponding to the image array
    spine: boolean numpy.ndarray
        A 2D boolean array corresponding to the central spine of the filament. It
        can be produced using FILFINDER
    sample_int: integer
        A integer corresponding to the sampling frequency (in pixels)
    distance: float
        Distance to the filament (in pc)
    cutdist: float or int
        The radial distance from the spine you'd like to search for the
        peak column density along each cut (in pc)
    avg: bool, optional
        If True, then mean radial profiles of North and South will also get 
        return. default: True

    Returns:

    radial distance and the respective profiles of North and South regions across
    the filament.

    north_radius : numpy.ndarray
        radial distance array of north region
    north_profile : numpy.ndarray
        profile array of north region
    south_radius : numpy.ndarray
        radial distance array of south region
    south_profile : numpy.ndarray
        profile array of south region

    "Below parameters gets return only when avg = True"

    north_radius_ : numpy.ndarray
        mean radial distance array of north region
    north_profile : numpy.ndarray
        mean profile array of north region
    south_radius : numpy.ndarray
        mean radial distance array of south region
    south_profile: numpy.ndarray
        mean profile array of south region
    ----------
    """

    fil_image, fil_header, fil_spine = image, header, spine
    radobj=radfil_class.radfil(fil_image, filspine = fil_spine, 
                           header=fil_header, distance= distance)
    
    radobj.build_profile(samp_int= sample_int, cutdist= cutdist)

    rad = radobj.dictionary_cuts['distance']
    prof = radobj.dictionary_cuts['profile']

    # Dividing the regions into two parts based on positive and negative values
    p_r = [] 
    n_r = []
    p_p = []
    n_p = []

    for i in range(0,len(rad)):
        # Radial distance
        pos_radius = np.delete(rad[i], np.where(rad[i] < 0))
        neg_radius = np.delete(rad[i], np.where(rad[i] > 0))
        p_r.append(pos_radius)
        n_r.append(np.absolute(neg_radius))
        
        # Column density profiles
        pos_profile1 = np.delete(prof[i], np.where(rad[i] < 0))
        neg_profile1 = np.delete(prof[i], np.where(rad[i] > 0))
        p_p.append(pos_profile1)
        n_p.append(np.absolute(neg_profile1))
    
    # Now dividing the regions to North or South based on the filament's slope

    y_spi, x_spi = np.where(fil_spine == True)
    res_slope = linregress(x_spi, y_spi).slope
    angle = np.rad2deg(np.arctan2(y_spi[-1] - y_spi[0], x_spi[-1] - x_spi[0]))

    # Saving the global variables for future use
    global fil_imgscale, xspline, yspline, fil__image, fil__header, fil_slope
    global fil_angle, sample__int
    fil_imgscale = radobj.imgscale.value
    xspline = radobj.xspline
    yspline = radobj.yspline
    fil_slope, fil_angle = res_slope, angle
    fil__header = fil_header
    fil__image = fil_image
    sample__int = sample_int


    if (res_slope >= 0):
      n_r1 = []
      n_p1 = []
      
      # Reversing To get the data in sequential order
      for i in range(0,len(n_r)):
        n_r1.append(n_r[i][::-1])
        n_p1.append(n_p[i][::-1])
  
      north_radius, north_profile = n_r1, n_p1
      south_radius, south_profile = p_r, p_p

    elif (res_slope < 0 or angle == 90):
      n_r1 = []
      n_p1 = []
      
      # Reversing To get the data in sequential order
      for i in range(0,len(n_r)):
        n_r1.append(n_r[i][::-1])
        n_p1.append(n_p[i][::-1])
      
      north_radius, north_profile = p_r, p_p
      south_radius, south_profile = n_r1, n_p1

    #Returing O/P depending upon the avg parameter

    if avg == False:
      return north_radius, north_profile, south_radius, south_profile
    else:
      #Mean radial profiles of North
      mean_north_radius, mean_north_profile = mean_profiles(north_radius, north_profile, fil_imgscale)

      #Mean radial profiles of South
      mean_south_radius, mean_south_profile = mean_profiles(south_radius, south_profile, fil_imgscale)
      
      return north_radius, north_profile, south_radius, south_profile, mean_north_radius,  mean_north_profile, mean_south_radius,  mean_south_profile 





def plummer_model(x, N0, p, rflat):
  """
    -------
    1D Plummer-like model. Equation for N(r)
    (Arzoumanian et al. 2011, Eswar et al. 2017)
    -------

    Arguments:
    
    x : float
        the radial distance from the filament ridge  
    N0 : float
        Amplitude of the Plummer-like function at 0 (the center).
    p : float
        The power-law index, p, in the function.
    rflat : float
        R_flat in the function.
  """

  return N0/((1.+(x/rflat)**2.)**((p-1.)/2.)) 

def volden_plummer_fit(radius, profile, bounds, init, error = False):
  """
  ----
  Calculating the plummer model parameters.
  ----

  Arguments:

  radial : numpy.ndarray
      radial distance of each cuts array 
  profile : numpy.ndarray
      extracted profiles of each cuts array
  bounds : list
      bounds on the plummer model parameters (nc, p and Rflat). If
      bounds = None, then then default values will be considered for 
      model fit
  init : list
      initialised values of the model parameters (nc, p and Rflat). If 
      init = None, then default values will be considered for model 
      fit. User can also enter their
      choice of initial values
  error : bool, optional
      If true then the error in the model parameters will also get returned

  Returns:

  nc : numpy.ndarray
      volume density profile at radial offset r to the the filament ridge array
  p : numpy.ndarray
      profile index array
  rflat : list
      the radius within which the N(r) profile is flat

  If error = True, then the fitting errors for the model parameters i.e. nc_err, p_err and rflat_err will also gets returned

  """

  N0 = []; p = []; rflat = []; covm = []
  N0_err = []; p_err=[]; rflat_err= []

  #plummer-fit for the extracted profiles
  for i in range(0, len(profile)):

    xdata = radius[i]
    ydata = profile[i]
    
    fittedParameters, pcov = curve_fit(plummer_model, xdata, ydata, p0=init, bounds=bounds , maxfev=3000)
    N0.append(fittedParameters[0])
    p.append(fittedParameters[1])
    rflat.append(fittedParameters[2])
    covm.append(np.array(pcov))

  #Error in each free parameter by considering the covariance matrix
  for i in range(0,len(covm)):
      err = np.sqrt(np.diag(covm[i]))
      N0_err.append(err[0])
      p_err.append(err[1])
      rflat_err.append(err[2])

  rflat, p = np.absolute(rflat), np.absolute(p)
  
  # Ap - finite constant factor for p > 1 that takes account the relative 
  # orientation of the filament in the plane of the sky. (Arzoumanian et al 2011)
  Ap = math.pi

  # Calculating nc
  nc = []
  nc_err = []

  '''
  for i in(range(len(rflat))):
    _N0 = ufloat(N0[i], N0_err[i])
    _rflat = ufloat(rflat[i], rflat_err[i])

    # 1pc = 3.08567758128e+18
    _nc = _N0/(Ap*_rflat*3.08567758128e+18)
    nc.append(_nc.nominal_value)
    nc_err.append(_nc.std_dev)
  '''

  nc = np.array(N0)/(Ap*np.array(rflat)*3.08567758128e+18)
  nc_err.append(np.std(nc))

  nc, nc_err, p_err, rflat_err = np.array(nc), np.array(nc_err), np.array(p_err), np.array(rflat_err)

  # Return the O/P based on the "err" input
  if error == True:
    return nc, p, rflat, nc_err, p_err, rflat_err

  else:
    return nc, p, rflat




def plummer(s, b, nc, p, rflat):

  """
    -------
    Equation for n(r)
    (Arzoumanian et al. 2011, Eswar et al. 2017, Wang et al. 2020)
    -------
    Arguments:

    s : float
        thickness value
    b : float
        intercept distance from the sightline to the filament crest
    nc : float
        volume density profile at radial offset r to the the filament ridge
    p : float
        The power-law index
    rflat : float
        R_flat in the function
    
    Returns: the number density value at a distance "r" from the filament's ridge
  """
  return nc / ((1.+((s**2 + b**2)/(rflat)**2.))**((p)/2.))


def cloud_boundary(radius, profile, nc, rflat, p, init_D = 0.01, threshold = 1e-8, ratio = 0.5):

  """
  ----
  Calculating the thickness values using trial and test method
  ----

  Arguments:

  radial : numpy.ndarray
      radial distance of each cuts array 
  profile : numpy.ndarray
      extracted profiles of each cuts array
  nc : numpy.ndarray
      volume density profile at radial offset r to the the filament ridge
  p : numpy.ndarray
      The power-law index
  rflat : numpy.ndarray
      R_flat in the function
  init_D : float, optional
      initialised value of thickness. If not defined by user then the default
      value will be considered. default = 0.01
  threshold : float, optional
     The threshold for error in the cloud boundary ratio equation as defined by
     Wang et al. 2019 in equation 4.If not defined by user then the default
     value will be considered. default = 1e-8
  ratio: float, optional
     default = 0.5. The ratio in the cloud boundary condition we are using in this analysis.

  Returns:

  rpos : numpy.ndarray
      radial distance(plane of the sky) from the filament ridge
  D : numpy.ndarray
      thickness values of each profiles

  """

  # For Storing the thickness value for a every cut
  D = []
  rpos = []

  for j in range(len(radius)):

      # Initialization 
      # value for updating delta (look inside the loop)
      alpha = 0.9

      # initial guess for D 
      guess = init_D
      
      #r = max(radius[j])

      # b is equivalent to the rpos
      b = radius[j] #np.linspace(0,r,int(r/fil_imgscale))

      nc0 = nc[j]
      Rflat0 = rflat[j]
      p0 = p[j]
      
      # For Storing the thickness value for a single cut
      d = []


      for i in b:
          # denominator of the boundary condition
          I0 = quad(plummer,-np.inf,np.inf,args=(i,nc0,p0,Rflat0))[0]

          # numerator of the boundary condition
          I = quad(plummer,-guess/2,guess/2,args=(i,nc0,p0,Rflat0))[0]

          # storing the deviated value
          err = (I/I0 - ratio)

          # for updating guess in every loop
          delta = 0.1
          while abs(err) > threshold:
              I2 = quad(plummer,-guess/2,guess/2,args=(i,nc0,p0,Rflat0))[0]
              err = (I2/I0 - ratio)

              # if the guess doesn't change after each iteration
              previous_guess = guess

              if err < 0:
                  guess += delta
              
              elif err > 0:
                  guess -= delta
                  
              delta *= alpha

              # if the guess doesn't change after each iteration
              if previous_guess == guess:
                  delta += 0.05
          
          d.append(guess)

      D.append(np.array(d))
      rpos.append(np.array(b))
  
  return rpos, D


def data_processing(xpix, ypix, ra, dec, profile, thickness, angle, direction):
  
  """
  Arguments:
  xpix: numpy.ndarray
      x-axis pixel coordinates of the extracted profiles
  ypix: numpy.ndarray
      y-axis pixel coordinates of the extracted profiles
  ra: numpy.ndarray
      right ascension
  dec: numpy.ndarray
      declination
  profile: numpy.ndarray 
      column density values at that particular (ra,dec)
  thickness: numpy.ndarray 
      thickness value at that positional coordinate
  angle: numpy ndarray
      variable that sets the angle of each of the profile cuts to classify them into various directions
  direction: string
      the direction at which the profiles are extracted. (can be North or South) 
  
  Returns:

  a 2D dataframe of the total data to construct thickness and number density map
  """

  df = pd.DataFrame(columns=['Cut Number','Pixel_X', 'Pixel_Y','RA(in deg)','DEC (in deg)','Column Density',
                                'Thickness(in pc)','Angle(in degrees)'])
  for i in range(len(xpix)):
    df1 = pd.DataFrame()
    dict = {'Cut Number': i,'Pixel_X': xpix[i],'Pixel_Y': ypix[i], 'RA(in deg)': ra[i],
            'DEC (in deg)': dec[i], 'Column Density': profile[i], 'Thickness(in pc)': thickness[i], 
            'Angle(in degrees)': np.rad2deg(angle)}

    data = pd.DataFrame(dict)
    df1 = df1.append(data, ignore_index = True)
    df = df.append(df1)

  df['Direction'] = direction
  return df





def filtering(x_pix, y_pix, rpos, rpos_pix, D, profile):
  '''
  ------------------
  Removing the outlier data that may go outside of the 2D map
  ------------------
  Arguments:

  x_pix : numpy.ndarray
       x-axis pixel arrays 
  y_pix : numpy.ndarray
       y-axis pixel arrays
  rpos : numpy.ndarray
      radial distance in plane of the sky for each cuts
  rpos_pix : numpy.ndarray
      radial distance (in pixels) in plane of the sky for each cuts
  D : numpy.ndarray
      thickness value array
  profile : numpy.ndarray
      column density profile array

  Returns:

  Filtered data (returned parameters are same as that of the input arguments)

  '''

  D1 = []
  rpos1 = []
  x_pix1 = []
  y_pix1 = []
  rpos_pix1 = []
  prof_1 = []

  for i in range(0, len(x_pix)):
      x = []
      y = []
      dd = []; rps = []; rps_pix = [] ; profff = []
      for j in range(len(x_pix[i])):
          if 0 <= x_pix[i][j] < np.shape(fil__image)[0]-1 and 0 <= y_pix[i][j] < np.shape(fil__image)[1]-1:
              x.append(x_pix[i][j])
              y.append(y_pix[i][j])
              dd.append(D[i][j])
              rps.append(rpos[i][j])
              rps_pix.append(rpos_pix[i][j])
              profff.append(profile[i][j])

      x_pix1.append(np.array(x))
      y_pix1.append(np.array(y))
      rpos1.append(np.array(rps))
      rpos_pix1.append(np.array(rps_pix))
      D1.append(np.array(dd))
      prof_1.append(np.array(profff))

  return x_pix1, y_pix1, rpos1, rpos_pix1, D1, prof_1


def pix_2_wcs(x_pix, y_pix):
  '''
  ------------------
  Converting pixel coordinates to WCS co-ordinates
  ------------------
  x_pix, y_pix

  Arguments:

  x_pix : numpy.ndarray
       x-axis pixel arrays 
  y_pix : numpy.ndarray
       y-axis pixel arrays
  wcs: astropy WCS object
       wcs information

  Returns:

  the sky co-ordinates of the corresponding pixel coordinates

  '''
  # Storing the WCS info for sky coordinates
  wcs = WCS(fil__header)

  sky_x = []
  sky_y = []

  for i in range(len(x_pix)):
      xxx = []
      yyy = []
      for j in range(len(x_pix[i])):
          sky0 = []
          pix_x = x_pix[i][j]
          pix_y = y_pix[i][j]
          sky0.append(wcs.pixel_to_world_values(pix_x, pix_y))
          xxx.append(sky0[0][0])
          yyy.append(sky0[0][1])
          
      sky_x.append(np.array(xxx))
      sky_y.append(np.array(yyy))

  return sky_x, sky_y




"""
def compute_thickness_map(data):
  '''
  ------------------
  Preparing the final thickness map
  ------------------

  Arguments:
  data : pandas.DataFrame
         total data catalog

  Returns:
  Computed thickness map

  '''
  # Storing the WCS info for sky coordinates
  wcs = WCS(fil__header)

  ra = np.array(data['RA(in deg)'])
  dec = np.array(data['DEC (in deg)'])
  depth = np.array(data['Thickness(in pc)'])
  cold = np.array(data['Column Density'])

  N,M = np.shape(fil__image)

  deptharr = np.zeros(np.shape(fil__image))

  depthreq = []
  
  if sample__int != 1:
    sampling_integer = int(round(sample__int/2))
  elif sample__int == 1:
    sampling_integer = int(round(sample__int))
  sampling_integer = sample__int

  for i in range(0, N):
      for j in range(0, M):
          ra0 = wcs.pixel_to_world_values(i,j)[0]
          dec0 = wcs.pixel_to_world_values(i,j)[1]
          cond1 = ra > ra0 + 2*sampling_integer*fil__header['CDELT1']
          cond2 = ra < ra0 - 2*sampling_integer*fil__header['CDELT1']
          cond12 = np.logical_and(cond1, cond2)

          cond3 = dec > dec0 - sampling_integer*fil__header['CDELT2']
          cond4 = dec < dec0 + sampling_integer*fil__header['CDELT2']
          cond34 = np.logical_and(cond3, cond4)

          condition = np.logical_and(cond12, cond34)
          condition1 = np.where(condition)
          depthreq = np.mean(depth[condition1])
          deptharr[j][i] = depthreq

  return deptharr
"""




def total_data(profile_north, profile_south, rpos_north, rpos_south, D_north, D_south):

  """
  ----
  Computing the total data for the thickness map
  ----

  Arguments:
  profile_north : numpy.ndarray
      extracted profiles in plane of the sky for each cuts in the North direction
  profile_south : numpy.ndarray
      extracted profiles in plane of the sky for each cuts in the South direction
  rpos_north : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  rpos_south : numpy.ndarray
      radial distance in plane of the sky for each cuts in the South direction
  D_north : numpy.ndarray
      thickness values in plane of the sky for each cuts in the North direction
  D_south : numpy.ndarray
      thickness values in plane of the sky for each cuts in the South direction

  Returns:

  total catalogue of the map as a pandas.dataframe
  """

  # Cartesian pixel coordinate in both the directions
  xpix_north = []; ypix_north = []
  xpix_south = []; ypix_south = []

  # WCS coordinate in both the direction
  ra_north = []; dec_north = []
  ra_south = []; dec_south = []

  # Converting the rpos from pc to pixel using the image scale
  rpos_pix_north = []
  rpos_pix_south = []

  for i in range(len(rpos_north)):
      rpos_pix_north.append(rpos_north[i]/fil_imgscale)

  for i in range(len(rpos_south)):
      rpos_pix_south.append(rpos_south[i]/fil_imgscale)

  # For North and South Angles
  if fil_slope >= 0:
      theta_north = np.radians(fil_angle + 90)
      theta_south = np.radians(fil_angle + 90 + 180)
  elif (fil_slope < 0 or fil_angle == 90):
      theta_north = np.radians(fil_angle - 90)
      theta_south = np.radians(fil_angle - 90 + 180)  

  for i in range(len(xspline)-1):
      # For storing the rpos values in Cartesian pixel cordinate for the ith cut
      xpix = []; ypix = []

      # X-pixel value of 'i'th cut
      x0 = xspline[i]

      # Y-pixel value of 'i'th cut
      y0 = yspline[i]

      x_pix_north = []; y_pix_north = []
      x_pix_south = []; y_pix_south = []
      
      # For North
      for r in rpos_pix_north[i]:
          x_val_N = x0 + r*np.cos(theta_north)
          y_val_N = y0 + r*np.sin(theta_north)
          x_pix_north.append(x_val_N)
          y_pix_north.append(y_val_N)

      # For South
      for r in rpos_pix_south[i]:
          x_val_S = x0 + r*np.cos(theta_south)
          y_val_S = y0 + r*np.sin(theta_south)
          x_pix_south.append(x_val_S)
          y_pix_south.append(y_val_S)

      xpix_north.append(np.array(x_pix_north))
      ypix_north.append(np.array(y_pix_north))

      xpix_south.append(np.array(x_pix_south))
      ypix_south.append(np.array(y_pix_south))

  # Filtering task
  xpix_north, ypix_north, rpos_north, rpos_pix_north, D_north, profile_north = filtering(xpix_north,
                                                                                         ypix_north,
                                                                                         rpos_north,
                                                                                         rpos_pix_north,
                                                                                         D_north,
                                                                                         profile_north)

  xpix_south, ypix_south, rpos_south, rpos_pix_south, D_south, profile_south = filtering(xpix_south,
                                                                                         ypix_south,
                                                                                         rpos_south,
                                                                                         rpos_pix_south,
                                                                                         D_south,
                                                                                         profile_south)

  ra_north, dec_north = pix_2_wcs(xpix_north, ypix_north)
  ra_south, dec_south = pix_2_wcs(xpix_south, ypix_south)

  # data processing task
  north_data = data_processing(xpix_north, ypix_north, ra_north, dec_north, 
                               profile_north, D_north, theta_north, direction = 'North')
  
  south_data = data_processing(xpix_south, ypix_south, ra_south, dec_south, 
                               profile_south, D_south, theta_south, direction = 'South')
  
  total_data = north_data.append(south_data)
  
  return total_data

