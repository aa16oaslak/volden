import pandas as pd
import numpy as np
from astropy.wcs import WCS

def compute_thickness_map(data, fil_image, fil_header, sample_int):
    """
    ---------------
    Computing the Thickness map
    ---------------
    Arguments:
    data: string
          the directory where the total 2D dataframe of the thickness values are saved
    fil_image: numpy.ndarray
          the filament image 2D numpy array
    fil_header: the header of the filament image fits file
    sample_int: integer
        The integer corresponding to the sampling frequency that the user used initially
        while extracting the radial profiles using radfil (in pixels)
        
    Returns:
    thickness map 2D numpy array
    
    """
    df = pd.read_csv(data)
    wcs = WCS(fil_header)

    ra = np.array(df['RA(in deg)'])
    dec = np.array(df['DEC (in deg)'])
    depth = np.array(df['Thickness(in pc)'])
    cold = np.array(df['Column Density'])

    N,M = np.shape(fil_image)

    deptharr = np.zeros(np.shape(fil_image))
    coldarr = np.zeros(np.shape(fil_image))
    volden = np.zeros(np.shape(fil_image))

    depthreq = []
    coldreq = []

    for i in range(0, N):
        for j in range(0, M):
            ra0 = wcs.pixel_to_world_values(i,j)[0]
            dec0 = wcs.pixel_to_world_values(i,j)[1]
            
            #By following the nquist criteria, we are giveing the value to each pixels according the the sample_int value
            cond1 = ra > ra0 + 2*sample_int*fil_header['CDELT1']
            cond2 = ra < ra0 - 2*sample_int*fil_header['CDELT1']
            cond12 = np.logical_and(cond1, cond2)

            cond3 = dec > dec0 - sample_int*fil_header['CDELT2']
            cond4 = dec < dec0 + sample_int*fil_header['CDELT2']
            cond34 = np.logical_and(cond3, cond4)

            condition = np.logical_and(cond12, cond34)
            condition1 = np.where(condition)
            
            #if a pixel have multiple values, then we will consider the mean value
            depthreq = np.mean(depth[condition1])
            coldreq = np.mean(cold[condition1])
            
            #preparing the thickness map array
            deptharr[j][i] = depthreq
  
    return deptharr
    


def number_density_map(colden_map, thickness_map):
    
    """
    ----
    Computing the number density map
    ----

    Arguments:
    colden_map : numpy.ndarray
    thickness_map : numpy.ndarray
    computed thickness map array

    Returns:

    volume density map (unit- n(H2) per cubic cm) : numpy.ndarray
    """

    return colden_map/ (2*thickness_map*3.086e+18)