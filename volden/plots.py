
def histogram_dist(data1, design1, data2, design2, labelx, name, binwidth = 10000):
  """
  ----
  Computing the histogram Distribution of two maps
  ----

  Arguments:
  data1 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design1 : char
      designation for the data1 for labelling purpose
  data2 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design2 : char
      designation for the data2 for labelling purpose
  labelx : char
      X- axis name of the histogram distribution. 
      Eg: "Number Density Values(in $cm^{-3}$)"
  name : char
      cloud name
  binwidth : positive int
      binwidth for the histogram plots
      
  Returns:
  the histogram distribution of two maps
  """

  # 1st Data
  img1 = data1


  img11 = np.nan_to_num(img1)
  img11 = np.array(img11.flatten())

  # 2nd Data
  img2 = data2

  img2 = np.nan_to_num(img2)
  img2 = np.array(img2.flatten())

  # configure and draw the histogram figure
  fig = plt.figure(figsize=(15,10))

  # seaborn histogram
  sns.distplot(img11, hist=True, kde=False, 
              bins= int(np.max(img11)/ binwidth), color = 'blue',
              hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"}, label = design1)

  sns.distplot(img2, hist=True, kde=False, 
              bins= int(np.max(img2)/ binwidth), color = 'blue',
              hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "red"}, label = design2)


  plt.title("{} vs {} Histogram distribution for {}".format(design1, design2, name), fontsize = 20)
  plt.xlabel(labelx, fontsize = 20)
  plt.ylabel("Pixel count", fontsize = 20)

  plt.tick_params(axis='both', which='major', labelsize=14)
  plt.legend(fontsize=20)
  plt.show()
    

# In[ ]:

def log_histogram_dist(data1, design1, data2, design2, labelx, name, min, max):
  """
  ----
  Computing the log-log histogram Distribution of two maps
  ----

  Arguments:
  data1 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design1 : char
      designation for the data1 for labelling purpose
  data2 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design2 : char
      designation for the data2 for labelling purpose
  labelx : char
      X- axis name of the histogram distribution. 
      Eg: "Number Density Values(in $cm^{-3}$)"
  name : char
      cloud name
  min : float
      minimum value for the X-axis
  max : float
      maximum value for the X-axis      
  
  Returns:
  log-log histogram Distribution of two maps
  """

  plt.figure(figsize=(15,10))
  plt.title("{} vs {} Log Histogram distribution for {}".format(design1, design2, name), fontsize = 25)
  plt.xlabel(labelx, fontsize = 25)
  plt.ylabel("Pixel count", fontsize = 25)
  MIN, MAX = min, max

  #For 1st data
  img1 = data1
  img11 = np.nan_to_num(img1)
  img11 = np.array(img11.flatten())
  plt.hist(img11, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50), label = design1, alpha = 0.7)
  plt.gca().set_xscale("log")

  #For 2nd data
  img2 = data2
  img22 = np.nan_to_num(img2)
  img22 = np.array(img22.flatten())
  plt.hist(img22, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50), color = 'red', label = design2, alpha = 0.4)
  plt.gca().set_xscale("log")

  plt.tick_params(axis='both', which='major', labelsize=20)

  plt.legend(fontsize=20)

# In[ ]:

def power_spectrum(data1, design1, data2, design2, name):
  """
  ----
  Computing the power spectrums of two maps
  
  Code taken from: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-  
  python/#:~:text=Calculating%20the%20power%20spectrum%20in%20Python&text=Convert%20the%20data%20set%20into,layout%20as%20the%20Fourier%20amplitudes.
  ----

  Arguments:
  data1 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design1 : char
      designation for the data1 for labelling purpose
  data2 : numpy.ndarray
      radial distance in plane of the sky for each cuts in the North direction
  design2 : char
      designation for the data2 for labelling purpose
      Eg: "Number Density Values(in $cm^{-3}$)"
  name : char
      cloud name
      
  Returns:
  power spectrum of the two maps 
  """

  #For data1

  img1 = data1
  img1 = np.nan_to_num(img1)

  #For data2

  img2 = data2
  img2 = np.nan_to_num(img2)

  npix = img1.shape[0]


  #Power Spectrum for data 1
  fourier_image = np.fft.fftn(img1)
  fourier_amplitudes = np.abs(fourier_image)**2
  kfreq = np.fft.fftfreq(npix) * npix
  kfreq2D = np.meshgrid(kfreq, kfreq)
  knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

  knrm = knrm.flatten()
  fourier_amplitudes = fourier_amplitudes.flatten()

  kbins = np.arange(0.5, npix//2+1, 1.)
  kvals = 0.5 * (kbins[1:] + kbins[:-1])
  Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                      statistic = "mean",
                                      bins = kbins)
  Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

  plt.figure(figsize=(15,10))

  plt.plot(kvals, Abins, label = design1)

  #Power Spectrum for data 2
  fourier_image = np.fft.fftn(img2)
  fourier_amplitudes = np.abs(fourier_image)**2
  kfreq = np.fft.fftfreq(npix) * npix
  kfreq2D = np.meshgrid(kfreq, kfreq)
  knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

  knrm = knrm.flatten()
  fourier_amplitudes = fourier_amplitudes.flatten()

  kbins = np.arange(0.5, npix//2+1, 1.)
  kvals = 0.5 * (kbins[1:] + kbins[:-1])
  Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                      statistic = "mean",
                                      bins = kbins)
  Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

  plt.plot(kvals, Abins, label = design2)

  plt.title("{} vs {} Log Histogram distribution for {}".format(design1, design2, name), fontsize = 25)
  plt.xlabel("$k$", fontsize = 25)
  plt.ylabel("$P(k)$", fontsize = 25)
  plt.yscale('log')
  plt.xscale('log')

  plt.tick_params(axis='both', which='major', labelsize=20)

  plt.legend(fontsize=20)