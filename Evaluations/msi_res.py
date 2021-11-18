import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import fftpack
from scipy import special
from scipy import optimize
from scipy import integrate



#define gaussian cdf+
esf = lambda x, sigma, mu, I: (0.5)*(I)*(1 + special.erf((x-mu)/(np.sqrt(2)*sigma)))

#define gaussian cdf with I0 and I1
esf_p = lambda x, sigma, mu, I1, I0: I0  + (0.5)*(I1 - I0)*(1 + special.erf((x-mu)/(np.sqrt(2)*sigma)))


        
#define normalised gaussian
def gaussian(x, sigma, A):
    return A*np.exp(-((x)**2)/(2*sigma**2))

def gaussian_mu(x, sigma, A, mu):
    return A*np.exp(-((x-mu)**2)/(2*sigma**2))



#define a function that sums over ROI linescans in the y direction
def esf1d(data):
    
    """sum over a 2d esf and obtain a 1d average; don't subtract"""
    
    data_avg = np.mean(data, axis = 0)
    
    
    return data_avg




#define a function that takes a 1D esf and returns a 1D LSF
def lsf1d(esf1d):
    
    """ differentiate a 1d est to obtain a 1d lsf """
    
    lsf_avg = np.gradient(esf1d)

    return lsf_avg


#define a function that computes an MTF from a 1D LSF
def mtf(lsf1d):
    
    """compute 1d mtf - absolute value of 1d lsf"""
    
    mtf = np.abs(((fftpack.fftshift(fftpack.fft(fftpack.ifftshift(lsf1d))))))
    
    return mtf

#define a function that fits gaussian cdf to to ESF
def esf_fit(data):
    
    xc = np.argmax(np.gradient(data)) #guess where xc is
    
    Ibar = np.average(data[xc:]) #guess what intensity is by computing average 
    
    #I0 = np.average(data[:xc])
    
    x = np.linspace(0,data.shape[0]-1, data.shape[0]) #create an array with data points (pixels) in x dir
    
    param, cov = curve_fit(esf_p, x, data, p0 = [0.5,0, Ibar, 0], maxfev = 5000) #fit cuntion
    

    
    err = np.sqrt(np.diag(cov)) # compute errors from covariance matrix
    
    return param, err
	
	#fit esf to data - input ESF ROI

def ff(x, p):
    return gaussian(x, *p)

def ffmu(x, p):
    return gaussian_mu(x, *p)


def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(100):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap 

	
def calculate_esf(data, pixel_size, disp=0):
    
    data_avg = esf1d(data) #turn into signal into 1d signal
    
    lsf = lsf1d(data_avg) # compute lsf
     
    xc = np.argmax(lsf) #get element corresponding to 50% of edge transition. need for surface calculations
    #print(xc) #use this to print x value of lsf max
   
    p, err = esf_fit(data_avg)  #fit gaussian cdf to data
    #print(p[1]) #use this to print x value of 50% esf transition
    #print(np.sqrt(np.diag(err)))
	
	
    #find where edge surface starts
    if xc > p[1]: # p[1] is the 50% x value from gaussian cdf fit ; xc is the x value of lsf maximum. theoretically they should match 
        xc = int(p[1] + 3) 
    else:
        xc = xc+3  # assume that surface starts  3 pixels away from x point corresponding to 50% of edge transition

    #print(xc) #use this to print x value of surface
    
    ibar = np.average(data_avg[xc:]) #compute average of surface
    vbar = np.var(data_avg[xc:], ddof=1) #compute variance of surface
    
    #add graph plotter
    if disp != 0 :
        x = np.linspace(0, data_avg.shape[0]-1, data_avg.shape[0])
        plt.plot(x,data_avg, 'o-', label = 'data')
        plt.plot(x,lsf, 'o-', label = 'lsf')
        plt.plot(x,esf_p(x, p[0],p[1],p[2], p[3]), 'o-', label = 'gaussian cdf fit')
        plt.title('average signal at {} $\mu m$'.format(pixel_size))
        plt.xlabel('pixels')
        plt.ylabel('Ion count')
        plt.legend()
    
    noise = np.zeros(data.shape)
    for i in range(data.shape[0]):
        noise[i,:] = np.abs(data[i,:] - data_avg)
    
    noisestd = np.sqrt(np.mean(noise**2, axis=0))
    noisestd = np.mean(noisestd)
    #calculate average snr
    snr = np.mean(data_avg)/noisestd
    
    
    #print(p[0], snr) #print curve fit values plus snr
    #print(p[1])
    return p[0], snr, ibar 




#define a function that calculates an MTF from ESF ROI
def calculate_mtf(data, pixel_size, disp):
       
    data_avg = esf1d(data) #turn into signal into 1d signal
    
    lsf = lsf1d(data_avg) # compute lsf from data

    p, err = esf_fit(data_avg)  #fit gaussian cdf to data

    ibar = p[2] #calculate average intensity of surface

    
    x = np.linspace(0,data_avg.shape[0]-1, data_avg.shape[0]) #create array with pixels 
    
    esf_theoretical = esf_p(x,p[0], p[1], p[2], p[3]) #create a gaussian cdf using parameters extracted from data
    
    mtf_data = mtf(lsf)
    
    
    f = 2*fftpack.fftfreq(np.shape(mtf_data[0:])[0], d=1) #create frequency array
    
    pstart = [0.25, np.abs(integrate.simps(f, mtf_data))]

    pfit, perr = fit_bootstrap(pstart, fftpack.fftshift(f), mtf_data, ff)
    
    mtf_gauss = gaussian(fftpack.fftshift(f), pfit[0], pfit[1])

    mtfeq = interp1d(fftpack.fftshift(f), mtf_gauss, fill_value="extrapolate") #create an equation of mtf from gaussian fit by interpolating
    
    noise = np.zeros(data.shape)
    noise2 = np.zeros(data.shape)
    noise_ft = np.zeros(data.shape)
    noise_ft2 = np.zeros(data.shape)
    
    for i in range(data.shape[0]):
        noise[i,:] = (data[i,:] -data_avg)
    for i in range(data.shape[0]):
        noise_ft[i,:] = np.sqrt(np.abs(fftpack.fftshift(fftpack.fft(fftpack.ifftshift(noise[i,:]))))**2)
    
    #for i in range(data.shape[1]):
        #noise2[:,i] = (data[:,i] -np.mean(data, axis=0))
    #for i in range(data.shape[1]):
        #noise_ft2[:,i] = np.sqrt(np.abs(fftpack.fftshift(fftpack.fft(fftpack.ifftshift(noise2[:,i]))))**2)
    
    
    
    
    nps = (1/(noise_ft.shape[0]-1))*np.sum(noise_ft,axis=0)/np.sqrt(data[:,int(p[1]+0.5):].shape[1]) #np.sqrt(data[:,int(p[1]+0.5):].shape[1])
        
    fnoise = 2*fftpack.fftfreq(nps.shape[0], d=1)
    
    #noise = (np.sqrt(data.shape[0]))*(data_avg - esf_theoretical)
            
    const = np.mean(nps)
    
    noiseq = interp1d(fftpack.fftshift(fnoise), const*np.ones(fnoise.shape), fill_value="extrapolate")
    #npsinterp = interp1d(fftpack.fftshift(fnoise), nps, fill_value = "extrapolate")

    #noisestd = np.std(npsinterp(fnoise)[int(np.where(fftpack.fftshift(f) == 0)[0]):])
    #print(noisestd)
    noisestd = np.std(nps)

    
   #fnoise_new = 2*fftpack.fftfreq(10*np.shape(noise[0:])[0], d=1)
    f_new = 2*fftpack.fftfreq(100*np.shape(mtf_data[0:])[0], d=1) #create a new array of frequencies with that has a 100times more points

    
    x2 = np.isclose(noiseq(f_new), fftpack.fftshift(mtfeq(f_new)), atol=1e-2*ibar).astype(int) #find intersection point between MTF and NPS in y dir
    x2_ind = np.argmax(x2) #get array value of intersection point (x dir)
    f_cutoff_gauss = np.abs(fftpack.fftshift(f_new)[x2_ind]) #get frequency value corresponding to array value - this is cut off freq; resolution point
    spectral_cut_off = 100*const/mtf_gauss[np.where(fftpack.fftshift(f)==0)][0] #compute percantge of average NPS 
    
    ferr = np.sqrt( ((noisestd**2) *(pfit[0]**2)) / (2*const**2 * np.log(pfit[1]/const)) + (2*perr[0]**2 * np.log(pfit[1]/const)) + ((perr[1]**2 * pfit[0]**2)/(2*pfit[1]**2 * np.log(pfit[1]/const))) )
    #print(const)
    #add plot
    if disp != 0 :

        plt.plot(fftpack.fftshift(f), mtf_data, label = 'data')
        plt.plot(fftpack.fftshift(f), mtf_gauss, label = 'gaussian')
        plt.plot(fftpack.fftshift(fnoise), nps, 'r')
        plt.plot(const*np.ones(fnoise.shape), 'r--', label = 'spectral cut-off $={0:.1f} \%$'.format(spectral_cut_off))
        plt.xlim(0,1)
        plt.xlabel('Normalised frequency')
        plt.ylabel('MTF')
        plt.title('MTF at {} $\mu m$'.format(pixel_size))
        plt.legend()
        
    
    return f_cutoff_gauss, spectral_cut_off, ferr

def subpl_calculate_esf(data, title, axis):
	data_avg = esf1d(data) #turn into signal into 1d signal
	lsf = lsf1d(data_avg) # compute lsf
	xc = np.argmax(lsf) #get element corresponding to 50% of edge transition. need for surface calculations
	print(xc) #use this to print x value of lsf max
	p, err = esf_fit(data_avg)  #fit gaussian cdf to data
    #print(p[1]) #use this to print x value of 50% esf transition
	
	
    #find where edge surface starts
	if xc > p[1]: # p[1] is the 50% x value from gaussian cdf fit ; xc is the x value of lsf maximum. theoretically they should match 
		xc = int(p[1] + 3) 
	else:
		xc = xc+3  # assume that surface starts  3 pixels away from x point corresponding to 50% of edge transition
	print(xc) #use this to print x value of surface
	ibar = np.average(data_avg[xc:]) #compute average of surface
	#vbar = np.var(data_avg[xc:], ddof=1) #compute variance of surface
	#add graph plotter
	
	noise = np.zeros(data.shape)
	for i in range(data.shape[0]):
		noise[i,:] = np.sqrt((data[i,:] -data_avg)**2)
	noisestd = np.sqrt(np.mean(noise**2, axis = 0))
	noise_std_mean = np.mean(noisestd[:])
	
	x = np.linspace(0, data_avg.shape[0]-1, data_avg.shape[0])
	axis.plot(x,data_avg, 'o-', label = 'data')
	axis.plot(x,lsf, 'o-', label = 'lsf')
	axis.plot(x,esf_p(x, p[0],p[1],p[2], p[3]), 'o-', label = 'gaussian cdf fit')
	axis.set_title('{}'.format(title))
	axis.set_xlabel('pixels')
	axis.set_ylabel('Ion count')
	axis.legend()   
    
    #calculate average snr
	#snr = (1/np.sqrt(data.shape[0]))*np.mean(data_avg[xc:])/np.std(data_avg[xc:])
    
	snr = np.mean(data_avg[:])/noise_std_mean
    #print(p[0], snr) #print curve fit values plus snr
	return p[0], snr, ibar 


	
def subpl_calculate_mtf(data, title, axis):
	data_avg = esf1d(data) #turn into signal into 1d signal
	lsf = lsf1d(data_avg) # compute lsf from data
	p, err = esf_fit(data_avg)  #fit gaussian cdf to data
	
	ibar = p[2] #calculate average intensity of surface
	x = np.linspace(0,data_avg.shape[0]-1, data_avg.shape[0]) #create array with pixels 
	esf_theoretical = esf_p(x,p[0], p[1], p[2], p[3]) #create a gaussian cdf using parameters extracted from data
	mtf_data = mtf(lsf)
	f = 2*fftpack.fftfreq(np.shape(mtf_data[0:])[0], d=1) #create frequency array
	f0 = np.where(fftpack.fftshift(f) == 0)[0] #find array element correspoding to zeroth frequency
	pstart = [0.25, np.abs(integrate.simps(f, mtf_data))]
	pfit, perr = fit_bootstrap(pstart, fftpack.fftshift(f), mtf_data, ff)
	mtf_gauss = gaussian(fftpack.fftshift(f), pfit[0], pfit[1])
	mtfeq = interp1d(fftpack.fftshift(f), mtf_gauss, fill_value="extrapolate") #create an equation of mtf from gaussian fit by interpolating
    
	noise = np.zeros(data.shape)
	noise_ft = np.zeros(data.shape)
	for i in range(data.shape[0]):
		noise[i,:] = data[i,:] - data_avg
	for i in range(data.shape[0]):
		noise_ft[i,:] = np.sqrt(np.abs(fftpack.fftshift(fftpack.fft(fftpack.ifftshift(noise[i,:]))))**2)
	nps = (1/(noise_ft.shape[0]-1))*np.sum(noise_ft, axis=0)/(np.sqrt(data[:,int(p[1]+1):].shape[1]))
	fnoise = 2*fftpack.fftfreq(np.shape(nps)[0], d=1)
    
    #noise = (np.sqrt(data.shape[0]))*(data_avg - esf_theoretical)
	#nps = mtf(noise) 
	const = np.mean(nps)
	noiseq = interp1d(fftpack.fftshift(fnoise), const*np.ones(fnoise.shape), fill_value="extrapolate")
	npsinterp = interp1d(fftpack.fftshift(fnoise), nps, fill_value = "extrapolate")

	noisestd = np.std(nps[:])/np.sqrt(nps.shape[0])
	#print(noisestd)
	
	#fnoise_new = 2*fftpack.fftfreq(10*np.shape(noise[0:])[0], d=1)
	f_new = 2*fftpack.fftfreq(100*np.shape(mtf_data[0:])[0], d=1) #create a new array of frequencies with that has a 100times more points
	
	x2 = np.isclose(noiseq(f_new), fftpack.fftshift(mtfeq(f_new)), atol=1e-2*ibar).astype(int) #find intersection point between MTF and NPS in y dir
	x2_ind = np.argmax(x2) #get array value of intersection point (x dir)
	f_cutoff_gauss = np.abs(fftpack.fftshift(f_new)[x2_ind]) #get frequency value corresponding to array value - this is cut off freq; resolution point
	spectral_cut_off = 100*const/mtf_gauss[f0][0] #compute percantge of average NPS 
	ferr = np.sqrt( ((noisestd**2) *(pfit[0]**2)) / (2*const**2 * np.log(pfit[1]/const)) + (2*perr[0]**2 * np.log(pfit[1]/const)) + ((perr[1]**2 * pfit[0]**2)/(2*pfit[1]**2 * np.log(pfit[1]/const))) )
	#add plot
	axis.plot(fftpack.fftshift(f), mtf_data, label = 'data')
	axis.plot(fftpack.fftshift(f), mtf_gauss, label = 'gaussian')
	axis.plot(fftpack.fftshift(fnoise), nps, 'r')
	axis.plot(const*np.ones(fnoise.shape[0]), 'r--', label = 'spectral cut-off $={0:.1f} \%$'.format(spectral_cut_off))
	axis.set_xlim(0,1)
	axis.set_xlabel('Normalised frequency')
	axis.set_ylabel('MTF')
	axis.set_title('{}'.format(title))
	axis.legend()
	
	return f_cutoff_gauss, spectral_cut_off, ferr