"""
We use this code to perform the Wilks ratio test and calculate the corresponding p-value.

"""
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.stats import chi2
import time
import matplotlib.pylab as plt
outfilepath = '/Users/Jatan/Google Drive/PHYS2010/'   #modify this path to your plot directory

#setting up x values
x=np.linspace(0, 10, 11)

#input data
data = [9.9078, 3.1797, 17.9771, 28.0620, 35.3188, 59.4874, 69.7478, 95.4985, 115.0069, 164.3853, 165.3513]
err_std  = [10 * np.ones(len(data))[i] for i in range(len(data))]   #for plotting error bars
err_large = [14 * np.ones(len(data))[i] for i in range(len(data))]

#likelihood function for a power law fit model: y = a*x^b 
def neg_loglhood_plaw(params):
    a, b = params

    ymod = a*pow(x, b)
    log_lik_plaw = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=err_large) )
    return(log_lik_plaw)

#likelihood function for a linear fit model: y = a*x
def neg_loglhood_lin_2d(params):
    m = params

    ymod = m*x
    log_lik_lin = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=err_large) )
    return(log_lik_lin)

#Wilks test statistic 
def test_stat(x, y):
    return -2*(x-y)   #where x and y is the ratio of log-likelihoods; y-> null model

#initial parameter guesses    
init_params_3d = [1, 1,]
init_params_2d = [1, ]

#minimize the log likelihood or equivalently maximize the likelihood
result_lin_2d = minimize(neg_loglhood_lin_2d, init_params_2d, method='nelder-mead')
equation_lin_2d = 'y =' + str(round(result_lin_2d.x[0], 4)) + '*' + 'x'

result_plaw = minimize(neg_loglhood_plaw, init_params_3d, method='nelder-mead')
equation_plaw = 'y =' + str(round(result_plaw.x[0], 4)) + '*' + 'x^' + str(round(result_plaw.x[1], 4))

#Wilks test statistic to compare models, only applicable when the degrees of freedom (d.o.f) of the alternate model are higher than the null model
D = test_stat(neg_loglhood_plaw(result_plaw.x), neg_loglhood_lin_2d(result_lin_2d.x))  
pval = chi2.sf(D, 1)
print D, pval
