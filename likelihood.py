"""
We use this code to minimize the negative log-likelihood of a normal probability density function (PDF) where we assume that the observed values are 
normally distributed around the mean with a certain standard deviation.

For an extensive discussion on how to fit a line to data see https://arxiv.org/abs/1008.4686

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
data = [14.9078, -8.8203, 7.9771, 28.0620, 35.3188, 49.4874, 74.7478, 85.4985, 115.0069, 164.3853, 200.351325]
err  = [10 * np.ones(len(data))[i] for i in range(len(data))]

#likelihood function for a linear fit model
def neg_loglhood_lin(params):
    c, m, stdev = params

    # Calculate the predicted values from the initial parameter guesses
    ymod = c + m*x
    log_lik_lin = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=stdev) )
    return(log_lik_lin)

def neg_loglhood_lin_2d(params):
    m, stdev = params

    # Calculate the predicted values from the initial parameter guesses
    ymod = m*x
    log_lik_lin = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=stdev) )
    return(log_lik_lin)

#likelihood function for a power law fit model
def neg_loglhood_plaw(params):
    a, b, stdev = params

    ymod = a*pow(x, b)
    log_lik_plaw = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=stdev) )
    return(log_lik_plaw)

#Wilkes test statistic 
def test_stat(x, y):
    return -2*(x-y)   #where x and y is the ratio of log-likelihoods; y-> null model

#initial parameter guesses    
init_params_3d = [1, 1, 1]
init_params_2d = [1, 1]

#minimize the log likelihood or equivalently maximize the likelihood
result_lin = minimize(neg_loglhood_lin, init_params_3d, method='nelder-mead')
equation_lin = 'y =' + str(round(result_lin.x[0], 4)) + '+' + str(round(result_lin.x[1], 4)) + 'x'

result_plaw = minimize(neg_loglhood_plaw, init_params_3d, method='nelder-mead')
equation_plaw = 'y =' + str(round(result_plaw.x[0], 4)) + '*' + 'x^' + str(round(result_plaw.x[1], 4))

result_lin_2d = minimize(neg_loglhood_lin_2d, init_params_2d, method='nelder-mead')


#print the results as a sanity check!
#print result_plaw.x

#plotting routine   #substitute _lin for _plaw to obtain plot for linear model
# fig, ax = plt.subplots(1,1)
# plt.plot(x, result_plaw.x[0]*pow(x, result_plaw.x[1]), lw=2, color='black', label = 'best-fit')
# plt.errorbar(x, data, yerr=err, fmt='o')
# plt.xlim(-1, 11)
# plt.suptitle("MLE: Maximum Likelihood Estimation")
# ax.text(0.5, 0.9, equation_plaw, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# plt.legend(loc='upper left', prop={'size':12}, frameon=False)
# #plt.show()
# plt.savefig(outfilepath + 'powerlawfit.pdf')   #'linearfit.pdf'

#have test statistic to compare models? introduce p-value?
D = test_stat(neg_loglhood_plaw(result_plaw.x), neg_loglhood_lin_2d(result_lin_2d.x))  
pval = chi2.sf(D, 1)
print pval