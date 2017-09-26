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
data = [9.9078, 3.1797, 17.9771, 28.0620, 35.3188, 59.4874, 69.7478, 95.4985, 115.0069, 164.3853, 165.3513]
err_std  = [10 * np.ones(len(data))[i] for i in range(len(data))]   #for plotting error bars
err_large = [14 * np.ones(len(data))[i] for i in range(len(data))]

def neg_loglhood_lin_2d(params):
    m, stdev= params

    ymod = m*x
    log_lik_lin = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=stdev) )
    return(log_lik_lin)

def neg_loglhood_parabolic(params):
    a, stdev= params

    ymod = a*pow(x, 2)
    log_lik_plaw = -np.sum(stats.norm.logpdf(data, loc=ymod, scale=stdev) )
    return(log_lik_plaw)

#initial parameter guesses    
init_params_2d = [1, 1]

#minimize the log likelihood or equivalently maximize the likelihood
result_parabolic = minimize(neg_loglhood_parabolic, init_params_2d, method='nelder-mead')
equation_parabolic = 'y =' + str(round(result_parabolic.x[0], 4)) + '*' + 'x^2' 

result_lin_2d = minimize(neg_loglhood_lin_2d, init_params_2d, method='nelder-mead')
equation_lin_2d = 'y =' + str(round(result_lin_2d.x[0], 4)) + '*' + 'x'

#print the results as a sanity check!
#print result_parabolic.x

#plotting routine   #substitute _lin for _plaw to obtain plot for linear model
fig, ax = plt.subplots(1,1)
plt.plot(x, result_parabolic.x[0]*pow(x,2), lw=2, color='black', label = 'best-fit') #result_lin_2d.x[0]*x #result_parabolic.x[0]*pow(x,2) #result_plaw.x[0]*pow(x, result_plaw.x[1]
plt.errorbar(x, data, yerr=err_std, color='red', fmt='o')
plt.xlim(-1, 11)
plt.suptitle("MLE: Maximum Likelihood Estimation (v3)")
ax.text(0.5, 0.9, equation_parabolic, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes) #equation_lin_2d #equation_parabolic 
plt.legend(loc='upper left', prop={'size':12}, frameon=False)
plt.savefig(outfilepath + 'parabolicfit.pdf')   #'linearfit.pdf' #'parabolicfit.pdf'

#plotting log-likelihood variations for linear model
a_lin = np.arange(12, 18, 0.01)
std_lin = [14 * np.ones(len(a_lin))[i] for i in range(len(a_lin))]  #substitute 10 for 'normal' error
params_lin = [[a_lin[i], std_lin[i]] for i in range(len(a_lin))]
plot_a = [-neg_loglhood_lin_2d(params_lin[i]) for i in range(len(a_lin))]

plt.plot(a_lin, plot_a, 'b')
plt.ylim(-50, -46.25)  #you may need to adjust the plot range for different error bars
plt.xlim(13, 16.7)
plt.axhline(y = max(plot_a), color='lawngreen', linewidth=2.0, linestyle='-') 
plt.axhline(y = (max(plot_a) - 0.5), color='lawngreen', linestyle='--') 
plt.axvline(x = 14.1, color='lawngreen', linestyle='--') 
plt.axvline(x = 15.5, color='lawngreen', linestyle='--') 
plt.xlabel(r'Slope (a)')
plt.ylabel(r'$\log (\mathcal{L})$')
plt.suptitle('Log likelihood for a linear model (v3)')
plt.grid()
plt.savefig(outfilepath + 'll_linear_largeerror.pdf') #'ll_linear_normalerror.pdf' #'ll_linear_largeerror.pdf'

#plotting log-likelihood variations for parabolic model
a_para = np.arange(1.5, 2.2, 0.001)
std_para = [14 * np.ones(len(a_para))[i] for i in range(len(a_para))]
params_para = [[a_para[i], std_para[i]] for i in range(len(a_para))]
plot_b = [-neg_loglhood_parabolic(params_para[i]) for i in range(len(a_para))]

plt.plot(a_para, plot_b, 'b')
plt.ylim(-43, -41.9)
plt.xlim(1.7, 2)
plt.axhline(y = max(plot_b), color='lawngreen', linewidth=2.0, linestyle='-') 
plt.axhline(y = (max(plot_b) - 0.5), color='lawngreen', linestyle='--') 
plt.axvline(x = 1.76, color='lawngreen', linestyle='--') 
plt.axvline(x = 1.935, color='lawngreen', linestyle='--') 
plt.xlabel(r'Constant (a)')
plt.ylabel(r'$\log (\mathcal{L})$')
plt.suptitle('Log likelihood for a parabolic model (v3)')
plt.grid()
plt.savefig(outfilepath + 'll_parabolic_largeerror.pdf') #'ll_parabolic_normalerror.pdf' 'll_parabolic_largeerror.pdf'

