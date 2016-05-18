import numpy as np
import matplotlib
from matplotlib import pylab, mlab, pyplot as plt

import ad
from ad import adnumber
from ad.admath import *

# Plotting imports 
# Comment these lines if you don't have seaborn installed
import seaborn as sns
sns.set_style("ticks")

import scipy.stats
import scipy.optimize

matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['text.usetex'] = True


# ## Simple examples

# In[ ]:

x = adnumber(5.0)
print "x = ", x


# In[ ]:

y = x**2
print "y = x**2"

print "y = ", y
print "dy/dx = ", y.d(x)
print "d^2y/dx^2 = ", y.d2(x)


# In[ ]:

y = tanh(x)
print "y = tanh(x)"

print "y = ", y
print "dy/dx = ", y.d(x)
print "d^2y/dx^2 = ", y.d2(x)


# ### Comparing AD vs finite-differences for $f(x) = x \sin(1/x)$

# In[ ]:

x = adnumber(np.linspace(0.001, 0.1, 101))

f = lambda x: x*sin(1./x)
gradf = lambda x: sin(1./x) - 1./x*cos(1./x)

# step size for finite difference
eps = 1.e-5

# centered finite-difference
y_fd = np.array([(f(v+0.5*eps) - f(v-0.5*eps))/eps for v in x])
# AD
y_ad = np.array([f(v).d(v) for v in x])
# symbolic
y_true = np.array([gradf(v) for v in x])

# plot relative errors
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
ax1.plot(x, (y_fd-y_true)/y_true, label="Finite difference")
ax2.plot(x, (y_ad-y_true)/y_true, label="AD")
for ax in [ax1, ax2]:
    ax.set_xlabel("$x$")
    ax.set_ylabel("Relative error in $df/dx$")
    ax.legend()
plt.savefig("errors_fd_ad.pdf", bbox_inches="tight")


# ### Gradients and Hessians (derivatives wrt multiple variables)

# In[ ]:

# compute log-likelihood with gradient and Hessian of a multinomial distribution

# parameters
p = np.array(adnumber([0.1, 0.2, 0.3, 0]))
# make the proportions sum to 1 
p[-1] = 1. - sum(p[:-1])

# observations
n = np.array([3, 20, 12, 3])

# log-likehood
log_lik = sum(n * np.array(log(p)))

# derivatives wrt each parameter
for i in range(len(p)-1):
    print "dL/dp%d" % i, log_lik.d(p[i])
print

# derivatives wrt each pair of parameters
for i in range(len(p)-1):
    for j in range(i, len(p)-1):
        print "d^2L/dp%dp%d" % (i,j), log_lik.d2c(p[i], p[j])
print


# In[ ]:

# convenience functions
print "Gradient"
print log_lik.gradient(p[:-1])
print

print "Hessian"
print np.array(log_lik.hessian(p[:-1]))
print


# ## Example application: maximum-likelihood haplotype frequency estimation from genotype data

# Haplotype frequencies at two loci $\mathbf{p} \in \Delta_3$ <br/>
# Sample $n$ haplotypes, $h_{i1}, h_{i2} \sim \text{Categorical}(\mathbf{p})$ <br/>
# Genotypes $g_i = h_{i1} + h_{i2}$ <br/>
# Problem: Given $n$ genotypes $\{ g_1, \ldots, g_n \}$, recover $\mathbf{p}$ <br/>

# In[ ]:

def neg_log_lik(x, obs_geno_locus1, obs_geno_locus2):
    """
    Returns the negative log-likelihood (because we will minimize) of observing the 
    genotype data given by a vector of genotypes obs_geno_locus1 and obs_geno_locus2 at two loci,
    for the haplotype frequences given in x
    """
    hap_freq = np.reshape(x, (2,2))
    hap_freq = hap_freq / np.sum(hap_freq)
    geno_freq = np.zeros((3, 3), dtype=object)
    
    log_lik = 0.

    # if genotypes at both loci are known
    # compute the two-fold convolution of the haplotype frequency distribution
    # go over each of the 4 possibilities for the first and second haplotypes
    for hap1_locus1 in range(0, 1+1):
        for hap1_locus2 in range(0, 1+1):
            hap1_freq = hap_freq[hap1_locus1, hap1_locus2]
            for hap2_locus1 in range(0, 1+1):
                for hap2_locus2 in range(0, 1+1):
                    hap2_freq = hap_freq[hap2_locus1, hap2_locus2]
                    # sum the haplotypes at each locus to get the genotypes at each locus
                    geno_locus1 = hap1_locus1 + hap2_locus1
                    geno_locus2 = hap1_locus2 + hap2_locus2
                    # add the contribution of this haplotype combination towards the genotype probability
                    geno_freq[geno_locus1, geno_locus2] += hap1_freq * hap2_freq

    for geno_locus1 in range(0, 2+1):
        for geno_locus2 in range(0, 2+1):
            n_geno = np.sum((obs_geno_locus1 == geno_locus1) & (obs_geno_locus2 == geno_locus2))
            log_lik += n_geno * log(geno_freq[geno_locus1, geno_locus2])

    # include data where one of the loci might have missing data
    for geno in range(0, 2+1):
        # locus 1 known, locus 2 unknown
        n_geno = np.sum((obs_geno_locus1 == geno) & np.isnan(obs_geno_locus2))
        log_lik += n_geno * log(sum(geno_freq[geno, :]))

        # locus 1 unknown, locus 2 known
        n_geno = np.sum(np.isnan(obs_geno_locus1) & (obs_geno_locus2 == geno))
        log_lik += n_geno * log(sum(geno_freq[:, geno]))

    return -log_lik


# In[ ]:

# create the Jacobian and Hessian of the negative log likelihood
grad_neg_log_lik, hess_neg_log_lik = ad.gh(neg_log_lik)


# In[ ]:

# We don't need equality constraints since we rescale the variables to get
# frequencies
#
# def eq_constraints(x):
#     """Only equality constraint is that the frequencies sum to 1"""
#     return np.array([np.sum(x) - 1.])

# jac_eq_constraints, hess_eq_constraints = ad.gh(eq_constraints)


# In[ ]:

def sample_genotypes(n, hap_freq, mask_frac=0.):
    """ Samples n genotypes, where the haplotypes are drawn independently from hap_freq,
    and a fraction mask_frac of the genotype entries are masked at one locus
    """
    obs_geno_locus1 = []
    obs_geno_locus2 = []

    hap_dist = scipy.stats.rv_discrete(values=(range(4), hap_freq))
    obs_haps = hap_dist.rvs(size=2*n)

    obs_geno_locus1 = obs_haps / 2
    obs_geno_locus1 = np.array(obs_geno_locus1[::2] + obs_geno_locus1[1::2], dtype=float)

    obs_geno_locus2 = obs_haps % 2
    obs_geno_locus2 = np.array(obs_geno_locus2[::2] + obs_geno_locus2[1::2], dtype=float)

    perm = np.random.permutation(n)
    obs_geno_locus1[perm[0:int(mask_frac*n)]] = np.nan

    perm = np.random.permutation(n)
    obs_geno_locus2[perm[0:int(mask_frac*n)]] = np.nan

    return obs_geno_locus1, obs_geno_locus2


# In[ ]:

# sample some data from the model
n = 1000
p = [0.1,0.2,0.3,0.4]
obs_geno_locus1, obs_geno_locus2 = sample_genotypes(1000, p, mask_frac=0.01)


# In[ ]:

# pick an initialization point
np.random.seed(0)
init_hap_freq = np.random.rand(4)
init_hap_freq /= np.sum(init_hap_freq)

mle = scipy.optimize.minimize(neg_log_lik, init_hap_freq, args=(obs_geno_locus1, obs_geno_locus2), 
                              method='SLSQP', 
                              jac=grad_neg_log_lik, bounds=[(1., None)]*4,
                              #constraints=({'type': 'eq', 'fun': eq_constraints}),
                              options={'disp': True})
x_sum = np.sum(mle.x)
print mle.x / x_sum


# In[ ]:

# using finite difference instead of the AD jacobian
np.random.seed(0)
init_hap_freq = np.random.rand(4)
init_hap_freq /= np.sum(init_hap_freq)

mle = scipy.optimize.minimize(neg_log_lik, init_hap_freq, args=(obs_geno_locus1, obs_geno_locus2), 
                              method='SLSQP', 
                              bounds=[(1., None)]*4,
                              options={'disp': True})
x_sum = np.sum(mle.x)
print mle.x / x_sum


# In[ ]:

# compute asymptotic confidence intervals using the Fisher Information matrix
fim_x = hess_neg_log_lik(mle.x, obs_geno_locus1, obs_geno_locus2)
x_sum = np.sum(mle.x)

#rescale into frequency coordinates
#using the Jacobian of the mapping x_i = \theta_i \sum_j x_j
jac = np.zeros((3,4))
jac[:3,:3] = np.eye(3)*x_sum
jac[:,-1] = -x_sum

fim = np.dot(np.dot(jac, fim_x), jac.T)
fim = fim[:3,:3]

estimate = mle.x[:3] / x_sum
asymptotic_covmat = np.linalg.inv(fim)
marginal_sd = np.sqrt(np.diag(asymptotic_covmat))


# In[ ]:

z = scipy.stats.norm.ppf(0.975)
ci = np.array(zip(estimate - z*marginal_sd, estimate + z*marginal_sd))
print ci


# ### Asymptotic confidence intervals

# In[ ]:

np.random.seed(0)

n = 1000
p = np.array([0.1,0.2,0.3,0.4])

# number of simulation replicates for the CI calibration
n_reps = 500

mean_estimates = []
sd_estimates = []

for rep in range(n_reps):
    # sample data
    obs_geno_locus1, obs_geno_locus2 = sample_genotypes(n, p, mask_frac=0.01)
    
    # optimize parameters
    init_hap_freq = np.random.rand(4)
    init_hap_freq /= np.sum(init_hap_freq)

    if rep % 50 == 0:
        print rep
        
    mle = scipy.optimize.minimize(neg_log_lik, init_hap_freq, args=(obs_geno_locus1, obs_geno_locus2), 
                              method='SLSQP', 
                              jac=grad_neg_log_lik, bounds=[(1., None)]*4,
                              options={'disp': False})
    
    # the Fisher information matrix is the Hessian of the negative log likelihood
    fim_x = hess_neg_log_lik(mle.x, obs_geno_locus1, obs_geno_locus2)
    x_sum = np.sum(mle.x)

    #rescale into frequency coordinates
    #using the Jacobian of the mapping x_i = \theta_i \sum_j x_j
    jac = np.zeros((3,4))
    jac[:3,:3] = np.eye(3)*x_sum
    jac[:,-1] = -x_sum
    
    fim = np.dot(np.dot(jac, fim_x), jac.T)
    fim = fim[:3,:3]
    
    asymptotic_covmat = np.linalg.inv(fim)
    marginal_sd = np.sqrt(np.diag(asymptotic_covmat))
    
    mean_estimates.append(mle.x[:3] / x_sum)
    sd_estimates.append(marginal_sd)


# In[ ]:

mean_estimates = np.array(mean_estimates)
sd_estimates = np.array(sd_estimates)
p_check = np.tile(p[:3], (n_reps, 1))
alphas = np.linspace(0.01, 1.00, 100)
outside_ci = []
for alpha in alphas:
    z = scipy.stats.norm.ppf(1. - alpha/2.)
    lbs = mean_estimates - z*sd_estimates
    ubs = mean_estimates + z*sd_estimates
    outside_ci.append(1. * np.sum((p_check < lbs) | (ubs < p_check), axis=0) / n_reps)
outside_ci = np.array(outside_ci).T


# In[ ]:

labels = ["$x_{00}$", "$x_{01}$", "$x_{10}$"]
f = plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(alphas, outside_ci[i], label=labels[i])
plt.plot([0,1], [0,1], label="Theoretical")
plt.legend(loc='best')
plt.savefig("confidence_interval_calibration.pdf", bbox_inches="tight")

# In[ ]:



