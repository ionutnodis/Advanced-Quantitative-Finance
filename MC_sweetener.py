# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:51:57 2024

@author: uctpwcu
"""

import numpy as np
from scipy.stats import norm

np.random.seed(1) # random number seed
S0 = 110
r  = 0.02
T  = 1
sigma = 0.3
K = 105
N = 10000000

##-------------- Define function: BS results for European Call and EuropeanPut
def blsprice(S0, K, r, T, sigma, q):
    d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return call_price, put_price

##--------------- Option with sweetener
# Simulate stock price in 6 months and 1 year:
# strike price drops 10 if S in 6 months is below S0
logS_std = sigma * np.sqrt(0.5)
logS05 = np.random.normal(np.log(S0) + (r - 0.5 * sigma**2) * 0.5, logS_std, N)
logS1 = np.random.normal(logS05 + (r - 0.5 * sigma**2) * 0.5, logS_std)

# Calculate option value same as before (where logS05 < log(S0) is a 0/1 function)
Vsw = np.exp(-r*T) * np.maximum(np.exp(logS1) - (K - 10 * (logS05 < np.log(S0))), 0)
C_sw = np.mean(Vsw)

# Display results:
print(f"Option with sweetener (1 year to maturity date): The MC price using {N} simulations is {C_sw}.")
sigvsw = np.std(Vsw)
CI = [np.mean(Vsw) - norm.ppf(0.995)*sigvsw/np.sqrt(N), np.mean(Vsw) + norm.ppf(0.995)*sigvsw/np.sqrt(N)]
print(f"A 99% confidence band for the MC price is ({CI[0]}; {CI[1]}).")
print(f"The width of the confidence band is {CI[1] - CI[0]}.")
print(" ")


##--------------- Control variate
Covariate = np.exp(-r*T) * np.maximum(np.exp(logS1) - K, 0) #C(X_i)
Covariate_true, P_BS = blsprice(S0, K, r, T, sigma, 0) #C; we know the call option price analytically
C_covariate = np.mean(Covariate)
sigcovariate = np.std(Covariate)
covar = np.cov(Vsw, Covariate)
beta = -covar[0, 1] / (sigcovariate ** 2)
C_cv = C_sw + beta * (C_covariate - Covariate_true)
sigcov = np.std(Vsw + beta * (Covariate - Covariate_true))
CI = [C_cv - norm.ppf(0.995) * sigcov / np.sqrt(N), C_cv + norm.ppf(0.995) * sigcov / np.sqrt(N)]

#Display results:
print('The MC price using', N, 'simulations is', C_cv, '.')
print('A 99% confidence band for the MC price is (', CI[0], ';', CI[1], ').')
print('The width of the confidence band is', CI[1] - CI[0], '.')
print(' ')

##-------------- Whatif without sweetener?
# Mean and variance of LogS_T
logS_mean = np.log(S0) + (r - 0.5*sigma**2) * T
logS_std = sigma * np.sqrt(T)
# Simulate Nx1 normal RVs:
logS = np.random.normal(logS_mean, logS_std, N)
# Payoff of call for each simulation:
V = np.exp(-r*T) * np.maximum(np.exp(logS) - K, 0)
# Value of call:
C_mc = np.mean(V)
# Display results:
print('Without sweetener, the MC price using', N, 'simulations is', C_mc, '.')
sigv = np.std(V)
CI = [np.mean(V) - norm.ppf(0.975)*sigv/np.sqrt(N), np.mean(V) + norm.ppf(0.975)*sigv/np.sqrt(N)]
print('A 95% confidence band for the MC price is (', CI[0], ';', CI[1], ').')
print('The width of the confidence band is', CI[1]-CI[0], '.')
print(' ')