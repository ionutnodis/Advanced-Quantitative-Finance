# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:38:49 2024

@author: uctpwcu
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

S0 = 110
r = 0.02
T = 2
sigma = 0.3
K = 105
N = 1000000

############## - Put option price using MC:
# Mean and variance of LogS_T
logS_mean = np.log(S0) + (r - 0.5*sigma**2) * T
logS_std = sigma * np.sqrt(T)

# Simulate Nx1 normal RVs
logS = np.random.normal(logS_mean, logS_std, N) 

# Payoff of put for each simulation
V = np.exp(-r*T) * np.maximum(K - np.exp(logS), 0)

# Value of put
P_mc = np.mean(V) #we take the average of all sampled paths and compute the present price of the option

# Display results
print(f"The MC price using {N} simulations is {P_mc}.")
sigv = np.std(V)
CI = [np.mean(V) - 1.96*sigv/np.sqrt(N), np.mean(V) + 1.96*sigv/np.sqrt(N)]
print(f"A 95% confidence band for the MC price is ({CI[0]}; {CI[1]}).")
print(f"The width of the confidence band is {CI[1] - CI[0]}.")
print()

# Plotting for illustration
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(logS, density=True, bins=50)
axs[0].set_xlabel('logS')
axs[0].set_ylabel('Probability Density')
axs[0].set_title('logS distribution')

axs[1].hist(np.exp(logS), density=True, bins=50)
axs[1].set_xlabel('S')
axs[1].set_ylabel('Probability Density')
axs[1].set_title('S distribution')

plt.tight_layout()
plt.show()


# Payoff of call for each simulation
V = np.exp(-r*T) * np.maximum(np.exp(logS) - K, 0)

# Value of call
C_mc = np.mean(V) #we take the average of all sampled paths and compute the present price of the option

# Display results
print(f"The MC price using {N} simulations is {C_mc}.")
sigv = np.std(V)
CI = [np.mean(V) - 1.96*sigv/np.sqrt(N), np.mean(V) + 1.96*sigv/np.sqrt(N)]
print(f"A 95% confidence band for the MC price is ({CI[0]}; {CI[1]}).")
print(f"The width of the confidence band is {CI[1] - CI[0]}.")
print()


############## - ANTITHETIC VARIABLES 
######### (please compare the range of the confidence interval with the previous result)
X = np.random.normal(0, logS_std, N)
logS1 = logS_mean + X
logS2 = logS_mean - X
V_av = 0.5 * (np.maximum(np.exp(logS1) - K, 0) + np.maximum(np.exp(logS2) - K, 0))
C_av = np.exp(-r*T) * np.mean(V_av)
sigv_av = np.std(V_av)
CI_av_u = np.exp(-r*T) * (np.mean(V_av) + 1.96*sigv_av/np.sqrt(N))
CI_av_l = np.exp(-r*T) * (np.mean(V_av) - 1.96*sigv_av/np.sqrt(N))
print('The MC price using anthithetic variables is', C_av)
print('A 95% confidence band for the MC price using anthithetic variables is', [CI_av_l, CI_av_u])
print('The width of the confidence band is', CI_av_u - CI_av_l)
print()

