
"""
MSIN0107, Advanced Quantitative Finance Coursework I
date created: 26-01-2025 ; timestamp: 13:20

author: @ionutnodis / CW 1 AQF 
"""



"""
Question 1 a) from CW 
Compute the price of a standard put option
using Monte Carlo simulation with N = 10, 000 draws and compute a
95% condence interval for the option price
"""

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm


#random normal distribution 

np.random.seed(1) 

# Parameters
S0 = 100
r = 0.015
T = 2
sigma = 0.3
K = 100
N = 10000


# d1 = [np.log(S0/K) + (r + 0.5*sigma**2)*T]/ (sigma*np.sqrt(T))

# d2 = d1 - sigma*np.sqrt(T)

# #Compute the price of the put option 
# put = K*np.exp(-r*T)*N(-d2)- S0*N(-d1)

#Put option using Black Scholes Model 

def black_scholes_price(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return put

# Compute the price of the put option
P_bs = black_scholes_price(S0, K, T, r, sigma)

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
print(f"The Black-Scholes price is {P_bs}.")
print()


""""
Question 1 b) from CW
Compute the return of a standard put option using Monte
Carlo simulation with N= 10,000 draws using Antithetic Variables and
compute a 95% condence interval for the option price. Is the con-
dence interval wider or narrower than that in the previous question? Is
the result closer to the theoretical B-S result compared to the previous
case?

"""

#Compute the option price using Antithetic Variables 

X = np.random.normal(0, logS_std, N)
logS1 = logS_mean + X
logS2 = logS_mean - X
V_av = 0.5 * (np.maximum(K - np.exp(logS1), 0) + np.maximum(K - np.exp(logS2), 0))
P_av = np.exp(-r*T) * np.mean(V_av)
sigv_av = np.std(V_av)
CI_av_u = np.exp(-r*T) * (np.mean(V_av) + 1.96*sigv_av/np.sqrt(N))
CI_av_l = np.exp(-r*T) * (np.mean(V_av) - 1.96*sigv_av/np.sqrt(N))
print('The MC price using antithetic variables is', P_av)
print('A 95% confidence band for the MC price using antithetic variables is', [CI_av_l, CI_av_u])
print('The width of the confidence band is', CI_av_u - CI_av_l)
print(f"The Black-Scholes price is {P_bs}.")
print()

"""
Question 1 c) from CW 

Option with knock-in barrier. A knock-in option is activated if the underlying asset reaches a predetermined barrier during its life. 
Assume that the put has the "barrier" of $80: 
If the stock price decreases below $80, the option is activated. 
Compute the price of the put with the barrier using Monte Carlo simulation with N = 10000 draws. 
Is it dierent from the price of a standard put (why)? 
Compute a 95% confidence interval for the option return.

"""

# Parameters
S0 = 100
r = 0.015
T = 2
sigma = 0.3
K = 100
N = 10000
barrier = 80

# Mean and variance of LogS_T
logS_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
logS_std = sigma * np.sqrt(T)

# Simulate Nx1 normal RVs
logS = np.random.normal(logS_mean, logS_std, N)

# Simulate stock paths
dt = T / 252  # daily steps
S_paths = np.zeros((N, 252 + 1))
S_paths[:, 0] = S0
for t in range(1, 252 + 1):
    Z = np.random.normal(0, 1, N)
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Check if the barrier is hit
knock_in = np.min(S_paths, axis=1) < barrier

# Payoff of put for each simulation
V = np.exp(-r * T) * np.maximum(K - np.exp(logS), 0) * knock_in

# Value of put
P_knock_in = np.mean(V)
sigv_knock_in = np.std(V)
CI_knock_in_u = np.mean(V) + 1.96 * sigv_knock_in / np.sqrt(N)
CI_knock_in_l = np.mean(V) - 1.96 * sigv_knock_in / np.sqrt(N)

# Display results
print(f"The MC price of the knock-in put option using {N} simulations is {P_knock_in}.")
print(f"A 95% confidence band for the MC price is [{CI_knock_in_l}, {CI_knock_in_u}].")
print(f"The width of the confidence band is {CI_knock_in_u - CI_knock_in_l}.")

"""
Question 1 d) from CW 

Compute the value of the option with knock-out barrier above using Monte Carlo simulation with N = 10, 000 draws and using the standard put option as a control variate. 
Also, compute a 95% confidence interval for the option price. 
Is the confidence interval wider or narrower than that in the previous question?

"""

# Parameters
S0 = 100
r = 0.015
T = 2
sigma = 0.3
K = 100
N = 10000
barrier = 80

# Mean and variance of LogS_T
logS_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
logS_std = sigma * np.sqrt(T)

# Simulate Nx1 normal RVs
logS = np.random.normal(logS_mean, logS_std, N)

# Simulate stock paths
dt = T / 252  # daily steps
S_paths = np.zeros((N, 252 + 1))
S_paths[:, 0] = S0
for t in range(1, 252 + 1):
    Z = np.random.normal(0, 1, N)
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Check if the barrier is hit
knock_out = np.min(S_paths, axis=1) >= barrier

# Payoff of knock-out put for each simulation
V_knock_out = np.exp(-r * T) * np.maximum(K - np.exp(logS), 0) * knock_out

# Payoff of standard put for each simulation
V_standard = np.exp(-r * T) * np.maximum(K - np.exp(logS), 0)

# Control variate method
P_standard = np.mean(V_standard)
covariance = np.cov(V_knock_out, V_standard)[0, 1]
variance_standard = np.var(V_standard)
c = -covariance / variance_standard

# Adjusted payoff
V_adjusted = V_knock_out + c * (V_standard - P_standard)

# Value of knock-out put using control variate
P_knock_out_cv = np.mean(V_adjusted)
sigv_knock_out_cv = np.std(V_adjusted)
CI_knock_out_cv_u = P_knock_out_cv + 1.96 * sigv_knock_out_cv / np.sqrt(N)
CI_knock_out_cv_l = P_knock_out_cv - 1.96 * sigv_knock_out_cv / np.sqrt(N)

# Display results
print(f"The MC price of the knock-out put option using {N} simulations and control variate is {P_knock_out_cv}.")
print(f"A 95% confidence band for the MC price is [{CI_knock_out_cv_l}, {CI_knock_out_cv_u}].")
print(f"The width of the confidence band is {CI_knock_out_cv_u - CI_knock_out_cv_l}.")

# Compare confidence intervals
width_knock_out = CI_knock_out_cv_u - CI_knock_out_cv_l
width_knock_in = CI_knock_in_u - CI_knock_in_l 

if width_knock_out < width_knock_in:
    print("The confidence interval is narrower than that in the previous question.")
else:
    print("The confidence interval is wider than that in the previous question.")
    
    
    
"""
Question 2 a) from CW 

Compute the price of the Lookback put option using Monte Carlo simulation with N = 10, 000 draws. 
Also, compute a 95% condence interval for the option price. Is the price higher or lower than
the value of the standard put option above with the given strike price K = 100? Why?

"""    
    
# Parameters
S0 = 100
r = 0.015
T = 2
sigma = 0.3
N = 10000

# Simulate stock paths
dt = T / 252  # daily steps
S_paths = np.zeros((N, 252 + 1))
S_paths[:, 0] = S0
for t in range(1, 252 + 1):
    Z = np.random.normal(0, 1, N)
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Determine the maximum stock price during the life of the option
K_max = np.max(S_paths, axis=1)

# Payoff of Lookback put for each simulation
V_lookback = np.exp(-r * T) * np.maximum(K_max - S_paths[:, -1], 0)

# Value of Lookback put
P_lookback = np.mean(V_lookback)
sigv_lookback = np.std(V_lookback)
CI_lookback_u = P_lookback + 1.96 * sigv_lookback / np.sqrt(N)
CI_lookback_l = P_lookback - 1.96 * sigv_lookback / np.sqrt(N)

# Display results
print(f"The MC price of the Lookback put option using {N} simulations is {P_lookback}.")
print(f"A 95% confidence band for the MC price is [{CI_lookback_l}, {CI_lookback_u}].")
print(f"The width of the confidence band is {CI_lookback_u - CI_lookback_l}.")

# Compare with the standard put option price
P_standard = np.mean(np.exp(-r * T) * np.maximum(100 - S_paths[:, -1], 0))  # Assuming K = 100 for standard put
if P_lookback > P_standard:
    print("The price of the Lookback put option is higher than the value of the standard put option.")
else:
    print("The price of the Lookback put option is lower than the value of the standard put option.")
    
    
    
    
"""
Question 2 b) from CW 

Compute the price of the Lookback put option using Monte Carlo simulation with N = 10, 000 draws. 
Also, compute a 95% condence interval for the option price. Is the price higher or lower than
the value of the standard put option above with the given strike price K = 100? Why?

"""    

# Parameters
S0 = 100
r = 0.015
T = 2
sigma = 0.3
N_initial = 10000  # Initial number of simulations to estimate sigma

# Simulate stock paths
dt = T / 252  # daily steps
S_paths = np.zeros((N_initial, 252 + 1))
S_paths[:, 0] = S0
for t in range(1, 252 + 1):
    Z = np.random.normal(0, 1, N_initial)
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Determine the maximum stock price during the life of the option
K_max = np.max(S_paths, axis=1)

# Payoff of Lookback put for each simulation
V_lookback = np.exp(-r * T) * np.maximum(K_max - S_paths[:, -1], 0)

# Estimate standard deviation
sigma_lookback = np.std(V_lookback)

# Calculate required N for standard Monte Carlo
desired_width = 0.01
N_standard = (2 * 1.96 * sigma_lookback / desired_width) ** 2

# Using Antithetic Variables
X = np.random.normal(0, sigma * np.sqrt(T), N_initial)
logS1 = logS_mean + X
logS2 = logS_mean - X
V_av = 0.5 * (np.maximum(K_max - np.exp(logS1), 0) + np.maximum(K_max - np.exp(logS2), 0))
sigma_av = np.std(V_av)

# Calculate required N for Monte Carlo with Antithetic Variables
N_antithetic = (2 * 1.96 * sigma_av / desired_width) ** 2

# Display results
print(f"Estimated standard deviation for Lookback put: {sigma_lookback}")
print(f"Number of simulations needed for standard Monte Carlo: {int(np.ceil(N_standard))}")
print(f"Estimated standard deviation for Lookback put using Antithetic Variables: {sigma_av}")
print(f"Number of simulations needed for Monte Carlo with Antithetic Variables: {int(np.ceil(N_antithetic))}")