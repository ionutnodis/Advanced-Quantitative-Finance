
'''
Pricing options with Binomial tree (adjusted for dividends)

author: @ionutnodis

'''






##Pricing options with Binomial tree (adjusted for dividends) 

import numpy as np

def binomial_tree_div(S0, K, T, r, sigma, N, div_times, div_amount, option_type='call'):
    """
    Binomial tree model for option pricing with two dividend payments.

    Parameters:
    S0 : float : Initial stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying stock (annualized)
    N : int : Number of steps in the binomial tree
    div_times : list : Times (in years) of dividend payments
    div_amount : list : Amounts of dividends paid at each dividend time
    option_type : str : "call" for a call option, "put" for a put option

    Returns:
    float : Option price
    """

    if len(div_times) != len(div_amount):
        raise ValueError('div_times and div_amount must have the same length')

    # Time step size
    dt = T / N

    # Compute the up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Compute the risk-neutral probability of going up and down
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p

    # Initialize the stock price tree
    stock_price = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_price[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Adjust for dividends
    div_indices = [int(t / dt) for t in div_times]
    for div_time, div_amt in zip(div_indices, div_amount):
        for i in range(div_time + 1):
            stock_price[i, div_time] = max(0, stock_price[i, div_time] - div_amt)

    # Initialize the option value tree
    option_values = np.zeros((N + 1, N + 1))

    # Compute option values at the terminal nodes
    if option_type == 'call':
        option_values[:, N] = np.maximum(0, stock_price[:, N] - K)
    elif option_type == 'put':
        option_values[:, N] = np.maximum(0, K - stock_price[:, N])
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + q * option_values[j + 1, i + 1])

    return option_values[0, 0]


# # Example from Lecture 1 slide 29
# S0 = 100
# K = 100
# T = 1
# sigma = 0.30
# r = 0.10
# N = 6
# div_times = [0.25, 0.75]
# div_amount = [1, 1]
# option_type = 'put'

# # Calculate the price
# price = binomial_tree_div(S0, K, T, r, sigma, N, div_times, div_amount, option_type)
# print(f"The price of the {option_type} option is: {price:.2f}")



# BLANK INPUT FOR FUTURE COMPUTATIONS 
S0 = 250
K = 300
T = 1
sigma = 0.40
r = 0.10
N = 10000
div_times = [0.25, 0.5, 0.75]
div_amount = [1, 1, 1]
option_type = 'put'

# Calculate the price
price = binomial_tree_div(S0, K, T, r, sigma, N, div_times, div_amount, option_type)
print(f"The price of the {option_type} option is: {price:.2f}")

# THIS NEEDS FIXING :(((((

# import matplotlib.pyplot as plt

# def plot_option_tree(option_values, N):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i in range(N + 1):
#         for j in range(i + 1):
#             ax.text(i, j, f'{option_values[j, i]:.2f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
#     ax.set_xticks(range(N + 1))
#     ax.set_yticks(range(N + 1))
#     ax.set_xlim(-1, N + 1)
#     ax.set_ylim(-1, N + 1)
#     ax.invert_yaxis()
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Nodes')
#     ax.set_title('Option Value Tree')
#     plt.show()

# # Calculate the option values tree
# option_values = np.zeros((N + 1, N + 1))
# if option_type == 'call':
#     option_values[:, N] = np.maximum(0, stock_price[:, N] - K)
# elif option_type == 'put':
#     option_values[:, N] = np.maximum(0, K - stock_price[:, N])
# for i in range(N - 1, -1, -1):
#     for j in range(i + 1):
#         option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + q * option_values[j + 1, i + 1])

# # Plot the option value tree
# plot_option_tree(option_values, N)