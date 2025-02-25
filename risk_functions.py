import numpy as np
from scipy.stats import norm


def Simulate_Price_Black_Scholes(initial_value, tendency, volatility, nb_simulations, Horizon=1, dt=1/365):
    """
    Simulates price trajectories using the Black-Scholes model.
                dS_t = mu S_t dt + sigma S_t dW_t
                
    Parameters:
    initial_value (float): The initial value of the price.
    tendency (float): The drift term (mu) in the Black-Scholes model.
    volatility (float): The volatility term (sigma) in the Black-Scholes model.
    nb_simulations (int): The number of simulations to run.
    Horizon (float, optional): The simulation horizon in years. Default is 1 year.
    dt (float, optional): The time step for the simulation. Default is 1/252 (1 day).
    
    Returns:
    np.ndarray: A 2D array where each row represents a simulated price trajectory.
    """
    # Calculate the number of time steps
    num_steps = round(Horizon / dt)
    
    # Initialize the array to store the simulated trajectories
    trajectories = np.zeros((nb_simulations, num_steps + 1))
    
    # Set the initial value for all simulations
    trajectories[:, 0] = initial_value
    
    # Simulate the trajectories
    for t in range(1, num_steps + 1):
        # Generate random normal values for the Wiener process
        dWt = np.random.normal(0, np.sqrt(dt), nb_simulations)
        
        # Calculate the price at the next time step
        trajectories[:, t] = trajectories[:, t-1] * np.exp((tendency - 0.5 * volatility**2) * dt + volatility * dWt)
    
    return trajectories

def Black_Scholes_Formula(vol, risk_free_rate, current_price, strike, maturity):
    r"""
    Black_Scholes_Formula(vol, risk_free_rate, current_price, strike, maturity)
    Function computing the Black-Scholes formula to obtain the price of a Call option
    Parameters
    ----------
    vol : Float
        volatility of the underlying
    risk_free_rate : Float
        value of the risk free rate
    current_price : Float
        Value of the underlying price (X_0)
    strike : Float
        Value of the strike (K)
    maturity : Float
        Maturity of the option (T-t)

    Returns
    -------
    Value : Float
        BS formua : X0 \Phi(d_1) - K e^{-r(T_t)} \Phi(d_2)
    """
    # Computation of d_1 and d_2 of the Black-Scholes Formula
    d_1 = (np.log(current_price/strike) + (risk_free_rate + 0.5 * vol**2)*maturity)/(vol * np.sqrt(maturity))
    d_2 = d_1 - vol * np.sqrt(maturity)
    #☻ Black-Scholes formula
    Value = current_price * norm.cdf(d_1) - strike * np.exp(- risk_free_rate * maturity) * norm.cdf(d_2)
    return Value

def Delta_Computation(vol, risk_free_rate, current_price, strike, maturity):
    r"""
    Delta_Computation(vol, risk_free_rate, current_price, strike, maturity)
    Compute the Delta of the Call option from the Black-Scholes formula
    Parameters
    ----------
    vol : Float
        volatility of the underlying
    risk_free_rate : Float
        value of the risk free rate
    current_price : Float
        Value of the underlying price (X_0)
    strike : Float
        Value of the strike (K)
    maturity : Float
        Maturity of the option (T-t)

    Returns
    -------
    Float
        Delta BS : \Phi(d_1)
    """
    # Computation of d_1 of the Black-Scholes Formula
    d_1 = (np.log(current_price/strike) + (risk_free_rate + 0.5 * (vol**2))*maturity)/(vol * np.sqrt(maturity))
    # Delta 
    return norm.cdf(d_1)
    
