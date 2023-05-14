import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

stocks = ['CHWY', 'QQQ', 'DIA', 'SPY', 'NDAQ', 'HLT', 'MAR', 'CSCO', 'AXP', 'ACN', 'NVDA', 'TEAM', 'CRM']

start_date = '2015-01-01'
end_date = date.today()

data = pd.DataFrame()
for stock in stocks:
    stock_data = yf.download(stock, start=start_date, end=end_date)
    data[stock] = stock_data['Adj Close']

returns = np.log(data / data.shift(1))

# Define the number of simulations to run for Monte Carlo simulation - set to 100 for runtime
num_simulations = 100

# Define the number of portfolios to generate for each simulation
num_portfolios = 1000

results = np.zeros((3+len(stocks), num_portfolios*num_simulations))

iters = 0

for i in range(num_simulations):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    for j in range(num_portfolios):
        portfolio_returns = np.sum(returns.mean() * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        results[0,i*num_portfolios+j] = portfolio_returns
        results[1,i*num_portfolios+j] = portfolio_volatility
        results[2,i*num_portfolios+j] = results[0,i*num_portfolios+j] / results[1,i*num_portfolios+j]
        for k in range(len(weights)):
            results[k+3,i*num_portfolios+j] = weights[k]
    iters += 1
    print(f"Monte Carlo - Simulation {iters}")

# Calculate the optimal portfolio using Markowitz portfolio optimization method
max_sharpe_idx = np.argmax(results[2])
optimal_weights = results[3:,max_sharpe_idx]
optimal_returns = results[0,max_sharpe_idx]
optimal_volatility = results[1,max_sharpe_idx]
optimal_sharpe_ratio = results[2,max_sharpe_idx]

print("Optimal Portfolio Weights:")
for i in range(len(stocks)):
    print(stocks[i], ": ", optimal_weights[i])
print("Optimal Portfolio Returns: ", optimal_returns)
print("Optimal Portfolio Volatility: ", optimal_volatility)
print("Optimal Sharpe Ratio: ", optimal_sharpe_ratio)
