from Stock_Info import Stock_Info
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # run regressions on the data we scrape
    ticker = input('Please enter a stock ticker: ')
    stock = Stock_Info(ticker)
    financials = stock.get_financials()

    # run a regression using the financial data to predict 
    # an average price for the stock for the following year
    y = financials['AVG Stock Price']
    X = financials.drop(columns = ['AVG Stock Price'], axis = 1)
    # drop columns with NaN
    X.dropna(axis=1, inplace=True)
    # drop the "total" columns
    cols = []
    for column in X.columns:
        if 'Total' in column:
            cols.append(column)

    X = X.drop(columns =cols, axis =1)
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    model = SVR(kernel = 'rbf')
    model.fit(scaled_X, y)
    print('Generating a prediction: ')
    values = []
    for column in X.columns:
        # in theory we would prompt values but for demonstration purposes,
        # I generated random values
        #prompt = 'Enter a value for the', column
        #value = input(prompt)
        
        value = np.random.randint(5000000, 10000000)
        values.append(float(value))
    scaled_test = scaler.transform(np.array(values).reshape(1, -1))
    prediction = model.predict(scaled_test)
    print('Predicted avg stock price for the next year:', prediction)

    # run a regression on daily stocks to build a model
    stock_data = stock.get_stock_prices()

    X = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'MA_5', 'MA_10', 'MA_20', 'RSI']]
    y = stock_data['Daily AVG']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print('Mean Squared Error for our model:', mse)
