import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Stock_Info import Stock_Info

class UI:
    def __init__(self, root):
        self.root = root
        self.root.title('Stock Analyzer')
        self.description_label = tk.Label(root, text='Hello! This is my project for the 2024 Tribehacks hackathon. In short, I created a program that can take a ticker for a stock and then retrieve financial data and stock price data for that company. Ideally, the program will then try to run some simplistic regressions\n that will return an estimated average stock price for the next year depending on the financial values you put in (code is currently set to generate random numbers for demonstration purposes). The number returned is more to see what the company\'s\n direction and future look like. The second regression performed will train a model to predict an average price for the day-to-day average price of the stock. Since I had more data points for the day-to-day pricing, I chose return the mean squared error of\n the model so that we could evaluate its performance.')
        self.description_label.pack(pady=10)
        self.ticker_label = tk.Label(root, text='Enter Stock Ticker:')
        self.ticker_label.pack()
        self.ticker_entry = tk.Entry(root)
        self.ticker_entry.pack()
        self.submit_button = tk.Button(root, text='Predict Stock Price', command=self.predict_stock_price)
        self.submit_button.pack(pady=10)
        self.stock_data_text = tk.Text(root, height=50, width=200)
        self.stock_data_text.pack(pady=10)

    def predict_stock_price(self):
        ticker = self.ticker_entry.get()
        if not ticker:
            messagebox.showerror('Error', 'Please enter a stock ticker.')
            return

        stock = Stock_Info(ticker)
        financials = stock.get_financials()

        financials.dropna(axis=1, inplace=True)

        # run a regression using the financial data to predict 
        # an average price for the stock for the following year
        y = financials['AVG Stock Price']
        X = financials.drop(columns = ['AVG Stock Price'], axis = 1)

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
        messagebox.showinfo('SVR Prediction', f'Predicted avg stock price for the next year: {prediction}')

        # run a regression on daily stocks to build a model
        stock_data = stock.get_stock_prices()

        self.stock_data_text.insert(tk.END, f'Last 10 Days\' Stock Price Data for {ticker}:\n')
        self.stock_data_text.insert(tk.END, stock_data.head(10).to_string(index=False)+'\n')

        X = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'MA_5', 'MA_10', 'MA_20', 'RSI']]
        y = stock_data['Daily AVG']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        messagebox.showinfo('Random Forest Result', f'Mean Squared Error for Random Forest model: {mse}')


