# setup
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

headers = { 
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36", 
	"Referer": "https://targetwebsite.com/page1" 
} 

class Stock_Info:
    def __init__(self, stock):
        self.stock = stock

    def get_financials(self):
        '''
        Function to scrape web data and process it
        '''
        url_data = f'https://finance.yahoo.com/quote/{self.stock}/financials'
        response_data = requests.get(url_data, headers=headers)
        print("response.ok : {} , response.status_code : {}".format(response_data.ok , response_data.status_code))
        soup_data = BeautifulSoup(response_data.text, 'html.parser')
        categories = soup_data.findAll('div', attrs = {'class' : 'column sticky svelte-1xjz32c'})
        labels = []
        for category in categories:
            labels.append(category.text)
        vals1 = soup_data.findAll('div', attrs = {'class' : 'column svelte-1xjz32c alt'})
        ytd = []
        vals2020 = []
        vals2022 = []
        for i in range(len(vals1)):
            if i%3 == 0:
                if vals1[i].text == '-- ':
                    ytd.append(np.NaN)
                else:
                    stringNum = vals1[i].text
                    ytd.append(float(stringNum.replace(',', '')))
            if i%3 == 1:
                if vals1[i].text == '-- ':
                    vals2022.append(np.NaN)
                else:
                    stringNum = vals1[i].text
                    vals2022.append(float(stringNum.replace(',', '')))
            if i%3 == 2:
                if vals1[i].text == '-- ':
                    vals2020.append(np.NaN)
                else:
                    stringNum = vals1[i].text
                    vals2020.append(float(stringNum.replace(',', '')))
        vals2 = soup_data.findAll('div', attrs = {'class' : 'column svelte-1xjz32c'})
        vals2023 = []
        vals2021 = []
        for i in range(len(vals2)):
            if i%2 == 0:
                if vals1[i].text == '-- ':
                    vals2023.append(np.NaN)
                else:
                    stringNum = vals1[i].text
                    vals2023.append(float(stringNum.replace(',', '')))
            if i%2 == 1:
                if vals1[i].text == '-- ':
                    vals2021.append(np.NaN)
                else:
                    stringNum = vals1[i].text
                    vals2021.append(float(stringNum.replace(',', '')))
        categories = soup_data.findAll('div', attrs = {'class' : 'rowTitle svelte-1xjz32c'})
        labels = []
        for category in categories:
            labels.append(category.text)
        data = pd.DataFrame({'YTD' : ytd, '2023' : vals2023, '2022' : vals2022,
                        '2021' : vals2021, '2020' : vals2020}, index = labels)
        data = data.T
        url_prices = f'https://finance.yahoo.com/quote/{self.stock}/history?filter=history&period1=1555791731&period2=1713644468&frequency=1mo'
        response_prices = requests.get(url_prices, headers=headers)
        print("response.ok : {} , response.status_code : {}".format(response_prices.ok , response_prices.status_code))
        soup_prices = BeautifulSoup(response_prices.text, 'html.parser')
        rows = soup_prices.findAll('tr', attrs = {'class' : 'svelte-ewueuo'})
        stocks2024 = []
        stocks2023 = []
        stocks2022 = []
        stocks2021 = []
        stocks2020 = []
        for row in rows:
            if '2024' in row.text and 'Dividend' not in row.text:
                stocks2024.append(row.text)
            if '2023' in row.text and 'Dividend' not in row.text:
                stocks2023.append(row.text)
            if '2022' in row.text and 'Dividend' not in row.text:
                stocks2022.append(row.text)
            if '2021' in row.text and 'Dividend' not in row.text:
                stocks2021.append(row.text)
            if '2020' in row.text and 'Dividend' not in row.text:
                if 'Stock Splits' not in row.text:
                    stocks2020.append(row.text) 
        avgs = []
        avgs.append(self.process_stock_prices(stocks2024))
        avgs.append(self.process_stock_prices(stocks2023))
        avgs.append(self.process_stock_prices(stocks2022))
        avgs.append(self.process_stock_prices(stocks2021))
        avgs.append(self.process_stock_prices(stocks2020))
        data['AVG Stock Price'] = avgs
        return data

    def process_stock_prices(self, stocks):
        '''
        Get the average stock price for the year
        '''
        result = []
        for row in stocks:
            row = row.split(' ')
            row = row[3:]
            row = row[:len(row) - 2]
            if len(row) == 5:
                for i in range(5):
                    result.append(float(row[i]))
            return np.mean(result)

    def get_stock_prices(self):
        '''
        Returns a dataframe of daily stock prices and other measures 
        '''
        url = f'https://finance.yahoo.com/quote/{self.stock}/history?period1=1555828231&period2=1713681029'
        response = requests.get(url, headers=headers)
        print("response.ok : {} , response.status_code : {}".format(response.ok , response.status_code))
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.findAll('tr', attrs = {'class' : 'svelte-ewueuo'})
        stocks = []
        for row in rows[1:]:
            if 'Dividend' not in row.text:
                if 'Stock Splits' not in row.text:
                    stocks.append(row.text)

        # process it all
        daily_avgs = []
        stock_data = pd.DataFrame(columns = ['Open', 'High', 'Low', 'Close', 'Adj Close'])
        for row in stocks:
            row = row.split(' ')
            row = row[3:]
            row = row[:len(row) - 2]
            daily = []
            for i in range(5):
                daily.append(float(row[i]))
            new_row = pd.DataFrame({'Open' : float(row[0]), 'High' : float(row[1]), 'Low' : float(row[2]),
            'Close' : float(row[3]), 'Adj Close' : float(row[4])}, index = [0])
            daily_avgs.append(np.mean(daily))
            stock_data = pd.concat([stock_data, new_row], ignore_index=True)
        stock_data['Daily AVG'] = daily_avgs
        # calculate 5, 10, and 20 day moving averages
        stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA_10'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])

        # Fill NaN values if any
        stock_data.fillna(method='bfill', inplace=True)
        return stock_data

    def calculate_rsi(self, data, window=14):
        '''
        Function to calculate RSI (relative strength index)

        A way to measure the speed and change of price movements
        '''
        delta = data.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi



