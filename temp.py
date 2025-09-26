import yfinance as yf
import pandas as pd
import datetime as dt

start = dt.datetime.today() - dt.timedelta(5)  # 設定開始日期，五年(1825天)前
end = dt.datetime.today()  # 設定結束日期，今天
ticker = "APPL"
rf_data = yf.download(ticker, start=start, end=end,interval="1y")
print(rf_data)