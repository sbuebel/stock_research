from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import operator
import os


# helper function for determining gradient - add by default
def combine_dicts(a, b, op=operator.add):
	res = {}
	for k in a.keys():
		if k in a.keys() and k in b.keys():
			res[k] = op(a[k], b[k])

	# don't lose data if they don't both have a key
	for key in a.keys():
		if key not in res.keys():
			res[key] = a[key]
	for key in b.keys():
		if key not in res.keys():
			res[key] = b[key]

	return res


# function to read data from API and store it locally in CSV
def data_to_csv(ticker, dire):
	print(f"Fetching historical data for {ticker}\nStoring data here: {dire}")

	# Your key here - keep it - not an issue, free
	key = '6LGG0QGAGBROB2M6'

	# make the directory if it doesn't exist
	if not os.path.isdir(dire):
		os.mkdir(ticker)
		os.mkdir(dire)
	else:
		print(f"Data already exists for {dire} - skipping API call.")

	ts = TimeSeries(key=key, output_format='pandas')
	ti = TechIndicators(key=key, output_format='pandas')
	
	# PRICE ###############################################################
	intra, meta = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')
	intra = intra[::-1]  # for some reason, it gives it backwards
	intra.to_csv(dire+ticker+"_prices.csv")

	# MACD ################################################################
	macd, meta = ti.get_macd(symbol=ticker, interval='1min')
	macd = macd[::-1]  # for some reason, it gives it backwards
	macd.to_csv(dire+ticker+"_macd.csv")


# helper function to buy
def buy(cash, shares, num, price):
	if cash > num*price:
		cash -= num*price
		shares += num
	else:
		shares += cash/price
		cash = 0

	return cash, shares


# helper function to sell
def sell(cash, shares, num, price):
	if shares > num:
		shares -= num
		cash += num*price
	else:
		cash += shares*price
		shares = 0

	return cash, shares