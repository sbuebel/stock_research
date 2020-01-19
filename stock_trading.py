from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import operator
import pandas as pd
import time
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
	# Your key here - keep it
	key = '6LGG0QGAGBROB2M6'

	# make the directory if it doesn't exist
	if os.path.isdir(dire) == False:
		os.mkdir(ticker)
		os.mkdir(dire)

	ts = TimeSeries(key=key, output_format='pandas')
	ti = TechIndicators(key=key, output_format='pandas')
	
	### PRICE ###############################################################
	intra, meta = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')
	intra.to_csv(dire+ticker+"_prices.csv")
	
	### SMA #################################################################
	sma8, meta = ti.get_sma(symbol=ticker, interval='1min', time_period=8)
	sma8.to_csv(dire+ticker+"_sma8.csv")

	### EMA #################################################################
	ema8, meta = ti.get_ema(symbol=ticker, interval='1min', time_period=8)
	ema8.to_csv(dire+ticker+"_ema8.csv")

	### VWAP ################################################################
	vwap, meta = ti.get_vwap(symbol=ticker, interval='1min')
	vwap.to_csv(dire+ticker+"_vwap.csv")

	### MACD ################################################################
	macd, meta = ti.get_macd(symbol=ticker, interval='1min')
	macd.to_csv(dire+ticker+"_macd.csv")

	# they limit user to 5 calls per minute and 500 per day for free access #
	print("waiting so we don't overload API (60 seconds...)")
	time.sleep(60)

	### STOCH ###############################################################
	stoch, meta = ti.get_stoch(symbol=ticker, interval='1min')
	stoch.to_csv(dire+ticker+"_stoch.csv")

	### RSI #################################################################
	rsi, meta = ti.get_rsi(symbol=ticker, interval='1min', time_period=60)
	rsi.to_csv(dire+ticker+"_rsi.csv")

	### ADX #################################################################
	adx, meta = ti.get_adx(symbol=ticker, interval='1min', time_period=60)
	adx.to_csv(dire+ticker+"_adx.csv")

	### CCI #################################################################
	cci, meta = ti.get_cci(symbol=ticker, interval='1min', time_period=60)
	cci.to_csv(dire+ticker+"_cci.csv")

	# TODO: These are tricky, several lines, parse differently, separate indices really
	### AROON ###############################################################
	aroon, meta = ti.get_aroon(symbol=ticker, interval='1min', time_period=60)
	aroon.to_csv(dire+ticker+"_aroon.csv")

	# they limit user to 5 calls per minute and 500 per day for free access #
	print("waiting so we don't overload API (60 seconds...)")
	time.sleep(60)

	### BBANDS ###############################################################
	bbands, meta = ti.get_bbands(symbol=ticker, interval='1min', time_period=60)
	bbands.to_csv(dire+ticker+"_bbands.csv")

	'''
	#################################################################################
	#################################################################################
	# LESS USED #####################################################################
	#################################################################################
	#################################################################################

	### WMA #################################################################
	wma8, meta = ti.get_wma(symbol=ticker, interval='1min', time_period=8)
	wma8.to_csv(dire+ticker+"_wma8.csv")

	### DEMA ################################################################
	dema8, meta = ti.get_dema(symbol=ticker, interval='1min', time_period=8)
	dema8.to_csv(dire+ticker+"_dema8.csv")

	### TEMA ################################################################
	tema8, meta = ti.get_tema(symbol=ticker, interval='1min', time_period=8)
	tema8.to_csv(dire+ticker+"_tema8.csv")

	### TRIMA ###############################################################
	trima8, meta = ti.get_trima(symbol=ticker, interval='1min', time_period=8)
	trima8.to_csv(dire+ticker+"_trima8.csv")

	### KAMA ################################################################
	kama8, meta = ti.get_kama(symbol=ticker, interval='1min', time_period=8)
	kama8.to_csv(dire+ticker+"_kama8.csv")

	### MAMA ################################################################
	mama8, meta = ti.get_mama(symbol=ticker, interval='1min', time_period=8)
	mama8.to_csv(dire+ticker+"_mama8.csv")

	### T3 ##################################################################
	t3, meta = ti.get_t3(symbol=ticker, interval='1min', time_period=8)
	t3.to_csv(dire+ticker+"_t3.csv")

	### MACDEXT #############################################################
	macdext, meta = ti.get_macdext(symbol=ticker, interval='1min')
	macdext.to_csv(dire+ticker+"_macdext.csv")

	### STOCHF ##############################################################
	stochf, meta = ti.get_stochf(symbol=ticker, interval='1min')
	stochf.to_csv(dire+ticker+"_stochf.csv")

	### STOCHRSI ############################################################
	stochrsi, meta = ti.get_stochrsi(symbol=ticker, interval='1min', time_period=8)
	stochrsi.to_csv(dire+ticker+"_stochrsi.csv")

	### WILLR ###############################################################
	willr, meta = ti.get_willr(symbol=ticker, interval='1min', time_period=8)
	willr.to_csv(dire+ticker+"_willr.csv")

	### ADXR #################################################################
	adxr, meta = ti.get_adxr(symbol=ticker, interval='1min', time_period=8)
	adxr.to_csv(dire+ticker+"_adxr.csv")

	#################################################################################
	#################################################################################
	'''

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