from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import time, os

# function to read data from API and store it locally in CSV
def data_to_csv(ticker):
	# Your key here - keep it
	key = '6LGG0QGAGBROB2M6'

	dire = 'DATA/'+ticker+'/'
	no_dir = True
	i = 0
	while no_dir:
		if os.path.isdir(dire+str(i)):
			i += 1
		else:
			dire += str(i) + '/'
			os.mkdir(dire)
			no_dir = False # break out of loop

	print(dire)

	ts = TimeSeries(key=key, output_format='pandas')
	ti = TechIndicators(key=key, output_format='pandas')
	
	### PRICE ###############################################################
	intra, meta = ts.get_intraday(symbol=ticker, interval='5min', outputsize='full')
	intra.to_csv(dire+ticker+"_prices.csv")
	
	### SMA #################################################################
	sma8, meta = ti.get_sma(symbol=ticker, interval='5min', time_period=8)
	sma8.to_csv(dire+ticker+"_sma8.csv")

	### EMA #################################################################
	ema8, meta = ti.get_ema(symbol=ticker, interval='5min', time_period=8)
	ema8.to_csv(dire+ticker+"_ema8.csv")

	### MACD ################################################################
	macd, meta = ti.get_macd(symbol=ticker, interval='5min')
	macd.to_csv(dire+ticker+"_macd.csv")

# function to read data from API and return the data struct
def data_to_struct(ticker):
	# Your key here - keep it
	key = '6LGG0QGAGBROB2M6'

	ts = TimeSeries(key=key, output_format='json')
	ti = TechIndicators(key=key, output_format='json')
	
	### PRICE ###############################################################
	intra, meta = ts.get_intraday(symbol=ticker, interval='5min', outputsize='full')
	
	### SMA #################################################################
	# sma8, meta = ti.get_sma(symbol=ticker, interval='5min', time_period=8)

	### EMA #################################################################
	# ema8, meta = ti.get_ema(symbol=ticker, interval='5min', time_period=8)

	### MACD ################################################################
	# macd, meta = ti.get_macd(symbol=ticker, interval='5min')

	return intra

if __name__ == "__main__":
	ticker = input("What stock: ")

	data_to_csv(ticker)