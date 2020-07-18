from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time

# FORMAT:	 TICKER, 	QUANTITY,	PRICE_SOLD, 		CUR_PRICE
TICKERS = [ 
			['GPRO', 	500, 		2.6,				0],
			['DIS',		15,			98.6,				0],
			['BA',		3,			161.5,				0],
			['VWAGY',	70,			12.82,				0],
			['TWTR',	40,			25.32,				0],
			['PYPL',	10,			95.14,				0],
			['AMZN',	1,			1915.33,			0],
			['GOOGL',	1,			1115.36,			0],
			['AAPL',	5,			249,				0],
			['QCOM',	10,			66.98,				0],
			['INTC',	15,			53.44,				0],
			['NVDA',	4,			252.57,				0],
			['AMD',		20,			46.71,				0],
			['TSLA',	3,			500,				0]
		  ]

# function to read data from API and store it locally in CSV
def get_prices():
	# Your key here - keep it
	key = '6LGG0QGAGBROB2M6'

	ts = TimeSeries(key=key, output_format='pandas')

	start_time = time.time()
	count = 0

	# track these throughout
	sell_value = 0.0
	curr_value = 0.0

	for stock in TICKERS:
		print(stock[0], end=' ', flush=True)
		count += 1
		stock[3] = float(ts.get_quote_endpoint(symbol=stock[0])[0]['05. price']['Global Quote'])

		if count == 5:
			print(' [ delay ]', flush=True)
			for i in range(0, 60):
				time.sleep(1)
				print(' .', end='', flush=True)
			count = 0
			print('', flush=True)

		# update values of holdings
		sell_value += stock[1]*stock[2]
		curr_value += stock[1]*stock[3]

	print("\n\nValue at sell time: $", sell_value, flush=True)
	print("Value now         : $", curr_value, flush=True)
	print("Change in value   : $", curr_value - sell_value, flush=True)

if __name__ == "__main__":
	get_prices()