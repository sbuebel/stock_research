import robin_stocks as r
import scraper as s
import pandas as pd

import datetime, time, os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# let's not put the whole password in plain text...
# learn from our past...
username='stbuebel@gmail.com'
password='Owii91'
p_file = open("C:/Users/stbue/Desktop/robi_code.txt", 'r')
password += p_file.read()
p_file.close()

# login
login = r.login(username, password)
ticker = 'GNUS'

# global variable to hold all the data
data = dict()

# get data from the scraper (alphavantage)
def get_init_data():
	global data

	# this gets the 5 minute price point for a stock
	today = r.stocks.get_historicals(ticker, span='day', bounds='regular')

	# flip it so last index is latest
	past = s.data_to_struct(ticker)
	data = dict()
	for key in reversed(past.keys()): 
		data[key] = past[key]

	# append today's data to past data
	for time_slice in today:
		# this is annoying to parse
		clock_time = time_slice['begins_at']
		day = clock_time.split('T')[0]
		clock_time = clock_time.split('T')[1][:-1]
		hour = 10*int(clock_time[0])+int(clock_time[1]) - 4
		if hour < 10:
			clock_time = '0'+str(hour)+clock_time[2:]
		else:
			clock_time = str(hour)+clock_time[2:]

		TIME_INDEX = day + " " + clock_time

		addon = dict()
		addon['1. open'] = time_slice['open_price']
		addon['2. high'] = time_slice['high_price']
		addon['3. low'] = time_slice['low_price']
		addon['4. close'] = time_slice['close_price']
		addon['5. volume'] = time_slice['volume']
		addon['6. pid'] = 0

		# we don't really get this datapoint
		if not TIME_INDEX.endswith('09:30:00'):
			data[TIME_INDEX] = addon
		
	# convert everyone to floats
	# use this for 'rolling' average
	last_price = 0
	for time_slice in data:

		time_slice = data[time_slice]
		time_slice['1. open'] = float(time_slice['1. open'])
		time_slice['2. high'] = float(time_slice['2. high'])
		time_slice['3. low'] = float(time_slice['3. low'])
		time_slice['4. close'] = float(time_slice['4. close'])
		time_slice['5. volume'] = int(time_slice['5. volume'])
		time_slice['6. pid'] = 0

		if last_price == 0:
			last_price = (time_slice['1. open']+time_slice['4. close'])/2
		# use a comp filter for this
		time_slice['0. price'] = 0.5*(time_slice['1. open']+time_slice['4. close'])/2 + 0.5*last_price
		last_price = (time_slice['1. open']+time_slice['4. close'])/2

# plot price over last 'x' days
def plot_data(num_days):
	global data

	dates = []
	opens = []
	closes = []
	highs = []
	lows = []
	prices = []
	spreads = []

	# for pretty plotting
	max_val = 0
	min_val = 0

	# 76: that many 5 minute intervals per day
	num_points = int(num_days*76)
	tot_points = len(data.keys())
	if num_points > tot_points:
		num_points = tot_points

	# for our plot, we want 100 points, so do that
	divider = int(num_points / 100)
	if divider == 0:
		divider = 1
	# print(num_points, divider)

	# format: 2020-06-15 12:40:00 {'1. open': '4.6000', '2. high': '4.6050', 
	#		'3. low': '4.5500', '4. close': '4.5701', '5. volume': '631196'}
	for i, time_slice in enumerate(data):
		# debugging
		# print(time_slice, data[time_slice])

		# only plot last i points
		if i < tot_points - num_points:
			continue

		if i % divider == 0:
			# ignore year and seconds
			cur_date = time_slice.split(':')[0][5:]+":"+time_slice.split(':')[1]

			dates.append(datetime.datetime.strptime(cur_date, '%m-%d %H:%M'))			
			opens.append(data[time_slice]['1. open'])
			highs.append(data[time_slice]['2. high'])
			lows.append(data[time_slice]['3. low'])
			closes.append(data[time_slice]['4. close'])
			prices.append(data[time_slice]['0. price'])

			if highs[-1] > max_val or max_val == 0:
				max_val = highs[-1]*1.025
			if lows[-1] < min_val or min_val == 0:
				min_val = lows[-1]*0.975

	dates = np.array(dates)
	opens = np.array(opens)
	closes = np.array(closes)
	lows = np.array(lows)
	highs = np.array(highs)
	prices = np.array(prices)
	spreads = np.array((prices - ((opens+closes)/2)) / ((opens+closes)/2))

	dates = matplotlib.dates.date2num(dates)
	hours = matplotlib.dates.HourLocator()

	fig, ax1 = plt.subplots()

	ax1.plot_date(dates, (opens+closes)/2, 'g-')
	ax1.plot_date(dates, prices, 'r-')
	
	date_form = matplotlib.dates.DateFormatter("%m-%d %H")
	fig.gca().xaxis.set_major_formatter(date_form)
	fig.gca().xaxis.set_minor_locator(hours)

	ax1.set_ylabel("Price ($)")
	ax1.set_xlabel("Date/Time")
	plt.title(ticker+": Time Series Data "+str(num_days)+" day(s)")
	ax1.set_ylim(min_val, max_val)

	ax2 = ax1.twinx()
	ax2.set_ylabel("Spread")
	ax2.plot_date(dates, spreads, 'b--')
	plt.show()

# just to update 5 min interval data
def data_updater():
	global data

	# track these for each 5 minute price interval
	price_open = 0
	price_close = 0
	price_high = 0
	price_low = 0
	last_minute = 0 # use this to make sure we just do one entry per 5 min
	rolling_price = 0
	last_price = 0

	# PID stuff
	_P_ = 5
	_I_ = 0.2
	_D_ = 20
	i_term = 0
	pid_term = 0

	# manage potfolio a little bit
	buy_open = False
	sell_open = False
	shares = 0 # assume we start with no shares

	while True:
		cur_time = datetime.datetime.now()
		cur_price = float(r.get_latest_price(ticker)[0])
		if rolling_price == 0:
			rolling_price = cur_price

		# calculate the 'rolling average'
		rolling_price = round(rolling_price*0.5 + cur_price*0.5, 3)
		spread = round((rolling_price - cur_price)/cur_price, 3)

		# i_term should roll
		i_term += spread*_I_
		pid_term = _P_*spread + i_term + _D_*(cur_price-last_price)

		# update for next time
		last_price = cur_price

		print(cur_price, rolling_price, pid_term)

		# update extremes for time interval
		if cur_price > price_high or price_high == 0:
			price_high = cur_price
		if cur_price < price_low or price_low == 0:
			price_low = cur_price
		if price_open == 0:
			price_open = cur_price

		# every 5 minutes, update another entry to our data struct
		if cur_time.minute % 5 == 0 and cur_time.minute != last_minute:

			# this will ensure we don't do more than one entry
			last_minute = cur_time.minute

			# convert to string to make my life easier
			cur_time = str(cur_time)

			# get the rigth index for the data dict addon
			day = cur_time.split(" ")[0]
			clock_time = cur_time.split(" ")[1].split(".")[0][:-2]+'00'
			TIME_INDEX = day + " " + clock_time

			# add an entry to the full data group so we can keep
			# making predictions
			addon = dict()
			addon['0. price'] = rolling_price
			addon['1. open'] = price_open
			addon['2. high'] = price_high
			addon['3. low'] = price_low
			addon['4. close'] = cur_price
			addon['5. volume'] = 0 # TODO
			addon['6. pid'] = pid_term

			data[TIME_INDEX] = addon

			# now, reset values we track
			price_open = price_close
			price_close = 0
			price_high = price_open
			price_low = price_open

		time.sleep(1)

if __name__ == "__main__":
	# populate global data variable
	get_init_data()

	# format: data, number of days
	plot_data(1)