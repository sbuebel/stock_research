import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean
from datetime import datetime
import itertools

import concurrent.futures

# helper functions
from stock_trading import combine_dicts, data_to_csv, buy, sell

class SmartMaximizer:
	# initialize
	def __init__(self, ticker, dire='run1'):
		# dict to hold all the lists of indicators, indexed by name
		self.indicators = {}

		# weights for each indicator
		self.weights = {}

		# stock ticker in questions
		self.ticker = ticker

		# this is arbitrary, weights should scale accordingly
		self.threshold = 1

		# this will be in the form of a list
		self.read_data_in(self.ticker, ticker+"/"+dire) # this populated indicators and prices

		# giant data struct of meta data for an entire run - store everything here
		self.run_data = {}
		self.run_data["weights"] = []
		self.run_data["scores"] = []

	# iterate through the prices, and calculate indicators as necessary
	def read_data_in(self, ticker, dire):
		self.prices = pd.read_csv(dire+"/"+ticker+"_prices.csv")['4. close'].tolist()
		sma8 = pd.read_csv(dire+"/"+ticker+"_sma8.csv")['SMA'].tolist()
		ema8 = pd.read_csv(dire+"/"+ticker+"_ema8.csv")['EMA'].tolist()
		macd = pd.read_csv(dire+"/"+ticker+"_macd.csv")['MACD'].tolist()
		rsi = pd.read_csv(dire+"/"+ticker+"_rsi.csv")['RSI'].tolist()
		vwap = pd.read_csv(dire+"/"+ticker+"_vwap.csv")['VWAP'].tolist()
		stoch_d = pd.read_csv(dire+"/"+ticker+"_stoch.csv")['SlowD'].tolist()
		stoch_k = pd.read_csv(dire+"/"+ticker+"_stoch.csv")['SlowK'].tolist()
		adx = pd.read_csv(dire+"/"+ticker+"_adx.csv")['ADX'].tolist()
		cci = pd.read_csv(dire+"/"+ticker+"_cci.csv")['CCI'].tolist()
		aroon_d = pd.read_csv(dire+"/"+ticker+"_aroon.csv")['Aroon Down'].tolist()
		aroon_u = pd.read_csv(dire+"/"+ticker+"_aroon.csv")['Aroon Up'].tolist()
		bbands_l = pd.read_csv(dire+"/"+ticker+"_bbands.csv")['Real Lower Band'].tolist()
		bbands_m = pd.read_csv(dire+"/"+ticker+"_bbands.csv")['Real Middle Band'].tolist()
		bbands_h = pd.read_csv(dire+"/"+ticker+"_bbands.csv")['Real Upper Band'].tolist()

		# get the dates as well
		price_dates = pd.read_csv(dire+"/"+ticker+"_prices.csv")['date'].tolist()
		sma8_dates = pd.read_csv(dire+"/"+ticker+"_sma8.csv")['date'].tolist()
		ema8_dates = pd.read_csv(dire+"/"+ticker+"_ema8.csv")['date'].tolist()
		macd_dates = pd.read_csv(dire+"/"+ticker+"_macd.csv")['date'].tolist()
		stoch_dates = pd.read_csv(dire+"/"+ticker+"_stoch.csv")['date'].tolist()
		rsi_dates = pd.read_csv(dire+"/"+ticker+"_rsi.csv")['date'].tolist()
		adx_dates = pd.read_csv(dire+"/"+ticker+"_adx.csv")['date'].tolist()
		cci_dates = pd.read_csv(dire+"/"+ticker+"_cci.csv")['date'].tolist()
		aroon_dates = pd.read_csv(dire+"/"+ticker+"_aroon.csv")['date'].tolist()
		bbands_dates = pd.read_csv(dire+"/"+ticker+"_bbands.csv")['date'].tolist()
		vwap_dates = pd.read_csv(dire+"/"+ticker+"_vwap.csv")['date'].tolist()

		# easier than finding end date also...
		lengths = []
		lengths.append(len(price_dates))
		lengths.append(len(sma8_dates))
		lengths.append(len(ema8_dates))
		lengths.append(len(macd_dates))
		lengths.append(len(stoch_dates))
		lengths.append(len(rsi_dates))
		lengths.append(len(adx_dates))
		lengths.append(len(cci_dates))
		lengths.append(len(aroon_dates))
		lengths.append(len(bbands_dates))
		lengths.append(len(vwap_dates))

		shortest = min(lengths)

		# trim all to latest start and go up to earliest finish
		s_dates = []
		s_dates.append(datetime.strptime(price_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(sma8_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(ema8_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(macd_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(stoch_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(rsi_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(adx_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(cci_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(aroon_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(bbands_dates[0], '%Y-%m-%d %H:%M:%S'))
		s_dates.append(datetime.strptime(vwap_dates[0], '%Y-%m-%d %H:%M:%S'))

		# get the latest start and earliest end for our data, and trim accordingly
		latest_start = max(s_dates)

		# this also works
		self.prices = self.prices[price_dates.index(str(latest_start)) : price_dates.index(str(latest_start)) + shortest]
		sma8 = sma8[sma8_dates.index(str(latest_start)) : sma8_dates.index(str(latest_start)) + shortest]
		ema8 = ema8[ema8_dates.index(str(latest_start)) : ema8_dates.index(str(latest_start)) + shortest]
		macd = macd[macd_dates.index(str(latest_start)) : macd_dates.index(str(latest_start)) + shortest]
		stoch_d = stoch_d[stoch_dates.index(str(latest_start)) : stoch_dates.index(str(latest_start)) + shortest]
		stoch_k = stoch_k[stoch_dates.index(str(latest_start)) : stoch_dates.index(str(latest_start)) + shortest]
		rsi = rsi[rsi_dates.index(str(latest_start)) : rsi_dates.index(str(latest_start)) + shortest]
		adx = adx[adx_dates.index(str(latest_start)) : adx_dates.index(str(latest_start)) + shortest]
		cci = cci[cci_dates.index(str(latest_start)) : cci_dates.index(str(latest_start)) + shortest]
		aroon_d = aroon_d[aroon_dates.index(str(latest_start)) : aroon_dates.index(str(latest_start)) + shortest]
		aroon_u = aroon_u[aroon_dates.index(str(latest_start)) : aroon_dates.index(str(latest_start)) + shortest]
		bbands_l = bbands_l[bbands_dates.index(str(latest_start)) : bbands_dates.index(str(latest_start)) + shortest]
		bbands_m = bbands_m[bbands_dates.index(str(latest_start)) : bbands_dates.index(str(latest_start)) + shortest]
		bbands_h = bbands_h[bbands_dates.index(str(latest_start)) : bbands_dates.index(str(latest_start)) + shortest]
		vwap = vwap[vwap_dates.index(str(latest_start)) : vwap_dates.index(str(latest_start)) + shortest]

		# print("prices:", len(self.prices))
		# print("sma8:", len(sma8))
		# print("ema8:", len(ema8))
		# print("vwap:", len(vwap))
		# print("macd:", len(macd))
		# print("stoch_d:", len(stoch_d))
		# print("stoch_k:", len(stoch_k))
		# print("rsi:", len(rsi))
		# print("adx:", len(adx))
		# print("cci:", len(cci))
		# print("aroon_d:", len(aroon_d))
		# print("aroon_u:", len(aroon_u))
		# print("bbands_l:", len(bbands_l))
		# print("bbands_m:", len(bbands_m))
		# print("bbands_h:", len(bbands_h))

		# NORMALIZE everything, otherwise it's useless

		# convert moving averages into a percent difference from price
		self.indicators["sma8"] = sma8
		self.indicators["sma8"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["sma8"],self.prices)]

		self.indicators["ema8"] = ema8
		self.indicators["ema8"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["ema8"],self.prices)]

		self.indicators["vwap"] = vwap
		self.indicators["vwap"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["vwap"],self.prices)]
				
		# already normalized
		self.indicators["macd"] = macd

		# stoch, rsi, adx in range 0, 100, so normalize to range [-1, 1]
		self.indicators["stoch_k"] = stoch_k # 'k' might be too erratic...
		self.indicators["stoch_d"] = stoch_d
		self.indicators["stoch_k"] = [float(x-50)/50 for x in stoch_k]
		self.indicators["stoch_d"] = [float(x-50)/50 for x in stoch_d]
	
		self.indicators["rsi"] = rsi
		self.indicators["rsi"] = [float(x-50)/50 for x in rsi]

		self.indicators["adx"] = adx
		self.indicators["adx"] = [float(x-50)/50 for x in adx]
	
		# cci range [-100, 100] is normal, convert to [-1, 1]
		self.indicators["cci"] = cci
		self.indicators["cci"] = [float(x)/500 for x in cci]

		# range 0, 100 to [-1, 1]
		self.indicators["aroon_d"] = aroon_d
		self.indicators["aroon_u"] = aroon_u
		self.indicators["aroon_d"] = [float(x-50)/50 for x in aroon_d]
		self.indicators["aroon_u"] = [float(x-50)/50 for x in aroon_u]
		
		# these behave like moving averages, so use the percent
		self.indicators["bbands_l"] = bbands_l
		self.indicators["bbands_m"] = bbands_m
		self.indicators["bbands_h"] = bbands_h
		self.indicators["bbands_l"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["bbands_l"],self.prices)]
		self.indicators["bbands_m"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["bbands_m"],self.prices)]
		self.indicators["bbands_h"] = [100*(float(x)/y-1) for x,y in zip(self.indicators["bbands_h"],self.prices)]

		# gradient will be used to determine best gradient
		self.weights["aggression"] = 0
		self.weights["sma8"] = 0
		self.weights["ema8"] = 0
		self.weights["vwap"] = 0
		self.weights["macd"] = 0
		self.weights["stoch_k"] = 0
		self.weights["stoch_d"] = 0
		self.weights["rsi"] = 0
		self.weights["adx"] = 0
		self.weights["cci"] = 0
		self.weights["aroon_d"] = 0
		self.weights["aroon_u"] = 0
		self.weights["bbands_l"] = 0
		self.weights["bbands_m"] = 0
		self.weights["bbands_h"] = 0

	# helper function to make sure indicators are working - plot them
	def plot_indicators(self):
		x = np.arange(0, len(self.prices))
		plt.plot(x, self.indicators["ema8"])
		plt.plot(x, self.indicators["sma8"])
		plt.plot(x, self.indicators["macd"])
		plt.plot(x, self.indicators["rsi"])
		plt.plot(x, self.indicators["aroon_u"])
		plt.plot(x, self.indicators["aroon_d"])
		plt.plot(x, self.indicators["cci"])
		plt.plot(x, self.indicators["adx"])
		plt.plot(x, self.indicators["vwap"])
		plt.plot(x, self.indicators["stoch_d"])
		plt.plot(x, self.indicators["stoch_k"])
		plt.plot(x, self.indicators["bbands_l"])
		plt.plot(x, self.indicators["bbands_m"])
		plt.plot(x, self.indicators["bbands_h"])
		plt.legend()
		plt.show()

	# hopefully better scoring function based on trading
	def trade_score(self, verbose=False):
		# will return gains in percent, so amount really isn't relevant
		cash = self.prices[0]/2
		shares = 0.5

		# for visualization if necessary
		cash_values = []
		shares_values = []
		balance_values = []
		predictors = []
		buys = np.array([[0, self.prices[0]], [0, self.prices[0]]])
		sells = np.array([[0, self.prices[0]], [0, self.prices[0]]])

		for i in range(0, len(self.prices)):
			# for each minute, get the predictor value based on weights
			# then see what the max is for next time period
			predictor = 0.0
			for key in self.indicators.keys():
				predictor += self.indicators[key][i]*self.weights[key]

			# visualizing - we want to plot this stuff at the end
			if verbose:
				predictors.append(predictor)
				cash_values.append(cash)
				shares_values.append(shares*self.prices[i])
				balance_values.append(cash_values[-1] + shares_values[-1])

			# trade based on aggression and strenght of indicator
			trade_amount = abs(self.weights["aggression"] * predictor / self.threshold)

			# predict price INCREASE, buy, then set limit order at some higher value
			if predictor >= self.threshold:
				# only add real buys for plotting
				if cash > 0:
					buys = np.append(buys, [[i, self.prices[i]]], axis=0)

				################ BUY #################
				cash, shares = buy(cash, shares, trade_amount, self.prices[i])
				# if verbose:
				# 	print("BUY:", trade_amount, cash, shares)

			# if we predict price will go down, SELL - more aggressive to sell
			elif predictor < -1*self.threshold:
				# only add real sells for plotting
				if shares > 0:
					sells = np.append(sells, [[i, self.prices[i]]], axis=0)

				################ SELL ################
				cash, shares = sell(cash, shares, trade_amount, self.prices[i])
				# if verbose:
				# 	print("SELL:", trade_amount, cash, shares)

		# plot if we want it
		if verbose:
			print("score: ", ((self.prices[-1]*shares+cash - self.prices[0]) / self.prices[0]) - ((self.prices[-1] - self.prices[0]) / self.prices[0]))

			self.run_data["cash"] = cash_values
			self.run_data["shares"] = shares_values
			self.run_data["balance"] = balance_values

			x = np.arange(0, len(self.prices))

			fig, ax1 = plt.subplots()

			ax1.plot(x, self.prices, label="$$$")

			ax1.scatter(buys[:,0], buys[:,1], label="BUYS")
			ax1.scatter(sells[:,0], sells[:,1], label="SELLS")

			ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
			# ax1.plot(x, predictors, label="indicators")
			# ax2.plot(x, cash_values, label="cash")
			# ax2.plot(x, shares_values, label="shares")
			# ax1.plot(x, balance_values, label="balance")
			# ax2.plot(x, self.prices, label="$TSLA")
			# ax2.plot(x, self.indicators["sma8"], label="SMA")
			# ax2.plot(x, self.indicators["ema8"], label="EMA")
			# ax2.plot(x, self.indicators["macd"], label="MACD")
			ax1.legend()
			# ax2.legend()
			plt.show()

		# account for the stock price increase
		return ((self.prices[-1]*shares+cash - self.prices[0]) / self.prices[0]) - ((self.prices[-1] - self.prices[0]) / self.prices[0])

	# prediction based scoring function, more useful for optimization potentially
	def prediction_score(self, verbose=False):
		patience = 30 # assume 10 minutes before prediction comes true
		total_score = 0

		ups = np.array([[0, self.prices[0]], [0, self.prices[0]]])
		downs = np.array([[0, self.prices[0]], [0, self.prices[0]]])

		for i in range(0, len(self.prices)):
			cur_price = self.prices[i]

			# for each minute, get the predictor value based on weights
			# then see what the max is for next time period
			predictor = 0.0
			for key in self.indicators.keys():
				predictor += self.indicators[key][i]*self.weights[key]

			# check next few minutes depending on patience to see if we were right
			next_high = cur_price
			next_low = cur_price

			if i + patience < len(self.prices):
				for j in range(i, i + patience):
					if self.prices[j] < next_low:
						next_low = self.prices[j]
					if self.prices[j] > next_high:
						next_high = self.prices[j]
			else:
				# garbage time, don't include
				predictor = 0

			# print(cur_price, next_low, next_high)

			# now we have the predictors, assume positive means price UP
			if predictor > self.threshold:
				# account for the drop too, it's relevant
				total_score += (abs(predictor)*(next_high - cur_price) -2*abs(predictor)*(cur_price - next_low))
				ups = np.append(ups, [[i, self.prices[i]]], axis=0)

			# if we predict decrease, see if we were right
			if predictor < -1*self.threshold:
				# account for increase too, means we were wrong
				total_score += (abs(predictor)*(cur_price - next_low) - 2*abs(predictor)*(next_high - cur_price))
				downs = np.append(downs, [[i, self.prices[i]]], axis=0)

		# plot if we want it
		if verbose:
			x = np.arange(0, len(self.prices))

			fig, ax1 = plt.subplots()

			ax1.plot(x, self.prices, label="$$$")
			ax1.scatter(ups[:,0], ups[:,1], label="p_UP")
			ax1.scatter(downs[:,0], downs[:,1], label="p_DOWN")

			# ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
			# ax2.plot(x, self.indicators["sma8"], label="SMA")
			# ax2.plot(x, self.indicators["ema8"], label="EMA")
			# ax2.plot(x, self.indicators["macd"], label="MACD")
			ax1.legend()
			# ax2.legend()
			plt.show()

		return total_score

	# function to try different gradients until we get max new results
	def gradient_max(self, last_score, precision=0.1):

		# need this to reset each time
		orig_weights = self.weights

		# threads will be stored here
		futures = []

		# results will be stored here
		results = []

		# predictor list of lists
		# predictor_indicators = []

		# value to count up to equal to number of params
		# basically a bitmask where first binary digit is sma8, and so on
		indexer = 0b00000

		# List of all indicators
		# self.weights["aggression"] = 0
		# self.weights["sma8"] = 0
		# self.weights["ema8"] = 0
		# self.weights["vwap"] = 0
		# self.weights["macd"] = 0
		# self.weights["stoch_k"] = 0
		# self.weights["stoch_d"] = 0
		# self.weights["rsi"] = 0
		# self.weights["adx"] = 0
		# self.weights["cci"] = 0
		# self.weights["aroon_d"] = 0
		# self.weights["aroon_u"] = 0
		# self.weights["bbands_l"] = 0
		# self.weights["bbands_m"] = 0
		# self.weights["bbands_h"] = 0

		# start some concurrent threads
		with concurrent.futures.ThreadPoolExecutor() as executor:
			while indexer <= 0b11111:
				gradient = {}

				# see if our bit is 1, then mult by 2, and sub 1, range(-1, 1)
				gradient["ema8"] = 		(((indexer & 	0b10000) >> 3) 	- 1) * precision
				gradient["cci"] = 		(((indexer &	0b01000) >> 2)	- 1) * precision
				gradient["stoch_d"] = 	(((indexer & 	0b00100) >> 1) 	- 1) * precision
				gradient["bbands_m"] = 	(((indexer & 	0b00010))	 	- 1) * precision
				gradient["macd"] = 		(((indexer & 	0b00001) << 1) 	- 1) * precision
				results.append([0, gradient])

				indexer += 1

				# add the gradient, and see results
				self.weights = combine_dicts(gradient, orig_weights)

				# basically start a new thread for each one
				futures.append(executor.submit(self.trade_score))

		# get all the future results
		for i, future in enumerate(futures):
			cur_score = future.result()

			# fill in this result
			results[i][0] = cur_score - last_score

		# reset weights
		self.weights = orig_weights

		best_gain = results[0][0]
		best_ind = 0
		# now, find best result and return that gradient
		for i in range(0, len(results)):
			if results[i][0] > best_gain:
				best_ind = i
				best_gain = results[i][0]

		# return gradient that gave best results
		return results[best_ind][1]

	# function to iterate through possible weights to find optimal values for highscore
	def gradient_based_maximizer(self):
		# need a better way....

		# just init to something - doesn't really matter
		last_score = 0

		cutoff = 0.00000001
		stale = 0

		# this will control our gradient better
		step_size = 0.1

		while stale < 10:
			# see what best direction is
			gradient = self.gradient_max(last_score, step_size)

			# update the weights, and get a new score
			self.weights = combine_dicts(self.weights, gradient)
			# print(self.weights)
			score = self.trade_score() # get new score

			# this is a 2D array itself, add current weights
			# self.run_data["weights"].append(self.weights)

			# see if we need to add here
			if abs(score - last_score) < cutoff:
				stale += 1
			else:
				stale = 0

			step_size = 5*abs(last_score-score) # this will make it converge, but maybe too fast?
			last_score = score

			print("score: ", score, "\r", end='')

		print("\nfinal weights:", self.weights)

		# at this point, we have ideal weights, run it again and show the output
		max_score = self.trade_score(verbose=True)

	# function to iterate through possible weights to find optimal values for highscore
	def brute_force_maximizer(self):
		# range of weights to look between
		weight_range = [-5, 5] # we get 10 steps in between
		granu = 2
		multi = 1

		if granu < 1:
			multi = 1 / granu
			granu = 1
			weight_range[0] = int(multi*weight_range[0])
			weight_range[1] = int(multi*weight_range[1])

		num_steps = pow(((weight_range[1] - weight_range[0]) / granu), 5)
		count = 0

		# eliminate this metric. we want to buy as much as possible
		self.weights["aggression"] = 100#0.01

		# keep track of high score and best gradients
		highscore = 0
		highgrad = {}

		# with concurrent.futures.ThreadPoolExecutor() as executor:
		for a in range(weight_range[0], weight_range[1], granu):
			self.weights["ema8"] = a/multi
			for b in range(weight_range[0], weight_range[1], granu):
				self.weights["cci"] = b/multi
				for c in range(weight_range[0], weight_range[1], granu):
					self.weights["stoch_d"] = c/multi
					for d in range(weight_range[0], weight_range[1], granu):
						self.weights["bbands_m"] = d/multi
						for e in range(weight_range[0], weight_range[1], granu):
							self.weights["macd"] = e/multi

							# score = self.trade_score()
							score = self.prediction_score()

							if count == 0 or score > highscore:
								highscore = score
								# hack to avoid copy constructor pointer redirect...
								highgrad = {}
								highgrad = combine_dicts(highgrad, self.weights)

							print("progress:", 100*count/num_steps, "\r", end='')

							count += 1

							# print("weights:", round(self.weights["ema8"], 2), 
							# 	round(self.weights["cci"], 2), 
							# 	round(self.weights["stoch_d"], 2), 
							# 	round(self.weights["bbands_m"], 2),
							# 	round(self.weights["macd"], 2))

		print("max score:", highscore)
		print("best weights:", highgrad)

		# set weights accordingly
		self.weights = highgrad

		print(self.prediction_score(True))

	# function to use all the indicators to see if they're helping, write data to text file
	def research_indicators(self):
		# range of weights to look between
		weight_range = [-10, 10]
		granu = 5
		multi = 1

		if granu < 1:
			multi = 1/granu
			granu = 1
			weight_range[0] = int(multi*weight_range[0])
			weight_range[1] = int(multi*weight_range[1])

		# this tells us progress
		num_steps = pow(((weight_range[1] - weight_range[0]) / granu), 5)
		count = 0

		range___ = np.arange(weight_range[0], weight_range[1], granu)

		self.weights["aggression"] = 100

		# track so we maximize
		highscore = 0
		highgrad = {}

		# 6 variables
		for combo in itertools.product(range___, range___, range___, range___, range___):
			self.weights["ema8"] = combo[0]
			self.weights["macd"] = combo[0]
			self.weights["bbands_h"] = combo[0]
			self.weights["stoch_d"] = combo[0]
			self.weights["cci"] = combo[0]

			score = self.trade_score()

			if count == 0 or score > highscore:
				highscore = score
				highgrad = self.weights

			count += 1

			print("progress:", 100*count/num_steps, "\r", end='')

		print(self.weights)
		print(self.trade_score(verbose = True))


if __name__ == "__main__":
	# helper function to read the data from the API
	# data_to_csv('MSFT', 'MSFT/run1/')

	# # 10 minutes to see how high the scores will go
	sam = SmartMaximizer('AAPL', 'run1')

	# we use this to get an initial idea, then gradient descent for optimization
	# sam.brute_force_maximizer()
	# sam.gradient_based_maximizer()

	# really good weights?
	sam.weights["aggression"] = 100
	sam.weights["ema8"] = -2
	sam.weights["macd"] = -5
	sam.weights["cci"] = 3.5
	sam.weights["bbands_m"] = -0.5

	print(sam.weights)
	print(sam.trade_score(True))

	# just to visualize our indicators - hard to see...
	# sam.plot_indicators()