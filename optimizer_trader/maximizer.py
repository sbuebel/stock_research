import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import argparse
from typing import List, Dict

import concurrent.futures

# helper functions
from trade_util import combine_dicts, data_to_csv, buy, sell


class SmartMaximizer:
	def __init__(self, ticker: str, dire: str = 'run1', prices: List[float] = None, ind: Dict[str, List[float]] = None):
		"""
		Start a maximizer based on the ticker provided - taking in prices from a file or parameter.

		params:
			ticker (str): stock ticker to trade (ex: NVDA)
			dire (str): directory from which to pull price and indicator data (default: run1)
			prices (list[float]): time ordered list of minute by minute price data for a given stock
			ind (dict{list[float]}): dict of lists of minute by minute indicator data for 1 or more indicators

		"""
		# dict to hold all the lists of indicators, indexed by name
		self.indicators = {}

		# weights for each indicator
		self.weights = {}

		# stock ticker in questions
		self.ticker = ticker

		# this is arbitrary, weights should scale accordingly
		self.threshold = 1

		self.weights["aggression"] = 0.5

		# if prices weren't specified, then read them - try a file first, then pull from API if none exists
		if not prices:
			try:
				self.read_data_from_csv(ticker, ticker + "/" + dire)  # this populates indicators and prices
			except IOError as e:
				print(f"No data file exists for {ticker} - pulling from API")
				data_to_csv(ticker, ticker + "/run1/")
				self.read_data_from_csv(ticker, ticker + "/run1/")  # this populates indicators and prices


		else:
			self.prices = prices
			if ind:
				self.indicators = ind
			else:
				print(f"ERROR: indicators must be specified if prices were passed as a parameter!")

	# iterate through the prices, and calculate indicators as necessary
	def read_data_from_csv(self, ticker, dire):
		self.prices = pd.read_csv(dire + "/" + ticker + "_prices.csv")['4. close'].tolist()
		macd = pd.read_csv(dire + "/" + ticker + "_macd.csv")['MACD'].tolist()

		# get the dates as well
		price_dates = pd.read_csv(dire + "/" + ticker + "_prices.csv")['date'].tolist()
		macd_dates = pd.read_csv(dire + "/" + ticker + "_macd.csv")['date'].tolist()

		# easier than finding end date also...
		lengths = {len(price_dates), len(macd_dates)}

		shortest = min(lengths)

		# trim all to latest start and go up to earliest finish
		s_dates = {
			datetime.strptime(price_dates[0], '%Y-%m-%d %H:%M:%S'),
			datetime.strptime(macd_dates[0], '%Y-%m-%d %H:%M:%S')
		}

		# get the latest start and earliest end for our data, and trim accordingly
		latest_start = max(s_dates)

		# this also works
		self.prices = self.prices[price_dates.index(str(latest_start)): price_dates.index(str(latest_start)) + shortest]
		macd = macd[macd_dates.index(str(latest_start)): macd_dates.index(str(latest_start)) + shortest]

		# already normalized
		self.indicators["macd"] = macd
		self.weights["macd"] = 0

	# helper function to make sure indicators are working - plot them
	def plot_indicators(self):
		x = np.arange(0, len(self.prices))
		plt.plot(x, self.indicators["macd"])
		plt.legend()
		plt.show()

	# hopefully better scoring function based on trading
	def flexy_trade_score(self, verbose=False, epoch_period=60, epoch_duration=1440):
		"""
		Iterates through the time series data and buys and sells based on a given set of indicators. BUTTTT, the weights
		are determined by optimizing for previous data every hour - so try to gauge the 'current' trading environment.

		Default to brute force optimization each epoch.

		params:
			verbose (bool): Whether or not to print out the trades made and plot results
			epoch_period (int): How often (in minutes) to re-evaluate the weights
			epoch_duration (int): How long (in minutes) to look at data to re-optimize weights

		returns:
			score (%): percent gain or loss - normalized


		"""

		# will return gains in percent, so amount really isn't relevant
		cash = self.prices[0]
		shares = 0

		# for visualization if necessary
		cash_values = []
		shares_values = []
		shares_list = []
		balance_values = []
		predictors = []
		buys = np.array([[0, self.prices[0]], [0, self.prices[0]]])
		sells = np.array([[0, self.prices[0]], [0, self.prices[0]]])

		for i in range(0, len(self.prices)):

			# new epoch, reoptimize our weights
			if i >= epoch_duration and i % epoch_period == 0:
				# need a new class with data up to this time point
				epoch_prices = self.prices[i - epoch_duration:i]
				epoch_indicators = {}
				for key, ind in self.indicators.items():
					epoch_indicators[key] = ind[i - epoch_duration:i]

				epoch_maximizer = SmartMaximizer(self.ticker, "tmp", epoch_prices, epoch_indicators)

				# update based on weights that the last hour's data produced
				old_weights = self.weights
				self.weights, max_score = epoch_maximizer.brute_force_maximizer()

				for key, weight in self.weights.items():
					if old_weights[key] != weight:
						print(f"Updated weights at epoch {i / epoch_period}: {self.weights} @ {max_score}")

			# for each minute, get the predictor value based on weights
			# then see what the max is for next time period
			predictor = 0.0
			for key in self.indicators.keys():
				predictor += self.indicators[key][i]*self.weights[key]

			# visualizing - we want to plot this stuff at the end
			if verbose:
				predictors.append(predictor)
				cash_values.append(cash)
				shares_list.append(shares)
				shares_values.append(shares*self.prices[i])
				balance_values.append(cash_values[-1] + shares_values[-1])

			# trade based on aggression and strenght of indicator
			trade_amount = abs(self.weights["aggression"] * predictor / self.threshold)

			# predict price INCREASE, buy, then set limit order at some higher value
			if predictor >= self.threshold:
				# only add real buys for plotting
				if cash > 0:
					buys = np.append(buys, [[i, self.prices[i]]], axis=0)

					# BUY #################
					cash, shares = buy(cash, shares, trade_amount, self.prices[i])

					if verbose:
						print(f"BUY: ${self.prices[i]}   -    C${cash}, SH{shares}")

			# if we predict price will go down, SELL - more aggressive to sell
			elif predictor < -1*self.threshold:
				# only add real sells for plotting
				if shares > 0:
					sells = np.append(sells, [[i, self.prices[i]]], axis=0)

					# SELL ################
					cash, shares = sell(cash, shares, trade_amount, self.prices[i])

					if verbose:
						print(f"SELL: ${self.prices[i]}   -    C${cash}, SH{shares}")

		final_score = ((self.prices[-1]*shares+cash - self.prices[0]) / self.prices[0]) - ((self.prices[-1] - self.prices[0]) / self.prices[0])

		# plot if we want it
		if verbose:
			print(f"Gain/Loss: {100 * final_score}%")

			x = np.arange(0, len(self.prices))

			fig, ax1 = plt.subplots()

			ax1.scatter(buys[:, 0], buys[:, 1], label="BUYS")
			ax1.scatter(sells[:, 0], sells[:, 1], label="SELLS")
			ax1.plot(x, balance_values, label="balance", color="Red")
			ax1.plot(x, self.prices, label=f"${self.ticker}", color="Black")
			ax1.legend(loc='upper right')

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
			# ax2.plot(x, predictors, label="indicators")
			ax2.plot(x, shares_list, label="shares", color="Green")

			# ax2.plot(x, self.indicators["macd"], label="MACD")
			ax2.legend(loc='upper left')
			plt.show()

		# account for the stock price increase
		return final_score

	# hopefully better scoring function based on trading
	def trade_score(self, verbose=False):
		"""
		Iterates through the time series data and buys and sells based on a given set of indicators and weights.

		params:
			verbose (bool): Whether or not to print out the trades made and plot results

		returns:
			score (%): percent gain or loss - normalized

		"""

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

					# BUY #################
					cash, shares = buy(cash, shares, trade_amount, self.prices[i])

					if verbose:
						print(f"BUY: ${self.prices[i]}   -    C${cash}, SH{shares}")

			# if we predict price will go down, SELL - more aggressive to sell
			elif predictor < -1*self.threshold:
				# only add real sells for plotting
				if shares > 0:
					sells = np.append(sells, [[i, self.prices[i]]], axis=0)

					# SELL ################
					cash, shares = sell(cash, shares, trade_amount, self.prices[i])

					if verbose:
						print(f"SELL: ${self.prices[i]}   -    C${cash}, SH{shares}")

		final_score = ((self.prices[-1]*shares+cash - self.prices[0]) / self.prices[0]) - ((self.prices[-1] - self.prices[0]) / self.prices[0])

		# plot if we want it
		if verbose:
			print(f"Gain/Loss: {100 * final_score}%")

			x = np.arange(0, len(self.prices))

			fig, ax1 = plt.subplots()

			ax1.scatter(buys[:, 0], buys[:, 1], label="BUYS")
			ax1.scatter(sells[:, 0], sells[:, 1], label="SELLS")

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
			# ax2.plot(x, predictors, label="indicators")
			# ax2.plot(x, cash_values, label="cash")
			# ax2.plot(x, shares_values, label="shares")
			# ax1.plot(x, balance_values, label="balance")
			ax1.plot(x, self.prices, label=f"${self.ticker}")
			ax2.plot(x, self.indicators["macd"], label="MACD")
			ax1.legend(loc='upper right')
			ax2.legend(loc='upper left')
			plt.show()

		# account for the stock price increase
		return final_score

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

		# start some concurrent threads
		with concurrent.futures.ThreadPoolExecutor() as executor:
			while indexer <= 0b11111:
				gradient = {}

				# see if our bit is 1, then mult by 2, and sub 1, range(-1, 1)
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
			print(self.weights)
			score = self.trade_score()  # get new score

			# see if we need to add here
			if abs(score - last_score) < cutoff:
				stale += 1
			else:
				stale = 0

			step_size = 5*abs(last_score-score)  # this will make it converge, but maybe too fast?
			last_score = score

			print("score: ", score, "\r", end='')

		print("\nfinal weights:", self.weights)

		# at this point, we have ideal weights, run it again and show the output
		max_score = self.trade_score(verbose=True)

	# function to iterate through possible weights to find optimal values for highscore
	def brute_force_maximizer(self, verbose=False):
		# range of weights to look between
		weight_range = [-2, 1]
		granu = 0.03
		multi = 1

		# normalize to integers
		if granu < 1:
			multi = 1 / granu
			granu = 1
			weight_range[0] = int(multi*weight_range[0])
			weight_range[1] = int(multi*weight_range[1])

		num_steps = (weight_range[1] - weight_range[0]) / granu
		count = 0

		# keep track of high score and best gradients
		highscore = 0
		highgrad = {}

		for e in range(weight_range[0], weight_range[1], granu):
			self.weights["macd"] = e / multi
			if verbose:
				print(f"macd: {self.weights['macd']}")

			score = self.trade_score()

			if count == 0 or score > highscore:
				highscore = score
				# hack to avoid copy constructor pointer redirect...
				highgrad = {}
				highgrad = combine_dicts(highgrad, self.weights)

			if verbose:
				print("progress:", 100*count/num_steps, "\r", end='')

			count += 1

		# set weights accordingly
		self.weights = highgrad

		if verbose:
			print(f"Best Return: {100 * highscore}%")
			print("best weights:", highgrad)

		return highgrad, 100 * highscore

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

		self.weights["aggression"] = 0.5

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


def trade_two_weeks(ticker):
	# helper function to read new data from the API -- don't need new data every run
	data_to_csv(ticker, ticker+'/run1/')

	# 10 minutes to see how high the scores will go
	sam = SmartMaximizer(ticker, 'run1')

	# starting weights?
	sam.weights["aggression"] = 0.5
	sam.weights["macd"] = -0.9

	# run our trader that optimizes weights every epoch
	sam.flexy_trade_score(True, epoch_period=60, epoch_duration=1440)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog='OptimizerTrader',
		description='Trade a stock based on some indicators to make a "profit"?',
	)
	parser.add_argument('ticker')
	parser.add_argument('run_number')
	parser.add_argument('new_data')  # bool - whether to take new data
	args = parser.parse_args()

	trade_two_weeks("TSLA")
