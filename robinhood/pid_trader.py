import datetime, time, os, ast, sys, signal, matplotlib
import matplotlib.pyplot as plt
import robin_stocks as r
import numpy as np
import zmq


class TraderPID:

	###########################################################################
	# __init__: 		initialize the trading class
	#					this includes logging into the API
	# params:			tick:			stock ticker to trade
	###########################################################################
	def __init__(self, tick):
		self.start_time = datetime.datetime.now()
		self.tick = tick
		self.user = 'stbuebel@gmail.com'
		self.pasw = 'Owii91'

		# initialize the trader by logging into robinhood API
		self.login()

		# initialize an empty dict by default
		self.real_time_data = dict()
		self.sim_data = dict()

		# we'll want to track this, init to 0
		self.cash = 0
		self.shares = 0
		self.loop_count = 1

		# trading and PID params
		self._P_ = 10
		self._I_ = 10
		self._D_ = 35
		self.PID_LIM = 0.75

		self.SHARES_PER_TRADE = 1
		self.SHARE_LIMIT = 5

		# method for passing data to the plotter
		self.context = zmq.Context()
		# we are the rerver, plotter will request data
		self.server = self.context.socket(zmq.REP)

		# let us run two
		try:
			self.server.bind("tcp://*:5555")
		except:
			self.server.bind("tcp://*:5556")

		self.server.RCVTIMEO = 1000
		self.server.linger = 250

		# register the signal to the handler we built
		signal.signal(signal.SIGINT, self.shutdown_)

	###########################################################################
	# login: 			login through robin_stocks class
	#					robust to password file placement
	#					secure, local machine requirement
	# params:			[none]
	###########################################################################
	def login(self):
		pasw_file = None
		try:
			pasw_file = open("C:/Users/stbue/Desktop/robi_code.txt", 'r')
		except:
			try:
				pasw_file = open("C:/Users/Spencer/Desktop/robi_code.txt", 'r')
			except:
				sys.exit("No password provided, login unsuccessful. IYKYK")
		self.pasw += pasw_file.read()
		pasw_file.close()

		self.login_handle = r.login(self.user, self.pasw)

	###########################################################################
	# get_price_safe:	safely fetch the price, we so don't break
	# params:			[none]
	###########################################################################
	def get_price_safe(self):
		try:
			return float(r.get_latest_price(self.tick)[0])
		except:
			return 0

	###########################################################################
	# get_rel_voluume:	helper to safely fetch the relative volume
	# params:			[none]
	###########################################################################
	def get_rel_volume(self):
		try:
			fund = r.get_fundamentals(self.tick)

			return float(fund['volume'])/float(fund['average_volume_2_weeks'])

		except:
			# be default, assume normal rel volume
			return 1

	###########################################################################
	# realtime_trader:	make trades in real time based on PID
	#					indicator
	# params:			hot:			specify whether to really trade
	#					cap_limit:		limit amount of cash to leverage
	###########################################################################
	def realtime_trader(self, hot=False, cap_limit=100):
		# make sure we get a reset, empty dict
		self.real_time_data = dict()
	
		# make sure we reset
		self.loop_count = 1
		
		# don't do this accidentally
		if hot:
			confirm = input("Authorize real-time trading?")

		# 'moving average' - init to real values so we don't get jumpy
		rolling_price = self.get_price_safe()
		last_price = rolling_price

		time.sleep(1)

		# PID stuff
		i_term = 0
		pid_term = 0

		# start trading here
		while True:
			# make sure we're getting a good price
			cur_price = self.get_price_safe()
			if cur_price == 0:
				pass

			# calculate the 'rolling average'
			rolling_price = rolling_price*0.9 + cur_price*0.1
			spread_percent = (rolling_price - cur_price)/cur_price
			change_percent = (last_price - cur_price)/cur_price

			# i_term should rollover from previous
			i_term += spread_percent*self._I_

			# CALCULATE PID
			pid_term = self._P_*spread_percent + i_term + self._D_*change_percent
			
			# adjust PID for relative volatility as well as shares we hold
			pid_term -= float(self.shares / self.SHARE_LIMIT)
			pid_term *= self.get_rel_volume()

			# update for next time - for 'd' term
			last_price = cur_price

			# by default, assume we don't have an order this second
			new_order = None # use this in the data struct

			# now, trade based on info
			if pid_term < -1*self.PID_LIM: 				# SELL

				# todo:
				price_target = cur_price
				trade_amount = self.SHARES_PER_TRADE

				# reset i term and moving average
				i_term = 0
				rolling_price = cur_price

				# don't oversell - this needs to be more robust
				if self.shares-trade_amount < 0:
					trade_amount = self.shares

				# place sell order
				if hot and trade_amount != 0:
					# let us know if it was placed
					print("[", self.loop_count, "] Order Placed: [ sell ] Q ", end='')
					print(trade_amount, "$", price_target)

					sell = r.order_sell_limit(self.tick, trade_amount, price_target)

					if sell != None:
						order_summary = dict()
						order_summary['id'] = sell['id']
						order_summary['quantity'] = float(sell['quantity'])
						order_summary['price'] = float(sell['price'])
						order_summary['side'] = sell['side']
						order_summary['state'] = 'placed'
						order_summary['average_price'] = None
						order_summary['created_at'] = sell['created_at']

						# store this, because we'll need to add it to datastruct
						new_order = order_summary
				else:
					# let us know if it was placed
					print("cold [", self.loop_count, "] Order Placed: [ sell ] Q ", end='')
					print(trade_amount, "$", price_target)
					
			elif pid_term > self.PID_LIM:				# BUY

				# todo:
				price_target = cur_price
				trade_amount = self.SHARES_PER_TRADE

				# reset i term if we order
				i_term = 0
				rolling_price = cur_price
				
				# cancel all other orders, and place buy order
				if hot and self.shares <= self.SHARE_LIMIT:
					# let us know if it was placed
					print("[", self.loop_count, "] Order Placed: [ buy ] Q ", end='')
					print(trade_amount, "$", price_target)

					buy = r.order_buy_limit(self.tick, trade_amount, price_target)

					if buy != None:
						order_summary = dict()
						order_summary['id'] = buy['id']
						order_summary['quantity'] = float(buy['quantity'])
						order_summary['price'] = float(buy['price'])
						order_summary['side'] = buy['side']
						order_summary['state'] = 'placed'
						order_summary['average_price'] = None
						order_summary['created_at'] = buy['created_at']

						# store this, because we'll need to add it to datastruct
						new_order = order_summary
				else:
					# let us know if it was placed
					print("(cold) [", self.loop_count, "] Order Placed: [ buy ] Q ", end='')
					print(trade_amount, "$", price_target)

			# add an entry to the full data group so we can keep
			# making predictions
			addon = dict()
			addon['tick'] = self.tick
			addon['price'] = cur_price
			addon['roll_price'] = rolling_price
			addon['pid'] = pid_term
			addon['order'] = new_order # if there was a trade, add it to the data we return

			self.real_time_data[self.loop_count] = addon
			self.loop_count += 1

			# see if any orders were filled, and update balanced
			self.update_orders()

			# see if we are making money
			self.calc_performance()

			# wait to recieve a request from plotting client, then send data
			# timeout set to 1000ms currently, so no need to sleep on except
			try:
				self.server.recv()
				self.server.send(bytes(str(self.real_time_data), 'utf-8'))
				time.sleep(1)
			except:
				pass
	
	###########################################################################
	# update_orders:	helper functions to see if orders have been filled
	#					and update our data struct accordingly
	# params:			[none]
	###########################################################################
	def update_orders(self):
		# iterate through the real-time data and see if orders have changed
		for key in self.real_time_data.keys():

			# if there was an order, update the status if necessary
			if self.real_time_data[key]['order'] == None:
				pass
			elif self.real_time_data[key]['order']['state'] != 'filled':
				# query api to get new status
				new_state = r.get_stock_order_info(self.real_time_data[key]['order']['id'])

				# check to see if this trade has been filled
				if new_state != None:

					# if the order was filled, update accordingly
					if new_state['state'] == 'filled' and new_state['average_price'] != None:
						
						# update original data struct
						self.real_time_data[key]['order']['state'] = 'filled'
						self.real_time_data[key]['order']['fill_time'] = datetime.datetime.now()
						self.real_time_data[key]['order']['fill_count'] = self.loop_count
						self.real_time_data[key]['order']['average_price'] = float(new_state['average_price'])

						if self.real_time_data[key]['order']['side'] == 'sell':
							self.cash += float(new_state['average_price'])*float(new_state['quantity'])
							self.shares -= float(new_state['quantity'])
						elif self.real_time_data[key]['order']['side'] == 'buy':
							self.cash -= float(new_state['average_price'])*float(new_state['quantity'])
							self.shares += float(new_state['quantity'])

						# let us know if it was filled
						print("[", self.loop_count, "] Order Filled: [", new_state['side'], "] Q ", end='')
						print(new_state['quantity'], "$", new_state['average_price'])

	###########################################################################
	# calc_performance:	see how we're doing so far based on completed trades
	#					run this every loop, based on cur price
	# params:			[none]
	###########################################################################
	def calc_performance(self):

		# hopefully these match...
		price = self.get_price_safe()
		if price != 0:
			print("$", self.cash, "Q", self.shares, "Profit (class_):", self.cash+self.shares*price, end='\r')

	###########################################################################
	# simulate_trader:	simulate trading based on PID settings
	#					using old data file
	# params:			verbose:		see if we want print statements
	#					fname:			file to pull data from
	###########################################################################
	def simulate_trader(self, fname, verbose=True):
		self.sim_data = dict()
		self.real_time_data = dict() # make sure we get a reset, empty dict

		self.extract_sim_data(fname)

		# make sure we reset
		self.loop_count = 1
		
		# 'moving average' - init to real values so we don't get jumpy
		rolling_price = self.sim_data[1]['price']
		last_price = rolling_price

		time.sleep(1)

		# PID stuff
		i_term = 0
		pid_term = 0

		# start trading here
		while True:
			try:
				# make sure we're getting a good price
				cur_price = self.sim_data[self.loop_count]['price']
			except:
				break

			if cur_price == 0:
				pass

			# calculate the 'rolling average'
			rolling_price = rolling_price*0.9 + cur_price*0.1
			spread_percent = (rolling_price - cur_price)/cur_price
			change_percent = (last_price - cur_price)/cur_price

			# i_term should rollover from previous
			i_term += spread_percent*self._I_

			# CALCULATE PID
			pid_term = self._P_*spread_percent + i_term + self._D_*change_percent
			
			# adjust PID for relative volatility as well as shares we hold
			pid_term -= float(self.shares / self.SHARE_LIMIT)
			# pid_term *= self.get_rel_volume()


			# update for next time - for 'd' term
			last_price = cur_price

			# now, trade based on info
			if pid_term < -1*self.PID_LIM: 				# SELL

				# todo:
				price_target = cur_price
				trade_amount = self.SHARES_PER_TRADE

				# reset i term and moving average
				i_term = 0
				rolling_price = cur_price

				# don't oversell - this needs to be more robust
				if self.shares-trade_amount < 0:
					trade_amount = self.shares

				# place sell order
				if trade_amount != 0:
					# let us know if it was placed
					print("[", self.loop_count, "] Order Placed: [ sell ] Q ", end='')
					print(trade_amount, "$", price_target)

					self.cash += trade_amount*price_target
					self.shares -= trade_amount

				else:
					# let us know if it was placed
					print("(cold) [", self.loop_count, "] Order Placed: [ sell ] Q ", end='')
					print(trade_amount, "$", price_target)
					
			elif pid_term > self.PID_LIM:				# BUY

				# todo:
				price_target = cur_price
				trade_amount = self.SHARES_PER_TRADE

				# reset i term if we order
				i_term = 0
				rolling_price = cur_price
				
				# cancel all other orders, and place buy order
				if self.shares <= self.SHARE_LIMIT:
					# let us know if it was placed
					print("[", self.loop_count, "] Order Placed: [ buy ] Q ", end='')
					print(trade_amount, "$", price_target)

					self.cash -= trade_amount*price_target
					self.shares += trade_amount

				else:
					# let us know if it was placed
					print("(cold) [", self.loop_count, "] Order Placed: [ buy ] Q ", end='')
					print(trade_amount, "$", price_target)
					

			# add an entry to the full data group so we can keep
			# making predictions
			addon = dict()
			addon['tick'] = self.tick
			addon['price'] = cur_price
			addon['roll_price'] = rolling_price
			addon['pid'] = pid_term
			addon['order'] = None

			self.real_time_data[self.loop_count] = addon
			self.loop_count += 1

			# wait to recieve a request from plotting client, then send data
			# timeout set to 1000ms currently, so no need to sleep on except
			# try:
			# 	self.server.recv()
			# 	self.server.send(bytes(str(self.real_time_data), 'utf-8'))
			# 	# time.sleep(1)
			# except:
			# 	pass

		self.cash += self.sim_data[self.loop_count-1]['price']*self.shares
		self.shares = 0

		print("Profit: $", round(self.cash, 2))

	###########################################################################
	# optimize_pid:		do a bunch of simulations and optimize the PID
	#					settings for best profit for given dataset
	# params:			[none]
	###########################################################################	
	def optimize_pid(self):
		# simulate for optimal pid
		profits = []
		for i in range(8, 11):
			for j in range(8, 11):
				for k in range(30, 80, 4):
					print(i, j, k)
					self._P_ = i
					self._I_ = j
					self._D_ = k

					profits.append([i, j, k, self.simulate_trader(verbose=False)])

		print(profits)

		max_ind = 0
		max_prof  = profits[0][3]
		for count, i in enumerate(profits):
			if i[3] > max_prof:
				max_ind = count

		print("MAX:", profits[max_ind])

	###########################################################################
	# extract_sim_data: extract simulation file in dict from
	#					from previous trading session
	# params:			fname 			file to get data out of
	###########################################################################
	def extract_sim_data(self, fname):
		# reinit to empty dict
		self.sim_data = dict()

		fname = fname[2:]

		# in case we want to use simulation testing
		data_file = None
		try:
			stub = 'C:/Users/Spencer/OneDrive/Coding/stock/robinhood/'
			data_file = open(stub+fname, 'r')
			# self.sim_data = ast.literal_eval(data_file.read())
			self.sim_data = eval(data_file.read())
		except:
			try:
				stub = 'C:/Users/stbue/OneDrive/Coding/stock/robinhood/'
				data_file = open(stub+fname, 'r')
				# self.sim_data = ast.literal_eval(data_file.read())
				self.sim_data = eval(data_file.read())
			except:
				sys.exit("Datafile invalid for simulation")
		data_file.close()

	###########################################################################
	# shutdown: 		graceful shutdown, including renaming the tmp data
	#					file so it stays forever
	# params:			sig:			signal (SIGINT)
	#					frame:			idk what this is
	###########################################################################
	def shutdown_(self, sig, frame):
		print("Trading exiting")

		fname = str(self.start_time)
		fname = '/D'+fname[5:7]+'-'+fname[8:10]+'__T'+fname[11:13]+'-'+fname[14:16]+'.txt'
		
		# write all the data to a file for potential later examination
		with open('data/'+self.tick+fname, 'w') as wf:
			wf.write(str(self.real_time_data))

		self.context.destroy()

		# for some reason, sometimes need to hit this twice...
		sys.exit(0)


if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python pid_trader.py TICKER [trade_file_to_visualize] [more to come]")

	# instantiate the class
	t = TraderPID(sys.argv[1])

	if len(sys.argv) == 2:
		# start realtime trading
		t.realtime_trader(hot=True)
	elif len(sys.argv) == 3:
		# simulate - filename
		t.simulate_trader(sys.argv[2])