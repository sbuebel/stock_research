import os, ast, datetime, time, matplotlib, sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import zmq

class TradePlotter:

	###########################################################################
	# __init__: 		initialize the trading class
	#					this includes logging into the API
	# params:			[none]
	###########################################################################
	def __init__(self, port=5555):
		# use a zmq client to get the data from trading script 
		self.context = zmq.Context()
		self.client = self.context.socket(zmq.REQ)

		self.client.connect("tcp://localhost:"+str(port))
		
		self.tick = '' # we'll get this from the datafile

		# class variables so we can call in the animate function
		self.fig, self.ax1 = plt.subplots()
		self.ax2 = self.ax1.twinx()

		self.data = dict()

	###########################################################################
	# get_columns:		helper function to extract columns in form of np  
	#					arrays so we can plot them
	# params:			[none]
	###########################################################################
	def get_columns(self):
		dates = []
		prices = []
		roll_prices = []
		pids = []

		# we want to plot trade times too- each plotted in diff color, so this matters..
		buy_fill_times = []
		buy_fills = []
		buy_order_times = []
		buy_orders = []

		sell_fill_times = []
		sell_fills = []
		sell_order_times = []
		sell_orders = []

		for i, time_slice in enumerate(self.data):

			dates.append(int(time_slice))
			prices.append(self.data[time_slice]['price'])
			roll_prices.append(self.data[time_slice]['roll_price'])
			pids.append(self.data[time_slice]['pid'])

			if self.data[time_slice]['order'] == None:
				pass
			elif self.data[time_slice]['order']['state'] == 'filled':
				if self.data[time_slice]['order']['side'] == 'buy':
					buy_fills.append(self.data[time_slice]['order']['average_price'])
					buy_fill_times.append(self.data[time_slice]['order']['fill_count'])
				elif self.data[time_slice]['order']['side'] == 'sell':
					sell_fills.append(self.data[time_slice]['order']['average_price'])
					sell_fill_times.append(self.data[time_slice]['order']['fill_count'])
			else:
				if self.data[time_slice]['order']['side'] == 'buy':
					buy_orders.append(self.data[time_slice]['order']['price'])
					buy_order_times.append(int(time_slice))
				elif self.data[time_slice]['order']['side'] == 'sell':
					sell_orders.append(self.data[time_slice]['order']['price'])
					sell_order_times.append(int(time_slice))


		dates = np.array(dates)
		prices = np.array(prices)
		roll_prices = np.array(roll_prices)
		pids = np.array(pids)

		buy_fill_times = np.array(buy_fill_times)
		buy_fills = np.array(buy_fills)
		buy_order_times = np.array(buy_order_times)
		buy_orders = np.array(buy_orders)

		sell_fill_times = np.array(sell_fill_times)
		sell_fills = np.array(sell_fills)
		sell_order_times = np.array(sell_order_times)
		sell_orders = np.array(sell_orders)

		return dates, prices, roll_prices, pids, buy_fill_times, buy_fills, buy_order_times, buy_orders, sell_fill_times, sell_fills, sell_order_times, sell_orders

	###########################################################################
	# show_realtime:	make the animation happen
	# params:			[none]
	###########################################################################
	def show_realtime(self):
		# This function is called periodically from FuncAnimation
		def animate(i):
			self.plot_data(1000) # plot last 500 points

		# Set up plot to call animate() function periodically
		ani = animation.FuncAnimation(self.fig, animate, fargs=(None), interval=1000)
		plt.show()

	###########################################################################
	# plot data:		function to plot data, which gets called by animate
	# params:			[none]
	###########################################################################
	def plot_data(self, num_points):
		# dummy request we'll send to get the data
		self.client.send(b"gimme data")
		rec = self.client.recv().decode('utf-8')
		# print(rec)
		# self.data = ast.literal_eval(rec)
		self.data = eval(rec)

		# extract np arrays from dict
		time, prices, roll_prices, pids, b_fill_t, b_fill, b_order_t, b_order, s_fill_t, s_fill, s_order_t, s_order = self.get_columns()

		if len(prices) != 0:

			# setup the bounds
			min_price = prices[-1]*0.975
			max_price = prices[-1]*1.025

			if len(prices) > num_points:
				for p in range(len(prices)-num_points, len(prices)):
					if prices[p] < min_price:
						min_price = prices[p]*0.975
					if prices[p] > max_price:
						max_price = prices[p]*1.025
			else:
				for p in range(len(prices)):
					if prices[p] < min_price:
						min_price = prices[p]*0.975
					if prices[p] > max_price:
						max_price = prices[p]*1.025

			self.ax1.clear()

			# plot buys and sells
			self.ax1.plot(b_order_t, b_order, '^', markersize=15, color='cyan')
			self.ax1.plot(s_order_t, s_order, 'rv', markersize=15, color='cyan')
			self.ax1.plot(b_fill_t, b_fill, 'g^', markersize=15)
			self.ax1.plot(s_fill_t, s_fill, 'rv', markersize=15)

			print('orders [b, s]:', b_order, s_order)
			print('fills  [b, s]:', b_fill, s_fill)

			self.ax1.plot(time, prices, 'g-', linewidth=2)
			self.ax1.plot(time, roll_prices, 'r--', linewidth=1)

			self.ax1.set_ylabel("Price ($)")
			self.ax1.set_xlabel("Time")
			if time[-1] > num_points:
				self.ax1.set_xlim(time[-1]-num_points, time[-1])
			self.ax1.set_ylim(min_price, max_price)
			self.ax1.set_title(self.data[1]['tick']+": Time Series Data")
			self.ax1.grid()

			self.ax2.clear()
			self.ax2.set_ylim(-1, 1)
			self.ax2.set_ylabel("PID")
			self.ax2.plot(time, pids, 'b--', linewidth=1)
			self.ax2.plot(time, 0.75*time/time, 'y--', linewidth=1.5)
			self.ax2.plot(time, -0.75*time/time, 'y--', linewidth=1.5)


if __name__ == "__main__":

	if len(sys.argv) > 1:
		port = sys.argv[1]

	# instantiate the class and make it work
	p = TradePlotter(port)
	p.show_realtime()