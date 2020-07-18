import robin_stocks as r

# let's not put the whole password in plain text...
# learn from our past...
username='stbuebel@gmail.com'
password='Owii91'
p_file = None
try:
	p_file = open("C:/Users/stbue/Desktop/robi_code.txt", 'r')
except:
	pass
try:
	p_file = open("C:/Users/Spencer/Desktop/robi_code.txt", 'r')
except:
	sys.exit("No password provided, login unsuccessful. IYKYK")
password += p_file.read()
p_file.close()

#print(r.get_all_open_stock_orders())
#print(r.get_quotes('LK'))
#print(r.get_latest_price('LK'))

#def get_latest_price(inputSymbols, includeExtendedHours=True):
#def get_quotes(inputSymbols, info=None):
#def order_buy_limit(symbol, quantity, limitPrice, timeInForce='gtc', extendedHours=False):
#def order_sell_limit(symbol, quantity, limitPrice, timeInForce='gtc', extendedHours=False):

if __name__ == "__main__":

	# ticker = input("Which stock would you like to trade: ")
	ticker = 'BYFC'

	# login
	login = r.login(username, password)


	# print out a summary and ask what to do
	cur_price = round(float(r.get_latest_price(ticker)[0]), 2)
	print("Current Price: ", cur_price)


	# now what?
	trade = input("Would you like to buy or sell (b/s): ")
	shares = int(input("How many shares: "))
	price_target = float(input("What price: "))


	if trade == 'b':
		# this means we want to buy
		print(ticker, ": Limit buy order for", shares, "shares has been placed at $", price_target)
		r.order_buy_limit(ticker, shares, price_target)

	elif trade == 's':
		# this means we want to sell
		print(ticker, ": Limit sell order for", shares, "shares has been placed at $", price_target)
		r.order_sell_limit(ticker, shares, price_target)

	else:
		print("ERROR: specify b/s for buy/sell")