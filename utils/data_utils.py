import pandas as pd
import os
import pickle


class Instance(object):
	def __init__(self, ticker, date, ret, close, vol,
			  n_before, n_after, d_before, d_after):
		self.ticker = ticker
		self.date = date
		self.year = int(self.date[0:4])
		self.ret = float(ret)
		self.close = float(close)
		self.vol = int(vol)
		self.n_before = int(n_before)
		self.n_after = int(n_after)
		self.d_before = d_before[['open', 'high', 'low', 'close', 'volume']].reset_index(drop = True)
		self.d_before.loc[:,['open', 'high', 'low', 'close']] /= self.close
		self.d_after = d_after[['open', 'high', 'low', 'close', 'volume']].reset_index(drop = True)
		self.d_after.loc[:,['open', 'high', 'low', 'close']] /= self.close
		self.anomalous = False
		self.label = []
		for l in 50:
			try:
				r = self.d_after.iloc[l, 3]
				if r > 10:
					self.anomalous = True
				self.label.append(r)
			except:
				self.label.append(0)

