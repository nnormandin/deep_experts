import pandas as pd
import numpy as np
import os
import pickle

ANOMALY_THRESH = 10
Y_len = 100

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
		for l in list(range(Y_len)):
			try:
				r = self.d_after.iloc[l, 3]
				if r > ANOMALY_THRESH:
					self.anomalous = True
				self.label.append(r)
			except:
				self.label.append(0)


def make_instances(directory, min_loss = -0.1, save_before = 150,
					 save_after = 50, save = True, save_file = './instances.p'):
	
	#os.chdir(directory)
	stock_files = os.listdir(directory)
	#stock_names = [x[0:-4] for x in stock_files]
	
	instances = []
	
	# iterate over all files in a directory
	for s in stock_files:
		stock = pd.read_csv(directory + s)
	
		# iterate over each day of returns
		n_days = len(stock)
		
		for i in range(n_days):
			if stock['return'][i] < min_loss:
				ticker = s[0:-4]
				date = stock['date'][i]
				close = stock['close'][i]
				vol = stock['volume'][i]
				n_before = i
				n_after = n_days - i
				ret = stock['return'][i]
				d_before = stock[max(0, i - save_before + 1):i+1]
				d_after = stock[i + 1 : min(i + n_after, i + save_after)]
				instances.append(Instance(ticker, date, ret, close, vol,
									n_before, n_after, d_before, d_after))

	if save:
		pickle.dump(instances, open(save_file, 'wb'))
		return
	
	else:
		return instances
	

def make_data(instances, tplus = 10, min_len = 50, max_len = 50,
			  exclude_anom = True):
		
	# remove positive return anomalies
	if exclude_anom:
		instances = [x for x in instances if x.anomalous == False]
	
	# select based on n_before and n_after
	instances = [x for x in instances if x.n_before >= min_len]
	X = [x.d_before.tail(max_len) for x in instances]
	
	y = np.asarray([x.label[tplus - 1] for x in instances])
	y = (y - 1) * 100
	
	N = len(X)
	M = X[0].shape[1]
	
	X_out = pd.DataFrame.as_matrix(pd.concat(X)).reshape(N, max_len, M)
	
	return X_out, y
	
def make_data_aux(instances, tplus = 10, min_len = 50, max_len = 50,
			  exclude_anom = True):
		
	# remove positive return anomalies
	if exclude_anom:
		instances = [x for x in instances if x.anomalous == False]
	
	# select based on n_before and n_after
	instances = [x for x in instances if x.n_before >= min_len]
	X = [x.d_before.tail(max_len) for x in instances]
	
	x_close = [x.close for x in instances]
	x_ret = [x.ret for x in instances]

	
	y = np.asarray([x.label[tplus - 1] for x in instances])
	y = (y - 1) * 100
	
	N = len(X)
	M = X[0].shape[1]
	
	#X_2 = np.reshape(X_2, (N, 1))
	X_2 = np.column_stack((x_close, x_ret))
	X_1 = pd.DataFrame.as_matrix(pd.concat(X)).reshape(N, max_len, M)
	
	return X_1, X_2, y

def clip_anomalies(y, iqr_multiple = 5):
	q3 = np.percentile(y, 75)
	iqr = q3 - np.percentile(y, 25)
	limit = q3 + (iqr_multiple * iqr)
	y[y>limit] = limit
	#n_clipped = len(y[y == limit])
	return y
	
	
	
