import utils.data_utils as util
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense #, BatchNormalization #, Dropout
from keras.layers import LSTM
import pickle
import os

# create an instance database if none exists
if 'instances.p' not in os.listdir():
	stock_data = '/home/nick/R/projects/pigasus2/data/data_by_ticker/'
	instances = util.make_instances(directory = stock_data, save=False,
					 min_loss = -0.07, save_before = 250, save_after = 150)

# pull into memory
if 'instances' not in dir():
	instances = pickle.load(open('./instances.p', 'rb'))

# choose input matrix and target; reshape to np array
X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)

# define graph
mod = Sequential()
mod.add(LSTM(256, return_sequences = True, input_shape = X.shape[1:]))
#mod.add(LSTM(128, return_sequences = True))
mod.add(LSTM(256))
mod.add(Dense(1))

# compile
mod.compile(optimizer='adam',
              loss='mean_squared_error')

# fit
mod1fit = mod.fit(X, y, validation_split =0.3, shuffle =	True, batch_size=128,
						epochs=30)

# reset graph
K.clear_session()