import utils.data_utils as util
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten #, BatchNormalization #, Dropout
from keras.layers.wrappers import TimeDistributed
from utils.yellowfin import YFOptimizer
from keras.optimizers import TFOptimizer
import pickle
import os

# create an instance database if none exists
#if 'instances.p' not in os.listdir():
#	stock_data = '/home/nick/R/projects/pigasus2/data/data_by_ticker/'
#	instances = util.make_instances(directory = stock_data, save=False,
#					 min_loss = -0.07, save_before = 250, save_after = 150)

# pull into memory
if 'instances' not in dir():
	instances = pickle.load(open('./instances.p', 'rb'))

# choose input matrix and target; reshape to np array
X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)

#y = [int(x>0) for x in y]

# define graph
mod = Sequential()
mod.add(TimeDistributed(Dense(16, activation = 'relu'),
						  input_shape=(X.shape[1:])))
mod.add(Flatten())
#mod.add(Dense(256, activation = 'relu'))

mod.add(Dense(1))

# compile
#mod.compile(optimizer=TFOptimizer(YFOptimizer()),
#              loss='binary_crossentropy', metrics = ['acc'])

mod.compile(optimizer='adam',
              loss='mean_squared_error')

# fit
modfit = mod.fit(X, y, validation_split =0.2, shuffle =	True, batch_size=64,
						epochs=30)


# reset graph
K.clear_session()