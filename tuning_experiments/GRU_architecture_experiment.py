import utils.data_utils as util
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense #, BatchNormalization #, Dropout
from keras.layers import GRU
from keras.callbacks import EarlyStopping
#from sklearn.metrics import mean_squared_error
import pickle
#import math


if 'instances' not in dir():
	instances = pickle.load(open('./instances24JUN.p', 'rb'))

# create x/y at tplus of 20
X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)

validation_history = []
mod_depths = [1, 2, 3]
layer_widths = [64, 128, 256, 512]
max_epochs = 100
stopping_patience = 10

for j in layer_widths:
	for i in mod_depths:
		
		print('training {}-layer GRU with {} neurons per layer'.format(i, j))
		
		# define graph
		mod = Sequential()
		
		if i == 1:
			mod.add(GRU(j, input_shape = X.shape[1:]))
		else:
			mod.add(GRU(j, return_sequences = True, input_shape = X.shape[1:]))
		
		if i == 2:
			mod.add(GRU(j))
		if i == 3:
			mod.add(GRU(j, return_sequences = True))
			mod.add(GRU(j))

		mod.add(Dense(1))
		
		# compile
		mod.compile(optimizer='adam',
		              loss='mean_squared_error')
		
		# fit
		mod_fitted = mod.fit(X, y, validation_split=0.4, shuffle=True, batch_size = 128,
									epochs=max_epochs,
									callbacks=[EarlyStopping(patience = stopping_patience)])
		
		validation_history.append(mod_fitted.history['val_loss'])
		
		# reset graph
		K.clear_session()