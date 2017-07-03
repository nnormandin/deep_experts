import utils.data_utils as util
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping
#from sklearn.metrics import mean_squared_error
import pickle
#import math


if 'instances' not in dir():
	instances = pickle.load(open('./instances24JUN.p', 'rb'))

# create x/y at tplus of 20
if 'X' not in dir():
	X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)

validation_history = []
mod_depths = [3, 4, 5]
layer_widths = [16, 32, 64]
max_epochs = 100
stopping_patience = 7

for j in layer_widths:
	for i in mod_depths:
		
		print('training {}-layer GRU with {} neurons per layer'.format(i, j))
		
		# define graph
		mod = Sequential()
		
		if i == 1:
			mod.add(Bidirectional(GRU(j, input_shape = X.shape[1:])))
			mod.add(Dropout(0.5))
		else:
			mod.add(Bidirectional(GRU(j, return_sequences = True), input_shape = X.shape[1:]))
			mod.add(Dropout(0.5))
		
		if i == 2:
			mod.add(Bidirectional(GRU(j)))
			mod.add(Dropout(0.5))
		if i == 3:
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j)))
			mod.add(Dropout(0.5))
		if i ==4:
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j)))
			mod.add(Dropout(0.5))
		if i ==5:
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j, return_sequences = True)))
			mod.add(Dropout(0.5))
			mod.add(Bidirectional(GRU(j)))
			mod.add(Dropout(0.5))

		mod.add(Dense(1))
		
		# compile
		mod.compile(optimizer='adam',
		              loss='mean_squared_error')
		
		# fit
		mod_fitted = mod.fit(X, y, validation_split=0.4, shuffle=True, batch_size = 256,
									epochs=max_epochs,
									callbacks=[EarlyStopping(patience = stopping_patience)])
		
		validation_history.append(mod_fitted.history['val_loss'])
		
		# reset graph
		K.clear_session()

min_val = [min(x) for x in validation_history]
