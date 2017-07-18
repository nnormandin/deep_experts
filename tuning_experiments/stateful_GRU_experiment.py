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
	instances = pickle.load(open('../data/instances25JUN.p', 'rb'))

# create x/y at tplus of 20
X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# make it a multiple of batch size for stateful network
batch_size = 256
n_x = int(X.shape[0] / batch_size) * batch_size
X = X[:n_x,:,:]
y = y[:n_x]

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)

validation_history = []
mod_depths = [1, 2, 3, 4, 5]
layer_widths = [16, 32]
max_epochs = 50
stopping_patience = 7

for j in layer_widths:
	for i in mod_depths:
		
		print('training {}-layer GRU with {} neurons per layer'.format(i, j))
		
		# define graph
		mod = Sequential()
		
		if i == 1:
			mod.add(GRU(j, stateful = True, batch_input_shape = (256, 100, 7)))
		else:
			mod.add(GRU(j, stateful = True, return_sequences = True,
			   batch_input_shape = (256, 100, 7)))
		
		if i == 2:
			mod.add(GRU(j, stateful = True))
		if i == 3:
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True))
		if i ==4:
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True))
		if i ==5:
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True, return_sequences = True))
			mod.add(GRU(j, stateful = True))

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