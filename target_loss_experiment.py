import utils.data_utils as util
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense #, BatchNormalization #, Dropout
from keras.layers import GRU
from sklearn.metrics import mean_squared_error
import pickle
import math

if 'instances' not in dir():
	instances = pickle.load(open('./instances24JUN.p', 'rb'))

# empty list for validation history
validation_history = []
y_history = []

for i in [1, 5, 10, 20, 30, 40, 50, 75, 100]:
	
	print('- fitting model with {} day target'.format(i))
	
	# choose input matrix and target; reshape to np array
	X, y = util.make_data(instances, tplus = i, min_len=100, max_len=100)
	
	# clip positive anomalies down to IQRx10
	y = util.clip_anomalies(y, iqr_multiple=10)
	y_history.append(y)
	
	print('- using {} instances'.format(X.shape[0]))
	
	# define graph
	mod = Sequential()
	mod.add(GRU(128, return_sequences = True, input_shape = X.shape[1:]))
	#mod.add(LSTM(128, return_sequences = True))
	mod.add(GRU(128))
	mod.add(Dense(1))
	
	# compile
	mod.compile(optimizer='adam',
	              loss='mean_squared_error')
	
	# fit
	mod_fitted = mod.fit(X, y, validation_split=0.4, shuffle=True,
							epochs=30)
	
	validation_history.append(mod_fitted.history['val_loss'])
	
	# reset graph
	K.clear_session()

y_means = [x.mean() for x in y_history]
min_val = [min(x) for x in validation_history]

rmse = [math.sqrt(x) for x in min_val]
coef_var = [x / y for x,y in zip(rmse, y_means)]

coef_var_dif = []

for x,y,z in zip(y_means, y_history, coef_var):
	coef_var_dif.append(((math.sqrt(mean_squared_error(y, [x]*len(y)))/x)-z))

