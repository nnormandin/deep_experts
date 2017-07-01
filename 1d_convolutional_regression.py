from keras.layers import Input, Dense, Flatten #, BatchNormalization, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import pickle
import utils.data_utils as util
from keras.callbacks import EarlyStopping
from keras.models import Model
import keras.backend as K


# pull from directory if not in memory
if 'instances' not in dir():
	instances = pickle.load(open('./data/instances25JUN.p', 'rb'))

if 'X' not in dir():
	X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)

# clip positive anomalies down to IQRx10
y = util.clip_anomalies(y, iqr_multiple=10)
y[y<-100] = -100

# time series data goes straight into GRU
data_in = Input(shape=X.shape[1:])
x = Conv1D(128, 3, activation = 'elu')(data_in)
x = Conv1D(128, 3, activation = 'elu')(x)
x = Conv1D(128, 3, activation = 'elu')(x)
x = Conv1D(128, 3, activation = 'elu')(x)

#x = MaxPooling1D()(x)
x = Conv1D(256, 6, activation='elu')(x)
x = Conv1D(256, 6, activation='elu')(x)
x = Conv1D(256, 12, activation='elu')(x)
x = Flatten()(x)

# main output for loss calculation #2
pred = Dense(1)(x)

# define inputs / outputs
model = Model(inputs=[data_in], outputs=[pred])

# weight losses and compile model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit([X], [y], epochs=50, validation_split=0.5, batch_size = 256,
			 callbacks=[EarlyStopping(patience=8)], shuffle = True)

# clear graph
K.clear_session()
