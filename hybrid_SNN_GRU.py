from keras.models import Model
import keras.backend as K
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers import GRU, concatenate
from keras.layers import Activation
from keras.layers.noise import AlphaDropout
from keras.optimizers import Adam
import pickle
import utils.data_utils as util
from keras.callbacks import EarlyStopping


# pull from directory if not in memory
if 'instances' not in dir():
	instances = pickle.load(open('./data/instances25JUN.p', 'rb'))

# use data making function to make inputs and output
# X1: time series tensor (OHCLV x number of periods x batch size)
# X2: close price and return on day of event
# y:  t+n return
X1, X2, y = util.make_data_aux(instances, tplus = 20, min_len=100, max_len=100)

y = util.clip_anomalies(y, iqr_multiple=10)
y[y<-100] = -100


# time series data goes straight into GRU
main_input = Input(shape=X1.shape[1:], name = 'main_in')

# stacked 128-neuron GRU output
gru_out = GRU(64, return_sequences = True)(main_input)
gru_out = GRU(64, return_sequences = True)(gru_out)
gru_out = GRU(64, return_sequences = True)(gru_out)
gru_out = GRU(64)(gru_out)

# GRU output for loss calculation #1
gru_pred = Dense(1, name = 'GRU_out')(gru_out)

# merge other data with GRU output
aux_input = Input(shape = X2.shape[1:], name = 'aux_in')
x = concatenate([gru_out, aux_input])

activation = 'selu'
layer_size = 128
n_layers = 4
drop_rate = 0.01

for n in range(n_layers):
	x = Dense(layer_size)(x)
	x = Activation(activation)(x)
	x = AlphaDropout(rate = drop_rate)(x)
	

# main output for loss calculation #2
main_pred = Dense(1, name = 'main_out')(x)

# define inputs / outputs
model = Model(inputs=[main_input, aux_input], outputs=[main_pred])

# weight losses and compile model
model.compile(optimizer=Adam(lr = 0.0001), loss='mean_squared_error',
              loss_weights=[1.])

model.fit([X1, X2], [y], epochs=50, validation_split=0.5, batch_size = 256,
			 callbacks=[EarlyStopping(patience=8)])

# clear graph
K.clear_session()
