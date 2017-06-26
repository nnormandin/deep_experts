from keras.models import Model
from keras.layers import Input, Dense #, BatchNormalization #, Dropout
from keras.layers import GRU
import pickle
import utils.data_utils as util


if 'instances' not in dir():
	instances = pickle.load(open('./instances25JUN.p', 'rb'))

X, y = util.make_data(instances, tplus = 20, min_len=100, max_len=100)


main_input = Input(shape=X.shape[1:], name = 'main_input')

lstm_out = GRU(32)(main_input)

main_out = Dense(1, activation = 'sigmoid', name = 'main_output')(lstm_out)

model = Model(inputs=[main_input], outputs=[main_out])

model.compile(optimizer='adam', loss='mean_squared_error',
              loss_weights=[1.])

model.fit([X], [y], epochs=5, batch_size=32)