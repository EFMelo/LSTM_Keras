from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop

class Petr4Model:

    @classmethod
    def build(cls, target_size):

        model = Sequential()

        # LSTM layers
        model.add(LSTM(units=39, return_sequences=True, input_shape=(target_size[0], target_size[1])))
        model.add(Dropout(0.11161976352718975))

        model.add(LSTM(units=39))
        model.add(Dropout(0.11161976352718975))

        # Output
        model.add(Dense(units=1, activation='linear'))

        # Configuring the network
        model.compile(optimizer=RMSprop(lr=0.005284515052060461), loss='mean_squared_error')

        return model