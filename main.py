from datasets import Petr4
from models import Petr4Model
from plots import Results

epochs = 100

# loading dataset
x_train, y_train, x_test, y_test, norm = Petr4.load_data(90, pp='mms')

# building CNN
model = Petr4Model.build(target_size=(x_train.shape[1], x_train.shape[2]))

# training
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64)

# prediction
prediction = model.predict(x_test)

# returning to the original values
y_test = norm.inverse_transform(y_test.reshape(y_test.shape[0], 1))
prediction = norm.inverse_transform(prediction)

# plotting the loss
Results.loss(epochs, history)

# plotting the curves
Results.curves(y_test, prediction)