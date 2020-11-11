# LSTM with Keras

> Using Long Short Term Memory (LSTM) from the keras framework.

### Datasets

**Petr4**

`Objective`: Predict the share price from 05/28/2018 to 06/22/2018. The default attribute to be predicted is `Open`.

**Cumulative Oil**

`Objective`: Cumulative oil production forecast from 06/30/2028 to 05/31/2033

### Example of Use

**Petr4**

- `time_step`: An integer informing the time interval.
- `pp`: A string informing the type of pre-processing: _'mms'_ is MinMaxScaler() and _'std'_ is StandardScaler(). _None_ is the default.
- `*args`: Strings informing the attributes used for LSTM. Values ​​can be: _'Open'_, _'High'_, _'Low'_, _'Close'_, _'Adj Close'_, _'Volume'_.

```python
from datasets import Petr4

x_train, y_train, x_test, y_test, norm = Petr4.load_data(time_step=90, pp='mms')
```

Using `*args`:

```python
x_train, y_train, x_test, y_test, norm = Petr4.load_data(90, 'Open', 'High', 'Low', 'Adj Close', pp='mms')
```

To predict another attribute, simply informe it first. The example below will use the LSTM to predict the `High` attribute:

```python
x_train, y_train, x_test, y_test, norm = Petr4.load_data(90, 'High', 'Low', 'Volume', pp='mms')
```

**Cumulative Oil**

- `time_step`: An integer informing the time interval.
- `pp`: A string informing the type of pre-processing: _'mms'_ is MinMaxScaler() and _'std'_ is StandardScaler(). _None_ is the default.

```python
from datasets import CumulativeOil

x_train, y_train, x_test, y_test, norm = CumulativeOil.load_data(time_step=20, pp='std')
```


### Some Results

**Petr4**

Predicting the `Open` attribute:

```python
from models import Petr4Model
from plots import Results

# building LSTM
model = Petr4Model.build(target_size=(x_train.shape[1], x_train.shape[2]))

# training
history = model.fit(x_train, y_train, epochs=100, batch_size=64)

# prediction
prediction = model.predict(x_test)

# returning to the original values
y_test = norm.inverse_transform(y_test.reshape(y_test.shape[0], 1))
prediction = norm.inverse_transform(prediction)

# plotting the loss
Results.loss(epochs, history)

# plotting the curves
Results.curves(y_test, prediction)
```

<p align="center">
  <img width="350" height="226" src="https://i.imgur.com/ueqakOt.png">
</p>


**Cumulative Oil**

Cumulative oil production forecast:

```python
from models import CumulativeOilModel
from plots import Results

# building LSTM
model = CumulativeOilModel.build(target_size=(x_train.shape[1], x_train.shape[2]))

# training
history = model.fit(x_train, y_train, epochs=150, batch_size=32)

# prediction
prediction = model.predict(x_test)

# returning to the original values
y_test = norm.inverse_transform(y_test.reshape(y_test.shape[0], 1))
prediction = norm.inverse_transform(prediction)

# plotting the loss
Results.loss(epochs, history)

# plotting the curves
Results.curves(y_test, prediction)
```

<p align="center">
  <img width="350" height="226" src="https://i.imgur.com/04fBLfA.png">
</p>


### Contact

[linkedin.com/in/edvaldo-melo/](https://www.linkedin.com/in/edvaldo-melo/)

emeloppgi@gmail.com

[github.com/EFMelo](https://github.com/EFMelo)