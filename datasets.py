from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import array, zeros

class Petr4:
   
    @classmethod
    def load_data(cls, time_step, *args, pp=None):

        """
        Description
        -----------
        Predict the share price from 05/28/2018 to 06/22/2018.
        The default attribute to be predicted is Open.
    
        Parameters
        ----------
        time_step: int
                  Time interval.
        pp : str or None
            Pre-processing type.
            The 'mms' or 'std' arguments can be entered, so the MinMaxScaler (mms) or StandardScaler (std) are calculated.
            The None argument doen nothing.
        *args: tuple of str.
              Informs the attributes to be read. Values ​​can be: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    
        Returns
        -------
        x_train : ndarray
        y_train : ndarray
        x_test : ndarray
        y_test : ndarray
        norm : sklearn.preprocessing
        """

        cls.__time_step = time_step
        cls.__atributes = args
        cls.__norm = None
        cls.__pp = pp

        # reading the data
        cls.__read_data()

        # buildind training and test data
        x_train, y_train = cls.__build_input_output(cls.__train_data)
        x_test, y_test = cls.__build_input_output(cls.__test_data)

        return x_train, y_train, x_test, y_test, cls.__norm


    @classmethod
    def __read_data(cls):

        # reading cvs files and deleting NaN values
        if len(cls.__atributes) == 0:            
            # data only with the Open attribute
            cls.__train_data = read_csv('datasets/petr4/train.csv').dropna().iloc[:, 1:2].values # train data 
            cls.__test = read_csv('datasets/petr4/test.csv').dropna().iloc[:, 1:2].values # test data
            
            # pre-processing
            cls.__train_data = cls.__pre_processing(cls.__train_data)
            cls.__test = cls.__pre_processing(cls.__test)
            
            # Finalizing the test data
            cls.__build_test_data()
        else:
            train = read_csv('datasets/petr4/train.csv').dropna().iloc[:, 1:7].values  # train data
            test = read_csv('datasets/petr4/test.csv').dropna().iloc[:, 1:7].values  # test data

            cls.__train_data = zeros((train.shape[0], len(cls.__atributes)))
            cls.__test = zeros((test.shape[0], len(cls.__atributes)))

            k = 0
            for atr in cls.__atributes:
                if atr == 'Open':
                    cls.__train_data[:, k] = train[:, 0]
                    cls.__test[:, k] = test[:, 0]
                elif atr == 'High':
                    cls.__train_data[:, k] = train[:, 1]
                    cls.__test[:, k] = test[:, 1]
                elif atr == 'Low':
                    cls.__train_data[:, k] = train[:, 2]
                    cls.__test[:, k] = test[:, 2]
                elif atr == 'Close':
                    cls.__train_data[:, k] = train[:, 3]
                    cls.__test[:, k] = test[:, 3]
                elif atr == 'Adj Close':
                    cls.__train_data[:, k] = train[:, 4]
                    cls.__test[:, k] = test[:, 4]
                elif atr == 'Volume':
                    cls.__train_data[:, k] = train[:, 5]
                    cls.__test[:, k] = test[:, 5]

                k += 1
                
            # pre-processing
            cls.__train_data = cls.__pre_processing(cls.__train_data)
            cls.__test = cls.__pre_processing(cls.__test)
            
            # Finalizing the test data
            cls.__build_test_data()


    @classmethod
    def __pre_processing(cls, data):
        
        size = len(cls.__atributes)
        if size == 0:
            size = 1
            
        # pre-processing        
        if cls.__pp == 'mms':
            cls.__norm = MinMaxScaler()
        elif cls.__pp == 'std':
            cls.__norm = StandardScaler()
        
        if cls.__pp != None:
            for i in range(size):
                data[:,i:i+1] = cls.__norm.fit_transform(data[:,i].reshape(data[:,i].shape[0], 1))
    
        return data
    

    @classmethod
    def __build_test_data(cls):

        dim = len(cls.__atributes)        
        if dim == 0:
            dim = 1
        
        # adding previous records to test data according to time_step
        cls.__test_data = zeros((cls.__time_step + cls.__test.shape[0], dim))
        cls.__test_data[0:cls.__time_step] = cls.__train_data[cls.__train_data.shape[0] - cls.__time_step:cls.__train_data.shape[0]]
        cls.__test_data[cls.__time_step:cls.__time_step + cls.__test.shape[0]] = cls.__test


    @classmethod
    def __build_input_output(cls, data):

        x, y = [], []

        for i in range(cls.__time_step, data.shape[0]):
            x.append(data[i-cls.__time_step:i])
            y.append(data[i, 0])

        x, y = array(x), array(y)

        return x, y


class CumulativeOil:

    @classmethod
    def load_data(cls, time_step, pp=None):

        """
        Description
        -----------
        Cumulative oil production forecast from 06/30/2028 to 05/31/2033
    
        Parameters
        ----------
        time_step: int
                  Time interval.
        pp : str or None
            Pre-processing type.
            The 'mms' or 'std' arguments can be entered, so the MinMaxScaler (mms) or StandardScaler (std) are calculated.
            The None argument doen nothing.
    
        Returns
        -------
        x_train : ndarray
        y_train : ndarray
        x_test : ndarray
        y_test : ndarray
        norm : sklearn.preprocessing
        """
        
        norm = None
        cls.__time_step = time_step

        # reading the data
        cls.__oil = read_csv('datasets/oil/cumulative_oil.csv').iloc[:, 2:3].values

        # pre-processing        
        if pp == 'mms':
            norm = MinMaxScaler()
        elif pp == 'std':
            norm = StandardScaler()
        
        if pp != None:
            cls.__oil = norm.fit_transform(cls.__oil)

        # buildind training and test data
        x_train, y_train = cls.__build_input_output(0, 182)  # 182 months for training data
        x_test, y_test = cls.__build_input_output(182 - time_step, 242)  # 60 months for test data

        return x_train, y_train, x_test, y_test, norm
    
    @classmethod
    def __build_input_output(cls, ini_size, end_size):

        x, y = [], []
        oil = cls.__oil[ini_size:end_size]

        for i in range(cls.__time_step, oil.shape[0]):
            x.append(oil[i-cls.__time_step:i])
            y.append(oil[i, 0])

        x, y = array(x), array(y)

        return x, y