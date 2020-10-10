import matplotlib.pyplot as plt

class Results:
    
    @classmethod
    def loss(cls, epochs, history):
        
        plt.plot(range(epochs), history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
    
    @classmethod
    def curves(cls, y, prevision):
        
        plt.plot(range(y.shape[0]), y, label='Actual Values')
        plt.plot(range(prevision.shape[0]), prevision, label='LSTM')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.show()