
class Statistics():
    def __init__(self):
        self.accuracies = {'train' : [], 'val' : []}
        self.losses = {'train' : [], 'val' : []}
        self.epochs = {'train' : [], 'val' : []}
        self.times = {'train' : [], 'val' : []}
        self.time_elapsed = -1
        self.best_acc = -1
        self.cnn_parameters = None

class CNNParameters():
    def __init__(self, resnet, batch_size, trained_layers, n_outputs, lr=0):
        self.resnet = resnet
        self.batch_size = batch_size
        self.trained_layers = trained_layers
        self.n_outputs = n_outputs
        self.lr = lr
