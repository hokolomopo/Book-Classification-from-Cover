
class Statistics():
    def __init__(self):
        self.accuracies = {'train' : [], 'val' : []}
        self.losses = {'train' : [], 'val' : []}
        self.epochs = {'train' : [], 'val' : []}
        self.times = {'train' : [], 'val' : []}
        self.time_elapsed = -1
        self.best_acc = -1
