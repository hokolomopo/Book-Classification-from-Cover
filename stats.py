
class Statistics():
    def __init__(self):
        self.accuracies = {'train' : [], 'val' : []}
        self.losses = {'train' : [], 'val' : []}
        self.epochs = {'train' : [], 'val' : []}
        self.times = {'train' : [], 'val' : []}
        self.time_elapsed = -1
        self.best_acc = -1

    def __repr__(self):
    	accuraciesRepr = "Accuracy train = {}\n".format(self.accuracies['train'])
    	accuraciesRepr += "Accuracy val = {}\n".format(self.accuracies['val'])
    	lossesRepr = "Losses train = {}\n".format(self.losses['train'])
    	lossesRepr += "Losses val = {}\n".format(self.losses['val'])
    	epochsRepr = "Epochs train = {}\n".format(self.epochs['train'])
    	epochsRepr += "Epochs val = {}\n".format(self.epochs['val'])
    	timesRepr = "Times train = {}\n".format(self.times['train'])
    	timesRepr += "Times val = {}\n".format(self.times['val'])
    	timeElapsedRepr = "Time elapsed = {}\n".format(self.time_elapsed)
    	bestAccRepr = "Best acc = {}\n".format(self.best_acc)

    	return accuraciesRepr + lossesRepr + epochsRepr + timesRepr + timeElapsedRepr + bestAccRepr
