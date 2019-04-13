from stats import Statistics, CNNParameters

def get_stats(filename):
    file = open(filename, "r")

    stats = []

    while True:
        line = file.readline()
        while line == '\n':
            line = file.readline()

        if not line:
            break
        
        splits = line.split(', ')

        resnet = int(splits[0][6:])
        lr = float(splits[1][5:])
        batch_size = int(splits[2][13:])
        trained_layers = int(splits[3][17:])
        n_outputs = int(splits[4][12:])

        stat = Statistics()
        stat.cnn_parameters = CNNParameters(resnet, batch_size, trained_layers, n_outputs, lr)

        x = "train"
        while True:
            line = file.readline()
            if line[:5] != x:
                break
            splits = line.split(' ')
            stat.epochs[x].append(int(splits[4]))
            stat.accuracies[x].append(float(splits[7][:len(splits[7]) - 1]))
            stat.times[x].append(float(splits[9][:len(splits[9]) - 1]) * 60 + float(splits[10][:len(splits[10]) - 2]))
            
        x = "val"
        while True:
            if line[:3] != x:
                break
            splits = line.split(' ')
            stat.epochs[x].append(float(splits[4]))
            stat.accuracies[x].append(float(splits[7][:len(splits[7]) - 1]))
            stat.times[x].append(float(splits[9][:len(splits[9]) - 1]) * 60 + float(splits[10][:len(splits[10]) - 2]))
            line = file.readline()
        
        splits = line.split(' ')
        stat.time_elapsed = float(splits[3][:len(splits[3]) - 1]) * 60 + float(splits[4][:len(splits[4]) - 1])

        splits = file.readline().split(' ')
        stat.best_acc = float(splits[3])

        stats.append(stat)

        

    file.close()
    return stats
