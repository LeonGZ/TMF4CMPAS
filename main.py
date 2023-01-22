import numpy as np
import csv
import os
import evaluation as ev

def Pretrain(file):
    reader = csv.reader(open(file))
    lon = dict()
    lat = dict()
    bias = dict()
    dataset = list()
    for row in reader:
        dataset.append([row[0], row[1], float(row[2])])
        if row[0] not in lon: # initialize the latent vectors of longitudes
            lon[row[0]] = np.array([0, 1])
            for i in range(0, 49):
                lon[row[0]] = np.concatenate((lon[row[0]], np.array([0, 1])))
            lon[row[0]] = lon[row[0]] * 0.0001
        if row[1] not in lat: # initialize the latent vectors of latitudes
            lat[row[1]] = np.array([1, 0])
            for i in range(0, 49):
                lat[row[1]] = np.concatenate((lat[row[1]], np.array([1, 0])))
            lat[row[1]] = lat[row[1]] * 0.0001
        if (row[0], row[1]) not in bias: #initialoze the biases
            bias[(row[0], row[1])] = 0.0
    for epoch in range(0, 10):
        for data in dataset:
            if float(data[2]) == 9998.95: # there are some errors in grid data that needs to skip
                continue
            error = float(data[2]) - np.dot(lon[data[0]],lat[data[1]]) - bias[(data[0], data[1])]
            bias[(data[0], data[1])] = bias[(data[0], data[1])] + 0.1 * error
            lon[data[0]] = lon[data[0]] + 0.001*error*lat[data[1]]
            lat[data[1]] = lat[data[1]] + 0.001*error*lon[data[0]]
    return lon, lat, bias

def Finetune(lon, lat, bias, file):
    reader = csv.reader(open(file))
    dataset = list()
    trans = 1.0 # global transfer
    for row in reader:
        dataset.append([row[0], row[1], float(row[2])])
    for epoch in range(0, 1000):
        for data in dataset:
            error = float(data[2]) - np.dot(lon[data[0]], lat[data[1]]) - trans * (bias[(data[0], data[1])])
            lon[data[0]] = lon[data[0]] + 0.0001 * error * lat[data[1]]
            lat[data[1]] = lat[data[1]] + 0.0001 * error * lon[data[0]]
            trans = trans + 0.001 * error * (bias[(data[0], data[1])])
    return lon, lat, trans

def TMF_writing(lon, lat, bias, trans, file): #output the reformed grid data to file
    wf_c = csv.writer(open(file, 'a', newline=''))
    for j in lat:
        for i in lon:
            wf_c.writerow([i, j, max(0.0, round(np.dot(lon[i], lat[j]) + trans * (bias[(i, j)]), 4))])

def Pretrain_(file):
    reader = csv.reader(open(file))
    lon = dict()
    lat = dict()
    dataset = list()
    for row in reader:
        dataset.append([row[0], row[1], float(row[2])])
        if row[0] not in lon: # initialize the latent vectors of longitudes
            lon[row[0]] = np.array([0, 1])
            for i in range(0, 49):
                lon[row[0]] = np.concatenate((lon[row[0]], np.array([0, 1])))
            lon[row[0]] = lon[row[0]] * 0.0001
        if row[1] not in lat: # initialize the latent vectors of latitudes
            lat[row[1]] = np.array([1, 0])
            for i in range(0, 49):
                lat[row[1]] = np.concatenate((lat[row[1]], np.array([1, 0])))
            lat[row[1]] = lat[row[1]] * 0.0001
    for epoch in range(0, 10):
        for data in dataset:
            if float(data[2]) == 9998.95: # there are some errors in grid data that needs to skip
                continue
            error = float(data[2]) - np.dot(lon[data[0]],lat[data[1]])
            lon[data[0]] = lon[data[0]] + 0.001*error*lat[data[1]]
            lat[data[1]] = lat[data[1]] + 0.001*error*lon[data[0]]
    return lon, lat

def TMF_writing_(lon, lat, file): #output the reformed grid data to file
    wf_c = csv.writer(open(file, 'a', newline=''))
    for j in lat:
        for i in lon:
            wf_c.writerow([i, j, max(0.0, round(np.dot(lon[i], lat[j]), 4))])

def Finetune_(lon, lat, file):
    reader = csv.reader(open(file))
    dataset = list()
    for row in reader:
        dataset.append([row[0], row[1], float(row[2])])
    for epoch in range(0, 1000):
        for data in dataset:
            error = float(data[2]) - np.dot(lon[data[0]], lat[data[1]])
            lon[data[0]] = lon[data[0]] + 0.0001 * error * lat[data[1]]
            lat[data[1]] = lat[data[1]] + 0.0001 * error * lon[data[0]]
    return lon, lat

if __name__ == '__main__':
    partition = '10%' # 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%

    for file in os.listdir('CMPAS/'):
        print(file)
        pretrainfile = 'CMPAS/' + file
        finetunefile = 'gauge-partition/' + partition + '/Train/' + file
        testfile = 'gauge-partition/' + partition + '/Test/' + file
        tmffile = 'TMF/' + partition + '/' + file
        lon, lat, bias = Pretrain(pretrainfile) # pre-training
        lon, lat, trans = Finetune(lon, lat, bias, finetunefile) # fine-tuning
        TMF_writing(lon, lat, bias, trans, tmffile)
        print('---------------------------------------------')


