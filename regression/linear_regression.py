import numpy as np
import pandas as pd
from random import shuffle

def data_preprocess():
    """[Pre-process the dataset to get input and output]
    
    Returns:
        [np.array] -- [input and output of train set, input of test set]
    """
    train_x = []
    test_x = []

    train_all = pd.read_csv('./ml2019spring-hw1/train.csv', encoding='big5')
    test_all = pd.read_csv('./ml2019spring-hw1/test.csv', encoding='big5')

    # get the train input
    day_cnt = train_all.shape[0] / 18
    for i in range(int(day_cnt)):
        train_x_element = train_all.iloc[18*i:18*(i+1), -10:-1].to_numpy()
        train_x_element[train_x_element == 'NR'] = 0.0
        train_x_element = train_x_element.astype(np.float).flatten()
        train_x.append(train_x_element)
    train_x = np.asarray(train_x)

    # get the train ouput
    train_y = train_all[train_all['測項'] == 'PM2.5']
    train_y = train_y.iloc[:,-1].to_numpy()
    train_y = np.asmatrix(train_y.astype(np.float)).T

    day_cnt = test_all.shape[0] / 18
    for i in range(int(day_cnt)):
        test_x_element = test_all.iloc[18*i:18*(i+1), 2:].to_numpy()
        test_x_element[test_x_element == 'NR'] = 0.0
        test_x_element = test_x_element.astype(np.float).flatten()
        test_x.append(test_x_element)
    test_x = np.asarray(test_x)

    return (train_x, train_y, test_x)


def gradient_descent(x, y, w, eta, iter):
    for i in range(iter):
        idx = list(range(x.shape[0]))
        shuffle(idx)
        for j in range(x.shape[0]):
            input_x = np.asmatrix(x[idx[j]]).T
            predict_y = w.T.dot(input_x)
            loss = predict_y - y[idx[j]]
            cost = np.sum(np.power(loss, 2))
            gradient = input_x.dot(loss)
            w -= eta * gradient
        if (i % 1000 == 0):
            print(f'iteration: {i}, cost: {cost}')
    
    return w


def save_res(predict_res):
    idx = list(range(predict_res.shape[0]))
    id = [f'id_{i}' for i in idx]
    res = {
        'id': id, 
        'value': list(map(int, predict_res.flatten()))
    }
    res = pd.DataFrame(res)
    res.to_csv('predict_result.csv', index=False)


def main():
    train_x, train_y, test_x = data_preprocess()
    train_x = np.append(train_x, np.ones((train_x.shape[0], 1)), axis=1)
    test_x = np.append(test_x, np.ones((test_x.shape[0], 1)), axis=1)
    w = np.zeros((train_x.shape[1], 1))

    w = gradient_descent(train_x, train_y, w, eta=1e-6, iter=20000)
    test_y = test_x.dot(w)
    save_res(test_y)

if __name__ == "__main__":
    main()