# just_plot.py
import re
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    result = []
    with open('result.txt', 'r') as f:
        result = [line.split('|') for line in f.readlines()]
    del result[0]
    X = []
    Y = []
    for line in result:
        X.append(int(line[0]))
        Y.append(float(line[1]))
    line_cross, = plt.plot(
        X, Y, 'bo-', label='Success rate to number of eigen faces')
    plt.xlabel('number of eigen faces')
    plt.ylabel('success rate')
    plt.xlim((0, 25))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()
