import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)
    plt.show()
    
    
# summarize scores
def summarize_results(config, scores, kernel):
    config.LOGGER_FILE.info(f'Scores: {scores}\n kernels: {kernel}')
    # summarize mean and standard deviation
    m, s = mean(scores), std(scores)
    # print('Score: %.3f%% (+/-%.3f)' % (m, s))
    config.LOGGER.info('Score: %.3f%% (+/-%.3f)' % (m, s))

    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        config.LOGGER.info('Kernel=%s: %.3f%% (+/-%.3f)' % (str(kernel[i]), m, s))
        # print('Kernel=%s: %.3f%% (+/-%.3f)' % (str(kernel[i]), m, s))
    # Box-plot of scores
    plt.boxplot(scores, labels=kernel)
    plt.title("Kernels")
    plt.show()