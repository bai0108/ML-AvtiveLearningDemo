import random
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_data(filename1, filename2, mode):
    y = scipy.io.loadmat(filename1)
    x = scipy.io.loadmat(filename2)

    if mode == 'train':
        x = x["trainingMatrix"]
        y = np.array(y['trainingLabels']).ravel()

    if mode == 'test':
        x = x["testingMatrix"]
        y = np.array(y['testingLabels']).ravel()
    if mode == 'unlabel':
        x = x["unlabeledMatrix"]
        y = np.array(y['unlabeledLabels']).ravel()
    return x, y


if __name__ == '__main__':
    train_x, train_y = read_data(filename1='MMI/trainingLabels_3.mat',
                                 filename2='MMI/trainingMatrix_3.mat',
                                 mode='train')
    test_x, test_y = read_data(filename1='MMI/testingLabels_3.mat',
                               filename2='MMI/testingMatrix_3.mat',
                               mode='test')
    unlabel_x, unlabel_y = read_data(filename1='MMI/unlabeledLabels_3.mat',
                                     filename2='MMI/unlabeledMatrix_3.mat',
                                     mode='unlabel')

    N = 50
    k = 10
    score_set = []

    for i in tqdm(range(N)):
        index = random.randint(0, 1099 - k)

        subset_x = unlabel_x[index: index + k]
        subset_y = unlabel_y[index: index + k]
        # step 1 train model
        model = LogisticRegression(solver='lbfgs', max_iter=50000).fit(train_x, train_y)
        # step 2 calculate accuracy
        score_set.append(model.score(test_x, test_y))
        # step 4 add subset
        train_x = np.append(train_x, subset_x, 0)
        train_y = np.append(train_y, subset_y, 0)
        # step 5 delete subset in original set
        unlabel_x = np.delete(unlabel_x, np.s_[index: index + k], 0)
        unlabel_y = np.delete(unlabel_y, np.s_[index: index + k], 0)

    NofI = [i for i in range(N)]

    plt.title("label data")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.plot(NofI, score_set, 'r', label='train_set')
    plt.legend()
    plt.show()
