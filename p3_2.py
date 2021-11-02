import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from p3_1 import read_data


def cal_entropy(model, sample):
    prob = model.predict_proba(sample)
    log_prob = model.predict_log_proba(sample)
    en = np.sum(-1 * prob * log_prob, axis=1)

    return en


# sort by entropy
def sorted_by_E(e, data, y):
    tuple_data = []
    for i in range(len(data)):
        tuple_data.append((data[i], y[i], e[i]))
        sorted_tuple = sorted(tuple_data, key=lambda x: x[2], reverse=True)

    return sorted_tuple


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
    acc_set = []
    for i in tqdm(range(N)):
        # step 1 train model
        LogReg = LogisticRegression(solver='lbfgs', max_iter=50000).fit(train_x, train_y)
        # step 2 calculate accuracy
        acc_set.append(LogReg.score(test_x, test_y))
        # step 3 apply model in unlabeled set to find top k subset
        Entropy = cal_entropy(LogReg, unlabel_x)
        sorted_data = sorted_by_E(Entropy, unlabel_x, unlabel_y)
        tot_num_x = np.array([list(x[0]) for x in sorted_data])
        tot_num_y = np.array([y[1] for y in sorted_data])
        subset_x = tot_num_x[:k]
        subset_y = tot_num_y[:k]
        # step 4 add subset(top k unlabeled set) to training set
        train_x = np.append(train_x, subset_x, 0)
        train_y = np.append(train_y, subset_y, 0)
        # step 5 delete subset in original set
        unlabel_x = tot_num_x[k:]
        unlabel_y = tot_num_y[k:]

    # plot
    num_of_it = [i for i in range(N)]

    plt.title("label data")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.plot(num_of_it, acc_set, 'r', label='train_set')
    plt.legend()
    plt.show()


