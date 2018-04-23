from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse
from utils import return_data


# parameter
def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--train_path', type=str, default='/home/enningxie/Downloads/Cell_/train')
    args.add_argument('--test_path', type=str, default='/home/enningxie/Downloads/Cell_/test')
    args.add_argument('--train_labels', type=str, default='/home/enningxie/Downloads/Cell_/train.txt')
    args.add_argument('--test_labels', type=str, default='/home/enningxie/Downloads/Cell_/test.txt')
    return args.parse_args()


def fn_func(true_y, pred_y):
    fn_num = [0] * 3
    assert len(true_y) == len(pred_y), 'error with fn_func.'
    for i in range(len(true_y)):
        if true_y[i] == 0:
            if pred_y[i] != 0:
                fn_num[pred_y[i]-1] += 1
    return fn_num


if __name__ == '__main__':
    FLAGS = parser()
    train_x, train_y, test_x, test_y = return_data(FLAGS.train_path, FLAGS.train_labels,
                                                   FLAGS.test_path, FLAGS.test_labels)
    # svm
    clf_svm = SVC()
    clf_svm.fit(train_x, train_y)
    pred_y_svm = clf_svm.predict(test_x)

    # svm evaluate
    fn_num_svm = fn_func(test_y, pred_y_svm)
    result_svm = np.array(fn_num_svm) / len(test_x) * 100
    print('svm_falseNegative: {0}%, {1}%, {2}%.'.format(result_svm[0], result_svm[1], result_svm[2]))

    # knn
    clf_knn = KNeighborsClassifier(n_neighbors=4)
    clf_knn.fit(train_x, train_y)
    pred_y_knn = clf_knn.predict(test_x)

    # knn evaluate
    fn_num_knn = fn_func(test_y, pred_y_knn)
    result_knn = np.array(fn_num_knn) / len(test_x) * 100
    print('knn_falseNegative: {0}%, {1}%, {2}%.'.format(result_knn[0], result_knn[1], result_knn[2]))



