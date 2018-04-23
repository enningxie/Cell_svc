from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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


# process labels 1, 2, 3 to 1
def process_label(labels):
    for i, label in enumerate(labels):
        if label != 0:
            labels[i] = 1


if __name__ == '__main__':
    FLAGS = parser()
    train_x, train_y, test_x, test_y = return_data(FLAGS.train_path, FLAGS.train_labels,
                                                   FLAGS.test_path, FLAGS.test_labels)
    # svm
    clf_svm = SVC()
    clf_svm.fit(train_x, train_y)
    pred_y_svm = clf_svm.predict(test_x)

    # knn
    clf_knn = KNeighborsClassifier(n_neighbors=4)
    clf_knn.fit(train_x, train_y)
    pred_y_knn = clf_knn.predict(test_x)

    # convert label_set
    process_label(pred_y_svm)
    process_label(pred_y_knn)
    process_label(test_y)

    confusion_svm = confusion_matrix(test_y, pred_y_svm)
    confusion_knn = confusion_matrix(test_y, pred_y_knn)

    # cal num of fn_svm
    fn_num_svm = confusion_svm[1, 0]
    result_svm = fn_num_svm / len(test_x) * 100
    print('svm_fn: {0}, svm_result: {1}%.'.format(fn_num_svm, result_svm))

    # cal num of fn_knn
    fn_num_knn = confusion_knn[1, 0]
    result_knn = fn_num_knn / len(test_x) * 100
    print('knn_fn: {0}, knn_result: {1}%.'.format(fn_num_knn, result_knn))


