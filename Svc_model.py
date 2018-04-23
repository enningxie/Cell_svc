from sklearn.svm import SVC
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
    clf = SVC()
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    # convert label_set
    process_label(pred_y)
    process_label(test_y)

    confusion = confusion_matrix(test_y, pred_y)

    # cal num of fn
    fn_num = confusion[1, 0]
    result = fn_num / len(test_x) * 100
    print('fn: {0}, result: {1}%.'.format(fn_num, result))


