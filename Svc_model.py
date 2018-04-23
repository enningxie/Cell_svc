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


if __name__ == '__main__':
    FLAGS = parser()
    train_x, train_y, test_x, test_y = return_data(FLAGS.train_path, FLAGS.train_labels,
                                                   FLAGS.test_path, FLAGS.test_labels)
    clf = SVC()
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    print(pred_y[:5])
    # confusion = confusion_matrix(test_y, pred_y)
    # print('FN:', confusion[1, 0])

