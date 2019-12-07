from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import util


def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = util.normalize(X_train, X_test)
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        """
        conf_mat = np.zeros((2, 2))
        for i in range(len(X_test)):
            pred = clf.predict(X_test[i])
            true = y_test[i]
            conf_mat[true][pred] += 1
        print(conf_mat)
        """
        predictions = clf.predict(X_test)
        conf_mat = confusion_matrix(y_test, predictions)
        print(conf_mat)

if __name__ == '__main__':
    main()
