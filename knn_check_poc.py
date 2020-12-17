from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_check(x_test_new_TP_TN, x_set_only_fp):
    """
    The function gets x_test_new_TP_TN:ndarray and x_set_only_fp:ndarray
    runs function with knn check for FP (is it closer to TP or TN from x_test_new_TP_TN)
    creates sorted dataframes with distances ,indexes and predicted labels of FP values to TN and TP
    calculates which percent of them was affiliated to TP on predicted and real sets.
    plots the barplot of the  distances from the FP feature intervals to TP and to TN
    calculates mean and median of the distances from the FP feature intervals to Class 1 and to Class 2.
    """
    # run knn
    indx, dist, pred = run_knn(x_test_new_TP_TN.iloc[:, 0:-3], x_test_new_TP_TN['label'], x_set_only_fp)
    # ________________________________________________
    # Percent of predicted  and real classes
    # _________________________________________________
    plt.figure(figsize=(6, 8))
    bar = sns.countplot(pred)
    plt.text(0.5, len(pred) * 1.1, f"Distribution of PREDICTED Classes",
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=15)
    for p in bar.patches:
        bar.annotate('{}%'.format(round(p.get_height() / len(pred), 3) * 100, '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center', xytext=(0, 10), textcoords='offset points', size=14)
    plt.xlabel('Classes')
    plt.show()

    plt.figure(figsize=(6, 8))
    bar = sns.countplot(x_test_new_TP_TN["label"])
    plt.text(0.5, len(x_test_new_TP_TN["label"]) * 1.1, "Distribution of REAL Classes",
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=15)
    for p in bar.patches:
        bar.annotate('{}%'.format(round(p.get_height() / len(x_test_new_TP_TN["label"]), 3) * 100, '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center', xytext=(0, 10), textcoords='offset points', size=14)
    plt.xlabel('Classes')
    plt.show()

    # create df with distances, indexes and pred labels
    df = pd.DataFrame(dist, index=np.hstack(indx))
    df.reset_index(inplace=True)
    df['pred'] = pred
    df.columns = ['indx', 'dist', 'pred']

    # sort values by distances
    class1 = df[df['pred'] == 1].sort_values(['dist'])['dist']
    class2 = df[df['pred'] == 2].sort_values(['dist'])['dist']

    # __________________________________________________
    # plot histogram for distances to class1 and class2
    # ____________________________________________________
    class1.hist(zorder=2, rwidth=0.9)
    plt.title(f'Histogram of the distances from FP to Class 1, neighbors = {i}', fontsize=13)
    plt.xlabel('distance from FP to Class 1', fontsize=12)
    plt.ylabel(f'item count from {len(class1)}', fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.show()

    class2.hist(zorder=2, rwidth=0.9)
    plt.title(f'Histogram of the distances from FP to Class 2, neighbors = {i}', fontsize=13)
    plt.xlabel('distance from FP to Class 2', fontsize=12)
    plt.ylabel(f'item count from {len(class2)}', fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.show()

    # print mean and median distances from FP to class1 and class2
    print(f'Mean distance from FP to Class 1: {class1.mean().round(2)}')
    print(f'Median distance from FP to Class 1: {class1.median().round(2)}')
    print(f'Mean distance from FP to Class 2: {class2.mean().round(2)}')
    print(f'Median distance from FP to Class 2: {class2.median().round(2)}')


def run_knn(X, y, fp):
    """
    The function get x_set: ndarray, y_labels, fp_set: ndarray, neighbo
    run KNeighborsClassifier with n_neighbors=2 and algorithm = 'auto'
    return euclidean distances: ndarray, indices:ndarray, predicted labels
    """
    nbrs = KNeighborsClassifier(n_neighbors=2).fit(np.nan_to_num(X), y)
    predict = nbrs.predict(np.nan_to_num(fp))

    distances, indxs = nbrs.kneighbors(np.nan_to_num(fp), return_distance=True)
    return indxs, distances, predict


if __name__ == '__main__':
    pass
