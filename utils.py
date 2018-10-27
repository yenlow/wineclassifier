import numpy as np
import pandas as pd


def onehot_traintest(train, test=None, sparse=True):
    '''
    Converts training vector of categories to one-hot matrix (sparse or dense);
    Also converts test vector (if provided) to one-hot matrix based on training vector's categories
    :param train: training list or vector as basis of categories
    :param test: test list or vector with same categories as train
    :param sparse: True for sparse output (Default: False)
    :return: output np array with indicator values
    '''
    from sklearn.preprocessing import LabelBinarizer

    encoder = LabelBinarizer(sparse_output=sparse)
    int_train = encoder.fit_transform(train)
    num_classes = len(encoder.classes_)
    if test is not None:
        int_test = encoder.transform(test)
        return int_train, int_test, num_classes
    else:
        return int_train, num_classes




def rare2other(pdseries, min_sparsity=0.01,
               newvalue='other', missing='other',
               verbose=False, plot=False, plotn=20, type=None):
    '''
    Groups rare values into a new value (e.g. 'other') of specified type (e.g. 'category')
    Optionally replaces missing value (e.g. 'other')
    Optionally plots histogram of plotn most frequent values
    :param pdseries: input pd.series with rare values to be replaced
    :param min_sparsity: threshold below which values will be replaced with newvalue (Default: 0.01)
    :param newvalue: newvalue replacing rare values (Default:'other')
    :param missing: value replacing missing values (Default:'other')
    :param verbose: True to print frequency counts before and after replacement (Default: False)
    :param plot: True to output histogram plot (Default: False)
    :param plotn: Number of categories in histogram (Default: 20)
    :param type: dtype e.g. 'category', 'object', 'float64' (Default: None, i.e. as is)
    :return: output pd.Series with rare values replaced
    '''
    import pandas as pd
    import matplotlib.pyplot as plt

    cnt = pdseries.value_counts(normalize=True)
    rarecat = cnt[cnt<min_sparsity].index
    ans = pdseries.fillna(missing)
    if len(rarecat)>0:
        ans.replace(rarecat, newvalue, inplace=True)

    if type is not None:
        ans = ans.astype(type)

    cnt_new = ans.value_counts(normalize=True)

    if verbose:
        print("Before:")
        print(cnt)
        print("\nAfter:")
        print(cnt_new)
        print("%d Categories" % len(cnt_new))
        print("%d Null values\n" % ans.isnull().sum())

    if plot:
        plotn = min(len(cnt_new),plotn)
        cnt_new[:plotn][::-1].plot.barh(figsize=(10, 10))
        plt.show()

    return ans
