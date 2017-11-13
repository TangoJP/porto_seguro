import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from feature_analysis import (Feature, ClassTarget, OrdinalFeature,
                              CategoricalFeatureCollection,
                              FeatureVsTarget,
                              BinaryComparison, CategoricalComparison)

cat_fusion_dict = {
 'ps_car_01_cat': [[6, 7], [3, 4, 5, 10], [2, 8, 11], [0, 1], [9], [-1]],
 'ps_car_02_cat': [[-1, 1], [0]],
 'ps_car_03_cat': [[-1], [0], [1]],
 'ps_car_04_cat': [[0, 4], [1, 2, 3, 8], [6, 9], [5], [7]],
 'ps_car_05_cat': [[-1], [0, 1]],
 'ps_car_06_cat': [[0, 1, 4, 11, 14], [3, 6, 7], [10, 12, 16], [9, 13, 15],
                   [2, 5, 8, 17]],
 'ps_car_07_cat': [[1], [0], [-1]],
 'ps_car_08_cat': [[1], [0]],
 'ps_car_09_cat': [[0, 2, 3], [1, 4], [-1]],
 'ps_car_10_cat': [[0], [1], [2]],
 'ps_car_11_cat': [[43],
                   [7, 9, 10, 11, 16, 19, 32, 39, 42, 44, 57, 62, 64, 66, 67,
                    82, 95, 99, 103],
                   [15,22,26,27,29,30,37,38,40,48,49,52,53,59,60,65,68,73,74,
                    76,77,84,85,86,87,88,92,96,98,102],
                   [1,2,5,6,8,12,14,23,24,25,28,31,34,35,36,46,47,51,54,70,78,
                    80,81,83,91,101],
                   [13, 17, 20, 45, 50, 79, 89, 90, 94, 104],
                   [3, 33, 55, 56, 61, 69, 71, 72, 100],
                   [4, 21, 58, 63, 75, 93, 97],
                   [18, 41]],
 'ps_ind_02_cat': [[1], [2, 3], [4], [-1]],
 'ps_ind_04_cat': [[0], [1], [-1]],
 'ps_ind_05_cat': [[0], [3], [1, 4, 5], [6], [-1, 2]]
}

ordinal_bw_dict = {
    'ps_calc_05': ,
    'ps_calc_06': ,
    'ps_calc_07': ,
    'ps_calc_10': ,
    'ps_calc_11': ,
    'ps_calc_12': ,
    'ps_calc_13': ,
    'ps_calc_14': ,
    'ps_car_11': ,
    'ps_ind_01': ,
    'ps_ind_03': ,
    'ps_ind_14': ,
    'ps_ind_15': ,
    'ps_reg_01': ,
    'ps_reg_02':
}


def fuseCategoricalFeatures(categoricals, dictionary='categorical'):
    collection = CategoricalFeatureCollection(categoricals)
    if dictionary == 'categorical':
        new_categoricals = collection.fuse_all_categories(cat_fusion_dict)
    elif dictionary == 'ordinal':
        new_categoricals = collection.fuse_all_categories(ordinal_fusion_dict)
    else:
        print('Error: Invalid dictionary option')
    return new_categoricals

def compareKDEs(bulk_kde, class_kde, span, level=1,
                        label=None, graph='gain', ax=None, output='both'):
    '''
    Calculates conditional probability and its ratio to the bulk frequecy
    of the class. It can output cond probability, its ratio to the bulk
    frequency, or both. If graph option is true, it plots the result
    '''
    kde1_size = len(bulk_kde.density)
    kde2_size = len(class_kde.density)
    density1 = kde1_size* bulk_kde.evaluate(span)
    density2 = kde2_size* class_kde.evaluate(span)
    cond_proba_class1_given_val = density2 / density1
    percent_gain = 100*((cond_proba_class1_given_val/level) - 1)

    if output == 'proba':
        results = cond_proba_class1_given_val
    elif output == 'gain':
        results = percent_gain
    elif output == None:
        results = None
    else:
        results = cond_proba_class1_given_val, percent_gain

    if graph is not None:
        if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if graph == 'proba':
            ax.axhline(y=level, color='0.8', ls='--', label='Comparison Level')
            ax.plot(span, cond_proba_class1_given_val, label=label)
            ax.set_title('Conditional Probability')
            y_label = 'Cond Proba of Class1 Given Value'
        else:
            ax.axhline(y=0, color='0.8', ls='--')
            ax.plot(span, percent_gain, label=label)
            ax.set_title('Percentange Gain from Bulk Frequency')
            y_label = '%% Gain from Bulk Class1 Freq'
        ax.set_ylabel(y_label)
        ax.set_xlabel('Feature Value')

    return results

def compareKDEs2(bulk_kde, class_kde, bulk_size, target_size,
                        span, level=1,
                        label=None, graph='gain', ax=None, output='both'):
    '''
    sklearn version
    Calculates conditional probability and its ratio to the bulk frequecy
    of the class. It can output cond probability, its ratio to the bulk
    frequency, or both. If graph option is true, it plots the result
    '''
    density1 = bulk_size * \
                np.exp(bulk_kde.score_samples(np.array(span).reshape(-1, 1)))
    density2 = target_size * \
                np.exp(class_kde.score_samples(np.array(span).reshape(-1, 1)))
    cond_proba_class1_given_val = density2 / density1
    percent_gain = 100*((cond_proba_class1_given_val/level) - 1)

    if output == 'proba':
        results = cond_proba_class1_given_val
    elif output == 'gain':
        results = percent_gain
    elif output == None:
        results = None
    else:
        results = cond_proba_class1_given_val, percent_gain

    if graph is not None:
        if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if graph == 'proba':
            ax.axhline(y=level, color='0.8', ls='--', label='Comparison Level')
            ax.plot(span, cond_proba_class1_given_val, label=label)
            ax.set_title('Conditional Probability')
            y_label = 'Cond Proba of Class1 Given Value'
        else:
            ax.axhline(y=0, color='0.8', ls='--')
            ax.plot(span, percent_gain, label=label)
            ax.set_title('Percentange Gain from Bulk Frequency')
            y_label = '%% Gain from Bulk Class1 Freq'
        ax.set_ylabel(y_label)
        ax.set_xlabel('Feature Value')

    return results

def myOrdinalFeatureAnalaysis1(feature, target, span, level=1, normed=True,
                                kernel='gau', bw=0.08, fft=True, hist_bins=10):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    kde_bulk = OrdinalFeature(feature).estimateKD(kernel=kernel, bw=bw, fft=fft,
                        graph=True, normed=normed, ax=ax1, hist_bins=hist_bins)
    kde_class1 = OrdinalFeature(feature[target == 1]).estimateKD(normed=normed,
                        kernel=kernel, bw=bw, fft=fft, ax=ax1,
                        graph=True, hist_bins=hist_bins, alpha=0.2, color='red')

    compareKDEs(kde_bulk, kde_class1, span, level=level,
                                 label=None, graph='proba', ax=ax2, output=None)
    compareKDEs(kde_bulk, kde_class1, span, level=level,
                                 label=None, graph='gain', ax=ax3, output=None)
    plt.tight_layout()
    return

def myOrdinalFeatureAnalaysis2(feature, target, span, level=1, normed=True,
                            bandwidth=1, algorithm='auto', kernel='gaussian',
                            metric='euclidean', atol=0, rtol=0,
                            breadth_first=True, leaf_size=40, metric_params=None,
                            hist_bins='auto'):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    kde_bulk = OrdinalFeature(feature).estimateKD2(
                        bandwidth=bandwidth, algorithm=algorithm, kernel=kernel,
                        metric=metric, atol=atol, rtol=rtol,
                        breadth_first=breadth_first, leaf_size=leaf_size,
                        metric_params=metric_params,
                        ax=ax1, hist_bins=hist_bins)

    kde_class1 = OrdinalFeature(feature[target == 1]).estimateKD2(
                        bandwidth=bandwidth, algorithm=algorithm, kernel=kernel,
                        metric=metric, atol=atol, rtol=rtol,
                        breadth_first=breadth_first, leaf_size=leaf_size,
                        metric_params=metric_params,
                        ax=ax1, hist_bins=hist_bins,
                        alpha=0.2, color='red')

    bulk_size = len(feature)
    class_size = np.sum(target)

    compareKDEs2(kde_bulk, kde_class1, bulk_size, class_size, span,
                    level=level, label=None, graph='proba', ax=ax2, output=None)
    compareKDEs2(kde_bulk, kde_class1, bulk_size, class_size, span,
                    level=level, label=None, graph='gain', ax=ax3, output=None)
    plt.tight_layout()
    return

def convertOrdinalFeatures():
    pass
