import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import cm
from feature_analysis import (Feature, ClassTarget,
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

def fuseCategoricalFeatures(categoricals):
    collection = CategoricalFeatureCollection(categoricals)
    new_categoricals = collection.fuse_all_categories(cat_fusion_dict)
    return new_categoricals
