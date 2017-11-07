import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import cm


def assess_feature_frequency(feature, target, mode='subtraction'):
    total = len(feature)
    num_class1 = np.sum(target)
    proba_class1 = num_class1 / total

    num_val0 = len(feature[feature == 0])
    num_val1 = len(feature[feature == 1])
    num_class1_given_val0 = len(feature[(feature == 0) & (target == 1)])
    num_class1_given_val1 = len(feature[(feature == 1) & (target == 1)])

    if num_val0 == 0:
        proba_class1_given_val0 = 0
    else:
        proba_class1_given_val0 = num_class1_given_val0 / num_val0

    if num_val1 == 1:
        proba_class1_given_val1 = 0
    else:
        proba_class1_given_val1 = num_class1_given_val1 / num_val1

    best_cond_proba = max(proba_class1_given_val0, proba_class1_given_val1)
    if mode == 'subtraction':
        differential = best_cond_proba - proba_class1
    elif mode == 'ratio':
        differential = best_cond_proba / proba_class1
    else:
        print('Error: the mode must be subtration or ratio')
    return differential

class FeatureComparison:
    '''
    Once this is implemented, Binary- and CategoricalComparison class
    will inherit from this parent class to reduce code redundancy.
    '''
    def __init__(self, feature1, feature2, target):
        self.features = pd.concat([feature1, feature2], axis=1)
        self.target = target
        self._table = sm.stats.Table.from_data(self.features)
        self.contingency_table_ = self._table.table_orig
        self.chi_result_ = self._table.test_nominal_association()
        self.chi_pvalue_ = self.chi_result_.pvalue

        self._f1 = self.features.iloc[:, 0]
        self._f2 = self.features.iloc[:, 1]

    def test_independence(self, significance_level=0.01):
        if self.chi_pvalue_ < significance_level:
            text = 'Feature association is significant (p-value=%.3f)' % self.chi_pvalue_
        else:
            text = 'Feature association is NOT significant (p-value=%.3f)' % self.chi_pvalue_
        print(text)
        return

class BinaryComparison(FeatureComparison):
    def __init__(self, feature1, feature2, target):
        self.super.__init__(self, feature1, feature2, target)



class CategoricalComparison:
    def __init__(self, feature1, feature2, target):
        self.super.__init__(self, feature1, feature2, target)
        self.category_values_ = {self._f1.name: self._f1.unique(),
                                 self._f2.name: self._f2.unique()}
        self.num_category_values_ = {self._f1.name: len(self._f1.unique()),
                                     self._f2.name: len(self._f2.unique())}
