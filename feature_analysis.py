import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import cm

# Classes for individual features and class target
class ColumnData:
    def __init__(self, data):
        self.data = data
        self.num_samples = len(self.data)
        self.unique_values = sorted(self.data.unique())
        self.num_unique_values = len(self.unique_values)
        self._counts = self.data.value_counts()
        self._frequencies = self._counts / self.num_samples

class Feature(ColumnData):
    def __init__(self, feature):
        super().__init__(feature)
        self.value_counts_ = self._counts
        self.value_frequencies_ = self._frequencies
        if self.num_unique_values == 2:
            self.feature_type = 'binary'
        elif self.num_unique_values > 2:
            self.feature_type = 'categorical'
        else:
            self.feature_type = 'unknown feature type'

    def fuse_categories(self, new_category_list):
        new_feature = self.data.copy()
        for i, list_ in enumerate(new_category_list):
            new_feature[self.data.isin(list_)] = i
        return new_feature

class ClassTarget(ColumnData):
    def __init__(self, target):
        super().__init__(target)
        self.class_counts_ = self._counts
        self.class_frequencies_ = self._frequencies

class FeatureCollection:
    def __init__(self, df):
        self.data = df
        self.feature_names = df.columns
        self.collection = {feature: Feature(df[feature]) \
                                 for feature in self.feature_names}

class CategoricalFeatureCollection(FeatureCollection):
    def __init__(self, df):
        super().__init__(df)

    def fuse_all_categories(self, dict_new_categories):
        new_categoricals = self.data.copy()
        for key, val in dict_new_categories.items():
            new_categoricals[key] = self.collection[key].fuse_categories(val)
        return new_categoricals

# Classes for comparing features with classes and among each other
class FeatureVsTarget:
    def __init__(self, feature, target):
        # Insert here to assert isinstance
        #
        self.feature = Feature(feature)
        self.target = ClassTarget(target)
        self._warning = False
        if self.feature.num_samples != self.target.num_samples:
            print('WARNING: Feature and Target lengths not the same.')
            self._warning = True
            return
        else:
            self.num_samples = self.feature.num_samples

        self.contingency_table_ = \
                            pd.crosstab(self.feature.data, self.target.data)
        self.frequency_table_ = self.contingency_table_ / self.num_samples
        self.conditional_probas_ = \
                 self.contingency_table_.div(self.feature.value_counts_, axis=0)

    def calculate_deviation(self, classes='all', mode='subtraction'):
        if self._warning:
            print('ERROR: Feature and Target lengths must be the same.')
            return

        table = self.conditional_probas_.copy()
        for key, val in self.target.class_frequencies_.to_dict().items():
            if mode == 'ratio':
                table[key] = table[key] / val
            else:
                table[key] = table[key] - val

        minmax = pd.DataFrame()
        minmax['max'] = table.max(axis=0)
        minmax['min'] = table.min(axis=0)

        return table, minmax

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

class BinaryComparison:
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
            text = 'Feature association is significant (p-value=%.3f)' \
                                                            % self.chi_pvalue_
        else:
            text = 'Feature association is NOT significant (p-value=%.3f)' \
                                                            % self.chi_pvalue_
        print(text)
        return


    def calculate_individual_probas(self):
        fs = [self._f1, self._f2]
        fs_labels = ['feature1_proba', 'feature2_proba']
        individual_probas = {}
        for i, f in enumerate(fs):
            num_val0 = len(f[f == 0])
            num_val1 = len(f[f == 1])

            num_class1_given_val0 = len(f[(f == 0) & (self.target == 1)])
            num_class1_given_val1 = len(f[(f == 1) & (self.target == 1)])
            try:
                proba_class1_given_val0 = num_class1_given_val0 / num_val0
            except ZeroDivisionError:
                proba_class1_given_val0 = 0

            try:
                proba_class1_given_val1 = num_class1_given_val1 / num_val1
            except ZeroDivisionError:
                proba_class1_given_val1 = 0

            individual_probas[fs_labels[i]] = \
                            (proba_class1_given_val0, proba_class1_given_val1)

        return individual_probas


    def calculate_join_probas(self):
        data = pd.concat([self.features, self.target], axis=1)
        f10_f20 = len(data[(data.iloc[:, 0] == 0) & (data.iloc[:, 1] == 0)])
        f11_f20 = len(data[(data.iloc[:, 0] == 1) & (data.iloc[:, 1] == 0)])
        f10_f21 = len(data[(data.iloc[:, 0] == 0) & (data.iloc[:, 1] == 1)])
        f11_f21 = len(data[(data.iloc[:, 0] == 1) & (data.iloc[:, 1] == 1)])
        class1_given_f10_f20 = len(data[(data.iloc[:, 0] == 0) & \
                            (data.iloc[:, 1] == 0) & (data.iloc[:, 2] == 1)])
        class1_given_f11_f20 = len(data[(data.iloc[:, 0] == 1) & \
                            (data.iloc[:, 1] == 0) & (data.iloc[:, 2] == 1)])
        class1_given_f01_f21 = len(data[(data.iloc[:, 0] == 0) & \
                            (data.iloc[:, 1] == 1) & (data.iloc[:, 2] == 1)])
        class1_given_f11_f21 = len(data[(data.iloc[:, 0] == 1) & \
                            (data.iloc[:, 1] == 1) & (data.iloc[:, 2] == 1)])

        try:
            proba_00 = class1_given_f10_f20 / f10_f20
        except ZeroDivisionError:
            proba_00 = 0

        try:
            proba_10 = class1_given_f11_f20 / f11_f20
        except ZeroDivisionError:
            proba_10 = 0

        try:
            proba_01 = class1_given_f01_f21 / f10_f21
        except ZeroDivisionError:
            proba_01 = 0

        try:
            proba_11 = class1_given_f11_f21 / f11_f21
        except ZeroDivisionError:
            proba_11 = 0

        join_proba_table = pd.DataFrame(
                   {self._f1.name: [0, 1, 0, 1],
                    self._f2.name: [0, 0, 1, 1],
                    'join_proba': [proba_00, proba_10, proba_01, proba_11]}
                    )

        return join_proba_table


    def assess_joint_result(self, mode='ratio'):
        individual_probas = self.calculate_individual_probas()
        ind_probas = [i
                      for k, v in individual_probas.items()
                      for i in v
                     ]
        joint_probas = self.calculate_join_probas()

        best_ind_probas = np.max(ind_probas)
        best_joint_probas = joint_probas['join_proba'].max()

        if mode == 'ratio':
            gain = best_joint_probas / best_ind_probas
        elif mode == 'subtraction':
            gain = best_joint_probas - best_ind_probas
        else:
            print('Error: mode has to be ratio or subtraction')

        return gain

class CategoricalComparison:
    def __init__(self, feature1, feature2, target):
        self.features = pd.concat([feature1, feature2], axis=1)
        self.target = target
        self._table = sm.stats.Table.from_data(self.features)
        self.contingency_table_ = self._table.table_orig
        self.chi_result_ = self._table.test_nominal_association()
        self.chi_pvalue_ = self.chi_result_.pvalue

        self._f1 = self.features.iloc[:, 0]
        self._f2 = self.features.iloc[:, 1]
        self.category_values_ = {self._f1.name: self._f1.unique(),
                                 self._f2.name: self._f2.unique()}
        self.num_category_values_ = {self._f1.name: len(self._f1.unique()),
                                     self._f2.name: len(self._f2.unique())}

    def test_independence(self, significance_level=0.01):
        if self.chi_pvalue_ < significance_level:
            text = 'Feature association is significant (p-value=%.3f)' \
                                                            % self.chi_pvalue_
        else:
            text = 'Feature association is NOT significant (p-value=%.3f)' \
                                                            % self.chi_pvalue_
        print(text)
        return

    def calculate_individual_probas(self):
        ind_probas_f1 = pd.DataFrame()
        ind_probas_f1['total_count'] = self._f1.value_counts()
        ind_probas_f1['class1_count'] = \
                                    self._f1[self.target == 1].value_counts()
        ind_probas_f1['proba_class1_given_val'] = \
                    ind_probas_f1['class1_count'] / ind_probas_f1['total_count']

        ind_probas_f2 = pd.DataFrame()
        ind_probas_f2['total_count'] = self._f2.value_counts()
        ind_probas_f2['class1_count'] = \
                                    self._f2[self.target == 1].value_counts()
        ind_probas_f2['proba_class1_given_val'] = \
                    ind_probas_f2['class1_count'] / ind_probas_f2['total_count']

        probas = pd.DataFrame()
        probas[(self._f1.name + '_probas')] = \
                                        ind_probas_f1['proba_class1_given_val']
        probas[(self._f2.name + '_probas')] = \
                                        ind_probas_f2['proba_class1_given_val']

        return {self._f1.name: ind_probas_f1,
                self._f2.name: ind_probas_f2,
                'probas': probas}

    def calculate_join_probas(self):
        total_contingency = pd.crosstab(self._f1, self._f2)
        class1_contingency = \
            pd.crosstab(self._f1[self.target == 1], self._f2[self.target == 1])
        joint_probas = class1_contingency / total_contingency
        return joint_probas

    def assess_joint_result(self, mode='ratio', printout=False):
        ind_probas = self.calculate_individual_probas()['probas']
        joint_probas = self.calculate_join_probas()

        best_ind_probas = ind_probas.replace({np.NaN: 0}).values.max()
        best_joint_probas = joint_probas.replace({np.NaN: 0}).values.max()

        if mode == 'ratio':
            gain = best_joint_probas / best_ind_probas
        elif mode == 'subtraction':
            gain = best_joint_probas - best_ind_probas
        else:
            print('Error: mode has to be ratio or subtraction')

        if printout:
            print('Max Individual Probability=%f' % best_ind_probas)
            print('Max Joint Probability=%f' % best_joint_probas)

        return gain
