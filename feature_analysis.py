import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KernelDensity

# Classes for individual features and class target
class ColumnData:
    def __init__(self, data, target=None):
        self.data = data
        self.num_samples = len(self.data)
        self.unique_values = sorted(self.data.unique())
        self.num_unique_values = len(self.unique_values)
        self._counts = self.data.value_counts()
        self._frequencies = self._counts / self.num_samples
        self.target = target

class Feature(ColumnData):
    def __init__(self, feature, target=None):
        super().__init__(feature, target=target)
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

    def calculate_IndividualClassFrequency(self, target=None):
        '''
        Create contingency table to count how many samples are in each class
        for each value of binary or categorical feature. This does not work well
        on continuous feature space.
        '''
        if self.target is None:
            if target is None:
                print('Error: Target must be set.')
            else:
                self.target = target

        contingency = pd.crosstab(self.data, self.target)
        return contingency

    def calculate_CondProba(self, target=None):
        '''
        Calcluate conditional probability of being in each clas given a certain
        value of a binary or categorical (or ordinal) feature.
        This does not work well on continuous feature space.
        '''
        if target is None:
            if self.target is None:
                print('Error: Target must be set.')
                return
        else:
            self.target = target

        contingency = self.calculate_IndividualClassFrequency(self.target)
        total = contingency.sum(axis=1)
        probas = contingency.div(total, axis=0)
        return probas

    def convert2CondProba(self, target_class=1, target=None):
        if target is None:
            if self.target is None:
                print('Error: Target must be set.')
                return
        else:
            self.target = target

        probas = self.calculate_CondProba()
        vec2proba_dict = probas[target_class].to_dict()
        feature_proba = self.data.replace(vec2proba_dict)

        return feature_proba

class OrdinalFeature(Feature):
    def __init__(self, feature):
        super().__init__(feature)
        self.feature_type = 'ordinal'

    def estimateKD(self, graph=True, ax=None, normed=True,
                   hist_bins='auto', color='skyblue', alpha=0.5,
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        '''
        KDE with statsmodels nonparametric estimator.
        '''
        kde = sm.nonparametric.KDEUnivariate(self.data.astype('float'))
        kde.fit(kernel=kernel, bw=bw, fft=fft, weights=weights,
                gridsize=gridsize, adjust=adjust, cut=cut, clip=clip)

        if hist_bins == 'auto':
            bins = self.num_unique_values
        elif type(hist_bins) == 'int':
            bins = hist_bins
        elif type(hist_bins == 'float'):
            bins = int(hist_bins)
        else:
            print('Error: bins must be an integer')

        if graph:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            if normed:
                ax.hist(self.data, bins=bins, normed=True,
                                                alpha=alpha, color=color)
                ax.plot(kde.support, kde.density, ls='--', color=color)
            else:

                ax.hist(self.data, bins=bins, normed=False,
                        alpha=alpha, color=color)
                ax.plot(kde.support,
                        self.num_samples*kde.density,
                        ls='--', color=color)
        return kde

    def estimateKD2(self, graph=True, ax=None,
                 span='auto', bandwidth=1, algorithm='auto', kernel='gaussian',
                 metric='euclidean', atol=0, rtol=0, breadth_first=True,
                 leaf_size=40, metric_params=None,
                 hist_bins='auto', color='skyblue', alpha=0.5):
        '''
        KDE with sklearn estimator.
        '''
        feature = self.data

        kde = KernelDensity(bandwidth=bandwidth, algorithm=algorithm,
                            kernel=kernel, metric=metric, atol=atol,
                            rtol=rtol, breadth_first=breadth_first,
                            leaf_size=leaf_size, metric_params=metric_params)
        kde.fit(np.array(feature).reshape(-1, 1))


        if hist_bins == 'auto':
            bins = len(feature.unique())
        elif type(hist_bins) == 'int':
            bins = hist_bins
        elif type(hist_bins == 'float'):
            bins = int(hist_bins)
        else:
            print('Error: bins must be an integer')

        if span == 'auto':
            span = np.linspace(np.min(feature.unique()),
                               np.max(feature.unique()), 30).reshape(-1, 1)
        if graph:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.hist(feature, bins=bins, normed=True,
                             color=color, alpha=alpha, label=None)
            ax.plot(span, np.exp(kde.score_samples(span)),
                             color=color, alpha=alpha, ls='--', label='Bulk')

        return kde

    def find_nearest_in_list(self, list_):
        '''
        For each entry in the feature (which is a single column),
        replace the original value with its nearest value in the list_.
        This is a utility function for KDE method to reduce number of
        processing.
        '''
        nearest = []
        for i in range(self.num_samples):
            idx = (np.abs(list_- self.data[i])).argmin()
            nearest.append(list_[idx])
        return pd.Series(nearest)

    def convertToGain(self, target, span, mode='percent',
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        '''
        Convert the feature space into space of gain in conditional probability
        for each value of the feature. It assumes that one is looking at the
        conditional probability of being in class1 (i.e. target == 1).

        statsmodel version of KDE used.
        '''
        if self.num_samples != len(target):
            print('Error: Target size must be the same as the feature size.')
            return

        class1 = OrdinalFeature(self.data[target == 1])
        bulk_size = len(target)
        class_size = np.sum(target)
        class1_freq = class_size / bulk_size

        kde_bulk = self.estimateKD(bw=bw, graph=False)
        kde_class1 = class1.estimateKD(bw=bw, graph=False)

        bulk_dens = bulk_size*kde_bulk.evaluate(np.array(span))
        class_dens = class_size*kde_class1.evaluate(np.array(span))
        proba = class_dens/bulk_dens
        if mode == 'fraction':
            gain = (proba/class1_freq) - 1
        else:
            gain = 100*((proba/class1_freq) - 1)
        conversion_dict = dict(zip(span, gain))

        converted = self.find_nearest_in_list(span).replace(conversion_dict)

        return converted

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
