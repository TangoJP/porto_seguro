import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def create_contingency_table(x, y, output=True, percent=False, printout=False):
    #if (x.shape[0] != 1) or (x.shape[0] != 1):
    #    print('Error: inputs must be column vectors')
    if len(x) != len(y):
        print('Error: inputs must be of the same length')
    total = len(x)
    
    xy = pd.DataFrame()
    xy['x'] = x
    xy['y'] = y
    
    zeroF_zeroT = len(xy[(xy.x == 0) & (xy.y == 0)])
    zeroF_oneT = len(xy[(xy.x == 0) & (xy.y == 1)])
    oneF_zeroT = len(xy[(xy.x == 1) & (xy.y == 0)])
    oneF_oneT = len(xy[(xy.x == 1) & (xy.y == 1)])
    unknown = total - (zeroF_zeroT + zeroF_oneT + oneF_zeroT + oneF_oneT)
    
    f0_frequency = zeroF_zeroT + zeroF_oneT
    f1_frequency = oneF_zeroT + oneF_oneT
    contingency = np.array([zeroF_zeroT, zeroF_oneT, oneF_zeroT, oneF_oneT, unknown])
                   
    if percent:
        f0_frequency = 100*(f0_frequency/total)
        f1_frequency = 100*(f1_frequency/total)
        contingency = 100*(contingency/total)
        total = 1
        
    if printout:
        if percent:
            print('Feature Distribution: f0=%3.1f%% f1=%3.1f%%' % (f0_frequency, f1_frequency))
            print('feature=0, class=0: %4.1f%%' % (contingency[0]))
            print('feature=0, class=1: %4.1f%%' % (contingency[1]))
            print('feature=1, class=0: %4.1f%%' % (contingency[2]))
            print('feature=1, class=1: %4.1f%%' % (contingency[3]))
            print('Unknown Data      : %4.1f' % (contingency[4]))
            
        else:
            print('Feature Distribution: f0=%7d f1=%7d' % (f0_frequency, f1_frequency))
            print('feature=0, class=0: %7d /%7d' % (contingency[0], total))
            print('feature=0, class=1: %7d /%7d' % (contingency[1], total))
            print('feature=1, class=0: %7d /%7d' % (contingency[2], total))
            print('feature=1, class=1: %7d /%7d' % (contingency[3], total))
            print('Uknown Data       : %7d /%7d' % (contingency[4], total))   

    if output:
        return contingency
    else:
        return


def calculate_conditional_prob_bin(contingency, output=True, printout=False):
    prob_class1_cond_feature0 = contingency[1]/np.sum(contingency[:2])
    prob_class1_cond_feature1 = contingency[3]/np.sum(contingency[2:])
    if printout:
        print('p(class=1|feature=0)=%.4f' % prob_class1_cond_feature0)
        print('p(class=1|feature=1)=%.4f' % prob_class1_cond_feature1)
    if output:
        return np.array([prob_class1_cond_feature0, prob_class1_cond_feature1])
    else:
        return

    
def encode_my_categorical_labels(categorical_feature_vector):
    '''
    Custome version of categorical label encoding. Takes a pd.Series of 
    a categorical feature. Creates a new DataFrame whose columnes 
    designate each category (including missing value) in the Series. 
    For each category, inser 1 if the sample is of that category, 
    otherwise insert 0.
    '''
    encoded = pd.DataFrame()
    feature = categorical_feature_vector.name
    categories = sorted(categorical_feature_vector.unique())
    for cat in categories:
        if cat == -1:
            cat_name = feature + '_NaN'
        else:
            cat_name = feature + '_' + str(cat)
        encoded[cat_name] = (categorical_feature_vector == cat).astype('int')
    return encoded


def calculate_conditional_prob_cat(categorical_feature_vector, target_vector, 
                                   output=True, printout=False):
    '''
    Custom encodes the categorical feature column (see encode_my_categorical_labels() method).
    Then for each encoded category, calculte the conditional probability of sample being in
    class1 (class information is stored in target_vector input. Therefore, the two input
    vectors must be of the same length and aligned) given the category
    (i.e p(class=1|category)). Output and printout options are there as output options.
    The output is returned as DataFrame
    '''
    # Create a dict to store cond. probs
    conditional_probs = {}
    
    # Custome encode the feature column (pd.Series)
    encoded = encode_my_categorical_labels(categorical_feature_vector)
    
    # Calculate conditional probability of being in classes 0 and 1 give category
    for category in encoded.columns:
        data = encoded[category]
        cat0 = len(data[data == 0])
        cat1 = len(data[data == 1])
        class1_given_cat0 = len(data[(data == 0) & (target_vector == 1)])
        class1_given_cat1 = len(data[(data == 1) & (target_vector == 1)])
        
        prob_class1_cond_cat0 = class1_given_cat0 / cat0
        prob_class1_cond_cat1 = class1_given_cat1 / cat1
        probs = np.array([prob_class1_cond_cat0, prob_class1_cond_cat1])
        
        conditional_probs[category] = probs
        
        if printout:
            print('p(class=1|%s=0)=%.4f' % (category, probs[0]))
            print('p(class=1|%s=1)=%.4f' % (category, probs[1]))
    
    result = pd.DataFrame(conditional_probs).T
    result = result.rename(columns={0: 'p(class1|category=0)', 1: 'p(class1|category=1)'})
        
    if output:
        return result
    else:
        return

    
def estimate_cond_prob_density(feature_vector, target_vector, printout=False, output=True):
    '''
    Calculate the probability of being class1 given a certain value of the 
    comtinuous/ordinal feature. It's almost identical to the function used
    for categorical features, except that each sample takes a certain value
    in a range here rather than being of a certain category. In this case,
    even the 'continous' features take quantum values (i.e. only increases)
    by a multiple of the same increment), so they can be treated analogously
    as in the category function.
    '''
    # Get the values the feature can take
    feature_space = sorted(feature_vector.unique())
    
    # Create a list to store cond. probs
    conditional_probs = []
    
    # Calculate conditional probability of being in classes 0 and 1 given
    # a certain value of the feature
    for val in feature_space:
        data = pd.concat([feature_vector, target_vector], axis=1).dropna()
        data_val = len(data[data.iloc[:, 0] == val])
        class1_given_val= len(data[(data.iloc[:, 0] == val) & (data.iloc[:, 1] == 1)])
        if data_val == 0:
            prob_class1_given_val = 0
        else:
            prob_class1_given_val = class1_given_val / data_val
        
        conditional_probs.append(prob_class1_given_val)
        
        if printout:
            print('p(class=1|%.1f=1)=%.4f' % (val, prob_class1_given_val))
    
    feature_name = feature_vector.name + '_value'
    result = pd.DataFrame({feature_name: feature_space, 'P(class1|value)': conditional_probs})
    result = result[[feature_name, 'P(class1|value)']]
    
    if output:
        return result
    else:
        return


def bin_myFeature(feature_vector, bin_min, bin_max, bins=10):
    '''
    Takes in a pd.Series, and convert the continuous variables there into digitized/binned
    semi-continuous/ordinal series. bin_min and bin_max define the ends of the available bins
    and 'bins' paramter is the number of bins to create. Each entry is converted to the
    minimum value of the range of the bin it belongs to. Which bin label/category it belongs
    to (i.e. index for bins, starting with 1. Designated as 'inds') is also returned along 
    with the converted feature (designated as 'binned_feature')
    '''
    x = np.array(feature_vector)
    
    # Creating bins
    bin_size = (bin_max - bin_min)/bins
    bin_cutoffs = []
    start = bin_min
    while start < bin_max:
        bin_cutoffs.append(start)
        start += bin_size
    
    #Create vecor containing the bin label/category for each x entry
    x_binned_inds = np.digitize(x, bin_cutoffs)
    
    # Replaced the original value with minimum of the bin.
    x_binned = [bin_cutoffs[x_binned_inds[i]-1] for i in range(len(x_binned_inds))]
    
    # Convert the arrays into pd.Series
    inds =  pd.Series(x_binned_inds, name=(feature_vector.name + '_binned_index'), index=feature_vector.index)
    binned_feature = pd.Series(x_binned, name=(feature_vector.name + '_binned'), index=feature_vector.index)
    
    # for testing purpose
    #print(len(bin_cutoffs))
    #print(len(binned_feature.unique()))
    #print(len(set(x_binned_inds)))
    #print(x.shape)
    #print(binned_feature.shape)
    #print(inds.shape)
    
    return inds, binned_feature

