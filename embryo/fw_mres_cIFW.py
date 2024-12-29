### Final CONTINUOUS IFW ###

import pandas as pd
import numpy as np
from functools import partial
from os.path import exists
import multiprocess
import scipy.sparse
import matplotlib.pyplot as plt
from p_tqdm import p_map
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.feature_selection import mutual_info_regression
import random

print('\nLoading data AND FILTERING.......')


# FIRST LOAD THE DATA WITH THE FILTERING FROM CESFW
# FIRST LOAD THE DATA WITH THE FILTERING FROM CESFW
raw_counts = pd.read_csv("Human_Embryo_Counts.csv",header=0,index_col=0)
#cESFW_subset = pd.read_csv('subsetted_genes_new_cESFW_embryo.csv', header=0,index_col=0)
#raw_counts_1 = raw_counts.loc[:, cESFW_subset]

subsetted_cESFW_matrix = pd.read_csv('subsetted_matrix_new_cESFW_embryo.csv',header=0,index_col=0)

cESFW_subset = subsetted_cESFW_matrix.columns

subsetted_cESFW_counts = raw_counts.loc[:,cESFW_subset.values]
#Feature_IDs = cESFW_subset.values

subsetted_cESFW_counts.to_csv('cESFW-filtered_count-matrix.csv')

# Normalisation function
def clip_counts(count_matrix, percentile = 97.5, full_distribution = False, normalise = False):
    count_matrix = count_matrix.astype(int)
    if full_distribution == False:
        #we only clip considering nonzero
        Upper = np.percentile(count_matrix[count_matrix > 0],percentile,axis=0) #get upper percentile of each gene (column)
    else:
        Upper = np.percentile(count_matrix,97.5,axis=0) #get upper percentile of each gene (column)

    count_matrix = count_matrix.clip(upper=Upper,axis=1) # this is for columns 
    # check that there are more than 2 unique counts, needed for entropy calculation
    for feature in range(count_matrix.shape[1]):
        if len(np.unique(count_matrix.iloc[:, feature])) < 3:
            raise ValueError(f'There are only 1 nonzero count in feature {feature}. Please change clipping parameters.')
    
    if normalise == True:
        count_matrix = count_matrix / np.max(count_matrix, axis = 0)
    return count_matrix

normalised_matrix = clip_counts(count_matrix=subsetted_cESFW_counts)
column_name = normalised_matrix.columns
Feature_IDs = column_name

############# NOW LOAD AND FILTER WITH CHISQ METHOD

chisq_subsetted_counts = pd.read_csv('chisq-filtered_count-matrix.csv', header=0,index_col=0)

normalised_matrix_2 = clip_counts(count_matrix=chisq_subsetted_counts)

column_name_2 = normalised_matrix_2.columns
Feature_IDs_2 = column_name_2

print(f'\nShape of cESFW filter is {normalised_matrix.shape} and of chisq filter is {normalised_matrix_2.shape}.')


print('\nLoading and filtering complete. Proceeding with ESS calculation.')

############################################################################################

print('\nProceeding with calculting CONTINUOUS ESS using sklearn MI.')

# MI and continuous ESS functions
## Now ESS using hvec
### implement this check in the function
def calculate_entropy(feature_ind, count_matrix, alternative_entropy = False, evaluation_check = False):
    feature = np.array(count_matrix.iloc[:,feature_ind]) 
    if isinstance(feature, np.ndarray) == False or np.issubdtype(feature.dtype,np.integer) == False:
        feature = np.array(feature.astype(int))
    
    if len(np.unique(feature)) < 3:
        raise ValueError(f"\nThere must be more than two unique counts for kde estimation. Please ensure the count matrix is being provided or change normalisation (clipping).")
    
    p0 = np.sum(feature == 0) / len(feature) if len(feature) > 0 else 0
    # Probability of zero
    non_zero_data = feature[feature != 0]

    # KDE for the non-zero part
    kde = gaussian_kde(non_zero_data)

    # Get the normalization factor by integrating KDE over the range of non-zero values
    #kde_normalisation_factor, _ = quad(lambda x: kde(x), 0, max(feature))

    # Combined PDF
    def combined_pdf(x):
        if x == 0:
            return p0 #/ kde_normalisation_factor
        else:
            return (1 - p0) * kde(x) #/ kde_normalisation_factor # sometimes we don't need it, but it helps

    integration_result, _ = quad(combined_pdf, 0, max(feature))

    # Check normalization (should be close to 1)
    normalisation_result = integration_result + p0  # Add p0 for the zero-inflated part
    if normalisation_result > 1 or normalisation_result < 0.95:
        print(f'\nNormalisation check unsuccessful. It should be close to 1 and it equals {normalisation_result}. This can lead to negative entropies. Proceeding with using summation instead of integration.')

    
    # Safe logarithm function
    def safe_log(x):
        return np.log(x) if x > 0 else 0

    # Combined PDF with log for entropy calculation
    def combined_pdf_log(x):
        fx = combined_pdf(x)
        return -fx * safe_log(fx)
    
    entropy_non_zero, _ = quad(combined_pdf_log, 0, max(feature))
    entropy_zero = -p0 * safe_log(p0)
    entropy = entropy_non_zero + entropy_zero

    if alternative_entropy == False:
        return entropy
    
    if alternative_entropy == True:
        # alternative entropy estimation: use sums
        x_values = np.arange(0, max(feature) + 1)
        pdf_log_values = [combined_pdf_log(x) for x in x_values]
        entropy_estimate = np.sum(pdf_log_values)
        return entropy, entropy_estimate

    if evaluation_check == True:
        print(f'\nNormalisation check, it should be close to 1: {normalisation_result}.')
        x_values = np.arange(0, max(feature) + 1)
        pdf_values = [combined_pdf(x) for x in x_values[0:np.random.randint(0,len(x_values), 10)]]
        log_values = [safe_log(fx) for fx in pdf_values[0:np.random.randint(0,len(x_values), 10)]]
        print("\n10 random x values:", x_values[0:np.random.randint(0,len(x_values), 10)])
        print("PDF values:", pdf_values[0:np.random.randint(0,len(x_values), 10)])
        print("Log values:", log_values[0:np.random.randint(0,len(x_values), 10)])

print(f'\nCalculating entropies...')

def Parallel_Calculate_entropy(working_matrix, Use_Cores=30):
    ## Provide indicies for parallel computing.
    global clipped_matrix
    clipped_matrix = working_matrix
    Feature_Inds = np.arange(clipped_matrix.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    #if Use_Cores == -1:
    #    Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
    #    if Use_Cores < 1:
    #        Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    print('\nCalculating entropy in parallel.')
    def initialize_worker(seed = 42):
        np.random.seed(seed)
        random.seed(seed)
    seed = 42
    p_map(initialize_worker, [seed]*Use_Cores)
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        entropies = p_map(partial(calculate_entropy, count_matrix=clipped_matrix), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(entropies)
    # Return results
    return results

########### Calculate entropies for both subsets...

print('\nCalculating entropies...')
entropy_vector = Parallel_Calculate_entropy(normalised_matrix)
entropy_vector = np.array(entropy_vector)
entropy = pd.Series(data = entropy_vector)
entropy.to_csv('embryo_entropy_cESFW-subset.csv')

entropy_vector = Parallel_Calculate_entropy(normalised_matrix_2)
entropy_vector = np.array(entropy_vector)
entropy = pd.Series(data = entropy_vector)
entropy.to_csv('embryo_entropy_chisq-subset.csv')


# Then have ESS use the entropy function
def continuous_ESS(feature_ind, count_matrix, entropy_vector, random_state = 42):
    f1 = np.array(count_matrix.iloc[:,feature_ind])
    num_genes = count_matrix.shape[1]
    cESS_vector = []
    MI_vector = []
    
    for i in range(num_genes):
        #define feature 2
        f2 = np.array(count_matrix.iloc[:,i])
        if len(f1) != len(f2):
                print("Fixed feature and features from matrix must be of the same length (same n of cells).")
        cMI = mutual_info_regression(f1.reshape(-1,1), f2, random_state=random_state)
        # for entropy
        S_f1 = entropy_vector[feature_ind]
        S_f2 = entropy_vector[i]
        cESS = cMI / max(S_f1, S_f2)
        
        cESS_vector.append(cESS)
        MI_vector.append(cMI)
        
    return [cESS_vector, MI_vector]

print(f'\nCalculating ESS...')

#set random seed?
np.random.seed(42)

def Parallel_Calculate_ESS(working_matrix, Use_Cores=30):
    ## Provide indicies for parallel computing.
    
    global clipped_matrix
    clipped_matrix = working_matrix
    
    Feature_Inds = np.arange(clipped_matrix.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    #if Use_Cores == -1:
    #    Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
    #    if Use_Cores < 1:
    #        Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    print('\nCalculating ESS in parallel.')
    def initialize_worker(seed = 42):
        np.random.seed(seed)
        random.seed(seed)
    seed = 42
    p_map(initialize_worker, [seed]*Use_Cores)
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        allscores = p_map(partial(continuous_ESS, count_matrix=clipped_matrix, entropy_vector = entropy_vector), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(allscores)
    ESS = results[:,0]
    MI = results[:,1]
    # Return results
    return ESS, MI

print('\nCalculating ESS for both subsets...')

ESS, MI = Parallel_Calculate_ESS(normalised_matrix)
ESS = np.squeeze(ESS, axis=2)
MI = np.squeeze(MI, axis=2)

print('\nProceeding with printing customESS results.')

continuousESS = pd.DataFrame(data = ESS, columns = column_name, index = column_name)
continuousESS.to_csv('final_embryo_continuous_ESS_cESFW-subset.csv')

c_MI = pd.DataFrame(data = MI, columns = column_name, index = column_name)
c_MI.to_csv('final_embryo_continuous_MI_cESFW-subset.csv')

####
ESS, MI = Parallel_Calculate_ESS(normalised_matrix_2)
ESS = np.squeeze(ESS, axis=2)
MI = np.squeeze(MI, axis=2)

print('\nProceeding with printing customESS results.')

continuousESS = pd.DataFrame(data = ESS, columns = column_name_2, index = column_name_2)
continuousESS.to_csv('final_embryo_continuous_ESS_chisq-subset.csv')

c_MI = pd.DataFrame(data = MI, columns = column_name_2, index = column_name_2)
c_MI.to_csv('final_embryo_continuous_MI_chisq-subset.csv')