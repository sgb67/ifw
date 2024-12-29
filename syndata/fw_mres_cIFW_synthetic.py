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

# Load synthetic data
raw_counts = pd.read_csv("Complete_Synthetic_Data.csv",header=0,index_col=0)

# Normalisation function
def normalise_counts(count_matrix):
    Upper = np.percentile(count_matrix,97.5,axis=0) #get upper percentile of each gene (column)
    #now for those which have 0 as the 97.5th percentile, just get the maximum value as the upper
    Upper[Upper == 0] = np.max(count_matrix,axis=0)[Upper == 0] 
    count_matrix = count_matrix.astype(float)
    count_matrix = count_matrix.clip(upper=Upper,axis=1) # this is for columns 
    count_max = np.max(count_matrix,axis=0) # get the max, the 97.5th percentile, to normalise
    output = count_matrix / count_max 
    return output

normalised_matrix = normalise_counts(count_matrix=raw_counts)

# Normalisation function
def clip_counts(count_matrix, percentile = 97.5, full_distribution = False):
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
    return count_matrix

clipped_counts = clip_counts(raw_counts)

column_name = normalised_matrix.columns

###########################################################################################

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

def Parallel_Calculate_entropy(Use_Cores=-1):
    ## Provide indicies for parallel computing.

    Feature_Inds = np.arange(raw_counts.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    def initialize_worker(seed = 42):
        np.random.seed(seed)
        random.seed(seed)
    seed = 42
    p_map(initialize_worker, [seed]*Use_Cores)
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        entropies = p_map(partial(calculate_entropy, count_matrix=clipped_counts), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(entropies)
    # Return results
    return results

entropy_vector = Parallel_Calculate_entropy()
entropy_vector = np.array(entropy_vector)
entropy = pd.Series(data = entropy_vector)
entropy.to_csv('synthetic_entropy_newfunction.csv')


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

def Parallel_Calculate_ESS(Use_Cores=-1):
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(raw_counts.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    
    def initialize_worker(seed = 42):
        np.random.seed(seed)
        random.seed(seed)
    seed = 42
    p_map(initialize_worker, [seed]*Use_Cores)
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        allscores = p_map(partial(continuous_ESS, count_matrix=normalised_matrix, entropy_vector = entropy_vector), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(allscores)
    ESS = results[:,0]
    MI = results[:,1]
    # Return results
    return ESS, MI

ESS, MI = Parallel_Calculate_ESS()
ESS = np.squeeze(ESS, axis=2)
MI = np.squeeze(MI, axis=2)

print('\nProceeding with printing customESS results.')

continuousESS = pd.DataFrame(data = ESS, columns = column_name, index = column_name)
continuousESS.to_csv('synthetic_continuous_ESS_newentropy.csv')

c_MI = pd.DataFrame(data = MI, columns = column_name, index = column_name)
c_MI.to_csv('synthetic_continuous_MI_newentropy.csv')

### Now chisq
print('\nNow calculating chisq')

# Discretisation function
def two_state_discretisation(feature):
    res = np.zeros(len(feature))
    res[np.where(feature == 0)] = 0 # set zeros as state 0
    nonzero = feature[np.where(feature != 0)]
    # 25th percentile of nonzero distribution as threshold, could tune it
    cutoff = np.percentile(nonzero, 25)
    bins = [cutoff]
    if cutoff >= 0.95: # if the 25th percentile is very big, just take nonzero values as state 1
        nonzero_vals = np.ones(len(nonzero))
    else:
        nonzero_vals = np.digitize(nonzero, bins = bins, right = True)
    res[np.where(feature != 0)] = nonzero_vals
    discrete = res.astype(int)
    return discrete

print('\nProceeding with calculting chi-squared.')

def two_state_chisq(feature_ind, normalised_matrix, zero_info = True, extra_info = False):
    
    f1_raw = np.array(normalised_matrix.iloc[:,feature_ind])
    discretization_f1 = 2
    f1 = two_state_discretisation(f1_raw)
    num_genes = normalised_matrix.shape[1]
    chip = []
    chistat = []
    
    # carry out test for every pairwise comparison
    for i in range(num_genes):
        #define feature 2
        f2_raw = np.array(normalised_matrix.iloc[:,i])
        f2 = two_state_discretisation(f2_raw)
        discretization_f2 = 2
            
        if len(f1) != len(f2):
            print("Fixed feature and features from matrix must be of the same length (same n of cells).")

        # Obtain co-occurrence counts (ns)
        n00 = 0
        n01 = 0

        n10 = 0
        n11 = 0
        
        for (cell1, cell2) in zip(f1, f2):
            if cell1 == cell2:
                if cell1 == 0:
                    n00 += 1
                elif cell1 == 1:
                    n11 += 1

                    
            elif cell1 == 0:
                if cell2 == 1:
                    n01 += 1
                    
            elif cell1 == 1:
                if cell2 == 0:
                    n10 += 1
                    
        #now calculate cs in reference or f1 (m) and query or f2 (q)
        #wrt to f1
        c_m0 = n00 + n01
        c_m1 = n10 + n11
        #wrt to f2
        c_q0 = n00 + n10
        c_q1 = n01 + n11


        ns = np.array([n00, n01,
                        n10, n11])

        cs = np.array([c_m0, c_m1,
                        c_q0, c_q1])

        C = normalised_matrix.shape[0]

        ps = cs / C #these are just the probabilities... divide cs by total # of cells
        
        # now simply obtain co-occurrence probabilities
        p_00 = ps[0]*ps[2]
        p_01 = ps[0]*ps[3]
        
        p_10 = ps[1]*ps[2]
        p_11 = ps[1]*ps[3]
        
        p_paired = np.array([p_00, p_01,
                            p_10, p_11])
        
        total = np.sum(ns) # should just be the number of cells, 1200 for synthetic data
        freq = p_paired * total # get expected frequencies

        #simply calculate Chi-squared p-value and statistic
        chip.append(scipy.stats.chisquare(ns, f_exp=freq).pvalue)
        chistat.append(scipy.stats.chisquare(ns, f_exp=freq).statistic)
        
    return [chip, chistat]

def Parallel_Calculate_chisq(Use_Cores=-1):
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(normalised_matrix.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        chisq = p_map(partial(two_state_chisq, normalised_matrix=normalised_matrix), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(chisq)
    chip = results[:,0]
    chistat = results[:,1]
    # Return results
    return chip, chistat

chipall, chistatall = Parallel_Calculate_chisq()

print('\nProceeding with printing chisq results.')

chipall = pd.DataFrame(data=chipall, columns=column_name, index=column_name)
chipall.to_csv('chip_synthetic_twostate_newentropy.csv')

chistatall = pd.DataFrame(data=chistatall, columns=column_name, index=column_name)
chistatall.to_csv('chistat_synthetic_twostate_newentropy.csv')

print('\nAnalysis completed.')