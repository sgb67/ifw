#### bIFW ####
# This code was extracted from a Jupyter notebook, and required slight changes


# import packages
import pandas as pd
import numpy as np
import scipy
from functools import partial
from os.path import exists
import multiprocess
import scipy.sparse
import matplotlib.pyplot as plt
from p_tqdm import p_map
import seaborn as sns

# load the example synthetic dataset - Bifurcation
path = 'c:/Users/sergi/Documents/GitHub/fw_mres/dev_syndata/'
syndata = pd.read_csv(path + "/bif1/bif1_data.csv",header=0,index_col=0)

# Preprocessing count matrix
## we can choose to clip the matrix, vary threshold and perform 'dummy' binarisation
def bIFW_preprocess_counts(count_matrix, clip = True, cutoff_percentile = 25, dummy = False):
    if clip == True:
        print(f"Clipping the matrix to remove outliers.\n")
        # the clipping from cESFW
        Upper = np.percentile(count_matrix,97.5,axis=0) # get upper percentile of each gene (column)
        # for those which have 0 as the 97.5th percentile, we simply get the maximum value as the upper bound
        Upper[Upper == 0] = np.max(count_matrix,axis=0)[Upper == 0] 
        count_matrix = count_matrix.astype(float)
        count_matrix = count_matrix.clip(upper=Upper,axis=1) #We do this for columns (for clip its 1)
    
    count_max = np.max(count_matrix,axis=0) #get the max, which is 97.5th percentile if clip == True, to normalise
    norm_orig = count_matrix / count_max #just normalise using max value
    if np.where(count_max == 0)[0].shape[0] > 0:
        print(f"There were {np.where(count_max == 0)[0].shape[0]} genes with 0 counts in all cells. Dropping them to prevent NaN formation.\n")
        # drop columns with all 0s
        norm_orig = norm_orig.loc[:, norm_orig.columns[norm_orig.sum() != 0]]
        count_matrix = count_matrix.loc[:, count_matrix.columns[count_matrix.sum() != 0]]
    print(f"Normalisation done. Matrix shape is {norm_orig.shape}. Now discretising.\n")
    Feature_IDs = norm_orig.columns    
    
    if dummy == False:
        # now discretisation: for bIFW, it is a two-state discretisation or binarisation
        def two_state_discretisation(feature, cutoff_percentile = cutoff_percentile):
            res = np.zeros(len(feature))
            res[np.where(feature == 0)] = 0
            nonzero = feature[np.where(feature != 0)]
            cutoff = np.percentile(nonzero, cutoff_percentile)
            bins = [cutoff]
            if cutoff >= 0.95: # if the cutoff is too high (i.e., the chosen percentile is close to the maximum value), then we binarise differently
                nonzero_vals = np.ones(len(nonzero))
            else:
                nonzero_vals = np.digitize(nonzero, bins = bins, right = True)
            res[np.where(feature != 0)] = nonzero_vals
            return res.astype(int)
        
        binarised_matrix = []
        for i in range(len(Feature_IDs)):
            binarised_matrix.append(two_state_discretisation(np.array(norm_orig.iloc[:, i])))
    
    elif dummy == True: # the dummy binarisation simply sets non-zero values to 1
        binarised_matrix = count_matrix.copy()
        binarised_matrix[binarised_matrix > 0] = 1
        binarised_matrix = binarised_matrix.T
        print(f"Dummy binarisation selected. All values above 0 will be set to 1.\n")
    
    print(f"Discretisation done. Returning resulting normalised and binarised matrices.\n")
    binarised_matrix = np.array(binarised_matrix).T
    binarised_df = pd.DataFrame(data=binarised_matrix, columns=Feature_IDs, index=norm_orig.index)
    
    if np.where(count_max == 0)[0].shape[0] > 0:
        return np.where(count_max == 0)[0], count_matrix, norm_orig, binarised_df
    return norm_orig, binarised_df

## we get the indexes of 'empty' genes, the raw count matrix, the normalised matrix and the binarised matrix
zerogenes, syndata, norm_orig, binarised_df = bIFW_preprocess_counts(count_matrix=syndata, dummy=True)
## we can also perform visual checks of the data to see effects of preprocessing - see notebooks

#####################################################################
# Significance testing: the chi-squared test
## Perform test for a single feature using the indices from the upper triangular matrix (triu, see next function)
def test_binarised_chisq(feature_ind, triu, binarised_df):
    #for some reason I had to import packages within function for multiprocessing to work
    #import numpy as np
    #import scipy
    
    triu_ind = [feat[1] for feat in triu if feat[0] == feature_ind][0]
    f1 = np.array(binarised_df.iloc[:,feature_ind])
    chip = []
    chistat = []
    
    for i in triu_ind:
        #define feature 2
        f2 = np.array(binarised_df.iloc[:,i])
        if len(f1) != len(f2):
            print("Fixed feature and features from matrix must be of the same length (same n of cells).")
        # get number of counts
        else:
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
                        
            allns = [n11, n10, n01, n00]
            n_by_ind = np.array(allns) #.T
            C = binarised_df.shape[0]

            p_11 = ((n11 + n10)/C) * ((n11 + n01)/C)
            p_10 = ((n11 + n10)/C) * (1 - (n11 + n01)/C)
            p_01 = (1 - (n11 + n10)/C) * ((n11 + n01)/C)
            p_00 = (1 - (n11 + n10)/C) * (1 - (n11 + n01)/C)
            probs = [p_11,p_10,p_01,p_00]
            probs_byind = np.array(probs) #.T

            #total = []
            #freq = []

            total = np.sum(n_by_ind) # should just be number of cells
            freq = probs_byind * total

            chip.append(scipy.stats.chisquare(n_by_ind, f_exp=freq).pvalue)
            chistat.append(scipy.stats.chisquare(n_by_ind, f_exp=freq).statistic)
                
    return [chip, chistat]

# Perform the test for all features in parallel
def parallel_chisq(binarised_data, Use_Cores=-1):
    import numpy as np
    global binarised_dataset
    binarised_dataset = binarised_data
    print(f"Data loaded. Shape: {binarised_dataset.shape}. Proceeding to obtain indices for efficient significance testing.\n")
    ## Provide indices for parallel computing and efficiency
    n = binarised_dataset.shape[1]
    def triu_list(n):
        triu_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
        triu = []
        for f in np.unique(triu_indices[0]):
            l = triu_indices[1][triu_indices[0] == f]
            triu.append([f,l])
        return triu, triu_indices
    triu, triu_indices = triu_list(n)
    Feature_Inds = [feat[0] for feat in triu]
    print(f"Indices obtained. Proceeding with performing statistical tests in parallel.\n")
    
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        chisq = p_map(partial(test_binarised_chisq, binarised_df=binarised_dataset, triu=triu), Feature_Inds, num_cpus=Use_Cores)
    chip = [row[0] for row in chisq]
    
    print(f"Calculations complete. Reconstructing significance matrix.\n")
    # Use allscores to build square matrix
    indices = (triu_indices[0], triu_indices[1])
    values = [value for sublist in chip for value in sublist]
    # Initialize a zero matrix
    matrix = np.zeros((n, n), dtype=float)
    for row, col, value in zip(indices[0], indices[1], values):
        #print(f"Placing value {value} at position ({row}, {col})")  # Debug print
        matrix[row, col] = value

    print(f"Matrix construction complete. Saving to dataframe.\n")
    m = pd.DataFrame(matrix)
    chip_m = m + m.T - np.diag(np.diag(m)) # make square
    nan_values = np.sum(np.isnan(chip_m.values))
    print(f"There are {nan_values} NaN values in the masked significance matrix ({nan_values/(chip_m.shape[0]*chip_m.shape[0])} %).")
    return chisq, chip_m

## we get both the chi-squared statistic and the p-value matrices
chisq, chip = parallel_chisq(binarised_data=binarised_df, Use_Cores=4)

# We then log-transform the significance matrix and mask it for calculating correlations
    # We use a Bonferroni correction for this example
def mask_non_sig(sign_matrix, threshold=0.05, bonferroni=True):
    n = sign_matrix.shape[0]
    np.fill_diagonal(sign_matrix.to_numpy(), 0)
    t_sign_matrix = np.absolute(sign_matrix)
    #avoid log(0) by setting a cutoff
    cutoff = 1e-300
    t_sign_matrix[t_sign_matrix < cutoff] = 0
    t_sign_matrix[t_sign_matrix > cutoff] = -np.log(t_sign_matrix[t_sign_matrix > cutoff])
    if bonferroni == True:
        corrected_thr = -np.log(threshold/(n*n))
        print(f"Bonferroni-corrected -log(p) threshold is {corrected_thr}, and the p-value is {threshold/(n*n)}")
    else:
        corrected_thr = -np.log(threshold)
        print(f"Non-corrected -log(p) threshold is {corrected_thr}, and the p-value is {threshold}")
    tm_sign_matrix = t_sign_matrix.copy()
    tm_sign_matrix[tm_sign_matrix < corrected_thr] = 0
    np.fill_diagonal(tm_sign_matrix.to_numpy(), 0) #remove inf values...
    print(f"The maximum value in the matrix is: {tm_sign_matrix.max().max()} and the minimum value is: {tm_sign_matrix.min().min()}\n")
    nan_values = np.sum(np.isnan(tm_sign_matrix.values))
    print(f"There are {nan_values} NaN values in the masked significance matrix ({nan_values/(tm_sign_matrix.shape[0]*tm_sign_matrix.shape[0])} %).")
    print(f"Returning the transformed significance matrix and the transformed and masked significance matrix.\n")
    return t_sign_matrix, tm_sign_matrix

## we get (i) the log-transformed matrix, and (ii) the log-transformed and masked matrix
chip_t, chip_masked = mask_non_sig(sign_matrix=chip, threshold=0.05, bonferroni=True)

###################################################################
# Omega (Ω), the correlation metric, equivalent to the Entropy Sort Score (ESS)

## The individual function to calculate the Omega score for a given feature
### sign_comp provides indices for significant comparisons
def bIFW_correlation(feature_ind, sign_comp, normalised_matrix, extra_vectors = False, zero_info = True, extra_info = False):
    #for some reason I had to import packages within function for multiprocessing to work
    #import numpy as np
    #import scipy
    
    # get the list of significant features
    sign_list = [feat[1] for feat in sign_comp if feat[0] == feature_ind][0]
    MI_vector = []
    S_q_vector = []
    S_m_vector = []
    f1 = np.array(normalised_matrix.iloc[:,feature_ind])
    
    for i in sign_list:
        #define feature 2
        f2 = np.array(normalised_matrix.iloc[:,i])
        n00 = 0
        n01 = 0
        n10 = 0
        n11 = 0
        
        if len(f1) != len(f2):
            print("Fixed feature and features from matrix must be of the same length (same n of cells).")
        else:
            c = len(f1)
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
                        
        # check discretization
        ns = np.array([n00, n01, n10, n11])
        n_str = ["n00", "n01", "n10", "n11"]
        nsum = np.sum(ns)
        if nsum != c:
            print("Sum of state counts do not add up.")
            MI = np.nan
            S_q = np.nan
            S_m = np.nan
            
        #calculate c's - need to have at least one of each
        else:
            #wrt to f1
            c_m0 = n00 + n01
            c_m1 = n10 + n11
            #wrt to f2
            c_q0 = n00 + n10
            c_q1 = n01 + n11
            
            cs_MI = np.array([[c_m0, c_q0], [c_m0, c_q1],
                              [c_m1, c_q0], [c_m1, c_q1]])
            cs_S = np.array([[c_m0, c_m1],
                             [c_q0, c_q1]])
            
            MI_terms = []
            zeroterms = []
            for ind, n in enumerate(ns):
                if n != 0 & np.all(cs_MI[ind]) == False: #if n and both cs are nonzero, calculate
                    MI_term = (n/c * np.log2(c * n / (cs_MI[ind][0] * cs_MI[ind][1])))
                    MI_terms.append(MI_term)
                    
                else:
                    zeroterms.append(n_str[ind])
            MI = np.sum(MI_terms)
            
            # entropies separately
            S_m_terms = []
            S_q_terms = []
            
            for ind in range(len(cs_S)):
                S_m_terms.append(cs_S[0][ind]/c * np.log2(cs_S[0][ind]/c))
                S_q_terms.append(cs_S[1][ind]/c * np.log2(cs_S[1][ind]/c))
                
            S_m = np.sum(S_m_terms) * (-1)
            S_q = np.sum(S_q_terms) * (-1)

            if extra_info == True:     
                exclude = str()
                for t in zeroterms:
                    exclude += (t + ", ")
                print("Be aware that the counts " + exclude + "were 0. This affects the calculations.")
                
        MI_vector.append(MI)
        S_q_vector.append(S_q)
        S_m_vector.append(S_m)
    
    max_entropy = [max(Sm, Sq) for Sm, Sq in zip(S_m_vector, S_q_vector)]
    
    #now calculate omega
    if len(MI_vector) != len(max_entropy):
        raise ValueError("All vectors (MI, x_max, S_q and S_m) must have the same length")    

    omega_vector = np.array(MI_vector) / np.array(max_entropy)
    if extra_vectors == True:
        return [omega_vector, MI_vector, max_entropy]            
    else:
        return [omega_vector]
    
## The parallelised function to calculate the Omega for all features
def parallel_bIFW_correlation(binarised_data, sign_matrix, Use_Cores=-1):
    global binarised_dataset
    binarised_dataset = binarised_data
    print(f"Data loaded. Shape: {binarised_dataset.shape}. Proceeding to obtain indices for efficient ESS calculation.\n")
    nonzero = np.nonzero(sign_matrix.to_numpy())
    sign_comp = []
    for f in np.unique(nonzero[0]):
        #print(f"Gene {f} has a significant interaction with genes {nonzero[1][nonzero[0] == f]}")
        l = nonzero[1][nonzero[0] == f]
        sign_comp.append([f,l])
        #print(f"Gene {f} has a significant interaction with {len(l)} genes")
    Feature_Inds = [feat[0] for feat in sign_comp]
    print(f"Indices obtained. Proceeding with calculating ESSs in parallel.\n")
    
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
        allscores = p_map(partial(bIFW_correlation, sign_comp=sign_comp, normalised_matrix=binarised_dataset), Feature_Inds, num_cpus=Use_Cores)
    print(f"Calculations complete. Proceeding with matrix reconstruction.\n")
    omega = [row[0] for row in allscores]
    
    # Use allscores to build square matrix
    n = binarised_dataset.shape[1]
    indices = (nonzero[0], nonzero[1])
    values = [value for sublist in omega for value in sublist]
    # Initialize a zero matrix
    matrix = np.zeros((n, n), dtype=float)
    for row, col, value in zip(indices[0], indices[1], values):
        #print(f"Placing value {value} at position ({row}, {col})")  # Debug print
        matrix[row, col] = value
    
    print("Matrix construction complete. Saving to dataframe.")
    m = pd.DataFrame(matrix)
    return allscores, m

## we get a list of all scores calculated and the reconstructed correlation matrix, omega_matrix
allscores, omega_matrix = parallel_bIFW_correlation(binarised_data=binarised_df, sign_matrix=chip_masked, Use_Cores=4)

### Note: calculating omega is faster than the significance testing...

# The weight calculation is simply the average of the Omega values weighted by the corresponding Phi (significance) value
def weights_IFW(masked_omega, masked_sign_matrix, normalise_by_edges = False):
    w = np.zeros(masked_omega.shape[1])
    for i in range(masked_omega.shape[0]):
        wsum = np.sum(masked_sign_matrix.iloc[i])
        prod = masked_omega.iloc[i].values * masked_sign_matrix.iloc[i].values
        if wsum > 0:
            w[i] = np.nansum(prod) / wsum
        else:
            w[i] = 0
            
    #w = np.average(masked_omega,weights=masked_sign_matrix,axis=0)
    if normalise_by_edges == True:
        edges = (masked_sign_matrix > 0).sum(1)
        n_w = w/edges
        return w, n_w, edges
    else:
        return w

weights = weights_IFW(omega_matrix, chip_masked)
#ranks = np.argsort(-weights)

# We have our data, we can save it:
omega_matrix.to_csv(path + "/bif1/bIFW_omega_matrix_nonans_dummy.csv")
chip_masked.to_csv(path + "/bif1/bIFW_sign_matrix_nonans_dummy.csv")
weights.to_csv(path + "/bif1/bIFW_weights")

# For plotting see Jupyter notebook.