### Final BINARY IFW ###

# Two-state custom ESS
import pandas as pd
import numpy as np
from functools import partial
from os.path import exists
import multiprocess
import scipy.sparse
import matplotlib.pyplot as plt
from p_tqdm import p_map

print('\nLoading data AND FILTERING.......')


# FIRST LOAD THE DATA WITH THE FILTERING FROM CESFW
raw_counts = pd.read_csv("Complete_Synthetic_Data.csv",header=0,index_col=0)

def normalise_counts(count_matrix):
    Upper = np.percentile(count_matrix,97.5,axis=0) #get upper percentile of each gene (column)
    #now for those which have 0 as the 97.5th percentile, just get the maximum value as the upper
    Upper[Upper == 0] = np.max(count_matrix,axis=0)[Upper == 0] 
    #now clip essentially makes values bigger than the Upper, the Upper limit. There is a deprecation warning for this function.
    count_matrix = count_matrix.astype(float)
    count_matrix = count_matrix.clip(upper=Upper,axis=1) #We do this for columns (for clip its 1)
    count_max = np.max(count_matrix,axis=0) #get the max, which is 97.5th percentile, to normalise
    output = count_matrix / count_max #just normalise
    return output

normalised_matrix = normalise_counts(count_matrix=raw_counts)

column_name = normalised_matrix.columns
Feature_IDs = column_name

# now binarise matrix

def two_state_discretization(feature):
    res = np.zeros(len(feature))
    res[np.where(feature == 0)] = 0
    nonzero = feature[np.where(feature != 0)]
    #med = np.median(nonzero)
    cutoff = np.percentile(nonzero, 25)
    bins = [cutoff]
    if cutoff >= 0.95:
        nonzero_vals = np.ones(len(nonzero))
    else:
        nonzero_vals = np.digitize(nonzero, bins = bins, right = True)
    res[np.where(feature != 0)] = nonzero_vals
    return res.astype(int)

binarised_matrix = []
for i in range(len(Feature_IDs)):
    binarised_matrix.append(two_state_discretization(np.array(normalised_matrix.iloc[:, i])))

binarised_matrix = np.array(binarised_matrix).T
binarised_df = pd.DataFrame(data=binarised_matrix, columns=normalised_matrix.columns, index=normalised_matrix.index)

##########################################################################################
# NOW CALCULATING BINARY ESS

def two_state_custom_ESS(feature_ind, normalised_matrix, zero_info = True, extra_info = False):
    MI_vector = []
    S_q_vector = []
    S_m_vector = []
    #direction_vector = []
    f1 = np.array(normalised_matrix.iloc[:,feature_ind])
    
    for i in range(normalised_matrix.shape[1]):
        
        # print("Calculating MI for feature number", i)
        
        #define feature 2
        f2 = np.array(normalised_matrix.iloc[:,i])
        
        n00 = 0
        n01 = 0
        n10 = 0
        n11 = 0
        
        # get number of counts
        
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
            #direction = np.nan
            
        #calculate c's - need to have at least one of each!
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
            
           # x_max = (c_m1 * c_q1) / c
            
            MI_terms = []
           
            zeroterms = []
            for ind, n in enumerate(ns):
                if n != 0 & np.all(cs_MI[ind]) == False: #if n and both cs are nonzero, calculate
                    MI_term = (n/c * np.log2(c * n / (cs_MI[ind][0] * cs_MI[ind][1])))
                    MI_terms.append(MI_term)
                    
                else:
                    zeroterms.append(n_str[ind])
            MI = np.sum(MI_terms)
            
            # entropies separately...
            S_m_terms = []
            S_q_terms = []
            
            for ind in range(len(cs_S)):
                S_m_terms.append(cs_S[0][ind]/c * np.log2(cs_S[0][ind]/c))
                S_q_terms.append(cs_S[1][ind]/c * np.log2(cs_S[1][ind]/c))
                
            S_m = np.sum(S_m_terms) * (-1)
            S_q = np.sum(S_q_terms) * (-1)

            #direction = np.sign(n11 - x_max)

            if extra_info == True:     
                exclude = str()
                for t in zeroterms:
                    exclude += (t + ", ")
                print("Be aware that the counts " + exclude + "were 0. This affects the calculations.")
    
        MI_vector.append(MI)
        #direction_vector.append(direction)
        S_q_vector.append(S_q)
        S_m_vector.append(S_m)
    
    max_entropy = [max(Sm, Sq) for Sm, Sq in zip(S_m_vector, S_q_vector)]
    
    #now calculate ESS
    if len(MI_vector) != len(max_entropy):
        raise ValueError("All vectors (MI, x_max, S_q and S_m) must have the same length")    

    ESS_vector = np.array(MI_vector) / np.array(max_entropy)
                
    return [ESS_vector, MI_vector, max_entropy]

def Parallel_Calculate_ESS(binarised_data, Use_Cores=30):
    
    global binarised_dataset
    binarised_dataset = binarised_data
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(binarised_dataset.shape[1])
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
        allscores = p_map(partial(two_state_custom_ESS, normalised_matrix=binarised_dataset), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(allscores)
    ESS = results[:,0]
    MI = results[:,1]
    entropy = results[:,2]
    # Return results
    return ESS, MI, entropy

ESS, MI, entropy = Parallel_Calculate_ESS(binarised_data=binarised_df)

pseudoESS = pd.DataFrame(data = ESS, columns = column_name, index = column_name)
pseudoESS.to_csv('final_binary_IFW_ESS_synthetic.csv')

twostate_MI = pd.DataFrame(data = MI, columns = column_name, index = column_name)
twostate_MI.to_csv('final_binary_IFW_MI_synthetic.csv')

entropyvec = pd.DataFrame(data=entropy, columns=column_name, index=column_name)
entropyvec.to_csv('final_binary_IFW_H_synthetic.csv')