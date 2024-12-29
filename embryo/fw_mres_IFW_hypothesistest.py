### BINARY IFW ### WITH TWO DIFFERENT FILTERING

# Two-state custom ESS
import pandas as pd
import numpy as np
from functools import partial
from os.path import exists
import multiprocess
import scipy.sparse
import matplotlib.pyplot as plt
from p_tqdm import p_map

print('\nLoading data...')

# FIRST LOAD THE DATA WITH THE FILTERING FROM CESFW
raw_counts = pd.read_csv("Human_Embryo_Counts.csv",header=0,index_col=0)
#cESFW_subset = pd.read_csv('subsetted_genes_new_cESFW_embryo.csv', header=0,index_col=0)
#raw_counts_1 = raw_counts[cESFW_subset]

normalised_matrix = pd.read_csv('subsetted_matrix_new_cESFW_embryo.csv',header=0,index_col=0)

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

# now do chisq test
def test_binarised_chisq(feature_ind, binarised_df):
    
    f1 = np.array(binarised_df.iloc[:,feature_ind])
    num_genes = binarised_df.shape[1]
    chip = []
    chistat = []
    
    for i in range(num_genes):
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

            total = np.sum(n_by_ind) # should just be 1200
            freq = probs_byind * total

            chip.append(scipy.stats.chisquare(n_by_ind, f_exp=freq).pvalue)
            chistat.append(scipy.stats.chisquare(n_by_ind, f_exp=freq).statistic)
                
    return [chip, chistat]
    
    
def Parallel_Calculate_chisq(binarised_data, Use_Cores=-1):
    
    ## Identify number of cores to use.
    global binarised_dataset
    binarised_dataset = binarised_data
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(binarised_dataset.shape[1])
    
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    with np.errstate(divide='ignore',invalid='ignore'):
        chisq = p_map(partial(test_binarised_chisq, binarised_df=binarised_dataset), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(chisq)
    chip = results[:,0]
    chistat = results[:,1]
    # Return results
    return chip, chistat

print('\nCalculating Chi-squared for cESFW subset...')

chipall, chistatall = Parallel_Calculate_chisq(binarised_data=binarised_df)

chipall = pd.DataFrame(data=chipall, columns=column_name, index=column_name)
chipall.to_csv('final_chip_embryo_twostate_cESFW-subset.csv')

chistatall = pd.DataFrame(data=chistatall, columns=column_name, index=column_name)
chistatall.to_csv('final_chistat_embryo_twostate_cESFW-subset.csv')

###################################################################################
#### NOW WITH CHISQ SUBSET ####

binarised_df_2 = pd.read_csv('chisq-filtered_binarised-matrix.csv',header=0,index_col=0)

column_name_2 = binarised_df_2.columns
Feature_IDs_2 = column_name_2

print(f'\nShape of cESFW filter is {binarised_df.shape} and of chisq filter is {binarised_df_2.shape}.')

print('\nCalculating Chi-squared for Chisq subset...')

chipall, chistatall = Parallel_Calculate_chisq(binarised_data=binarised_df_2)

chipall = pd.DataFrame(data=chipall, columns=column_name_2, index=column_name_2)
chipall.to_csv('final_chip_embryo_twostate_chisq-subset.csv')

chistatall = pd.DataFrame(data=chistatall, columns=column_name_2, index=column_name_2)
chistatall.to_csv('final_chistat_embryo_twostate_chisq-subset.csv')

########################################################################

# Now carry out FET

def test_FET(feature_ind, binarised_df):
    
    f1 = np.array(binarised_df.iloc[:,feature_ind])
    num_genes = binarised_df.shape[1]
    fetp = []
    fetstat = []
    
    for i in range(num_genes):
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
                        
            # allns = [n11, n10, n01, n00]
            arr = np.empty([2,2])
            arr[0] = ([n11, n10])
            arr[1] = ([n01, n00])
            arr = arr.astype(int)
            fetp.append(scipy.stats.fisher_exact(arr, alternative='two-sided').pvalue)
            fetstat.append(scipy.stats.fisher_exact(arr, alternative='two-sided').statistic)
    
    return [fetp, fetstat]


def Parallel_Calculate_FET(binarised_data, Use_Cores=-1):
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
        FET = p_map(partial(test_FET, binarised_df=binarised_dataset), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    results = np.asarray(FET)
    fetp = results[:,0]
    fetstat = results[:,1]
    # Return results
    return fetp, fetstat

print('\nCarrying out FET with cESFW subset')

fetpall, fetstatall = Parallel_Calculate_FET(binarised_data=binarised_df)

fetpall = pd.DataFrame(data=fetpall, columns=column_name, index=column_name)
fetpall.to_csv('final_fetp_embryo_twostate_cESFW-subset.csv')

fetstatall = pd.DataFrame(data=fetstatall, columns=column_name, index=column_name)
fetstatall.to_csv('final_fetstat_embryo_twostate_cESFW-subset.csv')


print('\nCarrying out FET with chisq subset')

fetpall, fetstatall = Parallel_Calculate_FET(binarised_data=binarised_df_2)

fetpall = pd.DataFrame(data=fetpall, columns=column_name_2, index=column_name_2)
fetpall.to_csv('final_fetp_embryo_twostate_chisq-subset.csv')

fetstatall = pd.DataFrame(data=fetstatall, columns=column_name_2, index=column_name_2)
fetstatall.to_csv('final_fetstat_embryo_twostate_chisq-subset.csv')

print('\nAnalysis complete.')