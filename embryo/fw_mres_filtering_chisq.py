### Analysis for embryo data
### IFW filtering ###
import scipy.sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import umap
#import pickle
# now the function to define the test

print('loading files...')
## Load human pre and post implantation embryo sample info and scRNA-seq counts matrix.
Human_Sample_Info = pd.read_csv("Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv("Human_Embryo_Counts.csv",header=0,index_col=0)
print('files loaded.')
raw_counts = Human_Embryo_Counts

## Drop highly expressed genes
Means = np.sum(Human_Embryo_Counts,axis=0) / np.sum(Human_Embryo_Counts > 0,axis=0)
Drop = np.where((Means > np.percentile(Means,97.5)))[0]
Human_Embryo_Counts = Human_Embryo_Counts.drop(Human_Embryo_Counts.columns[Drop],axis=1)

## Drop mitochondrial genes
Drop_Mitochondrial_Genes = Human_Embryo_Counts.columns[np.where(Human_Embryo_Counts.columns.str.contains("MTRNR"))[0]]
Human_Embryo_Counts = Human_Embryo_Counts.drop(Drop_Mitochondrial_Genes,axis=1)

## Remove genes that are active/inactive in less than 10 cells
Cell_Thresh = 10
Active_Cells = np.sum(Human_Embryo_Counts>5,axis=0)
Keep_Genes = np.where((Active_Cells > Cell_Thresh) & (Active_Cells < (Human_Embryo_Counts.shape[0]-Cell_Thresh)))[0]
Human_Embryo_Counts = Human_Embryo_Counts[Human_Embryo_Counts.columns[Keep_Genes]]

## Create the scaled matrix from the scRNA-seq counts matrix
Scaled_Matrix = Human_Embryo_Counts.copy()

## Clip expression of each gene
Upper = np.percentile(Scaled_Matrix,97.5,axis=0)
Upper[np.where(Upper == 0)[0]] = np.max(Scaled_Matrix,axis=0)[np.where(Upper == 0)[0]]
Scaled_Matrix = Scaled_Matrix.clip(upper=Upper,axis=1) 

## Normalise each feature/gene of the clipped matrix
Normalisation_Values = np.max(Scaled_Matrix,axis=0)
Scaled_Matrix = Scaled_Matrix / Normalisation_Values

Feature_IDs = Scaled_Matrix.columns
Initial_Used_Features = Feature_IDs.copy()
Dataset_IDs = np.unique(Human_Sample_Info["Dataset"])

## binarise matrix

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
    binarised_matrix.append(two_state_discretization(np.array(Scaled_Matrix.iloc[:, i])))

binarised_matrix = np.array(binarised_matrix).T
binarised_df = pd.DataFrame(data=binarised_matrix, columns=Scaled_Matrix.columns, index=Scaled_Matrix.index)

# dummy dataset ID
dummy_datasetID = []

for i in np.arange(Dataset_IDs.shape[0]):
    Dataset_Labels = np.zeros(Human_Sample_Info.shape[0])
    Dataset_Labels[np.where(Human_Sample_Info["Dataset"]==Dataset_IDs[i])[0]] = 1
    dummy_datasetID.append(Dataset_Labels.astype(int))

dummy_datasetIDs = np.array(dummy_datasetID)
dummy_df = pd.DataFrame(data = dummy_datasetIDs.T)


def batch_effect_chisq(dummy_matrix, binary_matrix, chistat_too = False):
    chip_all = []
    chistat_all = []
    
    for i in range(dummy_matrix.shape[1]):
        f1 = np.array(dummy_matrix.iloc[:,i])
        
        chip = []
        chistat = []
        
        for j in range(binary_matrix.shape[1]):
            f2 = np.array(binary_matrix.iloc[:,j])
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

                C = binary_matrix.shape[0]

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
                
        chip_all.append(chip)
        chistat_all.append(chistat)
        print(f'dataset {i} calculated.')
    
    chip_all = np.array(chip_all)
    chistat_all = np.array(chistat_all)   
    if chistat_too == True:
        return pd.DataFrame(data = chip_all, columns=binary_matrix.columns), pd.DataFrame(data = chistat_all, columns=binary_matrix.columns)
    else:
        return pd.DataFrame(data = chip_all, columns=binary_matrix.columns)
        
chipbatch = batch_effect_chisq(dummy_df, binarised_df, chistat_too = False)

# now figure out which is significant
def filter_chisq_batch(chipbatch, binarised_df, correction = 10e10):
    feature_orig = binarised_df.columns
    n_tests = chipbatch.shape[0] * chipbatch.shape[1]
    p_threshold = 0.05 / n_tests
    print(f'\nThe corrected threshold is {p_threshold/correction}')
    exclude_inds = np.where(chipbatch.min(axis=0) < p_threshold/correction)[0]
    print(f'\n{len(exclude_inds)} were excluded from the gene list.')
    new_feature_set = feature_orig[np.delete(np.arange(feature_orig.shape[0]),exclude_inds)]
    newdf = binarised_df.loc[:, new_feature_set]
    return newdf

newdf = filter_chisq_batch(chipbatch, binarised_df, correction=20e45)
newdf.to_csv('chisq-filtered_binarised-matrix.csv')


chisq_subset = newdf.columns
chisq_matrix = raw_counts.loc[:, chisq_subset]
chisq_matrix.to_csv('chisq-filtered_count-matrix.csv')

subsetted_genes = pd.Series(data=newdf.columns)
subsetted_genes.to_csv('subsetted_genes_chisq_embryo.csv')