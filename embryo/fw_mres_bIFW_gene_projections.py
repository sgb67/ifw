### CELL PROJECTIONS CESFW2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# first calculate the weights again... and plot to see if there are any negative weights?

print('Loading datasets...')

# Scaled_Matrix = pd.read_csv('subsetted_matrix_new_cESFW_embryo.csv', header=0,index_col=0)

print(f'\nLoading ESS matrices.')
# cIFW_ESS = pd.read_csv("final_embryo_continuous_ESS_cESFW-subset.csv",header=0,index_col=0)
bIFW_ESS = pd.read_csv("final_binary_IFW_ESS_embryo_cESFW-subset.csv",header=0,index_col=0)
bIFW_ESS = np.array(bIFW_ESS)
#bIFW_ESS_chisq_subset = pd.read_csv("final_binary_IFW_ESS_embryo_chisq-subset.csv",header=0,index_col=0)

#cESFW_ESS = pd.read_csv("ESSs_new_cESFW_embryo.csv",header=0,index_col=0)

print(f'\nLoading significance matrices.')
IFW_chip = pd.read_csv("final_chip_embryo_twostate_cESFW-subset.csv",header=0,index_col=0)
IFW_fetp = pd.read_csv("final_fetp_embryo_twostate_cESFW-subset.csv",header=0,index_col=0)
#IFW_chip_chisq_subset = pd.read_csv("final_chip_embryo_twostate_chisq-subset.csv",header=0,index_col=0)
#IFW_fetp_chisq_subset = pd.read_csv("final_fetp_embryo_twostate_chisq-subset.csv",header=0,index_col=0)
#cESFW_dEP = pd.read_csv("dEP_new_cESFW_embryo.csv",header=0,index_col=0)
#cESFW_ohEP = pd.read_csv("ohEP_new_cESFW_embryo.csv",header=0,index_col=0)


print('\nCalculating weights of the binary and custom IFW...')
print(f'\nNumber of genes (shape of matrices) of cESFW: {IFW_chip.shape}')

num_genes = IFW_chip.shape[0]
threshold = -np.log(0.05 / (num_genes * num_genes))

print(f'\nThe threshold is {threshold}')

#### chi-squared
significance_matrix_chip = IFW_chip
significance_matrix_fetp = IFW_fetp

print('\nObtaining used features...')

normalised_matrix = pd.read_csv('subsetted_matrix_new_cESFW_embryo.csv',header=0,index_col=0)
Used_Features_cESFW = normalised_matrix.columns

#normalised_matrix_2 = pd.read_csv('chisq-filtered_binarised-matrix.csv',header=0,index_col=0)
#Used_Features_chisq = normalised_matrix_2.columns

print(f'\nMasking and weight calculation...')
def updated_weight_calculation(ESS_Threshold, EPs_Threshold, Min_Edges, ESSs, EPs, Used_Features, test = 'chisq', method = 'bIFW'):
    Absolute_ESSs = np.array(np.absolute(ESSs))
    #transform p-values
    #np.fill_diagonal(EPs, 0.5)
    #np.fill_diagonal(ESSs, 0)
    cutoff = 1e-300 #This cutoff prevents introducing NaNs
    EPs[EPs > cutoff] = -np.log(EPs)
    EPs[EPs < cutoff] = 0
    EPs = np.array(EPs)
    
    #mask
    Mask_Inds = np.where((EPs <= EPs_Threshold)) # NO ESS THRESHOLD!
    ESSs_Graph = Absolute_ESSs.copy()
    ESSs_Graph[Mask_Inds] = 0
    Absolute_ESSs[Mask_Inds] = 0
    EPs_Graph = EPs.copy()
    EPs_Graph[Mask_Inds] = 0
    EPs[Mask_Inds] = 0
    #while loop
    Keep_Features = np.array([])
    while Keep_Features.shape[0] < EPs_Graph.shape[0]:
        print("Genes remaining: " + str(EPs_Graph.shape[0]))
        Keep_Features = np.where(np.sum(EPs_Graph > 0,axis=0) > Min_Edges)[0]
        Used_Features = Used_Features[Keep_Features]
        ESSs_Graph = ESSs_Graph[np.ix_(Keep_Features,Keep_Features)] #971 genes
        EPs_Graph = EPs_Graph[np.ix_(Keep_Features,Keep_Features)]
        EPs = EPs[np.ix_(Keep_Features,Keep_Features)]
        Absolute_ESSs = Absolute_ESSs[np.ix_(Keep_Features,Keep_Features)]
    
    Mask_Inds = np.where((EPs <= EPs_Threshold)) # NO ESS THRESHOLD!
    ESSs_Graph = Absolute_ESSs.copy()
    ESSs_Graph[Mask_Inds] = 0
    EPs_Graph = EPs.copy()
    EPs_Graph[Mask_Inds] = 0


    Feature_Weights = np.average(ESSs_Graph,weights=EPs_Graph,axis=0)
    Significant_Genes_Per_Gene = (EPs_Graph > 0).sum(1)
    Normalised_Network_Feature_Weights = Feature_Weights/Significant_Genes_Per_Gene
    
    #w = pd.DataFrame(data=Feature_Weights)
    #w.to_csv(test+'_'+method+'_weights_newweights.csv')
    
    #nw = pd.DataFrame(data=Normalised_Network_Feature_Weights)
    #nw.to_csv(test+'_'+method+'_normalised_weights_newweights.csv')
    
    #Subset_Used_Features = Used_Features
    #feat = pd.DataFrame(data=Subset_Used_Features)
    #feat.to_csv(test+'_'+method+'_used_feature_weights_newweights.csv')
    print('\nWeights, normalised weights and used features printed for'+test+'_'+method)
    
    return pd.DataFrame(data=Absolute_ESSs), EPs, Feature_Weights, Normalised_Network_Feature_Weights, Significant_Genes_Per_Gene

ESS_Threshold = 0.05
Min_Edges = 10
EPs_Threshold = threshold

print('\nCalculating weights and printing...')

Masked_ESS_chip, Masked_EP_chip, Feature_Weights_chip, Normalised_Network_Feature_Weights_chip, Significant_Genes_Per_Gene_chip = updated_weight_calculation(ESS_Threshold, EPs_Threshold, Min_Edges, bIFW_ESS, significance_matrix_chip, Used_Features_cESFW, test = 'chisq', method = 'bIFW')

Masked_ESS_fetp, Masked_EP_fetp, Feature_Weights_fetp, Normalised_Network_Feature_Weights_fetp, Significant_Genes_Per_Gene_fetp = updated_weight_calculation(ESS_Threshold, EPs_Threshold, Min_Edges, bIFW_ESS, significance_matrix_fetp, Used_Features_cESFW, test = 'fetp', method = 'bIFW')
#bIFW_fet, norm_bIFW_fet = calculate_IFW_weights_complete(significance_matrix_fetp, bIFW_ESS, test='fetp', method='bIFW')
#bIFW_chisq, norm_bIFW_chisq = calculate_IFW_weights_complete(significance_matrix_chip, bIFW_ESS, test='chip', method='bIFW')
#cIFW_fet, norm_cIFW_fet = calculate_IFW_weights_complete(significance_matrix_fetp, cIFW_ESS, test='fetp', method='cIFW')
#cIFW_chisq, norm_cIFW_chisq = calculate_IFW_weights_complete(significance_matrix_chip, cIFW_ESS, test='chip', method='cIFW')

mESS_chip = pd.DataFrame(data=Masked_ESS_chip)
mESS_chip.to_csv('final_bIFW_masked_ESS_chip')

mESS_fetp = pd.DataFrame(data=Masked_ESS_fetp)
mESS_fetp.to_csv('final_bIFW_masked_ESS_fetp')

#######################################
# print('\nNow do the same for the chisq dataset...')
# print gene embeddings for n top genes

n_genes = [9000, 7500, 6000,4000,2500]  
print(f'\nProceeding to plot the top ranked {n_genes} genes.')

random_seed = 42

def plot_gene_proj_normalised_rank(nw, w, score_matrix, test='chip'):
    for n in n_genes:

        normalised_weights = np.array(nw)
        Use_Inds = np.argsort(-normalised_weights)[np.arange(n)]
            
        Neighbours = 20
        Dist = 0.1
        Gene_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2, random_state=random_seed).fit_transform(score_matrix.iloc[Use_Inds,Use_Inds])

        # sort values so that they appear last
        sorted_indices = np.argsort(normalised_weights[Use_Inds])
        x = Gene_Embedding[:,0][sorted_indices]
        y = Gene_Embedding[:,1][sorted_indices]
        values_sorted = normalised_weights[Use_Inds][sorted_indices]

        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)

        plt.title("Colour = Normalised weights", fontsize=20)
        plt.scatter(x, y,s=7,c=values_sorted, vmax=np.percentile(normalised_weights[Use_Inds],97.5))
        plt.colorbar()
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
        plt.subplot(1,2,2)
        
        sorted_indices = np.argsort(w[Use_Inds])
        x = Gene_Embedding[:,0][sorted_indices]
        y = Gene_Embedding[:,1][sorted_indices]
        values_sorted = w[Use_Inds][sorted_indices]
        
        plt.title("Colour = Weights", fontsize=20)
        plt.scatter(x,y,s=7,c=values_sorted,vmax=np.percentile(w[Use_Inds],97.5))
        plt.colorbar()
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)

        plt.savefig(f"final_bIFW_gene_proj_nwproj_"+test+f"_{n}_genes_newweights_2.png",dpi=600)
        plt.close()
        
plot_gene_proj_normalised_rank(Normalised_Network_Feature_Weights_chip, Feature_Weights_chip, Masked_ESS_chip, test='chip')
plot_gene_proj_normalised_rank(Normalised_Network_Feature_Weights_fetp, Feature_Weights_fetp, Masked_ESS_fetp, test='fetp')


def plot_gene_proj_weight_rank(nw, w, score_matrix, test='chip'):
    for n in n_genes:

        weights = np.array(w)
        Use_Inds = np.argsort(-weights)[np.arange(n)]
            
        Neighbours = 20
        Dist = 0.1
        Gene_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2, random_state=random_seed).fit_transform(score_matrix.iloc[Use_Inds,Use_Inds])

        sorted_indices = np.argsort(nw[Use_Inds])
        x = Gene_Embedding[:,0][sorted_indices]
        y = Gene_Embedding[:,1][sorted_indices]
        values_sorted = nw[Use_Inds][sorted_indices]

        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)

        plt.title("Colour = Normalised weights", fontsize=20)
        plt.scatter(x, y, s=7,c=values_sorted,vmax=np.percentile(nw[Use_Inds],97.5))
        plt.colorbar()
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
        plt.subplot(1,2,2)
        
        sorted_indices = np.argsort(weights[Use_Inds])
        x = Gene_Embedding[:,0][sorted_indices]
        y = Gene_Embedding[:,1][sorted_indices]
        values_sorted = weights[Use_Inds][sorted_indices]
        
        plt.title("Colour = Weights", fontsize=20)
        plt.scatter(x, y, s=7,c=values_sorted,vmax=np.percentile(weights[Use_Inds],97.5))
        plt.colorbar()
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
        plt.close()
        plt.savefig(f"final_bIFW_gene_proj_weightproj_"+test+f"_{n}_genes_newweights_2.png",dpi=600)
        plt.close()
        
plot_gene_proj_weight_rank(Normalised_Network_Feature_Weights_chip, Feature_Weights_chip, Masked_ESS_chip, test='chip')
plot_gene_proj_weight_rank(Normalised_Network_Feature_Weights_fetp, Feature_Weights_fetp, Masked_ESS_fetp, test='fetp')

