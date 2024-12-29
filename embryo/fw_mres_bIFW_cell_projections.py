### CELL PROJECTIONS bIFW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import pickle
import plotly.express as px 
from functools import partial
import multiprocess
from p_tqdm import p_map

Colours = px.colors.qualitative.Dark24
Colours.remove('#222A2A')
Colours = np.concatenate((Colours,Colours))

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
    
    Subset_Used_Features = Used_Features
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

print('\nLoading embryo data...')
Human_Sample_Info = pd.read_csv("Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv("Human_Embryo_Counts.csv",header=0,index_col=0)

#######################################
# print('\nNow do the same for the chisq dataset...')
# print gene embeddings for n top genes

random_seed = 42

def Plot_UMAP(Embedding,Sample_Info, save_name):
    plt.figure(figsize=(8,6))
    plt.title("Timepoints",fontsize=18)
    Timepoints = np.asarray(Sample_Info["EmbryoDay"]).astype("f")
    Unique_Timepoints = np.unique(Timepoints)
    Unique_Timepoints = np.delete(Unique_Timepoints,np.where(np.isnan(Unique_Timepoints)))
    for i in np.arange(Unique_Timepoints.shape[0]):
        IDs = np.where(Timepoints == Unique_Timepoints[i])[0]
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=8,label=Unique_Timepoints[i])
    #
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('UMAP 1',fontsize=14)
    plt.ylabel('UMAP 2',fontsize=14)
    plt.savefig(f"timepoints_{save_name}_bIFW_ESsub_newweights_2.png",dpi=600)
    #
    plt.figure(figsize=(8,6))
    plt.title("Datasets",fontsize=18)
    Datasets = np.asarray(Sample_Info["Dataset"])
    Unique_Datasets = np.unique(Datasets)
    for i in np.arange(Unique_Datasets.shape[0]):
        IDs = np.where(Datasets == Unique_Datasets[i])[0]
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=8,label=Unique_Datasets[i])
    #
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('UMAP 1',fontsize=14)
    plt.ylabel('UMAP 2',fontsize=14)
    plt.savefig(f"datasets_{save_name}_bIFW_ESsub_newweights_2.png",dpi=600)
    #
    plt.figure(figsize=(8,6))
    plt.title("Cell annotations",fontsize=18)
    Annotations = np.asarray(Sample_Info["Manual_Annotations"])
    Unique_Annotations = np.unique(Annotations)
    Unique_Annotations = Unique_Annotations[np.array([0,9,7,6,3,14,2,4,5,1,8,10,13,11,15,12])]
    for i in np.arange(Unique_Annotations.shape[0]):
        IDs = np.where(Annotations == Unique_Annotations[i])[0]
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=8,label=Unique_Annotations[i],c=Colours[i])
    #
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.xlabel('UMAP 1',fontsize=14)
    plt.ylabel('UMAP 2',fontsize=14)
    plt.savefig(f"cellannot_{save_name}_bIFW_ESsub_newweights_2.png",dpi=600)
    plt.close()
    
n_genes = [9000, 7500, 6000,4000,2500]

def plot_cells_normalised(Normalised_Network_Feature_Weights, Subset_Used_Features, test='chisq'):
    for n in n_genes:
    ## Subset to top n genes
        Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)[np.arange(n)] 
        Selected_Genes = Subset_Used_Features[Use_Inds]
        #Selected_Genes.shape[0]
        #np.isin(Known_Important_Genes,Selected_Genes)
        #Known_Important_Genes[np.isin(Known_Important_Genes,Selected_Genes)]


        Cluster_Use_Gene_IDs = Selected_Genes[np.where(Normalised_Network_Feature_Weights[Use_Inds] < np.percentile(Normalised_Network_Feature_Weights[Use_Inds],97.5))[0]]

        ## Create UMAP of data subsetted to selected genes
        Reduced_Input_Data = Human_Embryo_Counts[Cluster_Use_Gene_IDs]
        Neighbours = 50
        Dist = 0.15

        Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
        Embedding = Embedding_Model.embedding_

        ## Plot UMAP
        Plot_UMAP(Embedding,Human_Sample_Info,save_name=f'_{n}_genes_{test}_normalised')
    #plt.show()

def plot_cells_weights(Feature_Weights, Subset_Used_Features, test='chisq'):
    for n in n_genes:
    ## Subset to top n genes
        Use_Inds = np.argsort(-Feature_Weights)[np.arange(n)] 
        Selected_Genes = Subset_Used_Features[Use_Inds]
        #Selected_Genes.shape[0]
        #np.isin(Known_Important_Genes,Selected_Genes)
        #Known_Important_Genes[np.isin(Known_Important_Genes,Selected_Genes)]
        Cluster_Use_Gene_IDs = Selected_Genes[np.where(Feature_Weights[Use_Inds] < np.percentile(Feature_Weights[Use_Inds],97.5))[0]]

        ## Create UMAP of data subsetted to selected genes
        Reduced_Input_Data = Human_Embryo_Counts[Cluster_Use_Gene_IDs]
        Neighbours = 50
        Dist = 0.15

        Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
        Embedding = Embedding_Model.embedding_

        ## Plot UMAP
        Plot_UMAP(Embedding,Human_Sample_Info,save_name=f'_{n}_genes_{test}_rawweights')
    #plt.show()

plot_cells_weights(Feature_Weights_chip, Subset_Used_Features=Used_Features_cESFW, test='chisq')
plot_cells_weights(Feature_Weights_fetp, Subset_Used_Features=Used_Features_cESFW, test='fet')

plot_cells_normalised(Normalised_Network_Feature_Weights_chip, Subset_Used_Features=Used_Features_cESFW, test='chisq')
plot_cells_normalised(Normalised_Network_Feature_Weights_fetp, Subset_Used_Features=Used_Features_cESFW, test='fet')

###############
