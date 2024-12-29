##### Dependencies #####

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
##### Dependencies #####

##### Functions #####

## Load human pre and post implantation embryo sample info and scRNA-seq counts matrix.
Human_Sample_Info = pd.read_csv("Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv("Human_Embryo_Counts.csv",header=0,index_col=0)

Masked_ESSs = pd.read_csv("cESFW_cESFW-subset_newweights_Masked_ESSs.csv",header=0,index_col=0)

Subset_Used_Features = np.array(pd.read_csv("cESFW_cESFW-subset_used_feature_weights_newweights.csv",header=0,index_col=0))

Feature_Weights = np.array(pd.read_csv("cESFW_cESFW-subset_weights_newweights.csv",header=0,index_col=0))

Normalised_Network_Feature_Weights = np.array(pd.read_csv("cESFW_cESFW-subset_normalised_weights_newweights.csv",header=0,index_col=0))


Subset_Used_Features = Subset_Used_Features.T[0]
Feature_Weights = Feature_Weights.T[0]
Normalised_Network_Feature_Weights = Normalised_Network_Feature_Weights.T[0]


n_genes = [9000,7500,6000,4000,2500,2250]  
print(f'\nProceeding to plot the top ranked {n_genes} genes, plotting CELLS.')
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
    plt.savefig(f"timepoints_{save_name}_cESFW2_newweights_2.png",dpi=600)
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
    plt.savefig(f"datasets_{save_name}_cESFW2_newweights_2.png",dpi=600)
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
    plt.savefig(f"cellannot_{save_name}_cESFW2_newweights_2.png",dpi=600)

n_genes = [9000, 7500, 6000,4000,2500, 2250]

def plot_cells_normalised(Normalised_Network_Feature_Weights, Subset_Used_Features):
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
        Plot_UMAP(Embedding,Human_Sample_Info,save_name=f'{n}_genes_normalised')
    #plt.show()

def plot_cells_weights(Feature_Weights, Subset_Used_Features):
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
        Plot_UMAP(Embedding,Human_Sample_Info,save_name=f'{n}_genes_rawweights')
    #plt.show()

plot_cells_normalised(Normalised_Network_Feature_Weights, Subset_Used_Features)
plot_cells_weights(Feature_Weights, Subset_Used_Features)


print(f'\nAnalysis completed.')
## May choose to subset the genes further by selecting clusters/boxes. In this case, we are going to trim off a few extra
# genes that have exceptionally high Normalised_Network_Feature_Weights
# Cluster_Use_Gene_IDs = Selected_Genes[np.where(Gene_Embedding[:,0] > -10)[0]]
#Cluster_Use_Gene_IDs = Selected_Genes[np.where(Normalised_Network_Feature_Weights[Use_Inds] < np.percentile(Normalised_Network_Feature_Weights[Use_Inds],97.5))[0]]

## Create UMAP of data subsetted to selected genes
#Reduced_Input_Data = Human_Embryo_Counts[Cluster_Use_Gene_IDs]
#Neighbours = 50
#Dist = 0.15

#Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
#Embedding = Embedding_Model.embedding_

## Plot UMAP
#Plot_UMAP(Embedding,Human_Sample_Info,save_name='subset_trimmed')
#plt.show()

## Plot gene expression
#Gene = "PDGFRA"
#plt.figure(figsize=(7,7))
#plt.scatter(Embedding[:,0],Embedding[:,1],s=4,c=np.log2(Human_Embryo_Counts[Gene]+1),cmap="seismic")
#plt.savefig("PDGFRA_expression_embedding_withmt_newweights.png",dpi=600)
#plt.show()


### In case of interest, here is a 3D plot of the resulting UMAP

# Embedding_Model_3D = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=3).fit(Reduced_Input_Data)
# Embedding_3D = Embedding_Model_3D.embedding_

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# Annotations = np.asarray(Human_Sample_Info["Manual_Annotations"])
# Unique_Annotations = np.unique(Annotations)
# Unique_Annotations = Unique_Annotations[np.array([0,9,7,6,3,14,2,4,5,1,8,10,13,11,15,12])]
# for i in np.arange(Unique_Annotations.shape[0]):
#     IDs = np.where(Annotations == Unique_Annotations[i])[0]
#     ax.scatter(Embedding_3D[IDs,0], Embedding_3D[IDs,1], Embedding_3D[IDs,2],c=Colours[i])

# plt.show()