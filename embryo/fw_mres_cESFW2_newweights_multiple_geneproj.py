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

## Subset to top n genes
#Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)[np.arange(2250)] 
#Selected_Genes = Subset_Used_Features[Use_Inds]
#Selected_Genes.shape[0]
#np.isin(Known_Important_Genes,Selected_Genes)
#Known_Important_Genes[np.isin(Known_Important_Genes,Selected_Genes)]


n_genes = [9000, 7500, 6000,4000,2500, 2250]  
print(f'\nProceeding to plot the top ranked {n_genes} genes.')

random_seed = 42


def plot_gene_proj_normalised_rank(nw, w, score_matrix, n_genes):
    for n in n_genes:

        normalised_weights = nw
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


        plt.savefig(f"final_cESFW_gene_proj_nwproj_{n}_genes_newweights_short_2.png",dpi=600)
        
plot_gene_proj_normalised_rank(Normalised_Network_Feature_Weights, Feature_Weights, Masked_ESSs, n_genes)
#plot_gene_proj_normalised_rank(Normalised_Network_Feature_Weights_fetp, Feature_Weights_fetp, Masked_ESS_fetp, test='fetp')


def plot_gene_proj_weight_rank(nw, w, score_matrix, n_genes):
    for n in n_genes:

        weights = w
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
        
        
        plt.savefig(f"final_cESFW_gene_proj_weightproj_{n}_genes_newweights_short_2.png",dpi=600)
        plt.close()
        
plot_gene_proj_weight_rank(Normalised_Network_Feature_Weights, Feature_Weights, Masked_ESSs, n_genes)
#plot_gene_proj_weight_rank(Normalised_Network_Feature_Weights_fetp, Feature_Weights_fetp, Masked_ESS_fetp, test='fetp')

print(f'\n Analysis complete')

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