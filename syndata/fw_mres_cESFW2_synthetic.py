### cESFW filtering + running SYNTHETIC ###
##### Dependencies #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import pickle
from functools import partial
import multiprocess
from p_tqdm import p_map

## Load human pre and post implantation embryo sample info and scRNA-seq counts matrix.
# Human_Sample_Info = pd.read_csv("Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv("Complete_Synthetic_Data.csv",header=0,index_col=0)

## Create the scaled matrix from the scRNA-seq counts matrix
Scaled_Matrix = Human_Embryo_Counts.copy()

## Clip expression of each gene USING THE ZERO!!! nonzero does not work for some reason...
Upper = np.percentile(Scaled_Matrix,97.5,axis=0)
Upper[np.where(Upper == 0)[0]] = np.max(Scaled_Matrix,axis=0)[np.where(Upper == 0)[0]]
Scaled_Matrix = Scaled_Matrix.clip(upper=Upper,axis=1) 

## Normalise each feature/gene of the clipped matrix
Normalisation_Values = np.max(Scaled_Matrix,axis=0)
Scaled_Matrix = Scaled_Matrix / Normalisation_Values

## Remove confounding genes that are specifically expressed in one of the datasets (possible batch effect genes)
# Currently this step relies on EP > 0 with regard to a dummy feature representing an individual dataset.
# This potentially biases the downstream analysis to Entropy Sorting, and as such, we should try creating an equivilent
# function that utalises chi-squared significant that is used for the chi-squared workflow.

##
Feature_IDs = Scaled_Matrix.columns 
#Initial_Used_Features = Feature_IDs.copy()
#Dataset_IDs = np.unique(Human_Sample_Info["Dataset"])
## Track ESSs and EPs of each gene in relation to each dataset
#Dataset_ESSs = np.zeros((Initial_Used_Features.shape[0],Dataset_IDs.shape[0]))
#Dataset_EPs = np.zeros((Initial_Used_Features.shape[0],Dataset_IDs.shape[0]))
print(f'\nShape is: {Scaled_Matrix.shape}')
#necessary functions

def Calc_ESSs_EPs(Feature_Ind,Sample_Cardinality,Feature_Sums):
    ## Extract the Fixed Feature (FF)
    Fixed_Feature = Global_Scaled_Matrix[:,Feature_Ind].reshape(Sample_Cardinality,1)
    Fixed_Feature_Cardinality = Feature_Sums[Feature_Ind]
    ##
    Minority_States = Feature_Sums.copy()
    Switch = np.where(Minority_States >= (Sample_Cardinality/2))[0]
    Minority_States[Switch] = Sample_Cardinality - Minority_States[Switch]
    ## Identify where FF is the QF or RF
    FF_QF_Vs_RF = np.zeros(Feature_Sums.shape[0])
    FF_QF_Vs_RF[np.where(Minority_States[Feature_Ind] > Minority_States)[0]] = 1 # 1's mean FF is QF
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair 
    RFms = Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 0)[0]
    RFms[Switch] = Minority_States[Feature_Ind]
    RFMs = Sample_Cardinality - RFms
    QFms = Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 1)[0]
    QFms[Switch] = Minority_States[Feature_Ind]
    QFMs = Sample_Cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    Max_Ent_x_mm = (RFms * QFms)/(RFms + RFMs)
    Max_Ent_x_Mm = (QFMs * RFms)/(RFms + RFMs)
    Max_Ent_x_mM = (RFMs * QFms)/(RFms + RFMs)
    Max_Ent_x_MM = (RFMs * QFMs)/(RFms + RFMs)
    Max_Ent_Options = np.array([Max_Ent_x_mm,Max_Ent_x_Mm,Max_Ent_x_mM,Max_Ent_x_MM])
    ######
    ## Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    All_Use_Cases, All_Overlaps_Options = Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF)
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, Divergent_EPs, Overhang_EPs, Max_CEs = Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases) 
    return ESSs, Divergent_EPs, Overhang_EPs, Max_CEs


def Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF):
    ## Set up an array to track which of ESE equations 1-4 the recorded observed overlap relates to (row), and if it is 
    # native correlation (1) or flipped anti-correlation (-1). Row 1 = mm, row 2 = Mm, row 3 = mM, row 4 = MM.
    # FIRST LETTER IS QUERY, LAST IS REFERENCE!
    
    All_Use_Cases = np.zeros((4,Feature_Sums.shape[0]))
    ## Set up and away to track the observed overlaps between the FF and the secondary features.
    All_Overlaps_Options = np.zeros((4,Feature_Sums.shape[0]))
    ## Identify the non-zero inds in the FF, since samples with 0 can never observe overlap, and hence no need to do calculations.
    Non_Zero_Inds = np.where(Fixed_Feature != 0)[0]
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality < (Sample_Cardinality / 2)):
        #######
        ## FF and other feature are minority states & FF is QF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        ### if FF is QF, then FF is in the first position; for anticorrelation, we flip secondary feature, so second letter, row 3
        
        All_Use_Cases[:,Calc_Inds] = np.array([1,0,-1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_mm  
        All_Overlaps_Options[2,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_mM
        ####### 
        ## FF and other feature are minority states & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        ### if FF is RF, then FF is in the second position; for anticorrelation, we flip secondary feature, so first letter, row 2
            ##### all other comparisons follow this logic

        All_Use_Cases[:,Calc_Inds] = np.array([1,-1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_mm
        All_Overlaps_Options[1,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_Mm
        #######
        ## FF is minority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,0,1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_mm
        All_Overlaps_Options[2,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_mM
        #######
        ## FF is minority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_mm
        All_Overlaps_Options[1,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_Mm
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality >= (Sample_Cardinality / 2)):
        #######
        ## FF is majority, other feature is minority & FF is QF 
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([0,1,0,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.        
        All_Overlaps_Options[1,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_Mm
        All_Overlaps_Options[3,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_MM
        #######
        ## FF is majority, other feature is minority & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,1,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations.     
        All_Overlaps_Options[2,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_mM
        All_Overlaps_Options[3,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_MM   
        #######
        ## FF is majority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        All_Use_Cases[:,Calc_Inds] = np.array([0,-1,0,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations. 
        All_Overlaps_Options[1,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_Mm
        All_Overlaps_Options[3,Calc_Inds] = np.sum(np.minimum(Fixed_Feature[Non_Zero_Inds],Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)]),axis=0) # Overlaps_MM
        #######
        ## FF is majority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,-1,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations 
        # and Global_Scaled_Matrix_Inverse for inverse observations. 
        All_Overlaps_Options[2,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix_Inverse[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0) # Overlaps_mM
        All_Overlaps_Options[3,Calc_Inds] = np.sum(np.minimum(Global_Scaled_Matrix[np.ix_(Non_Zero_Inds,Calc_Inds)],Fixed_Feature[Non_Zero_Inds]),axis=0)
        #
    return All_Use_Cases, All_Overlaps_Options


def Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases):
    ## Create variables to track caclulation outputs
    # Track the Observed, Local Minimum, Global Minimum and Maximum entropoes of the observed feature pairs
    Native_CEs = np.zeros((4,RFms.shape[0]))
    # Track the Observed, Local Minimum, Global Minimum and Maximum entropoes of the flipped secondary feature feature pairs
    Flipped_CEs = np.zeros((4,RFms.shape[0]))
    # Track all SDs, SWs and SGs
    All_SDs = np.zeros((4,RFms.shape[0]))
    All_SWs = np.zeros((4,RFms.shape[0]))
    All_SGs = np.zeros((4,RFms.shape[0]))
    All_Correlations = np.zeros((4,RFms.shape[0]))
    # Track the minumm and maximum boundary point overlap values for the relevent ESEs
    All_Min_xs = np.zeros((4,RFms.shape[0]))
    All_Max_xs = np.zeros((4,RFms.shape[0]))
    ###################
    ##### (1)  mm #####
    Use_Curve = 0
    ## Find the FF/SF pairs where we should use ESE (1) to calculate entropies
    Calc_Inds = np.where(All_Use_Cases[Use_Curve,:] != 0)[0]
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Min_x = np.repeat(0,Calc_Inds.shape[0])
        Max_x = np.minimum(RFms[Calc_Inds],QFms[Calc_Inds])
        Overlaps = All_Overlaps_Options[Use_Curve,Calc_Inds]
        # Track the Min and Max values for downstream EP calculations
        All_Min_xs[Use_Curve,Calc_Inds] = Min_x
        All_Max_xs[Use_Curve,Calc_Inds] = Max_x
        # Perform caclulations with ESE (1)
        Xs = np.stack((Overlaps,Min_x,Max_x,Max_Ent_x))
        G1 = (RFms[Calc_Inds]/Sample_Cardinality)*((-((Xs/RFms[Calc_Inds])*np.log2(Xs/RFms[Calc_Inds]))-(((RFms[Calc_Inds]-Xs)/RFms[Calc_Inds])*np.log2((RFms[Calc_Inds]-Xs)/RFms[Calc_Inds]))))
        G1[np.isnan(G1)] = 0
        G2 = (RFMs[Calc_Inds]/Sample_Cardinality)*((-(((QFms[Calc_Inds]-Xs)/RFMs[Calc_Inds])*np.log2((QFms[Calc_Inds]-Xs)/RFMs[Calc_Inds]))-(((RFMs[Calc_Inds]-QFms[Calc_Inds]+Xs)/RFMs[Calc_Inds])*np.log2((RFMs[Calc_Inds]-QFms[Calc_Inds]+Xs)/RFMs[Calc_Inds]))))
        G2[np.isnan(G2)] = 0
        CEs = G1 + G2
        # Save the CEs for the native FF/SF pairings
        Native_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == 1)[0]
        Native_CEs[:,Calc_Inds[Native_Inds]] = CEs[:,Native_Inds]
        # Save the CEs for the native FF/SF pairings
        Flipped_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == -1)[0]
        Flipped_CEs[:,Calc_Inds[Flipped_Inds]] = CEs[:,Flipped_Inds]
        # Caclulate ESSs
        SD = np.zeros(Calc_Inds.shape[0])
        Correlation = np.zeros(Calc_Inds.shape[0])
        #
        ESS_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD[ESS_Inds] = -1
        Correlation[ESS_Inds] = -1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[1,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[1,ESS_Inds])
        #
        ESS_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SD[ESS_Inds] = 1
        Correlation[ESS_Inds] = 1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[2,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[2,ESS_Inds])     
        # 
        # Flip_SDs = np.where(CEs[1,:] < CEs[2,:])[0]
        # SD[Flip_SDs] = SD[Flip_SDs] * -1
        All_SDs[Use_Curve,Calc_Inds] = SD
        All_Correlations[Use_Curve,Calc_Inds] = Correlation
    ###################
    ##### (2)  Mm #####
    Use_Curve = 1
    ## Find the FF/SF pairs where we should use ESE (2) to calculate entropies
    Calc_Inds = np.where(All_Use_Cases[Use_Curve,:] != 0)[0]
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Min_x = np.maximum(0, QFMs[Calc_Inds]-RFMs[Calc_Inds]) # QFMs[Calc_Inds]-RFMs[Calc_Inds]??
        Max_x = np.minimum(RFms[Calc_Inds],QFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        # Track the Min and Max values for downstream EP calculations
        All_Min_xs[Use_Curve,Calc_Inds] = Min_x
        All_Max_xs[Use_Curve,Calc_Inds] = Max_x
        # Perform caclulations with ESE (2)
        Xs = np.stack((Overlaps,Min_x,Max_x,Max_Ent_x))
        G1 = (RFms[Calc_Inds]/Sample_Cardinality)*((-(((RFms[Calc_Inds]-Xs)/RFms[Calc_Inds])*np.log2((RFms[Calc_Inds]-Xs)/RFms[Calc_Inds]))-(((Xs)/RFms[Calc_Inds])*np.log2((Xs)/RFms[Calc_Inds]))))
        G1[np.isnan(G1)] = 0
        G2 = (RFMs[Calc_Inds]/Sample_Cardinality)*((-(((RFMs[Calc_Inds]-QFMs[Calc_Inds]+Xs)/RFMs[Calc_Inds])*np.log2((RFMs[Calc_Inds]-QFMs[Calc_Inds]+Xs)/RFMs[Calc_Inds]))-(((QFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds])*np.log2((QFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds]))))
        G2[np.isnan(G2)] = 0
        CEs = G1 + G2
        # Save the CEs for the native FF/SF pairings
        Native_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == 1)[0]
        Native_CEs[:,Calc_Inds[Native_Inds]] = CEs[:,Native_Inds]
        # Save the CEs for the native FF/SF pairings
        Flipped_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == -1)[0]
        Flipped_CEs[:,Calc_Inds[Flipped_Inds]] = CEs[:,Flipped_Inds]
        # Caclulate ESSs
        SD = np.zeros(Calc_Inds.shape[0])
        Correlation = np.zeros(Calc_Inds.shape[0])
        #
        ESS_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD[ESS_Inds] = -1
        Correlation[ESS_Inds] = 1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[1,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[1,ESS_Inds])
        #
        ESS_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SD[ESS_Inds] = 1
        Correlation[ESS_Inds] = -1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[2,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[2,ESS_Inds])
        #  
        # Flip_SDs = np.where(CEs[1,:] < CEs[2,:])[0]
        # SD[Flip_SDs] = SD[Flip_SDs] * -1 
        All_SDs[Use_Curve,Calc_Inds] = SD
        All_Correlations[Use_Curve,Calc_Inds] = Correlation
    ###################
    ##### (3)  mM #####
    Use_Curve = 2
    ## Find the FF/SF pairs where we should use ESE (3) to calculate entropies
    Calc_Inds = np.where(All_Use_Cases[Use_Curve,:] != 0)[0]
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Min_x = np.maximum(0, RFMs[Calc_Inds]-QFMs[Calc_Inds])
        Max_x = np.minimum(QFms[Calc_Inds],RFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        # Track the Min and Max values for downstream EP calculations
        All_Min_xs[Use_Curve,Calc_Inds] = Min_x
        All_Max_xs[Use_Curve,Calc_Inds] = Max_x
        # Perform caclulations with ESE (3)
        Xs = np.stack((Overlaps,Min_x,Max_x,Max_Ent_x))
        G1 = (RFms[Calc_Inds]/Sample_Cardinality)*((-(((QFms[Calc_Inds]-Xs)/RFms[Calc_Inds])*np.log2((QFms[Calc_Inds]-Xs)/RFms[Calc_Inds]))-(((RFms[Calc_Inds]-QFms[Calc_Inds]+Xs)/RFms[Calc_Inds])*np.log2((RFms[Calc_Inds]-QFms[Calc_Inds]+Xs)/RFms[Calc_Inds]))))
        G1[np.isnan(G1)] = 0
        G2 = (RFMs[Calc_Inds]/Sample_Cardinality)*((-(((Xs)/RFMs[Calc_Inds])*np.log2((Xs)/RFMs[Calc_Inds]))-(((RFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds])*np.log2((RFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds]))))
        G2[np.isnan(G2)] = 0
        CEs = G1 + G2
        # Save the CEs for the native FF/SF pairings
        Native_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == 1)[0]
        Native_CEs[:,Calc_Inds[Native_Inds]] = CEs[:,Native_Inds]
        # Save the CEs for the native FF/SF pairings
        Flipped_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == -1)[0]
        Flipped_CEs[:,Calc_Inds[Flipped_Inds]] = CEs[:,Flipped_Inds]
        # Caclulate ESSs
        SD = np.zeros(Calc_Inds.shape[0])
        Correlation = np.zeros(Calc_Inds.shape[0])
        #
        ESS_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD[ESS_Inds] = -1
        Correlation[ESS_Inds] = 1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[1,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[1,ESS_Inds])
        #
        ESS_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SD[ESS_Inds] = 1
        Correlation[ESS_Inds] = -1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[2,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[2,ESS_Inds]) 
        #
        # Flip_SDs = np.where(CEs[1,:] < CEs[2,:])[0]
        # SD[Flip_SDs] = SD[Flip_SDs] * -1
        All_SDs[Use_Curve,Calc_Inds] = SD
        All_Correlations[Use_Curve,Calc_Inds] = Correlation
    ###################
    ##### (4)  MM #####
    Use_Curve = 3
    ## Find the FF/SF pairs where we should use ESE (4) to calculate entropies
    Calc_Inds = np.where(All_Use_Cases[Use_Curve,:] != 0)[0]
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Min_x = QFMs-RFms
        Max_x = np.minimum(QFMs[Calc_Inds],RFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        # Track the Min and Max values for downstream EP calculations
        All_Min_xs[Use_Curve,Calc_Inds] = Min_x
        All_Max_xs[Use_Curve,Calc_Inds] = Max_x
        # Perform caclulations with ESE (4)
        Xs = np.stack((Overlaps,Min_x,Max_x,Max_Ent_x))
        G1 = (RFms[Calc_Inds]/Sample_Cardinality)*((-(((RFms[Calc_Inds]-QFMs[Calc_Inds]+Xs)/RFms[Calc_Inds])*np.log2((RFms[Calc_Inds]-QFMs[Calc_Inds]+Xs)/RFms[Calc_Inds]))-(((QFMs[Calc_Inds]-Xs)/RFms[Calc_Inds])*np.log2((QFMs[Calc_Inds]-Xs)/RFms[Calc_Inds]))))
        G1[np.isnan(G1)] = 0
        G2 = (RFMs[Calc_Inds]/Sample_Cardinality)*((-(((RFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds])*np.log2((RFMs[Calc_Inds]-Xs)/RFMs[Calc_Inds]))-(((Xs)/RFMs[Calc_Inds])*np.log2((Xs)/RFMs))))
        G2[np.isnan(G2)] = 0
        CEs = G1 + G2
        # Save the CEs for the native FF/SF pairings
        Native_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == 1)[0]
        Native_CEs[:,Calc_Inds[Native_Inds]] = CEs[:,Native_Inds]
        # Save the CEs for the native FF/SF pairings
        Flipped_Inds = np.where(All_Use_Cases[Use_Curve,Calc_Inds] == -1)[0]
        Flipped_CEs[:,Calc_Inds[Flipped_Inds]] = CEs[:,Flipped_Inds]
        # Caclulate ESSs
        SD = np.zeros(Calc_Inds.shape[0])
        Correlation = np.zeros(Calc_Inds.shape[0])
        #
        ESS_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD[ESS_Inds] = -1
        Correlation[ESS_Inds] = -1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[1,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[1,ESS_Inds])
        #
        ESS_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SD[ESS_Inds] = 1
        Correlation[ESS_Inds] = 1
        All_SWs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[2,ESS_Inds]) / CEs[3,ESS_Inds]
        All_SGs[Use_Curve,Calc_Inds[ESS_Inds]] = (CEs[3,ESS_Inds] - CEs[0,ESS_Inds]) / (CEs[3,ESS_Inds] - CEs[2,ESS_Inds])   
        #
        # Flip_SDs = np.where(CEs[1,:] < CEs[2,:])[0]
        # SD[Flip_SDs] = SD[Flip_SDs] * -1
        All_SDs[Use_Curve,Calc_Inds] = SD
        All_Correlations[Use_Curve,Calc_Inds] = Correlation
    ## Multiply components to get ESS
    All_ESSs = All_SGs * All_SWs * All_Correlations
    ###################
    #### Calc EPs ####
    Divergent_EPs = np.zeros(RFms.shape[0])
    Overhang_EPs = np.zeros(RFms.shape[0])
    #
    Max_ESS_Inds = np.argmax(np.absolute(All_ESSs),axis=0)
    ESS_0_Inds = np.where(np.max(All_ESSs,axis=0) == 0)[0]
    Max_ESS_Inds[ESS_0_Inds] = np.argmax(All_SDs[:,ESS_0_Inds],axis=0)
    Max_ESS_Use_Cases = All_Use_Cases[Max_ESS_Inds,np.arange(RFms.shape[0])]
    Max_ESS_Max_Ents = Max_Ent_Options[Max_ESS_Inds,np.arange(RFms.shape[0])]
    Max_ESS_Overlaps = All_Overlaps_Options[Max_ESS_Inds,np.arange(RFms.shape[0])]
    Max_ESS_Min_xs = All_Min_xs[Max_ESS_Inds,np.arange(RFms.shape[0])]
    Max_ESS_Max_xs = All_Max_xs[Max_ESS_Inds,np.arange(RFms.shape[0])]
    Max_ESS_SDs = All_SDs[Max_ESS_Inds,np.arange(RFms.shape[0])]
    #
    CEs = np.zeros(Native_CEs.shape)
    Use_Case_Inds_1 = np.where(Max_ESS_Use_Cases==1)[0]
    CEs[:,Use_Case_Inds_1] = Native_CEs[:,Use_Case_Inds_1]
    Use_Case_Inds_2 = np.where(Max_ESS_Use_Cases==-1)[0]
    CEs[:,Use_Case_Inds_2] = Flipped_CEs[:,Use_Case_Inds_2]
    #
    Num_Divegent_Cells = np.zeros(Max_ESS_Inds.shape[0])
    Num_Overhang_Cells = np.zeros(Max_ESS_Inds.shape[0])
    ## mm (1), SD = -1
    Inds = np.where((Max_ESS_Inds == 0) & (Max_ESS_SDs == -1))[0]
    Num_Divegent_Cells[Inds] = Max_ESS_Overlaps[Inds]
    Num_Overhang_Cells[Inds] = Sample_Cardinality - (QFms[Inds] + RFms[Inds]) + Max_ESS_Overlaps[Inds]
    ## mm (1), SD = 1
    Inds = np.where((Max_ESS_Inds == 0) & (Max_ESS_SDs == 1))[0]
    Num_Divegent_Cells[Inds] = RFms[Inds] - Max_ESS_Overlaps[Inds]
    Num_Overhang_Cells[Inds] = QFms[Inds] - Max_ESS_Overlaps[Inds] 
    ## Mm (2), SD = -1
    Inds = np.where((Max_ESS_Inds == 1) & (Max_ESS_SDs == -1))[0]
    Num_Divegent_Cells[Inds] = Max_ESS_Overlaps[Inds]
    Num_Overhang_Cells[Inds] = QFms[Inds] - RFms[Inds] + Max_ESS_Overlaps[Inds]
    ## Mm (2), SD = 1
    Inds = np.where((Max_ESS_Inds == 1) & (Max_ESS_SDs == 1))[0]
    Num_Divegent_Cells[Inds] = (RFms[Inds] - Max_ESS_Overlaps[Inds]) # Changed
    Num_Overhang_Cells[Inds] = (QFMs[Inds] - RFms[Inds] + Num_Divegent_Cells[Inds]) # Changed
    ## mM (3), SD = -1
    Inds = np.where((Max_ESS_Inds == 2) & (Max_ESS_SDs == -1))[0]
    Num_Overhang_Cells[Inds] = Max_ESS_Overlaps[Inds]
    Num_Divegent_Cells[Inds] = (QFMs[Inds] - RFMs[Inds]) + Num_Overhang_Cells[Inds] # Changed
    ## mM (3), SD = 1
    Inds = np.where((Max_ESS_Inds == 2) & (Max_ESS_SDs == 1))[0]
    Num_Divegent_Cells[Inds] = QFms[Inds] - Max_ESS_Overlaps[Inds]
    Num_Overhang_Cells[Inds] = RFMs[Inds] - Max_ESS_Overlaps[Inds] # Changed Sample_Cardinality - (QFms[Inds] + RFms[Inds]) + Max_ESS_Overlaps[Inds]
    ## MM, SD = -1
    Inds = np.where((Max_ESS_Inds == 3) & (Max_ESS_SDs == -1))[0]
    Num_Divegent_Cells[Inds] = Max_ESS_Overlaps[Inds] - (Sample_Cardinality - (QFms[Inds] + RFms[Inds]))
    Num_Overhang_Cells[Inds] = Max_ESS_Overlaps[Inds] # Changed (Sample_Cardinality - (QFms[Inds] + RFms[Inds])) + Num_Divegent_Cells[Inds]
    ## MM, SD = 1
    Inds = np.where((Max_ESS_Inds == 3) & (Max_ESS_SDs == 1))[0]
    Num_Divegent_Cells[Inds] = QFMs[Inds] - Max_ESS_Overlaps[Inds] # Changed
    Num_Overhang_Cells[Inds] = RFMs[Inds] - QFMs[Inds] + Num_Divegent_Cells[Inds] # Changed
    #
    Num_Divegent_Cells[Num_Divegent_Cells < 0.0000001] = 0
    Num_Overhang_Cells[Num_Overhang_Cells < 0.0000001] = 0
    #
    EP_Inds = np.where(Max_ESS_SDs == -1)[0]
    DPCd = (CEs[0,EP_Inds] - CEs[1,EP_Inds]) / Num_Divegent_Cells[EP_Inds]
    DPCo = CEs[0,EP_Inds] / Num_Overhang_Cells[EP_Inds] # DPC overhang
    DPCi = CEs[3,EP_Inds] / (Max_ESS_Max_Ents[EP_Inds] - Max_ESS_Min_xs[EP_Inds])
    Divergent_EPs[EP_Inds] = DPCd - DPCi
    Overhang_EPs[EP_Inds] = DPCo - DPCi
    # EPs: SDs = 1
    EP_Inds = np.where(Max_ESS_SDs == 1)[0]
    # Divergent_Cells = Max_ESS_Max_xs[EP_Inds] - Max_ESS_Overlaps[EP_Inds]
    # Divergent_Cells[Divergent_Cells < 0.0000001] = 0
    DPCd = (CEs[0,EP_Inds] - CEs[2,EP_Inds]) / Num_Divegent_Cells[EP_Inds]
    DPCo = CEs[0,EP_Inds] / Num_Overhang_Cells[EP_Inds] # DPC overhang
    DPCi = CEs[3,EP_Inds] / (Max_ESS_Max_xs[EP_Inds] - Max_ESS_Max_Ents[EP_Inds])
    Divergent_EPs[EP_Inds] = DPCd - DPCi
    Overhang_EPs[EP_Inds] = DPCo - DPCi
    return All_ESSs[Max_ESS_Inds,np.arange(RFms.shape[0])], Divergent_EPs, Overhang_EPs, CEs[3,:]

########

def Calc_Individual_ESSs_EPs(Fixed_Feature, Scaled_Matrix):
    Sample_Cardinality = Fixed_Feature.shape[0]
    Fixed_Feature = Fixed_Feature.reshape(Sample_Cardinality,1)
    #
    ## Save feature ID names for later
    Feature_IDs = Scaled_Matrix.columns
    ## Create the global Global_Scaled_Matrix array for faster parallel computing calculations
    global Global_Scaled_Matrix
    Global_Scaled_Matrix = np.asarray(Scaled_Matrix)
    ## Create the global Global_Scaled_Matrix_Inverse array for faster parallel computing calculations
    global Global_Scaled_Matrix_Inverse
    Global_Scaled_Matrix_Inverse = 1 - Global_Scaled_Matrix
    ## Extract sample cardinality
    Sample_Cardinality = Global_Scaled_Matrix.shape[0]
    ## Calculate feature sums
    Feature_Sums = np.asarray(np.sum(Global_Scaled_Matrix,axis=0))
    #
    Fixed_Feature_Cardinality = np.sum(Fixed_Feature)
    Feature_Sums = np.asarray(np.sum(Global_Scaled_Matrix,axis=0))
    #
    FF_QF_Vs_RF = np.zeros(Feature_Sums.shape[0])
    FF_QF_Vs_RF[np.where(Fixed_Feature_Cardinality > Feature_Sums)[0]] = 1 # 1's mean FF is QF
    #
    RFms = Feature_Sums.copy()
    Switch = np.where(FF_QF_Vs_RF == 0)[0]
    RFms[Switch] = Fixed_Feature_Cardinality
    RFMs = Sample_Cardinality - RFms
    QFms = Feature_Sums.copy()
    Switch = np.where(FF_QF_Vs_RF == 1)[0]
    QFms[Switch] = Fixed_Feature_Cardinality
    QFMs = Sample_Cardinality - QFms
    #
    Max_Ent_x_mm = (RFms * QFms)/(RFms + RFMs)
    Max_Ent_x_Mm = (QFMs * RFms)/(RFms + RFMs)
    Max_Ent_x_mM = (RFMs * QFms)/(RFms + RFMs)
    Max_Ent_x_MM = (RFMs * QFMs)/(RFms + RFMs)
    Max_Ent_Options = np.array([Max_Ent_x_mm,Max_Ent_x_Mm,Max_Ent_x_mM,Max_Ent_x_MM])
    ###################
    All_Use_Cases, All_Overlaps_Options = Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF)
    ESSs, Divergent_EPs, Overhang_EPs, Max_CEs = Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases) 
    Divergent_EPs[np.isnan(Divergent_EPs)] = 0
    Divergent_EPs[np.isinf(Divergent_EPs)] = 0
    Overhang_EPs[np.isnan(Overhang_EPs)] = 0
    Overhang_EPs[np.isinf(Overhang_EPs)] = 0
    return pd.DataFrame(ESSs.reshape(1,Feature_IDs.shape[0]), columns=Feature_IDs), pd.DataFrame(Divergent_EPs.reshape(1,Feature_IDs.shape[0]), columns=Feature_IDs), pd.DataFrame(Overhang_EPs.reshape(1,Feature_IDs.shape[0]), columns=Feature_IDs)

### now we conduct the analysis

def Parallel_Calc_ESSs_EPs(Scaled_Matrix,Use_Cores=30):
    ## Save feature ID names for later
    Feature_IDs = Scaled_Matrix.columns
    ## Create the global Global_Scaled_Matrix array for faster parallel computing calculations
    global Global_Scaled_Matrix
    Global_Scaled_Matrix = np.asarray(Scaled_Matrix)
    ## Create the global Global_Scaled_Matrix_Inverse array for faster parallel computing calculations
    global Global_Scaled_Matrix_Inverse
    Global_Scaled_Matrix_Inverse = 1 - Global_Scaled_Matrix
    ## Extract sample and feature cardinality
    Sample_Cardinality = Global_Scaled_Matrix.shape[0]
    Feature_Cardinality = Global_Scaled_Matrix.shape[1]
    ## Calculate feature sums
    Feature_Sums = np.asarray(np.sum(Global_Scaled_Matrix,axis=0))
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(Feature_Cardinality)
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    if __name__ == '__main__':
        with np.errstate(divide='ignore',invalid='ignore'):
            Results = p_map(partial(Calc_ESSs_EPs,Sample_Cardinality=Sample_Cardinality,Feature_Sums=Feature_Sums), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results: [ESS, DivEP, OverhangEP, maxCE]
    Results = np.asarray(Results)
    ESSs = Results[:,0]
    Divergent_EPs = Results[:,1]
    Divergent_EPs[np.isnan(Divergent_EPs)] = 0
    Divergent_EPs[np.isinf(Divergent_EPs)] = 0
    Overhang_EPs = Results[:,2]
    Overhang_EPs[np.isnan(Overhang_EPs)] = 0
    Overhang_EPs[np.isinf(Overhang_EPs)] = 0
    Max_CEs = Results[:,3]
    #
    # ESSs[ESSs < 0.0000001] = 0
    ESS_0_Inds = np.where(ESSs == 0)
    Inverse_ESS_0_Inds = (ESS_0_Inds[1],ESS_0_Inds[0])
    # why is this here???
    Values = Divergent_EPs[ESS_0_Inds]
    Inverse_Values = Divergent_EPs[Inverse_ESS_0_Inds]
    Substitute_Inds = np.where(Values < Inverse_Values)[0]
    Values[Substitute_Inds] = Inverse_Values[Substitute_Inds]
    Divergent_EPs[ESS_0_Inds] = Values
    # Same for overhangs
    Values = Overhang_EPs[ESS_0_Inds]
    Inverse_Values = Overhang_EPs[Inverse_ESS_0_Inds]
    Substitute_Inds = np.where(Values < Inverse_Values)[0]
    Values[Substitute_Inds] = Inverse_Values[Substitute_Inds]
    Overhang_EPs[ESS_0_Inds] = Values
    #
    return ESSs, Divergent_EPs, Overhang_EPs, Max_CEs


## Calculate ESSs and EPs
ESSs, Divergent_EPs, Overhang_EPs, Max_CEs = Parallel_Calc_ESSs_EPs(Scaled_Matrix)

## Accepted EPs are the maximum of Divergent_EPs and Overhang_EPs
EPs = np.maximum(Divergent_EPs,Overhang_EPs)
column_name = Scaled_Matrix.columns

ESSs = pd.DataFrame(data=ESSs, columns=column_name, index=column_name)
ESSs.to_csv('final_ESSs_new_cESFW_synthetic.csv')

EP = pd.DataFrame(data=EPs, columns=column_name, index=column_name)
EP.to_csv('final_EP_new_cESFW_synthetic.csv')

#dEP = pd.DataFrame(data=Divergent_EPs, columns=column_name, index=column_name)
#dEP.to_csv('dEP_new_cESFW_embryo.csv')

#ohEP = pd.DataFrame(data=Overhang_EPs, columns=column_name, index=column_name)
#ohEP.to_csv('ohEP_new_cESFW_embryo.csv')

maxCE = pd.DataFrame(data=Max_CEs, columns=column_name, index=column_name)
maxCE.to_csv('maxCE_new_cESFW_synthetic.csv')