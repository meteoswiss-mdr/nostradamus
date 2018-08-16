#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def pd_rr_create_y_yhsaf_Xraw(all_data, all_data_names, tarvar, cut_precip=True):
    
    ##############################

    ## input:
    # all_data: as created by pd_create_sets_ody_seviriIR_nwcsaf_addparams_2017.ipynb (or rr_....) for m samples and n features: (m,n)
    # all_data_names: names of the n features
    # tarvar: 'pd' for precip detection, 'rr' for rain rate
    # cut_precip: tells me wether I want to keep only the actually precipitating ody pixel (True) or all the predicted precipitating pixels (False)

    ## output:
    # y, yhsaf, Xraw
    
    ##############################


    ##############################        
    ## create y & yhsaf   
    ##############################

    print('input all_data shape:', all_data.shape)
    
    if tarvar == 'pd':
        # create y & yhsaf
        y = all_data[:,0] >= 0.3
        yhsaf = np.logical_and(all_data[:,1] < 9999,all_data[:,1] >= 0.3)
        

    if tarvar == 'rr':

        if cut_precip == True:
            # only keep the precipitating pixels:
            idx_rr = all_data[:,0] >= 0.3  # just a lower bc because the unreasonably high values are cut in the *_create_sets_* scripts already
            all_data = all_data[idx_rr,:]
            print('cut all the samples where ody < 0.3 mm/h')

        if cut_precip == False:
            # keep all the input pixels:
            print('keep all the input samples')


        # create y & yhsaf
        y = all_data[:,0]
        yhsaf = all_data[:,1]
        yhsaf[yhsaf>9998] = 0.0

    print('target variable:', tarvar)
    print('all_data shape:', all_data.shape)
    ##############################


    ##############################        
    ## create X_raw   
    ##############################

    
    ## set up arrays
    
    nsamp = all_data.shape[0]
    nvars = (all_data.shape[1]-3)+ 1 + 21 + 1 + 8 + 2
            # (original-ody&hsaf&idx) + 1 add lst + all IR_chan_diff + 1 add CMa + 8 add CT + 2 add CT_PHASE
            
    X_raw = np.empty([nsamp,nvars])
    feature_list = np.empty(nvars).astype(list)

    n = 0
    
    print('X_raw shape:', X_raw.shape)

    
    ## constants fields
    
    # local solar time
    X_raw[:,n] = np.sin(all_data[:,2])
    feature_list[n] = 'sin lst'
    n+=1
    X_raw[:,n] = np.cos(all_data[:,2])
    feature_list[n] = 'cos lst'
    n+=1

    # lon/lat, land/sea mask, topo + viewing geomatry meteosat-9
    X_raw[:,n:n+6] = all_data[:,3:9]
    feature_list[n:n+6] = all_data_names[3:9]
    n+=6


    ## seviriIR

    # all IR channels
    X_raw[:,n:n+7] = all_data[:,9:16]
    feature_list[n:n+7] = all_data_names[9:16]
    n+=7

    # all IR channel differences
    for i in np.arange(9,16):
        for j in np.arange((i+1),16):
            X_raw[:,n]=all_data[:,i]-all_data[:,j]
            feature_list[n]=all_data_names[i]+' - '+all_data_names[j]
            n+=1   


    ## nwcsaf

    # continuous nwcsaf vars (CTT,CTP,CTH)
    X_raw[:,n:n+3] = all_data[:,19:22]
    feature_list[n:n+3] = all_data_names[19:22]
    n+=3

    # dummy vars for categorical nwcsaf vars (CMa, CT, CT_PHASE)
        # PROBLEMS: - It would be better if I used OneHotEncoder of the sklearn package but I have not figured out yet, how it works
        #           - I code a seperate dummy var for each cat. Full information I'd also get with 1 less dummy var (i.e. meaning in all 0) -> no clue how that affects my algorithm (& the problems I have with correlated vars)    
    all_cat_CMa = [2,3]
    names_CMa = ['CMa cont','CMa filled']
    for i,name in zip(all_cat_CMa,names_CMa): 
        X_raw[:,n]=all_data[:,16]==i 
        feature_list[n]=name
        n+=1
    all_cat_CT = [6,8,10,12,14,15,16,17,18]
    names_CT = ['vl','l','m','h o','vh o','h st thin','h st mthick', 'h st thick','h st above lm']
    for i,name in zip(all_cat_CT,names_CT): 
        X_raw[:,n]=all_data[:,17]==i 
        feature_list[n]=name
        n+=1
    all_cat_CT_PHASE = [1,2,3]
    names_CT_PHASE = ['w','i','undef']
    for i,name in zip(all_cat_CT_PHASE,names_CT_PHASE): 
        X_raw[:,n]=all_data[:,18]==i 
        feature_list[n]=name
        n+=1

    return y, yhsaf, X_raw, feature_list
