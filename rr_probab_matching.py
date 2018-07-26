#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# to create y, yhsaf & Xraw
from pd_rr_create_y_yhsaf_Xraw_2017 import pd_rr_create_y_yhsaf_Xraw


def probab_match_rr_refprovide(ody_rr_ref, pred_rr_ref, rr_tmp):

    ##############################
    ## purpose:
    # calculate probability matched rainfall rates based on a given reference set (i.e. ody rain rates & the associated predicted rain rates)
    
    ## input:
    # ody_rr_ref: 1d array with the reference ody rain rates
    # pred_rr_ref: 1d array with the reference predicted rain rates
    # rr_tmp: 1d array with the predicted rain rates for which we want to know the probability matches rain rates

    ## output:
    # rr_tmp_pm: 1d array with the probability matched predicted rain rates
    ##############################
    
    perc_pred_ref=np.percentile(pred_rr_ref,np.arange(0,100.001,0.001)) # used to be 0.01 but led to too many predictions at exactly same rr in predictions (however, as is now, is extremely slow! -> not feasible!)
    perc_ody_ref=np.percentile(ody_rr_ref,np.arange(0,100.001,0.001))

    idx = [(np.abs(perc_pred_ref-pred)).argmin() for pred in rr_tmp ]
    rr_tmp_pm = perc_ody_ref[idx]
    
    return rr_tmp_pm

def probab_match_rr_refcreate(all_data_ref,all_data_names_ref,scaler_rr,clf_rr,rr_tmp):

    ##############################
    ## purpose:
    # calculate probability matched rainfall rates based on a reference set (i.e. ody rain rates & the associated predicted rain rates) which is created from raw data insdie the function
    
    ## input:
    # all_data_ref: ody rr, hsaf rr, sat channels, nwcsaf products + add params mxn matrix with m samples & n variables of the reference data set
    # all_data_names_ref: list of the varialbes in all_data_ref
    # scaler_rr: scaler to standardize all_data according to the model employed for the prediction
    # clf_rr: classifier to generate the predicted reference rain rates
    # rr_tmp: 1d array with the predicted rain rates for which we want to know the probability matches rain rates

    ## output:
    # rr_tmp_pm: 1d array with the probability matched predicted rain rates
    ##############################
        
    # create the reference set for the pm
    ody_rr_ref, rr_hsaf_ref, X_raw_ref, feature_list_ref = pd_rr_create_y_yhsaf_Xraw(all_data_ref,all_data_names_ref,'rr',cut_precip=True)
    X_ref=scaler_rr.transform(X_raw_ref) 

    # check whether the X is reasonable
    #for i in range(len(feature_list_ref)):
    #    print(feature_list_ref[i],X_ref[:,i].max(),X_ref[:,i].min(),X_ref[:,i].mean())

    # make predictions fro the reference set
    pred_rr_ref=clf_rr.predict(X_ref)
        
    rr_tmp_pm = probab_match_rr_refprovide(ody_rr_ref,pred_rr_ref,rr_tmp)
        
    return rr_tmp_pm
