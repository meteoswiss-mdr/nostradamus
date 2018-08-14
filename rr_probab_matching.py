#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function 

import numpy as np
import time
from copy import deepcopy

# to create y, yhsaf & Xraw
from pd_rr_create_y_yhsaf_Xraw import pd_rr_create_y_yhsaf_Xraw


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

    print ("... rainrate min/max before prob matching", rr_tmp.min(), rr_tmp.max(), rr_tmp.shape) 
    
    t0 = time.time()

    # used to be 0.01 but led to too many predictions at exactly same rr in predictions
    # (however, as is now, is extremely slow with index matching! -> use spline interpolation )
    delta=0.001
    perc_min =   0.0
    perc_max = 100.0
    perc_grid = np.arange( perc_min, perc_max+delta, delta)
    
    #### 0.25 seconds
    perc_pred_ref = np.percentile( pred_rr_ref, perc_grid )
    
    t1 = time.time()
    #print("... elapsed time for pdf (predicted rain) in seconds: "+str(t1-t0))

    #### 0.20 seconds
    perc_ody_ref  = np.percentile( ody_rr_ref,  perc_grid )
    t2 = time.time()
    #print("... elapsed time for pdf (ody rain reference) in seconds: "+str(t2-t1))

    spline_interpolation = True
    if spline_interpolation:
        # add a few fake points for reasonable extrapolation
        # first index: predicted rainrate =-1.0mm/h -> odyssey rainrate =0.3mm/h
        pred = np.insert( perc_pred_ref,0, -1.0)
        ody  = np.insert( perc_ody_ref, 0,  0.3)
        # last index: predicted rainrate =20.0mm/h -> odyssey rainrate =140.0mm/h
        pred = np.insert(pred,-1,  20)
        ody  = np.insert(ody, -1, 140)
        # fit cubic spline function 
        #### linear 0.003 s / cubic 0.4 seconds
        from scipy.interpolate import interp1d
        spline = interp1d(pred, ody, kind='linear')
        #spline = interp1d(pred, ody, kind='cubic')
        t3 = time.time()
        #print("... elapsed time for fitting cubic spline: "+str(t3-t2))

        # interp1d does not support masked arrays
        #### linear 0.005 s / cubic 0.2 seconds
        if isinstance(rr_tmp, np.ma.MaskedArray):
            idx = (rr_tmp.mask==False)
            rr_data = rr_tmp.data[idx]
            rr_tmp_pm = deepcopy(rr_tmp)
            rr_tmp_pm[idx] = spline(rr_data)
        else:
            rr_tmp_pm = spline(rr_tmp)
                   
        t4 = time.time()
        #print("... elapsed time for evaluating cubic spline in seconds: "+str(t4-t3))

        if False:
            import matplotlib.pyplot as plt
            plt.plot(perc_pred_ref, perc_ody_ref)
            rr_grid=np.arange(-1, 18, 0.1)
            rr_ody_fit = spline(rr_grid)
            plt.plot(rr_grid, rr_ody_fit)
            plt.show()
    else:
        # search index with minimum absolute difference between predicted rain and reference pdf 
        # 44 seconds
        idx = [(np.abs(perc_pred_ref-pred)).argmin() for pred in rr_tmp ]
        t3 = time.time()
        #print("... elapsed time for searching indices in seconds: "+str(t3-t2))

        # get reference rainfall rate for all indices and save as probability matched rainrate
        # 0.003 seconds
        rr_tmp_pm = perc_ody_ref[idx]
        t4 = time.time()
        #print("... elapsed time for index matching in seconds: "+str(t4-t3))

    print ("... rainrate min/max after prob matching", rr_tmp_pm.min(), rr_tmp_pm.max(), rr_tmp_pm.shape) 
        
    return rr_tmp_pm

######################################################################################################

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
