#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original name: pd_rr_preprocess_data_single_scene_ody_seviriIR_nwcsaf_addparams_2017.py

import numpy as np
from datetime import datetime

from mpop.satellites import GeostationaryFactory

# to get the local time
import ephem

# to load all the input data 
from pd_rr_load_fields import load_input, load_constant_fields

# to be able to create a sepearte copy of the dates which I shuffle around
import copy

# to be able to carry out a bash command from my script 
import os


def solartime(observer, sun=ephem.Sun()):
    sun.compute(observer)
    # sidereal time == ra (right ascension) is the highest point (noon)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm  # norm for 24h

def pd_rr_preprocess_data_single_scene(sat_nr, area, time_slot, par_fill, tarvar, read_HSAF=True):
    #########
    ## input
    # area: projection area
    # time_slot: time in UTC
    # par_fill: parallax corr gap filler: choose between 'False', 'nearest', and 'bilinear'
    # tar_var: target variable: choose between 'pd', 'rr'


    ## output
    # all_data: mxn matrix with m places where prediction is carried out & n input vars for pd_rr_create_y_yhsaf_Xraw
    # all_data_names: name of the input vars
    # mask_h: field indicating where NWCSAF products are available & thus where predictions are carried out: True if NWCSAF products available
    # mask_r: field indicating where radar products are available: True if radar product is available
    # mask_rnt: field indicating where radar product available but not trustworthy: i.e. in threshold_mask, 0<rr<0.3, rr>130 overlaid: True if radar product is NOT trustworthy
    # lon,lat: lon and lat field for plotting of the map
    #########
    
    ##set up the variable names
    
    # sat channel used
    channel_sat=['WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134']

    # nwc products used
    channel_nwc = ['CMa','CT','CT_PHASE','CTT','CTP']

    # list all var
    all_data_names = ['odyRR','hsafRR','lst','lon','lat','lsmask','topo','vaa','vza','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134','CMa','CT','CT_PHASE','CTT','CTP','CTH','idx']
    nr_vars = len(all_data_names)


    ## read in all the constants files
    
    print('=================================')
    print('*** load the constant fields (radar mask, viewing geometry, and land/sea mask plus surface elevation)')
    mask_rad_thres, vg, ls_ele = load_constant_fields(sat_nr, area) # mask_rad_thres = True where ody not trustworthy
    print('=================================')

    
    ## read in the time slot specific fields
    
    print('=================================')
    print('load the time slot specific fields with par_fill:', par_fill)
    global_radar, global_sat, global_nwc, global_cth, global_hsaf = load_input(sat_nr, area, time_slot,par_fill, read_HSAF=read_HSAF)
    print('=================================')

    
    ## create target variable
    print('=================================')
    print('create target variable:',tarvar )

    if tarvar == 'pd':
        # create y & yhsaf
        y = global_radar['RATE'].data.data >= 0.3
        if read_HSAF:
            yhsaf = np.logical_and(global_hsaf['h03_PC'].data.data < 9999, global_hsaf['h03_PC'].data.data >= 0.3)
        else:
            yhsaf = None

    if tarvar == 'rr':
        # create y & yhsaf (do not throw out any pixels yet)
        y = copy.copy(global_radar['RATE'].data.data)
        y[np.logical_or(y < 0.3,y >= 130.0)] = 0.0 # say: do not trust below 0.3 & above 130 -> do not consider it rain + set all fill values to 0 as well

        if read_HSAF:
            yhsaf = copy.copy(global_hsaf['h03_PC'].data.data)
            yhsaf[np.logical_or(yhsaf < 0.3,yhsaf > 9998)] = 0.0 # same as for ody (but no explicit upper boundary apart from the fill values)
        else:
            yhsaf = None
    
    print('=================================')

    
    
    ## create all data

    print('=================================')
    print('create all data of time slot:',time_slot)

    all_data = np.empty([global_radar['RATE'].data.flatten().shape[0],nr_vars])

    n=0

    # add radar
    all_data[:,n]=global_radar['RATE'].data.flatten()
    n+=1

    if read_HSAF:
        # add hsaf
        all_data[:,n]=global_hsaf['h03_PC'].data.data.flatten() ## rr with fill value 9999
        n+=1
    else:
        # add dummy data
        all_data[:,n]=0
        n+=1
        
    # add local solar time
    o = ephem.Observer()
    o.date = time_slot # some utc time
    lon_rad = np.deg2rad(vg['lon'].data.data.flatten())
    lst = np.empty(lon_rad.shape) #local solar time as np.vec
    for i in range(len(lon_rad)):
        o.lon = lon_rad[i]
        lst[i]=solartime(o) # angle in rad -> if want to turn back to hours use '%s' %ephem.hours(lst[i])
    all_data[:,n]=lst 
    n+=1
        
    # add lon / lats
    all_data[:,n]=vg['lon'].data.data.flatten() # add lon from nominal pos file
    n+=1
    all_data[:,n]=vg['lat'].data.data.flatten() # add lat from nominal pos file
    n+=1
        
    # add lsmask
    all_data[:,n]=ls_ele['lsmask'].data.data.flatten() # add lon from nominal pos file
    n+=1
        
    # add topo
    all_data[:,n]=ls_ele['ele'].data.data.flatten() # add lon from nominal pos file
    n+=1
        
    # add in viewing geom from nominal file
    all_data[:,n]=vg['vaa'].data.data.flatten() # viewing azimuthal angle
    n+=1
    all_data[:,n]=vg['vza'].data.data.flatten() # viewing zenith angle
    n+=1
        
    # add sat chan
    for chn in channel_sat:
        print(n,chn)
        all_data[:,n]=global_sat[chn+'_PC'].data.data.flatten()
        n+=1

    # add nwc products
    for chn in channel_nwc:
        print(n,chn)
        all_data[:,n]=global_nwc[chn+'_PC'].data.data.flatten()
        n+=1

    # add CTH
    all_data[:,n]=global_cth['CTH_PC'].data.data.flatten()
    n+=1

    # add idx
    all_data[:,n]= 0.0 # since have no idx just fill it with anything    
    
    # remove some vars to not run into memory problems
    os.system('rm /tmp/SEVIRI_DECOMPRESSED/*'+time_slot.strftime('%Y%m%d%H%M')+'*') # that don't fill up the diskspace    
    print('=================================')

    
    
    ## create masks
    
    print('=================================')
    print('create masks of time slot:',time_slot)

    # radar mask to see where ody ground truth exists
    mask_r = global_radar['RATE-MASK'].data.data==False # True where rad product available

    mask_r_lt = np.logical_and(global_radar['RATE'].data.data>0.0,global_radar['RATE'].data.data<0.3) #low threshold mask: True where spurious RR values we cannot trust
    mask_r_ht = global_radar['RATE'].data.data>=130.0 # high threshold mask: True where unphysically high RR


    mask_rnt = np.logical_and(mask_r, np.logical_or(mask_rad_thres, np.logical_or(mask_r_lt,mask_r_ht)))
               # not trustworthy available radar product mask bec too high RR, too low RR (0-0.3mm/h) or part of threshold mask

    # find cloud mask so that I know which pixels to not consider predictions at
    mask_cth = global_cth['CTH_PC'].data.data.flatten()>0 # mask to show where all NWCSAF products available

    CT = global_nwc['CT_PC'].data.data.flatten()
    mask_CT = np.logical_and(CT>4, CT!=20) # -> - 6 cat -> left with 15 cat
    # exclude regions which are masked out
    CT_mask = global_nwc['CT_PC'].data.mask.flatten()
    mask_CT[CT_mask] = False
    PHASE = global_nwc['CT_PHASE_PC'].data.data.flatten()
    mask_PHASE = PHASE!=0 # -> -1 cat -> left with 3 cat  
    mask_cat = np.logical_and(mask_CT, mask_PHASE)

    if 1==0:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
        
        plt.imshow( mask_cat.reshape(global_sat['IR_108'].shape) )
        plt.colorbar()
        fig.savefig("mask_cat.png")
        print("... display mask_cat.png &")
        #plt.show()
        quit()
    
    mask_h = np.logical_and(mask_cth,mask_cat) # True where NWCSAF products Cloud Top Height and Cloud Type and Cloud Phase available
    print('=================================')

    print('=================================')
    print('reduce all_data set to points where NWCSAF products are available')
    all_data=all_data[mask_h,:]

    mask_h = mask_h.reshape(global_sat['IR_108'].shape) # doesn't matter which field I take... just one with appropriate size
    print('=================================')

    # create additional vars I need for plotting
    print('=================================')
    print('create lon lat vars')
    lon = vg['lon'].data.data
    lat = vg['lat'].data.data
    print('=================================')

    
    return all_data,all_data_names,mask_h,mask_r,mask_rnt,y,yhsaf,lon,lat
