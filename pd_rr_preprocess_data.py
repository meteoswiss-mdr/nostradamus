#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original name: pd_rr_preprocess_data_single_scene_ody_seviriIR_nwcsaf_addparams_2017.py

import numpy as np
from datetime import datetime

from mpop.satellites import GeostationaryFactory

# to get the local time
import ephem

# to be able to create a sepearte copy of the dates which I shuffle around
import copy

# to be able to carry out a bash command from my script 
import os


def solartime(observer, sun=ephem.Sun()):
    sun.compute(observer)
    # sidereal time == ra (right ascension) is the highest point (noon)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm  # norm for 24h


def project_data(area, global_radar_mask, global_vg, global_ls_ele, global_radar, global_sat, global_nwc, global_cth, global_hsaf, read_HSAF=True):

    print('=================================')
    print('reproject constant fields to ' + area)  
    # Odyssey radar mask 
    radar_mask = global_radar_mask.project(area, precompute=True)
    # viewing geometry
    vg     = global_vg.project(area,precompute=True)
    # land sea mask and elevation 
    ls_ele = global_ls_ele.project(area, precompute=True)

    print('=========================')
    print('reproject observational data to ' + area)  
    data_radar = global_radar.project(area, precompute=True)
    data_sat   = global_sat.project  (area, precompute=True)
    data_nwc   = global_nwc.project  (area, precompute=True)
    data_cth = global_cth.project(area, precompute=True)
    if read_HSAF:
        data_hsaf = global_hsaf.project(area, precompute=True)
    else:
        data_hsaf = None

    if 1==0:
        print ( radar_mask['mask_radar'].data.shape)
        print ( vg['lat'].data.shape)
        print ( ls_ele['lsmask'].data.shape)
        print ( vg['lat'].data.shape)
        print ( data_radar['RATE'].data.shape )
        print ( data_sat['IR_108'].data.shape )
        print ( data_nwc['CT'].data.shape )
        print ( data_cth['CTH'].data.shape )        
        
    print('ok')
    print('=========================')

    return radar_mask, vg, ls_ele, data_radar, data_sat, data_nwc, data_cth, data_hsaf


def parallax_correct_sat_data(data_sat, data_nwc, data_cth, data_hsaf, par_fill, read_HSAF=True):
    # carry out appropriate parralax corr: 
    print('start parrallax corr:')

    print('=========================')
    print('start sat')
    data_sat = data_sat.parallax_corr(cth=data_cth['CTTH'].height, fill=par_fill)
    print('done sat')
    print('=========================')

    print('=========================')
    print('start nwc')
    data_nwc = data_nwc.parallax_corr(cth=data_cth['CTTH'].height, fill=par_fill)
    print('done nwc')
    print('=========================')

    if read_HSAF:
        print('=========================')    
        print('start hsaf')
        data_hsaf = data_hsaf.parallax_corr(cth=data_cth['CTTH'].height, fill=par_fill)
        print('done hsaf')
        print('=========================')

    print('=========================')
    print('start cth')
    data_cth=data_cth.parallax_corr(cth=data_cth['CTTH'].height, fill=par_fill)
    print('done cth')
    print('=========================')
    print('done parrallax corr:')

    return data_sat, data_nwc, data_cth, data_hsaf


def create_target_variable(tarvar, global_radar, global_hsaf, read_HSAF=True):
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
    return y, yhsaf


def create_ANN_input(all_data_names, time_slot, vg, ls_ele, global_radar, global_sat, channel_sat, global_nwc, channel_nwc, global_cth, global_hsaf, read_HSAF=True):

    ## create all data

    print('=================================')
    print('create all data of time slot:',time_slot)

    nr_vars = len(all_data_names)
    all_data = np.empty([global_radar['RATE'].data.flatten().shape[0],nr_vars])

    n=0

    # add radar
    print(n,"Odyssey radar")
    all_data[:,n]=global_radar['RATE'].data.flatten()
    n+=1

    if read_HSAF:
        # add hsaf
        print(n,"HSAF rainrate")
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
    print(n,"local time")
    all_data[:,n]=lst 
    n+=1
        
    # add lon / lats
    print(n,"longitude")
    all_data[:,n]=vg['lon'].data.data.flatten() # add lon from nominal pos file
    n+=1
    print(n,"latitude")
    all_data[:,n]=vg['lat'].data.data.flatten() # add lat from nominal pos file
    n+=1
        
    # add lsmask
    print(n,"land/sea mask")
    all_data[:,n]=ls_ele['lsmask'].data.data.flatten() # add lon from nominal pos file
    n+=1
        
    # add topo
    print(n,"topography")
    all_data[:,n]=ls_ele['ele'].data.data.flatten() # add lon from nominal pos file
    n+=1
        
    # add in viewing geom from nominal file
    print(n,"viewing azimith angle")
    all_data[:,n]=vg['vaa'].data.data.flatten() # viewing azimuthal angle
    n+=1
    print(n,"viewing zenith angle")
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
    
    # remove some vars to not run into disk space problems
    os.system('rm /tmp/SEVIRI_DECOMPRESSED/*'+time_slot.strftime('%Y%m%d%H%M')+'*') # that don't fill up the diskspace    
    print('=================================')

    return all_data

# -------------------------------------------------------------------------

def create_masks(radar_mask, global_radar, global_sat, global_nwc, global_cth):

    print('=================================')
    print('create masks of time slot:')

    # create radar masks (mask_r and mask_rnt)
    ###########################
    
    # radar mask to see where odyssey ground truth exists
    mask_r = global_radar['RATE-MASK'].data.data==False # True where rad product available

    # mask for rainrates > 0.3mm/h and below 130mm/h
    mask_r_lt = np.logical_and(global_radar['RATE'].data.data>0.0,global_radar['RATE'].data.data<0.3) #low threshold mask: True where spurious RR values we cannot trust
    mask_r_ht = global_radar['RATE'].data.data>=130.0 # high threshold mask: True where unphysically high RR

    # Odyssey radar mask  # mask_rad_thres = True where ody not trustworthy
    mask_rad_thres = radar_mask['mask_radar'].data

    mask_rnt = np.logical_and(mask_r, np.logical_or(mask_rad_thres, np.logical_or(mask_r_lt,mask_r_ht)))
               # not trustworthy available radar product mask bec too high RR, too low RR (0-0.3mm/h) or part of threshold mask

    # create satellite mask (mask_h)
    ############################
    
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
    
    mask_h = np.logical_and(mask_cth,mask_cat) # True where NWCSAF products Cloud Top Height and Cloud Type and Cloud Phase available
    mask_h = mask_h.reshape(global_sat['IR_108'].shape) # doesn't matter which field I take... just one with appropriate size
    print('=================================')
        
    return mask_h, mask_r, mask_rnt

# -------------------------------------------------------------------------

def pd_rr_preprocess_data_single_scene(area, time_slot, radar_mask, vg, ls_ele, data_radar, data_sat, data_nwc, data_cth, data_hsaf, par_fill, tarvar, read_HSAF=True):
    #########
    ## input
    # area: projection area
    # time_slot: time in UTC
    # par_fill: parallax corr gap filler: choose between 'False', 'nearest', and 'bilinear'
    # tar_var: target variable: choose between 'pd', 'rr'
    # 
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
    
    ## parallax correct the satellite observations 
    data_sat, data_nwc, data_cth, data_hsaf = \
        parallax_correct_sat_data(data_sat, data_nwc, data_cth, data_hsaf, par_fill, read_HSAF=read_HSAF)

    # create target variables (truth)
    y, yhsaf = create_target_variable(tarvar, data_radar, data_hsaf, read_HSAF=read_HSAF)

    # create input matrix for the ANN 
    all_data = create_ANN_input(all_data_names, time_slot, vg, ls_ele, data_radar, data_sat, channel_sat, data_nwc, channel_nwc, data_cth, data_hsaf, read_HSAF=read_HSAF)    

    ## create masks mask_h(CTH>0, CT>4&&CT!=20, PHASE!=0), mask_r(Odyssey trustworthy), mask_rnt(rr>0.3mm/h+Odyssey trustworthy)
    mask_h, mask_r, mask_rnt = create_masks(radar_mask, data_radar, data_sat, data_nwc, data_cth)

    print('=================================')
    print('reduce all_data set to points where NWCSAF products are available')
    all_data=all_data[mask_h.flatten(),:]
    
    # create additional vars I need for plotting
    print('=================================')
    print('create lon lat vars')
    lon = vg['lon'].data.data
    lat = vg['lat'].data.data
    print('=================================')
    
    return all_data, all_data_names, mask_h, mask_r, mask_rnt, y, yhsaf, lon, lat
