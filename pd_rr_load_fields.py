#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mpop.satellites import GeostationaryFactory
from mpop.projector import get_area_def

import sys

from datetime import datetime

# to import netcdf files
from netCDF4 import Dataset


# to get the cloud mask from the NWC SAF
from my_msg_module import get_NWC_pge_name

# to convert NWCSAF variable to channel type where we can carry out parallax correction
from my_msg_module import convert_NWCSAF_to_radiance_format

# add a channel
from mpop.channel import Channel

from copy import deepcopy

def load_constant_fields(sat_nr):
    
    # radar threshold mask:
    radar_mask = GeostationaryFactory.create_scene("odyssey", "", "radar", datetime(1900,1,1,0))

    # reproject this to the desired area:
    mask_rad_thres=np.load('./odyssey_mask/threshold_exceedance_mask_avg15cut2_cut04_cutmistral_201706_201707_201708.npy')        
    from mpop.projector import get_area_def
    area_radar_mask = 'EuropeOdyssey00'
    radar_mask.channels.append(Channel(name='mask_radar', wavelength_range=[0.,0.,0.], data=mask_rad_thres[:,:]))
    radar_mask['mask_radar'].area     = area_radar_mask
    radar_mask['mask_radar'].area_def = get_area_def(area_radar_mask)
    
    # nominal viewing geometry
    print('*** read nominal viewing geometry', "meteosat", sat_nr, "seviri" )
    # time_slot has NO influence at all just goes looking for the nominal position file -> will use these fields for all dates 
    vg = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", datetime(1900,1,1,0)) 
    vg.load(['vaa','vza','lon','lat'], reader_level="seviri-level6")  
    msg_area       = deepcopy(vg['vaa'].area)
    msg_area_def   = deepcopy(vg['vaa'].area_def)
    msg_resolution = deepcopy(vg['vaa'].resolution)
    
    # read land sea mask (full SEVIRI Disk seen from 0 degree East)
    ls_file = './SEVIRI_data/LandSeaMask_SeviriDiskFull00.nc'
    fh = Dataset(ls_file, mode='r')
    lsmask = fh.variables['lsmask'][:]

    # read topography (full SEVIRI Disk seen from 0 degree East)
    ls_file = './SEVIRI_data/SRTM_15sec_elevation_SeviriDiskFull00.nc'
    fh = Dataset(ls_file, mode='r')
    ele = fh.variables['elevation'][:]

    # create  a dummy satellite object (to reproject the land/sea mask and elevation)
    ls_ele = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri",datetime(1900,1,1,0))
    #ls_ele.load(['CTTH'], calibrate=True, reader_level="seviri-level3")  
    #convert_NWCSAF_to_radiance_format(ls_ele, None,'CTH', False, True)

    # add land sea mask as a dummy channel 
    ls_ele.channels.append(Channel(name='lsmask', wavelength_range=[0.,0.,0.], resolution=msg_resolution, data=lsmask[::-1,:]))
    #ls_ele['lsmask'].area = ls_ele['CTH'].area
    #ls_ele['lsmask'].area_def = ls_ele['CTH'].area_def
    ls_ele['lsmask'].area     = msg_area
    ls_ele['lsmask'].area_def = msg_area_def

    # add elevation as a dummy channel 
    ls_ele.channels.append(Channel(name='ele', wavelength_range=[0.,0.,0.], resolution=msg_resolution, data=ele[::-1,:]))
    #ls_ele['ele'].area     = ls_ele['CTH'].area
    #ls_ele['ele'].area_def = ls_ele['CTH'].area_def
    ls_ele['ele'].area     = msg_area
    ls_ele['ele'].area_def = msg_area_def

    return radar_mask, vg, ls_ele


def load_input(sat_nr, time_slot, par_fill, read_HSAF=True):
    #########
    # time_slot: time in UTC
    # par_fill: parallax corr gap filler: choose between 'False', 'nearest', and 'bilinear'
    #########
    
    # RADAR
    prop_rad='RATE'

    # SATELLITE
    channel_sat=['WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134']

    # Cloud Mask
    prop_cma='CMa'
    pge_cma = get_NWC_pge_name(prop_cma)

    # cloud type
    prop_ct='CT'
    pge_ct = get_NWC_pge_name(prop_ct)

    # cloud phase
    prop_ctph='CT_PHASE'
    pge_ctph = get_NWC_pge_name(prop_ctph)

    # cloud top temperature
    prop_ctt = 'CTT'
    pge_ctt = get_NWC_pge_name(prop_ctt)

    # cloud top pressure
    prop_ctp = 'CTP'
    pge_ctp = get_NWC_pge_name(prop_ctp)

    # put all the strings I want to load in the same obj together
    prop_nwc = [prop_cma, prop_ct,prop_ctph,prop_ctt,prop_ctp] 
    pge_nwc = [pge_cma, pge_ct,pge_ctph,pge_ctt,pge_ctp]

    # Cloud height
    prop_cth='CTH'
    pge_cth = get_NWC_pge_name(prop_cth) # separate so can correct all others before also correcting it

    # hsaf
    prop_hsaf = 'h03' # <- what is this?! apparently estimated rain rate in mm/h


    print('=========================')
    print('start:',time_slot)

    print('read Odyssey radar composite')
    global_radar = GeostationaryFactory.create_scene("odyssey", "", "radar", time_slot)
    global_radar.load([prop_rad])
    print(global_radar)
    print('=========================')

    print('read satellite data')
    try:
        global_sat = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot)
        global_sat.load(channel_sat)
        print(global_sat)
        print('=========================')
    except AttributeError:
        date_missed=time_slot
        #sys.exit() # move on to the next iteration if the NWCSAF does not have a product at this time instance
        text = ['skipped because SEVIRI product missing']
        return date_missed,text

    if read_HSAF:
        print('read HSAF data')
        try:
            global_hsaf = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot)
            global_hsaf.load([prop_hsaf], reader_level='seviri-level7')
            print('=========================')
        except ValueError:
            date_missed = time_slot
            text = ['skipped because HSAF product missing']
            #sys.exit() # move on to the next iteration if the NWCSAF does not have a product at this time instance
            return date_missed,text  
    else:
        global_hsaf = None
    
    print('read Nowcasting SAF data')
    global_nwc = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot)
    nwcsaf_calibrate=True   # converts data into physical units
    global_nwc.load(pge_nwc, calibrate=nwcsaf_calibrate, reader_level="seviri-level3")  
    print("=========================")

      
    print('read CTH data')
    global_cth = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot)
    nwcsaf_calibrate=True   # converts data into physical units
    global_cth.load([pge_cth], calibrate=nwcsaf_calibrate, reader_level="seviri-level3") 
    print('=========================')
        
    # convert NWCSAF input to channels to be able to carry out parallax corr   
    try:
        for var in prop_nwc:
            convert_NWCSAF_to_radiance_format(global_nwc, None, var, False, True) 

        convert_NWCSAF_to_radiance_format(global_cth, None, prop_cth, False, True) 
    except KeyError:
        date_missed = time_slot
        text = ['skipped because NWC SAF product missing']
        #sys.exit() # move on to the next iteration if the NWCSAF does not have a product at this time instance
        return date_missed,text


    return global_radar,global_sat,global_nwc,global_cth,global_hsaf
