
# coding: utf-8
      
# ATTENTION: ONCE I HAVE THE NEWLY TRAINED MODELS I'LL NEED TO SWITCH THE FOLLOWING:
    # pd_rr_preprocess_data_single_scene_ody_seviriIR_nwcsaf_addparams_2017_BTvg09 to the one without BTvg09
    # change input dir of precip detection + rr model + probab matching database
    # area switch to EuropeOdyssey00
    # when loading IR_108, switch to Meteosat-10

from __future__ import division
from __future__ import print_function 

#get_ipython().magic(u'matplotlib inline')

import numpy as np
import sys
from sklearn.externals import joblib # same purpose as pickle but more efficient with big data / can only pickle to disk
import copy
from datetime import datetime, timedelta
import time

# to create y, yhsaf & Xraw
from pd_rr_create_y_yhsaf_Xraw import pd_rr_create_y_yhsaf_Xraw

# to create all_data etc for a single time slot
from pd_rr_preprocess_data import project_data, pd_rr_preprocess_data_single_scene

# to carry out probability matching
from rr_probab_matching import probab_match_rr_refprovide

# to load all the input data 
from pd_rr_load_fields import load_input, load_constant_fields

# for plotting
import matplotlib.pyplot as plt

from matplotlib.colors import from_levels_and_colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting_tools import smart_colormap, map_plot, save_RR_as_netCDF

from my_msg_module import get_last_SEVIRI_date

from pydecorate import DecoratorAGG
from os.path import isfile
from os import chmod
from postprocessing import postprocessing

#########################################################################

def nostradamus_rain(in_msg):
            
    if in_msg.datetime is None:
        in_msg.get_last_SEVIRI_date()

    if in_msg.end_date is None:
        in_msg.end_date = in_msg.datetime
        #in_msg.end_date = in_msg.datetime + timedelta(15)
      
    delta     = timedelta(minutes=15) 

    # automatic choise of the FULL DISK SERVICE Meteosat satellite
    if in_msg.datetime <  datetime(2008, 5, 13, 0, 0):   # before 13.05.2008 only nominal MSG1 (meteosat8), no Rapid Scan Service yet
        sat_nr = "08" 
    elif in_msg.datetime <  datetime(2013, 2, 27, 9, 0): # 13.05.2008 ...  27.02.2013 
        sat_nr = "09"                              # MSG-2  (meteosat9) became nominal satellite, MSG-1 (meteosat8) started RSS
    elif in_msg.datetime <  datetime(2018, 3, 9, 0, 0):  # 27.02.2013 9:00UTC ... 09.03.2013                                   
        sat_nr = "10"                              # MSG-3 (meteosat10) became nominal satellite, MSG-2 started RSS (MSG1 is backup for MSG2)
    else:
        sat_nr = "11"
    print ("... work with Meteosat"+str(sat_nr))
    
    print ("")
    if in_msg.verbose:
        print ('*** Create plots for ')
        print ('    Satellite/Sensor: ' + in_msg.sat_str()) 
        print ('    Satellite number: ' + in_msg.sat_nr_str() +' // ' +str(in_msg.sat_nr))
        print ('    Satellite instrument: ' + in_msg.instrument)
        print ('    Start Date/Time:      '+ str(in_msg.datetime))
        print ('    End Date/Time:        '+ str(in_msg.datetime))
        print ('    Areas:           ', in_msg.areas)
        for area in in_msg.plots.keys():
            print ('    plots['+area+']:            ', in_msg.plots[area])
        #print ('    parallax_correction: ', in_msg.parallax_correction)
        #print ('    reader level:    ', in_msg.reader_level)
    
    ## read in all the constants files
    print('=================================')
    print('*** load the constant fields (radar mask, viewing geometry, and land/sea mask plus surface elevation)')
    global_radar_mask, global_vg, global_ls_ele = load_constant_fields(sat_nr) 
    
    ###############################################
    ## load the mlp for the precip detection (pd) #
    ###############################################

    if in_msg.model == 'mlp':
        dir_start_pd= './models/precipitation_detection/mlp/2hl_100100hu_10-7alpha_log/'
        dir_start_rr= './models/precipitation_rate/mlp/2hl_5050hu_10-2alpha_log/'

    if not in_msg.read_from_netCDF:

        clf_pd = joblib.load(dir_start_pd+'clf.pkl') 
        scaler_pd = joblib.load(dir_start_pd+'scaler.pkl')
        feature_list_pd = joblib.load(dir_start_pd+'featurelist.pkl')
        thres_pd=np.load(dir_start_pd+'opt_orig_posteriorprobab_thres.npy')

        #########################################
        ## load the mlp for the rain rates (rr) #
        #########################################

        reg_rr = joblib.load(dir_start_rr+'reg.pkl') 
        scaler_rr = joblib.load(dir_start_rr+'scaler.pkl')
        feature_list_rr = joblib.load(dir_start_rr+'featurelist.pkl')

    ####################################
    ## load the reference sets for a climatological probab matching (pm) if requested
    ####################################

    if in_msg.probab_match:
        # load in the ref data sets created with the script: rr_probab_matching_create_refset.ipynb
        ody_rr_ref=np.load(dir_start_rr+'pm_valid_data_ody_rr_ref.npy')
        pred_rr_ref=np.load(dir_start_rr+'pm_valid_data_pred_rr_ref.npy')

    # initialize processed RGBs
    plots_done={}
    
    time_slot = copy.deepcopy(in_msg.datetime)
    while time_slot <= in_msg.end_date:
        print('... processing for time: ', time_slot)

        ################################################
        ## CHOOSE THE SETUP (time_slot, area, model)
        ################################################
 
        ##########################
        ## LOAD THE NEEDED INPUTS
        ##########################

        if not in_msg.read_from_netCDF:
            ## read observations at the specific time
            print('=================================')
            print('*** load the time slot specific fields with in_msg.parallax_gapfilling:', in_msg.parallax_gapfilling)
            global_radar, global_sat, global_nwc, global_cth, global_hsaf = load_input(sat_nr, time_slot, in_msg.parallax_gapfilling, read_HSAF=in_msg.read_HSAF)
            #                                                               def load_input(sat_nr, time_slot, par_fill, read_HSAF=True):
        else:
            print('read Odyssey radar composite')
            from mpop.satellites import GeostationaryFactory
            global_radar = GeostationaryFactory.create_scene("odyssey", "", "radar", time_slot)
            global_radar.load(['RATE'])
            print(global_radar)
            print('=========================')
    
        for area in in_msg.areas:

            print ("================================")
            print ("*** PROCESSING FOR AREA: "+area)

            # declare "precipitation detection" and "rainrate dictionary", the applied model (e.g. MLP) is used as key
            pd = {}
            rr = {}
            plots_done[area]=[]
            
            if in_msg.read_from_netCDF:

                # reproject Odyssey radar mask to area of interest 
                #radar_mask = global_radar_mask.project(area, precompute=True)
                data_radar = global_radar.project(area, precompute=True)
                # radar mask to see where odyssey ground truth exists
                mask_r = data_radar['RATE-MASK'].data.data==False
                rr['ody'] = copy.deepcopy(data_radar['RATE'].data.data)
                # do not trust values below 0.3 & above 130 -> do not consider it as rain and set all values to 0 
                rr['ody'][np.logical_or(rr['ody'] < 0.3,rr['ody'] >= 130.0)] = 0.0
                print (rr['ody'].min(), rr['ody'].max(), rr['ody'].shape, type(rr['ody']))
                
                from netCDF4 import Dataset
                # read from file
                outdir_netCDF = time_slot.strftime('/data/COALITION2/database/meteosat/nostradamus_RR/%Y/%m/%d/')
                file_netCDF = time_slot.strftime('MSG_rr-'+in_msg.model+'-'+area+'_%Y%m%d%H%M.nc')
                print ("*** read precip prediction from", outdir_netCDF+"/"+file_netCDF)

                ncfile = Dataset(outdir_netCDF+"/"+file_netCDF,'r')
                rr_tmp    = ncfile.variables['rainfall_rate'][:,:]

                ### now, we read radar data directly from odyssey file 
                #rr['ody'] = ncfile.variables['rainfall_rate (odyssey)'][:,:]
                #print (rr['ody'].min(), rr['ody'].max(), rr['ody'].shape, type(rr['ody']))

                ### now, we read radar mask directly from odyssey file 
                #mask_r    = ncfile.variables['odyssey_mask'][:,:]
                #print ("... convert mask_r (1, 0) from int to bolean (True, False)")
                #mask_r = (mask_r == 1)
                
                # create fake mask_h (where rainfall is larger than 0 mm/h)
                mask_h = rr_tmp>0
                pd[in_msg.model] = rr_tmp>0
                rr_tmp = rr_tmp.flatten()
                # remove 0 entries 
                rr_tmp = rr_tmp [ rr_tmp != 0 ]

                if False:
                    import matplotlib.pyplot as plt
                    #fig = plt.figure()
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
                    plt.subplot(2, 1, 1)
                    plt.imshow(mask_h)
                    #plt.colorbar()
                    plt.subplot(2, 1, 2)
                    plt.imshow(mask_r)
                    #plt.colorbar()
                    fig.savefig("mask_h_mask_r_netCDF.png")
                    print("... display mask_h_mask_r_netCDF.png &")
                    #plt.show()
                    #quit()

                
            else:
                
                ## project all data to the desired projection 
                radar_mask, vg, ls_ele, data_radar, data_sat, data_nwc, data_cth, data_hsaf = \
                    project_data(area, global_radar_mask, global_vg, global_ls_ele, global_radar, global_sat, global_nwc, global_cth, global_hsaf, read_HSAF=in_msg.read_HSAF)

                ###########################################################
                ## SINGLE TIME SLOT TO CARRY OUT A FULL RAIN RATE RETRIEVAL 
                ###########################################################

                # preprocess the data 
                # mask_h: field indicating where NWCSAF products are available & thus where predictions are carried out: True if NWCSAF products available
                # mask_r: field indicating where radar products are available: True if radar product is available
                # mask_rnt: field indicating where radar product available but not trustworthy: i.e. in threshold_mask, 0<rr<0.3, rr>130 overlaid: True if radar product is NOT trustworthy
                all_data, all_data_names, mask_h, mask_r, mask_rnt, rr['ody'], rr['hsaf'], lon, lat = \
                         pd_rr_preprocess_data_single_scene( area, time_slot,
                                                             radar_mask, vg, ls_ele,
                                                             data_radar, data_sat, data_nwc, data_cth, data_hsaf,
                                                             in_msg.parallax_gapfilling, 'rr', read_HSAF=in_msg.read_HSAF)
                         #pd_rr_preprocess_data_single_scene( sat_nr, area, time_slot, 'nearest', 'rr', read_HSAF=False)

                if False:
                    import matplotlib.pyplot as plt
                    #fig = plt.figure()
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
                    plt.subplot(2, 1, 1)
                    plt.imshow(mask_h)
                    #plt.colorbar()
                    plt.subplot(2, 1, 2)
                    plt.imshow(mask_r)
                    #plt.colorbar()
                    fig.savefig("mask_h_mask_r.png")
                    print("... display mask_h_mask_r.png &")
                    #plt.show()
                    #quit()

                del rr['hsaf'] # since not actually needed in this script

                # project all data to desired projection 
                # ...

                print('... predictions at ' + str(mask_h.sum())+' out of ' +str(mask_h.flatten().shape[0])+ ' points')

                ####################################
                ## precip detection
                ####################################

                # create y_pd, y_hsaf_pd, X_raw_pd 
                y_pd_vec, y_hsaf_pd_vec, X_raw, feature_list = pd_rr_create_y_yhsaf_Xraw( all_data, all_data_names, 'pd', cut_precip=False )

                del y_pd_vec, y_hsaf_pd_vec # (since not actually ever needed in this script)

                if in_msg.remove_vg==True:
                    print('... remove viewing geometry from predictors')
                    feature_list = np.append(feature_list[:6],feature_list[8:])
                    X_raw = np.hstack([X_raw[:,:6],X_raw[:,8:]])
                    print('    new X_raw.shape:', X_raw.shape)
                    feature_list

                # check features
                if np.array_equal(feature_list, feature_list_pd):
                    print('OK, input features correspond to input features required by the loaded model')
                else:
                    print('ATTENTION, input features do not correspond to input features required by the loaded model')
                    quit()

                # create X_pd
                X_pd=scaler_pd.transform(X_raw) 

                # create final precip detection fields: opera + hsaf
                pd['ody']=rr['ody']>=0.3

                # make precip detection predictions
                print ("***  make precip detection predictions")
                pd_probab = clf_pd.predict_proba(X_pd)[:,1]   # probab precip balanced classes
                pd_vec_h = pd_probab>=thres_pd  
                pd[in_msg.model] = np.zeros(lon.shape,dtype=bool)
                pd[in_msg.model][mask_h] = pd_vec_h

                ####################################
                ## rain rate on above identified precipitating pixels
                ####################################

                # reduce X_raw to the points where rain was predicted by the mlp
                X_raw= X_raw[pd_vec_h,:] 

                # check, if read features correspond to the trained model
                if np.array_equal(feature_list, feature_list_rr):
                    print('OK, input features correspond to input features required by the loaded model')
                else:
                    print('ATTENTION, input features do not correspond to input features required by the loaded model')
                    quit()

                # create X_rr
                X_rr=scaler_rr.transform(X_raw)    

                # rain rate prediction at places where precip detected by mlp
                rr_tmp=reg_rr.predict(X_rr)  
                
            # carry out a probability machting if requested    
            if in_msg.probab_match:
                print("... do probability matching for:", in_msg.model)
                pm_str = str(in_msg.model)+'_pm'
                rr_tmp_pm = probab_match_rr_refprovide(ody_rr_ref,pred_rr_ref,rr_tmp)  
                #rr[pm_str] =  np.zeros_like(lon) 
                rr[pm_str] =  np.zeros_like(rr['ody'])   # also casts the type float
                rr[pm_str][pd[in_msg.model]]=rr_tmp_pm
                print("... probability matching done for:", in_msg.model)

            # copy rainrate data to the final place
            # replace all prediction lower than precipitation detection threshold with threhold rain rate        
            rr_tmp[rr_tmp<0.3]=0.3 # correct upward all too low predictions (i.e. the ones below the precip detection threshold)
            rr[in_msg.model] = np.zeros_like(rr['ody'])
            rr[in_msg.model][pd[in_msg.model]]=rr_tmp
                
            #####################################
            ## SAVE RESULT AS NETCDF 
            #####################################
            if area in in_msg.save_netCDF and (not in_msg.read_from_netCDF):
                outdir_netCDF = time_slot.strftime(in_msg.outdir_netCDF)
                file_netCDF   = time_slot.strftime(in_msg.file_netCDF)
                file_netCDF   = file_netCDF.replace("%(area)s", area)
                file_netCDF   = file_netCDF.replace("%(model)s", in_msg.model)
                #save_RR_as_netCDF(outdir_netCDF, file_netCDF, rr[in_msg.model], save_rr_ody=True, rr_ody=rr['ody'], save_ody_mask=True, ody_mask=mask_r, zlib=True)
                save_RR_as_netCDF(outdir_netCDF, file_netCDF, rr[in_msg.model])
                
                
            #####################################
            ## SINGLE TIME SLOT TO DRAW THE MAPS 
            #####################################

            print ("*** start to create plots")
            
            ####################################
            ## plot precip detection
            ####################################
            
            if 'pdMlp' in in_msg.plots[area]:
                
                mask_rt = np.logical_and(mask_r, mask_rnt==False) # trusted radar i.e. True where I have a trustworthy radar product available

                mod_ss = [in_msg.model] + ['ody']

                # ver for verification;
                ver={}    
                for x in mod_ss:
                    ver[x]=np.zeros_like(lon) #  sat: no 
                    ver[x][pd[x]>0] = 1 # sat: yes
                    ver[x][np.logical_and(ver[x]==0,mask_rnt)] = 2 # sat: no (rad clutter)
                    ver[x][np.logical_and(ver[x]==1,mask_rnt)] = 3 # sat: yes (rad clutter)
                    ver[x][np.logical_and(mask_rt,np.logical_and(pd[x]==1,pd['ody']==1))] = 4 # hit
                    ver[x][np.logical_and(mask_rt,np.logical_and(pd[x]==1,pd['ody']==0))] = 5 # false alarm
                    ver[x][np.logical_and(mask_rt,np.logical_and(pd[x]==0,pd['ody']==0))] = 6 # correct reject
                    ver[x][np.logical_and(mask_rt,np.logical_and(pd[x]==0,pd['ody']==1))] = 7 # miss

                # define colorkey 
                v_pd=np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
                cmap_pd, norm_pd = from_levels_and_colors(v_pd,
                                    colors =['darkgrey', '#984ea3','lightgrey','plum', '#377eb8', '#e41a1c','ivory','#ff7f00'],
                                    extend='neither')

                plot_precipitation_detection=False
                if plot_precipitation_detection:    

                    # single prediction plot
                    #fig,ax= plt.subplots(figsize=(20, 10))
                    #plt.rcParams.update({'font.size': 16})
                    fig,ax= plt.subplots(figsize=(10, 5))
                    plt.rcParams.update({'font.size': 8})
                    plt.rcParams.update({'mathtext.default':'regular'}) 

                    m = map_plot(axis=ax,area=area)
                    m.ax.set_title('precip detection based on sat vs opera')

                    # plot sat precip detection product against opera product
                    tick_label_pd_nr=np.array([0,1,2,3,4,5,6,7])
                    tick_label_pd=['sat: no','sat: yes','sat: no (rad unr)','sat: yes (rad unr)','hit','false alarm','correct reject','miss']


                    im=m.pcolormesh( lon, lat, ver['mlp'], cmap=cmap_pd, norm=norm_pd, latlon=True )
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4%", pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, ticks=tick_label_pd_nr, spacing='uniform')
                    a=cbar.ax.set_yticklabels(tick_label_pd)
                    outfile= 'precip_detection_sat'+in_msg.model+'_vs_opera_%s'
                    fig.savefig((in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
                    print('... create figure: display ' + in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

                plots_done[area].append('pdMlp')
                    
            ####################################
            ## plot rain rate with matplotlib
            ####################################

            if 'rrMatplotlib' in in_msg.plots[area]:  

                # create the combi rr field
                rr['combi']=copy.deepcopy(rr[in_msg.model+'_pm'])
                rr['combi'][mask_r]=rr['ody'][mask_r]

                # determine where I have >0.3 mm/h precip on the permanent mask -> overlay end picture with a pink(?) color there
                pd_nt=np.logical_and(mask_rnt,pd['ody']>=0.3) #precip detected but not trusted

                t = time.time()

                #fig, axes = plt.subplots(1, 2,figsize=(19, 6))
                #plt.rcParams.update({'font.size': 16})
                fig, axes = plt.subplots(1, 2,figsize=(9.5, 3))
                plt.rcParams.update({'font.size': 8})
                plt.rcParams.update({'mathtext.default':'regular'}) 

                ## 1st subplot
                m =  map_plot(axis=axes[0],area=area)
                m.ax.set_title('Rain Rate (opera + MSG ANN), '+str(time_slot))

                # plot a white colored background where I have data available
                v_pd_nt=np.array([0.5,1.5]) 
                cmap_pd_nt, norm_pd_nt = from_levels_and_colors(v_pd_nt, colors=['white'], extend='neither')
                im4=m.pcolormesh(lon,lat,np.ones(lon.shape),cmap=cmap_pd_nt,norm=norm_pd_nt,latlon=True)

                # plot mask which contains no rad & not trusted rad values
                nr_ntr = copy.deepcopy(ver['ody']) 
                nr_ntr=np.ma.masked_greater(nr_ntr,2)
                nr_ntr=np.ma.masked_equal(nr_ntr,1)
                im2=m.pcolormesh(lon,lat,nr_ntr,cmap=cmap_pd,norm=norm_pd,latlon=True)

                # plot combined precip opera + sat
                v_rr = [0.3,0.6,1.2,2.4,4.8,9.6]
                cmap_rr,norm_rr=smart_colormap(v_rr,name='coolwarm',extend='max')
                im=m.pcolormesh(lon,lat,rr['combi'],cmap=cmap_rr,norm=norm_rr,latlon=True)

                # plot pink pixels everywhere on permanently not trusted radar mask where we observe > 0.3 mm/h precip
                v_pd_nt=np.array([0.5,1.5])
                cmap_pd_nt, norm_pd_nt = from_levels_and_colors(v_pd_nt, colors=['plum'], extend='neither')
                im3=m.pcolormesh(lon,lat,pd_nt,cmap=cmap_pd_nt,norm=norm_pd_nt,latlon=True)

                ## 2nd subplot
                # plot purely satellite based precip product
                m =  map_plot(axis=axes[1],area=area)
                m.ax.set_title('Rain Rate (MSG ANN), '+str(time_slot))

                if in_msg.IR_108 and not in_msg.read_from_netCDF:
                    # plot the IR_108 channel
                    clevs = np.arange(225,316,10)
                    cmap_sat,norm_sat=smart_colormap(clevs,name='Greys',extend='both')
                    im4 = m.pcolormesh(lon,lat,data_sat['IR_108_PC'].data,cmap=cmap_sat,norm=norm_sat,latlon=True)
                else:
                    # plot a white surface to distinguish between the regions where the produ
                    v_pd_nt=np.array([0.5,1.5])
                    cmap_pd_nt, norm_pd_nt = from_levels_and_colors(v_pd_nt, colors=['white'], extend='neither')
                    im4=m.pcolormesh(lon,lat,np.ones(lon.shape),cmap=cmap_pd_nt,norm=norm_pd_nt,latlon=True)

                if in_msg.probab_match:
                    im=m.pcolormesh(lon, lat,rr[in_msg.model+'_pm'], cmap=cmap_rr, norm=norm_rr, latlon=True)
                else:
                    im=m.pcolormesh(lon, lat,rr[in_msg.model],       cmap=cmap_rr, norm=norm_rr, latlon=True)

                if in_msg.IR_108 and in_msg.probab_match:
                    outfile= 'rr_combioperasat'+in_msg.model+'pm_satIR108'+in_msg.model+'pm_%s'    
                elif in_msg.IR_108 and (in_msg.probab_match==False):
                    outfile= 'rr_combioperasat'+in_msg.model+'_satIR108'+in_msg.model+'_%s'
                elif (in_msg.IR_108==False) and in_msg.probab_match:
                    outfile= 'rr_combioperasat'+in_msg.model+'pm_sat'+in_msg.model+'pm_%s'
                elif (in_msg.IR_108==False) and (in_msg.probab_match==False):
                    outfile= 'rr_combioperasat'+in_msg.model+'_sat'+in_msg.model+'_%s'    

                fig.subplots_adjust(bottom=0.15)
                cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])
                cbar=fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cbar.set_label('$mm\,h^{-1}$')


                fig.savefig((in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
                print('... create figure: display ' + in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

                elapsed = time.time() - t
                print("... elapsed time for creating the rainrate image in seconds: "+str(elapsed))

                plots_done[area].append('rrMatplotlib')
                
            ####################################
            ## plot rain rate with trollimage
            ####################################
            plot_trollimage=True
            if plot_trollimage:

                from plotting_tools import create_trollimage
                from plot_msg import add_title
                
                print ("*** create plot with trollimage")
                from copy import deepcopy
                from trollimage.colormap import RainRate
                colormap = deepcopy(RainRate)

                # define contour write for coasts, borders, rivers
                from pycoast import ContourWriterAGG
                cw = ContourWriterAGG(in_msg.mapDir)

                from plot_msg import choose_map_resolution
                resolution = choose_map_resolution(area, None) 
                #resolution='l' # odyssey, europe
                #resolution='i' # ccs4
                print ("    resolution=", resolution)

                IR_file=time_slot.strftime(in_msg.outputDir+'MSG_IR-108-'+area+'_%Y%m%d%H%M.png')
                
                if 'IR_108' in in_msg.plots[area] and not in_msg.read_from_netCDF:
                    # create black white background
                    #img_IR_108 = data_sat.image.channel_image('IR_108_PC')
                    img_IR_108 = data_sat.image.ir108()
                    img_IR_108.save(IR_file)
                                    
                for rgb in in_msg.plots[area]:
                        if rgb == 'RATE':
                            prop = np.ma.masked_equal(rr['ody'], 0)
                            mask2plot=deepcopy(mask_r)
                        elif rgb =='rrMlp':
                            prop = np.ma.masked_equal(rr[in_msg.model], 0)
                            mask2plot=None
                        elif rgb == 'rrMlpPm':
                            prop = np.ma.masked_equal(rr[in_msg.model+'_pm'], 0)
                            mask2plot=None
                        elif rgb == 'rrOdyMlp':
                            rr['combi']=copy.deepcopy(rr[in_msg.model])
                            rr['combi'][mask_r]=rr['ody'][mask_r]
                            prop = np.ma.masked_equal(rr['combi'], 0)
                            mask2plot=deepcopy(mask_r)
                        elif rgb == 'rrOdyMlpPm':
                            rr['combi']=copy.deepcopy(rr[in_msg.model+'_pm'])
                            rr['combi'][mask_r] = rr['ody'][mask_r]
                            prop = np.ma.masked_equal(rr['combi'], 0)
                            mask2plot=deepcopy(mask_r)
                        elif rgb == 'IR_108':
                            continue
                        else:
                            "*** Error, unknown product requested"
                            quit()
                        filename = None
                        if area in in_msg.postprocessing_composite:
                            composite_file = in_msg.outputDir+"/"+'MSG_'+in_msg.postprocessing_composite[area][0]+"-"+area+'_%Y%m%d%H%M.png'
                            composite_file = composite_file.replace("%(rgb)s", rgb)
                        else:
                            composite_file = None
                            
                        PIL_image = create_trollimage(rgb, prop, colormap, cw, filename, time_slot, area, composite_file=composite_file,
                                          background=IR_file, mask=mask2plot, resolution=resolution, scpOutput=in_msg.scpOutput)

                        # add title to image
                        dc = DecoratorAGG(PIL_image)
                        if in_msg.add_title:
                            add_title(PIL_image, in_msg.title, rgb, 'MSG', sat_nr, in_msg.datetime, area, dc, in_msg.font_file, True,
                                      title_color=in_msg.title_color, title_y_line_nr=in_msg.title_y_line_nr ) # !!! needs change
                            
                        # save image as file
                        outfile = time_slot.strftime(in_msg.outputDir+"/"+in_msg.outputFile).replace("%(rgb)s", rgb).replace("%(area)s", area).replace("%(model)s", in_msg.model)

                        PIL_image.save(outfile, optimize=True)
                        if isfile(outfile):
                            print ("... create figure: display "+outfile+" &")
                            chmod(outfile, 0777)
                            plots_done[area].append(rgb)
                        else:
                            print ("*** Error: "+outfile+" could not be generated")
                            quit()
                            
                            

                print('=================================')

            ##############################################
            ## potential other map setups
            ##############################################
            ##############################################

            ##############################################
            ## opera composite vs the prediction... but I think it'd be less confusing to only show the prediction
            ##############################################

            if 'OdyVsRr' in in_msg.plots[area]:

                fig, axes = plt.subplots(1, 2,figsize=(23.5, 5))
                    # will be switched to basemap once have new training set together
                plt.rcParams.update({'font.size': 16})
                plt.rcParams.update({'mathtext.default':'regular'}) 

                # set up nn subplot       
                m =  map_plot(axis=axes[0],area=area)
                m.ax.set_title('precip detection based on mlp vs opera')
                v_pd=np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
                cmap_pd, norm_pd = from_levels_and_colors(v_pd, colors=['darkgrey', '#984ea3','lightgrey','plum', '#377eb8', '#e41a1c','ivory','#ff7f00'], extend='neither')
                tick_label_pd_nr=np.array([0,1,2,3,4,5,6,7])
                tick_label_pd=['sat: no','sat: yes','sat: no (rad unr)','sat: yes (rad unr)','hit','false alarm','correct reject','miss']

                im=m.pcolormesh(lon,lat,ver['mlp'],cmap=cmap_pd, norm=norm_pd, latlon=True)
                divider = make_axes_locatable(m.ax)
                cax = divider.append_axes("right", size="4%", pad=0.05)
                cbar = fig.colorbar(im,cax=cax, ticks=tick_label_pd_nr, spacing='uniform')
                cbar.ax.set_yticklabels(tick_label_pd, fontsize=14)

                m =  map_plot(axis=axes[1],area=area)
                m.ax.set_title('precip detection based on opera vs opera')

                v_pd=np.array([-0.5,0.5,2.5,3.5,4.5,6.5])
                cmap_pd, norm_pd = from_levels_and_colors(v_pd, colors=['darkgrey','lightgrey','plum', '#377eb8','ivory'], extend='neither')
                tick_label_pd_nr=np.array([0,1.5,3,4,5.5])
                tick_label_pd=['no rad','rad clutter: no','rad clutter: yes','rad: yes','rad: no']

                im=m.pcolormesh(lon,lat,ver['ody'],cmap=cmap_pd,norm=norm_pd,latlon=True)
                divider = make_axes_locatable(m.ax)
                cax = divider.append_axes("right", size="4%", pad=0.05)
                cbar = fig.colorbar(im,cax=cax,ticks=tick_label_pd_nr,spacing='uniform')
                a=cbar.ax.set_yticklabels(tick_label_pd,fontsize=14)

                outfile= 'test_%s'
                fig.savefig((in_msg.outputDir+ outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
                print('... create figure: display ' + in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

                plots_done[area].append('OdyVsRr')
                
            ##############################################
            ## cth visualisation without parallax corr for a test
            ##############################################

            if 'CTH' in in_msg.plots[area]:
                fig, axes = plt.subplots(1, 1,figsize=(5, 3))
                plt.rcParams.update({'font.size': 16})
                plt.rcParams.update({'mathtext.default':'regular'}) 

                ## 1st subplot
                m =  map_plot(axis=axes,area=area)
                m.ax.set_title('CTH (without parallax corr)')

                v_rr = np.arange(6000,12001,1000)
                cmap_rr,norm_rr=smart_colormap(v_rr,name='coolwarm',extend='neither')

                im4 = m.pcolormesh(lon,lat,data_cth['CTTH'].height,cmap=cmap_rr,norm=norm_rr,latlon=True)
                fig.colorbar(im4)

                data_cth['CTTH'].height

                outfile= 'CTH_without_parallax_%s'
                fig.savefig((in_msg.outputDir+ outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
                print('... create figure: display ' + in_msg.outputDir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

                plots_done[area].append('CTH')
                
        # end of area loop
        ## start postprocessing
        for area in in_msg.postprocessing_areas:
            postprocessing(in_msg, time_slot, int(sat_nr), area)
            
        # increase the time by a time delta
        time_slot += delta
        # end of time loop

    return plots_done

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def print_usage():
   
   print ("***           ")
   print ("*** Error, not enough command line arguments")
   print ("***        please specify at least an input file")
   print ("***        possible calls are:")
   print ("*** python "+inspect.getfile(inspect.currentframe())+" input_rr ")
   print ("*** python "+inspect.getfile(inspect.currentframe())+" -d 2014 07 23 16 10 input_rr  ")
   print ("                                 date and time must be completely given")
   print ("***           ")
   quit() # quit at this point
#----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

   import sys
   from get_input_msg import get_date_and_inputfile_from_commandline
   in_msg = get_date_and_inputfile_from_commandline(print_usage=print_usage)
   
   plots_done = nostradamus_rain(in_msg)
   print ("*** Satellite pictures produced for ", plots_done)
   print (" ")
