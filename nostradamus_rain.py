
# coding: utf-8

# In[42]:

# ATTENTION: NEED TO INTRODUCE REAL-TIME -> Uli or I?
      
# ATTENTION: ONCE I HAVE THE NEWLY TRAINED MODELS I'LL NEED TO SWITCH THE FOLLOWING:
    # pd_rr_preprocess_data_single_scene_ody_seviriIR_nwcsaf_addparams_2017_BTvg09 to the one without BTvg09
    # change input dir of precip detection + rr model + probab matching database
    # area switch to EuropeOdyssey00
    # when loading IR_108, switch to Meteosat-10

# !!!hau!!!:

from __future__ import division
from __future__ import print_function 

#get_ipython().magic(u'matplotlib inline')

import numpy as np

from sklearn.externals import joblib # same purpose as pickle but more efficient with big data / can only pickle to disk

import copy

from datetime import datetime
import time

# to create y, yhsaf & Xraw
from pd_rr_create_y_yhsaf_Xraw_2017 import pd_rr_create_y_yhsaf_Xraw

# to create all_data etc for a single time slot
from pd_rr_preprocess_data import pd_rr_preprocess_data_single_scene

# to carry out probability matching
from rr_probab_matching import probab_match_rr_refprovide

# for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting_tools import smart_colormap,map_plot
from mpop.satellites import GeostationaryFactory

from datetime import datetime, timedelta

from my_msg_module import get_last_SEVIRI_date

#----------------------------------------------------------------------------------------------------------------

def create_trollimage(prop, colormap, cw, filename, time_slot, fill_value=None, composite_file=None, background=None,
                      add_borders=True, add_rivers=False, resolution='l', bits_per_pixel=8, mask=None):

    fill_value=None
    img = trollimage(prop, mode="L", fill_value=fill_value)
    img.colorize(colormap)
    PIL_image = img.pil_image()
            
    # define area
    from mpop.projector import get_area_def
    obj_area = get_area_def(area)
    proj4_string = obj_area.proj4_string            
    # e.g. proj4_string = '+proj=geos +lon_0=0.0 +a=6378169.00 +b=6356583.80 +h=35785831.0'
    area_extent = obj_area.area_extent              
    # e.g. area_extent = (-5570248.4773392612, -5567248.074173444, 5567248.074173444, 5570248.4773392612)
    area_tuple = (proj4_string, area_extent)
    
    from plot_msg import add_borders_and_rivers
    add_borders_and_rivers(PIL_image, cw, area_tuple,
                           add_borders=add_borders, add_rivers=add_rivers,
                           resolution=resolution, verbose=False)
    
    # indicate mask
    if mask!=None:
        print ("    indicate measurement mask")

        #from skimage import feature
        #mask = feature.canny(mask) - mask

        # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve2d.html
        from scipy import signal
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                           [-10+0j, 0+ 0j, +10 +0j],
                           [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
        grad = signal.convolve2d(mask, scharr, boundary='symm', mode='same')
        mask = np.absolute(grad)
        mask /= mask.max()
        mask = 1 - mask
        print (mask.max(),mask.min()) 

        img = trollimage(mask, mode="L", fill_value=None) #fill_value,[1,1,1], None
        from trollimage.colormap import greys
        img.colorize(greys)
        
        ##img.putalpha(mask*0 + 0.5)
        img.putalpha((mask.max()-mask) * 0.5)
        PIL_mask = img.pil_image()
        from PIL import Image as PILimage 
        PIL_image = PILimage.alpha_composite(PIL_mask, PIL_image)
        #PIL_image = PIL_mask

    # save image as file 
    outfile = time_slot.strftime(filename)
    PIL_image.save(outfile, optimize=True)
    print ("... create figure: display "+outfile+" &")
        
    if composite_file != None:
        bg_file = time_slot.strftime(background)
        comp_file = time_slot.strftime(composite_file)
        
        command="/usr/bin/composite -depth "+str(bits_per_pixel)+" "+outfile+" "+bg_file+" "+comp_file
        print ("    "+command)
        print ("")
        import subprocess
        subprocess.call(command, shell=True) #+" 2>&1 &"
    
#----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    near_real_time=True
    if near_real_time:
        RSS=False
        time_slot = get_last_SEVIRI_date(RSS, delay=10)
        end_date  = time_slot
    else:
        time_slot = datetime(2018,7,13,15,45)
        end_date  = datetime(2018,7,13,15,45)

    delta     = timedelta(minutes=15) 

    while time_slot <= end_date:
        print(time_slot)

        ################################################
        ## CHOOSE THE SETUP (time_slot, area, model)
        ################################################

        #infile='/data/COALITION2/PicturesSatellite/results_BEL/data/precip_estimate/dates/test400_201706_201707_201708.npy'
        #test_dates = np.load(infile)
        #len(test_dates)

        # Area
        #area='EuropeOdyssey00' # will need to be switched to EuropeOdyssey00 once I've done the new training
        #area='EuropeCanaryS95' ## does not work, as radar Odyssey masks saved on the 'EuropeOdyssey00' projection
        area='ccs4'

        # automatic choise of the FULL DISK SERVICE Meteosat satellite
        if time_slot <  datetime(2008, 5, 13, 0, 0):   # before 13.05.2008 only nominal MSG1 (meteosat8), no Rapid Scan Service yet
            sat_nr = "08" 
        elif time_slot <  datetime(2013, 2, 27, 9, 0): # 13.05.2008 ...  27.02.2013 
            sat_nr = "09"                              # MSG-2  (meteosat9) became nominal satellite, MSG-1 (meteosat8) started RSS
        elif time_slot <  datetime(2018, 3, 9, 0, 0):  # 27.02.2013 9:00UTC ... 09.03.2013                                   
            sat_nr = "10"                              # MSG-3 (meteosat10) became nominal satellite, MSG-2 started RSS (MSG1 is backup for MSG2)
        else:
            sat_nr = "11"
        print ("... work with Meteosat"+str(sat_nr))

        # Model
        model = 'mlp'

        if model == 'mlp':
            dir_start_pd= './models/precipitation_detection/mlp/2hl_100100hu_10-7alpha_log/'
            dir_start_rr= './models/precipitation_rate/mlp/2hl_5050hu_10-2alpha_log/'

        # remove viewing geometry
        remove_vg = True
        # Probability matching    
        probab_match = True

        # plot IR_108 channel below the sat derived rainfall
        IR_108 = True


        ##########################
        ## LOAD THE NEEDED INPUTS
        ##########################

        ###############################################
        ## load the mlp for the precip detection (pd) #
        ###############################################

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

        if probab_match:
                # load in the ref data sets created with the script: rr_probab_matching_create_refset.ipynb
                ody_rr_ref=np.load(dir_start_rr+'pm_valid_data_ody_rr_ref.npy')
                pred_rr_ref=np.load(dir_start_rr+'pm_valid_data_pred_rr_ref.npy')


        #############################################################################
        ## load the IR_108 channel for plotting black white background if requested
        #############################################################################

        if IR_108:
            # load the IR_108 channel 
            global_sat = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot) # needs to be switched to
                                                                                                    # 10 for final product!
            print("*** read IR_108")
            global_sat.load(['IR_108'])
            global_sat = global_sat.project(area, precompute=True)
            print (global_sat)

            # load the CTH for the parallax corr    
            print("*** read CTH")
            global_cth = GeostationaryFactory.create_scene("meteosat", sat_nr, "seviri", time_slot)
            global_cth.load(['CTTH'], calibrate=True, reader_level="seviri-level3")         
            global_cth = global_cth.project(area, precompute=True)

            # carry out the parallax corr  
            print("***  carry out the parallax correction")
            global_sat = global_sat.parallax_corr(cth=global_cth['CTTH'].height, fill='nearest')


        ###########################################################
        ## SINGLE TIME SLOT TO CARRY OUT A FULL RAIN RATE RETRIEVAL 
        ###########################################################

        # preprocess the data 
        rr={}
        all_data,all_data_names,mask_h,mask_r,mask_rnt,rr['ody'],rr['hsaf'],lon,lat = pd_rr_preprocess_data_single_scene( sat_nr, area, time_slot, 'nearest', 'rr', read_HSAF=False)

        if 1==0:
            import matplotlib.pyplot as plt
            #fig = plt.figure()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
            plt.subplot(2, 1, 1)
            plt.imshow(mask_h)
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.imshow(mask_r)
            plt.colorbar()
            fig.savefig("mask_h_mask_r.png")
            print("... display mask_h_mask_r.png &")
            #plt.show()
            quit()

        del rr['hsaf'] # since not actually needed in this script

        print('predictions at ' + str(mask_h.sum())+' out of ' +str(mask_h.flatten().shape[0])+ ' points')

        ####################################
        ## precip detection
        ####################################

        # create y_pd, y_hsaf_pd, X_raw_pd 
        y_pd_vec, y_hsaf_pd_vec, X_raw, feature_list = pd_rr_create_y_yhsaf_Xraw( all_data, all_data_names, 'pd', cut_precip=False )

        del y_pd_vec, y_hsaf_pd_vec # (since not actually ever needed in this script)


        if remove_vg==True:
            feature_list = np.append(feature_list[:6],feature_list[8:])
            X_raw = np.hstack([X_raw[:,:6],X_raw[:,8:]])
            print(X_raw.shape)
            feature_list

        # check features
        if np.array_equal(feature_list, feature_list_pd):
            print('ok, input features correspond to input features required by the loaded model')
        else:
            print('ATTENTION, input features do not correspond to input features required by the loaded model')

        # create X_pd
        X_pd=scaler_pd.transform(X_raw) 

        # create final precip detection fields: opera + hsaf
        pd = {}
        pd['ody']=rr['ody']>=0.3

        # make precip detection predictions
        print ("***  make precip detection predictions")
        pd_probab = clf_pd.predict_proba(X_pd)[:,1]   # probab precip balanced classes
        pd_vec_h=pd_probab>=thres_pd  
        pd[model] = np.zeros(lon.shape,dtype=bool)
        pd[model][mask_h]=pd_vec_h

        ####################################
        ## rain rate on above identified precipitating pixels
        ####################################

        # reduce X_raw to the points where rain was predicted by the mlp
        X_raw= X_raw[pd_vec_h,:] 

        # check, if read features correspond to the trained model
        if np.array_equal(feature_list, feature_list_rr):
            print('ok, input features correspond to input features required by the loaded model')
        else:
            print('ATTENTION, input features do not correspond to input features required by the loaded model')

        # create X_rr
        X_rr=scaler_rr.transform(X_raw)    

        # rain rate prediction at places where precip detected by mlp
        rr_tmp=reg_rr.predict(X_rr)  

        # carry out a probability machting if requested    
        if probab_match:
            print(model)
            pm_str = str(model)+'_pm'
            rr_tmp_pm = probab_match_rr_refprovide(ody_rr_ref,pred_rr_ref,rr_tmp)  
            rr[pm_str] =  np.zeros_like(lon)
            rr[pm_str][pd[model]]=rr_tmp_pm

        # correct all too low predictions to the threhold rain rate        
        rr_tmp[rr_tmp<0.3]=0.3 # correct upward all too low predictions (i.e. the ones below the precip detection threshold)
        rr[model] = np.zeros_like(lon)
        rr[model][pd[model]]=rr_tmp    


        #####################################
        ## SAVE RESULT AS NETCDF 
        #####################################

        #outdir = '/data/COALITION2/PicturesSatellite/results_BEL/plots/precip_estimate/sat_live/'
        outdir = '/data/cinesat/out/'

        save_as_netCDF = True
        if save_as_netCDF:

            # save result as netCDF
            from netCDF4 import Dataset
            outfile= outdir+'MSG_rr-'+model+'-'+area+'_%s' % time_slot.strftime('%Y%m%d%H%M')+'.nc'
            ncfile = Dataset(outfile,'w',format='NETCDF4_CLASSIC')

            nx=rr[model].shape[1]
            ny=rr[model].shape[0]

            #create dimensions
            ncfile.createDimension('x',nx)
            ncfile.createDimension('y',ny)

            #define variables
            # data types: 'f4' (32-bit floating point), 'f8' (64-bit floating point), 'i4' (32-bit signed integer), 'i2' (16-bit signed integer), 
            x = ncfile.createVariable('x','i4',('x',))
            y = ncfile.createVariable('y','i4',('y',))
            #pd_nc = ncfile.createVariable('precipitation_detection','f4',('y','x'))
            rr_nc = ncfile.createVariable('rainfall_rate','f4',('y','x'))
            if probab_match:
                rr_pm_nc = ncfile.createVariable('rainfall_rate (probability matched)','f4',('y','x'))

            # write data into variables 
            x[:] = range(nx)
            y[:] = [ny-1-i for i in range(ny)]
            #pd_nc[:]    = pd[model]        
            rr_nc[:]    = rr[model]
            if probab_match:
                rr_pm_nc[:] = rr[pm_str]

            #close ncfile
            ncfile.close()
            print ("... saved results in: "+outfile)
            print('=================================')


        #####################################
        ## SINGLE TIME SLOT TO DRAW THE MAPS 
        #####################################

        ####################################
        ## plot precip detection
        ####################################

        mask_rt = np.logical_and(mask_r, mask_rnt==False) # trusted radar i.e. True where I have a trustworthy radar product available

        mod_ss = [model] + ['ody']

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
        cmap_pd, norm_pd = from_levels_and_colors(v_pd, colors=['darkgrey', '#984ea3','lightgrey','plum', '#377eb8', '#e41a1c','ivory','#ff7f00'], extend='neither')

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
            outfile= 'precip_detection_sat'+model+'_vs_opera_%s'
            fig.savefig((outdir+outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
            print('... create figure: display ' + outdir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')


        ####################################
        ## plot rain rate
        ####################################

        # with matplotlib
        plot_precipitation_rate=False
        if plot_precipitation_rate:    

            # create the combi rr field
            rr['combi']=copy.deepcopy(rr[model+'_pm'])
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

            if IR_108:
                # plot the IR_108 channel
                clevs = np.arange(225,316,10)
                cmap_sat,norm_sat=smart_colormap(clevs,name='Greys',extend='both')

                im4 = m.pcolormesh(lon,lat,global_sat['IR_108_PC'].data,cmap=cmap_sat,norm=norm_sat,latlon=True)
            else:
                # plot a white surface to distinguish between the regions where the produ
                v_pd_nt=np.array([0.5,1.5])
                cmap_pd_nt, norm_pd_nt = from_levels_and_colors(v_pd_nt, colors=['white'], extend='neither')
                im4=m.pcolormesh(lon,lat,np.ones(lon.shape),cmap=cmap_pd_nt,norm=norm_pd_nt,latlon=True)


            if probab_match:
                im=m.pcolormesh(lon, lat,rr[model+'_pm'], cmap=cmap_rr, norm=norm_rr, latlon=True)

            else:
                im=m.pcolormesh(lon, lat,rr[model],       cmap=cmap_rr, norm=norm_rr, latlon=True)

            if IR_108 and probab_match:
                outfile= 'rr_combioperasat'+model+'pm_satIR108'+model+'pm_%s'    
            elif IR_108 and (probab_match==False):
                outfile= 'rr_combioperasat'+model+'_satIR108'+model+'_%s'
            elif (IR_108==False) and probab_match:
                outfile= 'rr_combioperasat'+model+'pm_sat'+model+'pm_%s'
            elif (IR_108==False) and (probab_match==False):
                outfile= 'rr_combioperasat'+model+'_sat'+model+'_%s'    

            fig.subplots_adjust(bottom=0.15)
            cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])
            cbar=fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('$mm\,h^{-1}$')


            fig.savefig((outdir+outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
            print('... create figure: display ' + outdir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

            elapsed = time.time() - t
            print("... elapsed time for creating the rainrate image in seconds: "+str(elapsed))


        plot_trollimage=True
        if plot_trollimage:
            
            print ("*** create plot with trollimage")
            from trollimage.image import Image as trollimage
            from copy import deepcopy
            from trollimage.colormap import RainRate
            colormap = deepcopy(RainRate)
            
            # define contour write for coasts, borders, rivers
            from pycoast import ContourWriterAGG
            mapDir='/opt/users/common/shapes/'
            cw = ContourWriterAGG(mapDir)
            
            from plot_msg import choose_map_resolution
            resolution = choose_map_resolution(area, None, None) 
            #resolution='l' # odyssey, europe
            #resolution='i' # ccs4
            print ("resolution=", resolution)
            
            if IR_108:  
                # create black white background
                #img_IR_108 = global_sat.image.channel_image('IR_108_PC')
                img_IR_108 = global_sat.image.ir108()
                IR_file=outdir+'MSG_IR-108-'+area+'_%s' % time_slot.strftime('%Y%m%d%H%M')+'.png'
                img_IR_108.save(IR_file)

            # reference Odyssey radar composite
            prop = np.ma.masked_equal(rr['ody'], 0)
            filename       = outdir+'ODY_rainrate-'+area+'_%Y%m%d%H%M.png'
            composite_file = outdir+'ODY_rainrate-IR-108-'+area+'_%Y%m%d%H%M.png'
            create_trollimage(prop, colormap, cw, filename, time_slot, composite_file=composite_file, background=IR_file, mask=mask_r, resolution=resolution)

            # rainrate as produced by model 
            prop = np.ma.masked_equal(rr[model], 0)
            filename       = outdir+'MSG_rainrate-'+model       +"-"+area+'_%Y%m%d%H%M.png'
            composite_file = outdir+'MSG_rainrate-'+model+"-IR-108-"+area+'_%Y%m%d%H%M.png'
            create_trollimage(prop, colormap, cw, filename, time_slot, composite_file=composite_file, background=IR_file, resolution=resolution)
                
            # rainrate with probability matching 
            if probab_match:
                prop = np.ma.masked_equal(rr[model+'_pm'], 0)
                filename       = outdir+'MSG_rainrate-'+model       +"-"+area+'_%Y%m%d%H%M.png'
                composite_file = outdir+'MSG_rainrate-'+model+"-IR-108-"+area+'_%Y%m%d%H%M.png'
                create_trollimage(prop, colormap, cw, filename, time_slot, composite_file=composite_file, background=IR_file, resolution=resolution)    

            # rainrate Odyssey radar composite with satellite rainrate beyond radar range
            combi_image=True
            if combi_image:
                # create the combi rr field
                rr['combi']=copy.deepcopy(rr[model+'_pm'])
                rr['combi'][mask_r]=rr['ody'][mask_r]
                prop = np.ma.masked_equal(rr['combi'], 0)
                filename       = outdir+'MSG_rainrate-ody-'+model       +"-"+area+'_%Y%m%d%H%M.png'
                composite_file = outdir+'MSG_rainrate-ody-'+model+"-IR-108-"+area+'_%Y%m%d%H%M.png'
                create_trollimage(prop, colormap, cw, filename, time_slot, composite_file=composite_file, background=IR_file, mask=mask_r, resolution=resolution)

            print('=================================')

        ##############################################
        ## potential other map setups
        ##############################################
        ##############################################

        ##############################################
        ## opera composite vs the prediction... but I think it'd be less confusing to only show the prediction
        ##############################################

        plot_opera_vs_prediction=False
        if plot_opera_vs_prediction:

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
            fig.savefig((outdir+ outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
            print('... create figure: display ' + outdir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

        ##############################################
        ## global cth visualisation without parallax corr for a test
        ##############################################

        plot_cth=False
        if plot_cth:
            fig, axes = plt.subplots(1, 1,figsize=(5, 3))
            plt.rcParams.update({'font.size': 16})
            plt.rcParams.update({'mathtext.default':'regular'}) 

            ## 1st subplot
            m =  map_plot(axis=axes,area=area)
            m.ax.set_title('CTH (without parallax corr)')

            v_rr = np.arange(6000,12001,1000)
            cmap_rr,norm_rr=smart_colormap(v_rr,name='coolwarm',extend='neither')

            im4 = m.pcolormesh(lon,lat,global_cth['CTTH'].height,cmap=cmap_rr,norm=norm_rr,latlon=True)
            fig.colorbar(im4)

            global_cth['CTTH'].height

            outfile= 'CTH_without_parallax_%s'
            fig.savefig((outdir+ outfile %time_slot.strftime('%Y%m%d%H%M')), dpi=300, bbox_inches='tight')
            print('... create figure: display ' + outdir+outfile %time_slot.strftime('%Y%m%d%H%M') + '.png')

        # increase the time by a time delta
        time_slot += delta
