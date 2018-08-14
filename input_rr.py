
def input(in_msg):

    import inspect
    in_msg.input_file = inspect.getfile(inspect.currentframe()) 
    print "*** read input from ", in_msg.input_file

    # THIS HAS TO BE THE FULL DISK SATELLITE, AS THE ANN WAS TRAINED FOR 0 degree position  
    # 8=MSG1, 9=MSG2, 10=MSG3, 11=MSG4
    in_msg.sat = "meteosat"
    in_msg.sat_nr=11
    in_msg.RSS=False
    
    # specify an delay (in minutes), when you like to process a time some minutes ago
    # e.g. current time               2015-05-31 12:33 UTC
    # delay 5 min                     2015-05-31 12:28 UTC
    # last Rapid Scan Service picture 2015-05-31 12:25 UTC (Scan start) 
    in_msg.delay=15

    # initialize self.datetime with a fixed date (for testing purpose)
    if False:
        # offline mode (always a fixed time) # ignores command line arguments
        year=2015
        month=2
        day=10
        hour=11
        minute=45
        in_msg.update_datetime(year, month, day, hour, minute)
        # !!!  if archive is used, adjust meteosat09.cfg accordingly !!!

    in_msg.end_date = None
        
    # Model
    in_msg.model = 'mlp'
    
    # remove viewing geometry
    in_msg.remove_vg = True
        
    # calculate Probability matching    
    in_msg.probab_match = True

    # plot IR_108 channel below the sat derived rainfall
    in_msg.IR_108 = True

    # NOT YET IMPLEMENTED 
    #in_msg.parallax_correction = True
    in_msg.parallax_gapfilling  = 'nearest' # 'False' (default), 'nearest'
    #in_msg.parallax_gapfilling = 'bilinear' # 'False' (default), 'nearest'
    
    in_msg.read_HSAF=False

    ##############
    # chose area
    ##############
    ##in_msg.areas.append('EuropeCanary')    # upper third of MSG disk, satellite at 0.0 deg East, full resolution 
    ##in_msg.areas.append('EuropeCanary95')  # upper third of MSG disk, satellite at 9.5 deg East, full resolution 
    ##in_msg.areas.append('EuropeCanaryS95') # upper third of MSG disk, satellite at 9.5 deg East, reduced resolution 1000x400
    ##in_msg.areas.append('EuroMercator')    # same projection as blitzortung.org
    ##in_msg.areas.append('germ')            # Germany 1024x1024
    ##in_msg.areas.append('euro4')           # Europe 4km, 1024x1024
    ##in_msg.areas.append('eurotv4n')        # Europe TV4 -  4.1x4.1km 2048x1152
    ##in_msg.areas.append('eurol')           # Europe 3.0km area - Europe 2560x2048
    in_msg.areas.append('euroHDready')      # Europe in HD resolution 1280 x 720
    ##in_msg.areas.append('euroHDfull')      # Europe in full HD resolution 1920 x 1080
    ##in_msg.areas.append('SwitzerlandStereo500m')
    in_msg.areas.append('ccs4')             # CCS4 Swiss projection 710x640
    ##in_msg.areas.append('alps95')          # area around Switzerland processed by NWCSAF software 349x151 
    ##in_msg.areas.append('ticino')          # stereographic proj of Ticino 342x311
    ##in_msg.areas.append('opera_odyssey')
    in_msg.areas.append('odysseyS25')
    in_msg.check_RSS_coverage()

    # save rainrate (ANN) prediction as netCDF file for different areas 
    in_msg.save_netCDF = ['ccs4','euroHDready']
    in_msg.outdir_netCDF = '/data/COALITION2/database/meteosat/nostradamus_RR/%Y/%m/%d/'
    in_msg.file_netCDF   = 'MSG_rr-%(model)s-%(area)s_%Y%m%d%H%M.nc'
    # read rainrate from netCDF file instead of running the ANN
    in_msg.read_from_netCDF = False

    #-------------------------------------
    # choose products as function of area
    #-------------------------------------
    in_msg.plots={}
    # possible plots:
    # 'RATE':        odyssey rainrate
    # 'pdMlp':       precipitation_detection with Multi Layer Perceptrion (MLP) ANN
    # 'rrMatplotlib' rain rate (Leas Matplotlib diagram)
    # 'rrMlp':       rain rate Multi Layer Perceptrion (MLP) Aritifical Neural Network (ANN)
    # 'rrMlpPm':     rain rate MLP ANN with probability matching (PM)
    # 'rrOdyMlp':    same as rrMlp, but replaced with Odyssey when available
    # 'rrOdyMlpPm':  same as rrMlpPm, but replaced with Odyssey when available
    # 'OdyVsRr':     Odyssey vs rain rate with MLP ANN
    # 'IR_108':      background IR_108 image
    # 'CTH':         Cloud Top Height (as given by NWCSAF) 
    in_msg.plots['ccs4']        = ['rrMlp','rrMlpPm']
    in_msg.plots['odysseyS25']  = ['rrMlp']
    in_msg.plots['euroHDready'] = ['rrOdyMlpPm']

    # NOT YET IMPLEMENTED 
    # in_msg.check_input = False
    
    in_msg.make_plots = True
    in_msg.fill_value = (0,0,0)  # black (0,0,0) / white (1,1,1) / transparent None  
    in_msg.add_title = True
    #in_msg.title = [" %(sat)s, %Y-%m-%d %H:%MUTC, %(area)s, %(rgb)s"]
    in_msg.title = ["2nd layer:  %(sat)s ANN %(rgb)s"]
    in_msg.title_y_line_nr = 2
    #in_msg.title_color = (255,255,255)
    in_msg.add_borders = True
    in_msg.add_rivers = False
    in_msg.add_logos = False
    in_msg.logos_dir = "/opt/users/common/logos/"
    in_msg.add_colorscale = True
    in_msg.HRV_enhancement = False

    in_msg.outputFile = 'MSG_%(rgb)s-%(area)s_%y%m%d%H%M.png'
    #in_msg.outputDir='./pics/'
    #in_msg.outputDir = "./%Y-%m-%d/%Y-%m-%d_%(rgb)s-%(area)s/"
    #if in_msg.nrt:
    #    in_msg.outputDir = '/data/cinesat/out/'
    #else:
    #    in_msg.outputDir = '/data/COALITION2/PicturesSatellite/%Y-%m-%d/%Y-%m-%d_%(rgb)s_%(area)s/'
    #    #in_msg.outputDir = '/data/COALITION2/PicturesSatellite/%(rgb)s/%Y-%m-%d/'
    #    #in_msg.outputDir = '/data/COALITION2/PicturesSatellite/GPM/%Y-%m-%d/'
    #in_msg.outputDir = '/data/COALITION2/PicturesSatellite/%Y-%m-%d/%Y-%m-%d_%(rgb)s_%(area)s/'
    in_msg.outputDir = '/data/cinesat/out/'
        
    in_msg.compress_to_8bit=False

    in_msg.scpOutput = True
    #default: in_msg.scpOutputDir="las@lomux240:/www/proj/OTL/WOL/cll/satimages"
    #default: in_msg.scpID="-i /home/cinesat/.ssh/id_dsa_las"
    in_msg.scpProducts = [["MSG_rrMlp-HRVir108",   "MSG_radar-HRVir108", "MSG_CRR-HRVir108", \
                           "MSG_rrMlpPm-HRVir108", "MSG_h03-HRV"    , "MSG_CRPh-HRVir108"],\
                          ["MSG_rrMlp-VIS006ir108","MSG_RATE-VIS006ir108","MSG_h03-VIS006ir108"],\
                          "rrOdyMlpPm-VIS006ir108"]

    # please download the shape file 
    in_msg.mapDir='/opt/users/common/shapes/'

    #in_msg.postprocessing_areas=["ccs4"]   
    #in_msg.postprocessing_areas=['EuropeCanaryS95']
    in_msg.postprocessing_areas=["ccs4","odysseyS25","euroHDready"]

    in_msg.postprocessing_composite={}
    # possible backgrounds: IR-108 (not yet implemented: HRV, VIS006ir108, HRVir108)
    in_msg.postprocessing_composite['ccs4']       = ["rrMlp-HRVir108","rrMlpPm-HRVir108"]
    in_msg.postprocessing_composite['odysseyS25'] = ["rrMlp-VIS006ir108"]
    in_msg.postprocessing_composite['euroHDready'] = ["rrOdyMlpPm-VIS006ir108"]
    
    in_msg.postprocessing_montage={}
    #in_msg.postprocessing_montage['ccs4']       = [["MSG_radar-HRVir108","MSG_rrMlp-HRVir108", "MSG_CRR-HRVir108", "MSG_CRPh-HRVir108"]]
    in_msg.postprocessing_montage['ccs4']      = [["MSG_rrMlp-HRVir108",   "MSG_radar-HRVir108", "MSG_CRR-HRVir108", \
                                                   "MSG_rrMlpPm-HRVir108", "MSG_h03-HRV"    , "MSG_CRPh-HRVir108"]]
    in_msg.postprocessing_montage['odysseyS25'] = [["MSG_rrMlp-VIS006ir108","MSG_RATE-VIS006ir108","MSG_h03-VIS006ir108"]]
    
    #in_msg.resize_composite = 100
    in_msg.resize_montage = 70

