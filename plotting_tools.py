#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function



#################
## Mapfigure from Nicolas Piaget iac ethz (adjusted)
#################

import numpy as np
#from cartopy import crs as ccrs # module was not available -> dont think I need it
#from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.pyplot import get_cmap, subplot

class Mapfigure_Nicolas:
    """
        Class based on Basemap with additional functionallity
        such as plot_trajectories
    """
    def __init__(self, resolution='i', projection='cyl',
                 domain=None, lon=None, lat=None, basemap=None, **kwargs):

            if basemap is None:
                if (domain is None) & (lon is not None):
                    domain = [lon.min(), lon.max(), lat.min(), lat.max()]
                if projection == 'eqc':
                    projection = 'cyl'
                if domain is not None:
                    kwargs['llcrnrlon'] = domain[0]
                    kwargs['urcrnrlon'] = domain[1]
                    kwargs['llcrnrlat'] = domain[2]
                    kwargs['urcrnrlat'] = domain[3]
                kwargs['resolution'] = resolution
                kwargs['projection'] = projection
                self.m = Basemap(**kwargs)
            else:
                self.m = basemap
            if lon is not None:
                self.x, self.y = self.m(lon, lat)

    def __getattr__(self, item):
        return getattr(self.m, item)

    def __call__(self, *args, **kwargs):
        return self.m(*args, **kwargs)

    def __dir__(self):
        return self.m.__dir__() + ['drawmap', 'plot_traj']

    def drawmap(self, continent=False, nbrem=5, nbrep=5,
                coastargs={}, countryargs={},
                meridiansargs={}, parallelsargs={}):
        """
        draw basic features on the map
        nbrem: interval bewteen meridians
        nbrep: interval between parallels
        """
        self.drawcoastlines(**coastargs)
        self.drawcountries(**countryargs)
        merid = np.arange(-180, 180, nbrem)
        parall = np.arange(-90, 90, nbrep)
        self.drawmeridians(merid, labels=[0, 0, 0, 1], **meridiansargs)
        self.drawparallels(parall, labels=[1, 0, 0, 0], **parallelsargs)
        if continent:
            self.fillcontinents(color='lightgrey')


#################
## Basemap configuration adapted to EuropeOdyssey00 & 95
#################
from mpl_toolkits.basemap import Basemap

def map_plot(axis,area='EuropeOdyssey00',fill_color='black'):
    if area == 'EuropeOdyssey00':
        m = Basemap(projection='geos',lon_0=0,resolution='l',llcrnrx=-1215000,llcrnry=3585000,urcrnrx=1780000,urcrnry=5273000)
    if area == 'EuropeOdyssey95':
        m = Basemap(projection='geos',lon_0=9.5,resolution='l',llcrnrx=-1250000,llcrnry=3753000,urcrnrx=1365000,urcrnry=5250000)
        
    m.ax = axis
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(40,71,10),labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-20,61,10), labels=[0, 0, 0, 1])
    m.drawmapboundary(fill_color=fill_color)
    m.ax.set_axis_bgcolor('black')
    
    return (m)

#----------------------------------------------------------------------------------------------------------------

#################
## Colorbar for imshow / pcolormesh from Loris Foresti (adjusted)
#################            
            
from matplotlib.colors import from_levels_and_colors 
from pylab import get_cmap

def smart_colormap(clevs, name='jet', extend='both'):
    '''
    Automatically grabs the colors to extend the colorbar from the colormap
    '''
    
    # Define number of colors
    if extend == 'both':
        nrColors = len(clevs)+1
    elif (extend == 'min') | (extend == 'max'):
        nrColors = len(clevs)
    elif (extend == 'neither'):
        nrColors = len(clevs)-1
    else:
        nrColors = len(clevs)-1
        extend = 'neither'
    
    # Get colormap
    cmap = get_cmap(name, nrColors)
    
    
    # Get the list of colors
    colors = []
    for i in range(0, nrColors):
        colors.append(cmap(i/(nrColors-1)))
    
    # Use utility function to get cmap and norm at the same time
    cmap, norm = from_levels_and_colors(clevs, colors, extend=extend)

    return(cmap, norm)

#----------------------------------------------------------------------------------------------------------------

def create_trollimage(prop, colormap, cw, filename, time_slot, area, fill_value=None, composite_file=None, background=None,
                      add_borders=True, add_rivers=False, resolution='l', bits_per_pixel=8, mask=None):

    from trollimage.image import Image as trollimage

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
    
