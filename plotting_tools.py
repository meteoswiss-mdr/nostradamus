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
