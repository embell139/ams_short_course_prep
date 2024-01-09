import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import itertools as it
from datetime import datetime
import netCDF4 as nc
import glob
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mc
import os

#------------------------
#   GLOBAL VARIABLES
#------------------------    
boundaries={
    'Global':[-180,180,-90,90],
    'Louisiana': [-95.9,-87.50,28.7,33.5],
    'CONUS':[-127.08,-63.87,23.55,49.19],   #   conus
    'Florida':[-84.07,-79.14,24.85,30.5],
    'Northeast':[-74.88,-69.81,40.48,42.88]
}

lpj_dir = './data/lpj/monthly_nc4/gmao.gsfc.nasa.gov/gmaoftp/lott/CH4/wetlands/'
merra_soil_moisture_dir = './data/merra2/MERRA2_400.tavgM_2d_lnd_Nx/'
merra_t2m_dir = './data/merra2/MERRA2_400.instM_2d_asm_Nx/'
merra_precip_rate_dir = './data/merra2/MERRA2_400.tavgM_2d_flx_Nx/'
merra_t2m_clim_dir = '/home/embell/ams_short_course_prep/data/merra2/MERRA2.tavgC_2d_ltm_Nx/1991_2020/'
merra_precip_rate_clim_dir = merra_t2m_clim_dir 
merra_soil_moisture_clim_dir = merra_t2m_clim_dir

savedir = '/data8/embell/ams_short_course/'

params={
    'LPJ CH4 Wetland Emissions':
        {'var':'ch4_wl',
        'cmap':'magma',
        'dir':lpj_dir,
        'nickname':'lpj_ch4_wl'},
    'MERRA-2 T2M':
        {'var':'T2M',
        'cmap':'Spectral_r',
        'dir':merra_t2m_dir,
        'nickname':'merra2_t2m',
        'climdir':merra_t2m_clim_dir,
        'climvar':'T2MMEAN'},
    'MERRA-2 Surface Soil Moisture':
        {'var':'GWETTOP',
        'cmap':'Blues',
        'dir':merra_soil_moisture_dir,
        'nickname':'merra2_sm',
        'climdir':merra_soil_moisture_clim_dir,
        'climvar':'GWETTOP'},
    'MERRA-2 Precipitation Rate':
        {'var':'PRECTOT',
        'cmap':'Spectral_r',
        'dir':merra_precip_rate_dir,
        'nickname':'merra2_pr',
        'climdir':merra_precip_rate_clim_dir,
        'climvar':'PRECTOT'}
}

#------------    
#------------ 
#   creates Spectral_r with white at the bottom
def custom_cmap():
    color_list = [
        [0, 0, 0, 0],
        [0.36862745, 0.30980392, 0.63529412, 1.0],
        [0.53794694, 0.81476355, 0.64505959, 1.0],
        [0.99992311, 0.9976163, 0.74502115, 1.0],
        [0.97347174, 0.54740484, 0.31810842, 1.0],
        [0.61960784, 0.00392157, 0.25882353, 1.0]
    ]
    custom = LinearSegmentedColormap('mycmap',color_list,255)
    return custom

#------------    
#------------ 

def heatmap(x,y,vals,vmin=0,vmax=0.004,cmap='magma',norm='linear',colbar_label='',boundaries=None,**kwargs):
    fig1 = plt.figure(1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    if boundaries:
        ax.set_extent(boundaries)
        ax.add_feature(cfeature.STATES,edgecolor='white',linewidth=0.5)
    else: 
        ax.set_global()
    ax.coastlines(color='white',linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND,color='black')
    ax.add_feature(cartopy.feature.OCEAN,color='black')
    mapp = ax.pcolormesh(x,y,vals,cmap=cmap,transform=ccrs.PlateCarree(),
        vmin=vmin,vmax=vmax)#,norm=norm)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
            
    # colorbar
    tick_locator = mpl.ticker.MaxNLocator(nbins=6)
    colbar_l2 = fig1.colorbar(mapp,extend='max',shrink=0.6,
        label=colbar_label,orientation='horizontal')
    colbar_l2.locator = tick_locator
    colbar_l2.update_ticks()
    for l in colbar_l2.ax.xaxis.get_ticklabels():
        l.set_fontsize(9)
    plt.tight_layout()

    if 'text' in kwargs:
       plt.text(0.6,0.02,kwargs['text'],transform=ax.transAxes)

    if 'savename' in kwargs:
        print('Saving to '+kwargs['savename'])
        plt.figure(1).savefig(kwargs['savename'],dpi=300,bbox_inches='tight')

    return

#------------    
#------------ 

#   NOT IN USE
#def contour_map(x,y,vals,vmin=0,vmax=100,cmap='magma',norm='linear',colbar_label='',boundaries=None,**kwargs):
#    fig1 = plt.figure(1)
#    ax = plt.axes(projection=ccrs.PlateCarree())
#    if boundaries:
#        ax.set_extent(boundaries)
#        ax.add_feature(cfeature.STATES,edgecolor='white',linewidth=0.5)
#    else: 
#        ax.set_global()
#    ax.coastlines(linewidth=0.5)
#    levels = np.linspace(vmin,vmax,20)
#    mapp = ax.contourf(x,y,vals,levels,transform=ccrs.PlateCarree(),
#        vmin=vmin,vmax=vmax)#,norm=norm)
#    if 'title' in kwargs:
#        ax.set_title(kwargs['title'])
#            
#    # colorbar
#    tick_locator = mpl.ticker.MaxNLocator(nbins=6)
#    colbar_l2 = fig1.colorbar(mapp,extend='max',shrink=0.6,
#        label=colbar_label,orientation='horizontal')
#    colbar_l2.locator = tick_locator
#    colbar_l2.update_ticks()
#    for l in colbar_l2.ax.xaxis.get_ticklabels():
#        l.set_fontsize(9)
#    plt.tight_layout()
#
#    if 'text' in kwargs:
#       plt.text(0.6,0.02,kwargs['text'],transform=ax.transAxes)
#
#    if 'savename' in kwargs:
#        print('Saving to '+kwargs['savename'])
#        plt.figure(1).savefig(kwargs['savename'],dpi=300,bbox_inches='tight')
#
#    return

#------------    
#------------ 
def get_lpj_timeseries(year,focus,p):
    files = glob.glob(params[p]['dir']+'*%s*.nc'%(year))
    data = nc.Dataset(files[0])
    wlat = np.logical_and(
        data['longitude'][:] < boundaries[focus][1],
        data['longitude'][:] > boundaries[focus][0]
    )
    wlon = np.logical_and(
        data['latitude'][:] < boundaries[focus][3],
        data['latitude'][:] > boundaries[focus][2]
    )
    #wbox = np.logical_and(wlat,wlon)
    month_labels = []
    month_field = []
    box_totals = []
    for t in range(0,len(data['time'])):
        month_labels.append(datetime(year,t+1,1).strftime('%B')) 
        box_totals.append(np.sum(data[params[p]['var']][t,wlon,wlat]))
        month_field.append(data[params[p]['var']][t,:,:])

    data_return = {'month_labels':month_labels,
        'box_totals':box_totals,
        'month_fields':month_field,
        'units':data[params[p]['var']].units,
        'lat':data['latitude'][:],
        'lon':data['longitude'][:]
    }
    data.close()
    return data_return 

#------------    
#------------ 

def get_merra2_timeseries(year,focus,p,anomaly):
    files = glob.glob(params[p]['dir']+'%s/*.nc4'%(year))
    if anomaly:
        try:
            clim_files = glob.glob(params[p]['climdir']+'*.nc4')
        except:
            print('Climatological mean files (climdir) not found for specified parameter.')
            breakpoint()
    month_labels = []
    box_totals = []
    month_field = []
    dt = []
    for i,f in enumerate(files):
        data = nc.Dataset(f)
        
        #   Get bounding box
        wlat = np.logical_and(
            data['lat'][:] < boundaries[focus][3],
            data['lat'][:] > boundaries[focus][2]
        )
        wlon = np.logical_and(
            data['lon'][:] < boundaries[focus][1],
            data['lon'][:] > boundaries[focus][0]
        )

        datestamp = f.split('.')[-2]
        month = int(datestamp[-2::])

        dt.append(datetime(year,month,1))
        month_labels.append(datetime(year,month,1).strftime('%B'))

        if anomaly:
            #   Make sure you read the climatology for the right month (whichfile)
            whichfile = [datetime(2020,month,1).strftime('%y%m') in f for f in clim_files]
            climdata = nc.Dataset(np.array(clim_files)[whichfile][0])
            
            #   Calculate sum (emissions) or mean (met params) over your bounding box
            if 'LPJ' in p:
                clim_box_total = np.nansum(climdata[params[p]['climvar']][0,wlat,wlon])
                now_box_total = np.nansum(data[params[p]['var']][0,wlat,wlon])
            elif 'MERRA' in p:
                clim_box_total = np.nanmean(climdata[params[p]['climvar']][0,wlat,wlon])
                now_box_total = np.nanmean(data[params[p]['var']][0,wlat,wlon])

            #   Replace fill values with NaN 
            #   Otherwise differencing might give wild results? (Just be safe)
            wfillclim = np.where(climdata[params[p]['climvar']][0,:,:] == climdata[params[p]['climvar']]._FillValue)
            climfield = climdata[params[p]['climvar']][0,:,:]
            climfield[wfillclim] = np.nan
            wfillnow = np.where(data[params[p]['var']][0,:,:] == data[params[p]['var']]._FillValue)
            nowfield = data[params[p]['var']][0,:,:]
            nowfield[wfillnow] = np.nan

            #   And finally, difference current month and long-term mean 
            box_totals.append(now_box_total - clim_box_total)
            month_field.append(nowfield - climfield)
            climdata.close()
        else:
            if 'LPJ' in p:
                box_totals.append(np.nansum(data[params[p]['var']][0,wlat,wlon]))
            elif 'MERRA' in p:
                box_totals.append(np.nanmean(data[params[p]['var']][0,wlat,wlon]))
            #   Replace fill values with NaN (otherwise maps are hard to read) 
            month_field.append(data[params[p]['var']][0,:,:])
            wfill = np.where(month_field[-1] == data[params[p]['var']]._FillValue)
            month_field[-1][wfill] = np.nan
            #breakpoint()

    #   Sort in case months are out of order
    dti = np.argsort(dt)
    month_labels = np.array(month_labels)[dti]
    box_totals = np.array(box_totals)[dti]
    month_field = np.array(month_field)[dti]

    print('mean ',np.nanmean(month_field))
    print('std ',np.nanstd(month_field))

    data_return = {
        'month_labels':month_labels,
        'box_totals':box_totals,
        'month_fields':month_field,
        'units':data[params[p]['var']].units,
        'lat':data['lat'][:],
        'lon':data['lon'][:]
    }
    data.close()
    return data_return 

#------------    
#------------ 
#   output monthly time series summed over boundary
def monthly_timeseries(year,focus,param,anomaly):
    labels = []
    cmap = plt.get_cmap('gnuplot') 
    colors = cmap(np.linspace(0,1,len(param)))
    for i,p in enumerate(param):
    #   In theory this loop is to iterate over multiple parameters
    #   and put them on the same plot,
    #   but I never got around to incorporating multiple axes for 
    #   parameters of different magnitudes/scales.
    #   So right now, [param] should be one element long. 
    #   Don't pass multiple at a time.
        if 'LPJ' in p:
            ts = get_lpj_timeseries(year,focus,p)
        elif 'MERRA' in p:
            ts = get_merra2_timeseries(year,focus,p,anomaly)
            
        if i == 0:
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)

        #breakpoint()
        try:
            ax.plot(
                list(range(0,12)),
                ts['box_totals'],
                linestyle='-',
                linewidth=2,
                color=colors[i],
                markersize=4,
                marker='o',
                label=p
            )
        except ValueError:
            print('Double check that you have all twelve months of MERRA-2 data downloaded!')
            print(params[p]['dir'])
            breakpoint()

        #   Construct plot title
        title = '%s\n%s Mean Monthly %s'%(focus,year,p)
        if anomaly:
           title+=' Anomaly' 
        if 'LPJ' in p:
            title = title.replace('Mean','Total')
        plt.title(title)
        
        plt.xticks(list(range(0,12)))
        ax.set_xticklabels(ts['month_labels'],rotation=40,ha='right')


        if p == param[-1]:
            if i > 0:
                ax.legend(loc='best')
                nickname = '_'.join(params[p]['nickname'] for p in params)
                savename = '%s/box_summed_%s_%s_%s.png'% \
                    (savedir,nickname,year,focus)
            else:
                nickname = params[p]['nickname']
                savename = '%s/%s/%s/box_summed_%s_%s_%s.png'% \
                    (savedir,nickname,focus,nickname,year,focus)
            if anomaly:
                ax.plot(list(range(-1,13)),np.zeros(14),linewidth=0.4)
                savename = savename.replace('.png','_Anomaly.png')
            ax.set_xlim(-1,12)
            ax.set_ylim(-4e-5,4e-5)     #   manual per parameter
            print('Saving to '+savename)
            plt.figure(1).savefig(savename,dpi=300,bbox_inches='tight')

    return ts 
    
    
#------------    
#------------ 

def monthly_heatmaps(year,focus,p,anomaly):
    if 'LPJ' in p:
        ts = get_lpj_timeseries(year,focus,p)
    if 'MERRA-2' in p:
        ts = get_merra2_timeseries(year,focus,p,anomaly)
    
    for t in range(0,len(ts['month_labels'])):    #   monthly data
        plt.close()
    
        #   Construct plot title
        title = '%s\n%s %s'% \
            (datetime(int(year),t+1,1).strftime('%B %Y'),focus,p)
        #   Construct file savename
        savename = savedir+'%s/%s/%s_%s_%s_%s.png'% \
            (params[p]['nickname'],
            focus,
            params[p]['nickname'],
            focus,
            year,
            datetime(int(year),t+1,1).strftime('%m')
            )

        if anomaly:
            savename = savename.replace('.png','_Anomaly.png')
            title+=' Anomaly'
            cmap = 'RdBu_r'
        else:
            cmap = params[p]['cmap']
    
        #   vmin and vmax currently set manually per parameter
        heatmap(
            ts['lon'],
            ts['lat'][:],
            ts['month_fields'][t],
            colbar_label=ts['units'],
            boundaries=boundaries[focus],
            cmap=cmap,
            #vmin = 0,
            #vmax = 0.0025,
            vmin = -4e-5,
            vmax = 4e-5,
            #vmin = 260,
            #vmax = 315,
            #vmin = 0,
            #vmax = np.nanmean(ts['month_fields'])+2.5*np.nanstd(ts['month_fields']),
            #vmin = -4*np.nanstd(ts['month_fields']),
            #vmax = 4*np.nanstd(ts['month_fields']),
            **{'title':title,'savename':savename}
        )
#   save(year)

    return


#------------------------------------------------------------------------    
#------------------------------------------------------------------------    
#------------------------------------------------------------------------ 

year=2020
focus = 'Northeast'
anomaly = 1
param = ['MERRA-2 Precipitation Rate']
#   param:
#    'LPJ CH4 Wetland Emissions'
#    'MERRA-2 T2M'
#    'MERRA-2 Surface Soil Moisture'
#    'MERRA-2 Precipitation Rate'
#    This is a list because I was developing the ability to input
#    multiple params and have them all on the same time series plot..
#    Never got around to it. Do one at a time. 
#

#   output timeseries of 
#   emissions summed over boundary box
#   (or met parameter averaged within boundary box)
monthly_timeseries(year,focus,param,anomaly)
#breakpoint()

#   output maps
#   (I built contour_map in case that was preferred,
#    but didn't end up using it)
for p in param:
    monthly_heatmaps(year,focus,p,anomaly)

