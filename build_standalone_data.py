"""Observed cell-day dataset, 2021--2024; freeze scaling on 2021--2022."""
from pathlib import Path
import json,hashlib,calendar
import numpy as np,pandas as pd,xarray as xr,yaml
from src.preprocess import process_firms,_relative_humidity

ROOT=Path(__file__).resolve().parent

def main():
    original=yaml.safe_load((ROOT/'config.yaml').read_text())
    cfg={k:original[k] for k in ['region','time','grid','firms','era5','paths']};cfg['time'].update(end='2024-12-31',train_years=[2021,2022],val_years=[2023],test_years=[2024])
    cfg['firms']['map_key']=''
    (ROOT/'config_standalone.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    modis=xr.open_dataset(ROOT/'data/raw/modis_public/MOD13A1_2020_2024_samples.nc')
    lat=modis.latitude.values;lon=modis.longitude.values; ncell=len(lat)
    dates=pd.date_range('2021-01-01','2024-12-31'); weather=[];provenance={'era5':{},'target':'VIIRS hotspot occurrence; not incident ignition','train':[2021,2022],'selection_calibration':[2023],'test':[2024]}
    for year in range(2021,2025):
        for month in range(1,13):
            path=ROOT/f'data/raw/era5/era5_{year}_{month:02d}.nc'
            with xr.open_dataset(path) as ds:
                if 'valid_time' in ds:ds=ds.rename({'valid_time':'time'})
                ds=ds.sortby('latitude').sortby('longitude').sortby('time')
                expected=pd.date_range(f'{year}-{month:02d}-01',periods=calendar.monthrange(year,month)[1]*24,freq='h')
                assert np.array_equal(ds.time.values,expected.values),path.name
                ilat=np.argmin(abs(ds.latitude.values[:,None]-lat[None,:]),axis=0);ilon=np.argmin(abs(ds.longitude.values[:,None]-lon[None,:]),axis=0)
                a={v:ds[v].transpose('time','latitude','longitude').values[:,ilat,ilon].reshape(-1,24,ncell) for v in ['t2m','d2m','u10','v10','swvl1']}
                assert all(np.isfinite(v).all() for v in a.values()),path.name
                imax=np.argmax(a['t2m'],axis=1)[:,None,:];tmax=np.take_along_axis(a['t2m'],imax,axis=1)[:,0,:];td=np.take_along_axis(a['d2m'],imax,axis=1)[:,0,:]
                weather.append({'t_max':tmax-273.15,'rh_min':_relative_humidity(tmax,td),'u10_max':np.sqrt(a['u10']**2+a['v10']**2).max(axis=1),'sm_top':a['swvl1'].mean(axis=1)})
            provenance['era5'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest();print('Weather',year,month,flush=True)
    raw={k:np.concatenate([m[k] for m in weather]) for k in weather[0]}
    # Reject invalid/snow/cloud pixels; carry valid completed composites forward, at most 48 days old.
    ndvi=modis['500m_16_days_NDVI'].values;nir=modis['500m_16_days_NIR_reflectance'].values;mir=modis['500m_16_days_MIR_reflectance'].values;qa=modis['500m_16_days_pixel_reliability'].values
    valid=np.isin(qa,[0,1])&(ndvi>=-2000)&(ndvi<=10000)&(nir>=0)&(nir<=10000)&(mir>=0)&(mir<=10000)&((nir+mir)>0)
    # Day 353 year-end composites terminate at the year boundary.
    starts=pd.DatetimeIndex(modis.time.values)
    availability=pd.DatetimeIndex([min(t+pd.Timedelta(days=16),pd.Timestamp(t.year+1,1,1)) for t in starts])
    sample_times=availability.values
    for key,values in [('ndvi',ndvi*.0001),('ndwi',(nir-mir)/(nir+mir+1e-8))]:
        out=np.full((len(dates),ncell),np.nan)
        for cell in range(ncell):
            good=np.flatnonzero(valid[:,cell]);available=sample_times[good];idx=np.searchsorted(available,dates.values,side='right')-1
            ok=idx>=0;chosen=good[np.maximum(idx,0)] if len(good) else np.zeros(len(dates),int)
            if len(good):
                age=(dates.values-sample_times[chosen])/np.timedelta64(1,'D');ok &=age<=48;out[ok,cell]=values[chosen[ok],cell]
        raw[key]=out
    with xr.open_dataset(ROOT/'data/raw/srtm/srtm_slope.nc') as slope:
        ilat=np.argmin(abs(slope.latitude.values[:,None]-lat),axis=0);ilon=np.argmin(abs(slope.longitude.values[:,None]-lon),axis=0)
        raw['slope']=np.broadcast_to(slope.slope.transpose('latitude','longitude').values[ilat,ilon],(len(dates),ncell))
    firms=process_firms(ROOT/'data/raw',cfg)
    ilat=np.searchsorted(firms.latitude.values,lat);ilon=np.searchsorted(firms.longitude.values,lon)
    for key in ['frp_hist','count_hist']:raw[key]=firms[key].values[:,ilat,ilon]
    labels=firms.ignition.values[:,ilat,ilon].astype(np.int8)
    # Retain raw observation arrays so mathematical experiments need not reacquire data.
    rawds=xr.Dataset({k:(['time','cell'],v) for k,v in raw.items()},coords={'time':dates,'cell':np.arange(ncell),'latitude':('cell',lat),'longitude':('cell',lon)})
    rawds['ignition']=(['time','cell'],labels)
    outdir=ROOT/'data/standalone';outdir.mkdir(exist_ok=True);rawds.to_netcdf(outdir/'raw_cell_days.nc')
    norm={};train=dates.year<=2022;scaled={}
    for k,a in raw.items():
        lo=float(np.nanmin(a[train]));hi=float(np.nanmax(a[train]));assert hi>lo
        norm[k]={'min':lo,'max':hi};scaled[k]=np.clip((a-lo)/(hi-lo),0,1)
    ds=xr.Dataset({k:(['time','cell'],a) for k,a in scaled.items()},coords=rawds.coords);ds['ignition']=rawds.ignition
    ds.to_netcdf(outdir/'features.nc');(outdir/'normalization.json').write_text(json.dumps(norm,indent=2))
    frame=ds.to_dataframe().reset_index();columns=list(raw);complete=frame[columns].notna().all(axis=1)
    provenance['annual']={str(year):{'source_rows':int((frame.time.dt.year==year).sum()),'source_positives':int(frame.loc[frame.time.dt.year==year,'ignition'].sum()),'complete_rows':int(((frame.time.dt.year==year)&complete).sum()),'complete_positives':int(frame.loc[(frame.time.dt.year==year)&complete,'ignition'].sum())} for year in range(2021,2025)}
    provenance['modis']='MOD13A1.061 native 500m sample; reliability0/1; completed windows; maximum age48 days'
    metadata=[json.loads(p.read_text()) for p in (ROOT/'data/raw/modis_public').glob('*.json')]
    gaps=[m for m in metadata if m.get('status')=='no source granule in public archive']
    provenance['modis_source_gaps']={'tile_composites':len(gaps),'dates':sorted(set(m['date'] for m in gaps))}
    provenance['modis_samples_sha256']=hashlib.sha256((ROOT/'data/raw/modis_public/MOD13A1_2020_2024_samples.nc').read_bytes()).hexdigest()
    provenance['features_sha256']=hashlib.sha256((outdir/'features.nc').read_bytes()).hexdigest()
    (ROOT/'outputs/stage3/data_audit.json').write_text(json.dumps(provenance,indent=2));print('Dataset complete',provenance['annual'],flush=True)

def load_frames():
    with xr.open_dataset(ROOT/'data/standalone/features.nc') as ds:frame=ds.to_dataframe().reset_index()
    from research_models import NAMES
    frame=frame.dropna(subset=NAMES+['ignition']);frame['year']=frame.time.dt.year
    return frame[frame.year<=2022].copy(),frame[frame.year==2023].copy(),frame[frame.year==2024].copy()

if __name__=='__main__':main()
