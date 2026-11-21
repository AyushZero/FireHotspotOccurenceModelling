"""Sample observed MOD13A1.061 COGs consistently for 2021--2024.

Uses the public Microsoft Planetary Computer read token; no user credential.
Stores raw sampled NDVI/NIR/MIR/reliability and unsigned source URLs, not tokens.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import time,json,hashlib,xml.etree.ElementTree as ET
import requests,numpy as np,pandas as pd,xarray as xr
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
from src.grid import grid_info
import yaml

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'data/raw/modis_public';OUT.mkdir(exist_ok=True)
BASE='https://modiseuwest.blob.core.windows.net/modis-061-cogs'
TOKEN_URL='https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs'
BANDS=['500m_16_days_NDVI','500m_16_days_NIR_reflectance','500m_16_days_MIR_reflectance','500m_16_days_pixel_reliability']
CRS='+proj=sinu +R=6371007.181 +nadgrids=@null +wktext'

def main():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text());gi=grid_info(cfg)
    ca=gpd.read_file('zip://'+str(ROOT/'outputs/cb_2024_us_state_500k.zip'));geom=ca.loc[ca.STUSPS=='CA','geometry'].iloc[0]
    cells=np.array([(lat,lon) for lat in gi['lats'] for lon in gi['lons'] if geom.covers(Point(float(lon),float(lat)))])
    xx,yy=transform('EPSG:4326',CRS,cells[:,1],cells[:,0]);xx=np.array(xx);yy=np.array(yy)
    hs=np.floor((xx+20015109.354)/1111950.5196667).astype(int);vs=np.floor((10007554.677-yy)/1111950.5196667).astype(int)
    tiles=sorted(set(zip(hs,vs)));print('Cells',len(cells),'tiles',tiles,flush=True)
    response=requests.get(TOKEN_URL,timeout=60);response.raise_for_status();token=response.json()['token']
    dates=[pd.Timestamp('2020-12-18')]+[pd.Timestamp(year,1,1)+pd.Timedelta(days=day) for year in range(2021,2025) for day in range(0,365,16)]
    jobs=[(date,int(h),int(v)) for date in dates for h,v in tiles]
    def one(job):
        date,h,v=job;stamp=date.strftime('%Y%j');target=OUT/f'{stamp}_h{h:02d}v{v:02d}.npz';meta=target.with_suffix('.json')
        if target.exists() and meta.exists():return target.name,'cached'
        ix=np.flatnonzero((hs==h)&(vs==v));prefix=f'MOD13A1/{h:02d}/{v:02d}/{stamp}/'
        for attempt in range(5):
            try:
                r=requests.get(BASE+'?restype=container&comp=list&prefix='+prefix+'&maxresults=100&'+token,timeout=60);r.raise_for_status()
                names=[b.findtext('Name') for b in ET.fromstring(r.content).findall('./Blobs/Blob')]
                if not names:
                    np.savez_compressed(target,cell_indices=ix,**{band:np.full(len(ix),np.nan) for band in BANDS})
                    meta.write_text(json.dumps({'date':str(date.date()),'tile':[h,v],'status':'no source granule in public archive','unsigned_prefix':BASE+'/'+prefix,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
                    return target.name,'source gap (retained as missing)'
                values={};sources={}
                for band in BANDS:
                    matching=sorted(n for n in names if n.endswith('_'+band+'.tif'));assert matching,(stamp,h,v,band)
                    url=BASE+'/'+matching[-1];sources[band]=url
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_TIMEOUT='60',GDAL_HTTP_MAX_RETRY='3',GDAL_HTTP_RETRY_DELAY='2'):
                        with rasterio.open(url+'?'+token) as ds:
                            rr,cc=rasterio.transform.rowcol(ds.transform,xx[ix],yy[ix]);rr=np.array(rr);cc=np.array(cc)
                            assert ((rr>=0)&(rr<ds.height)&(cc>=0)&(cc<ds.width)).all()
                            bh,bw=ds.block_shapes[0];sample=np.full(len(ix),np.nan)
                            blocks=sorted(set(zip(rr//bh,cc//bw)))
                            for br,bc in blocks:
                                mask=(rr//bh==br)&(cc//bw==bc)
                                block=ds.read(1,window=Window(bc*bw,br*bh,min(bw,ds.width-bc*bw),min(bh,ds.height-br*bh)),masked=True)
                                sample[mask]=np.ma.filled(block[rr[mask]-br*bh,cc[mask]-bc*bw].astype(float),np.nan)
                            values[band]=sample
                np.savez_compressed(target,cell_indices=ix,**values)
                meta.write_text(json.dumps({'date':str(date.date()),'tile':[h,v],'sources':sources,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
                return target.name,'downloaded'
            except Exception as exc:
                # GDAL exceptions may contain signed URLs, so do not print messages.
                print('Retry',stamp,h,v,attempt+1,type(exc).__name__,flush=True)
                if attempt==4:raise RuntimeError(f'MODIS sample failed {stamp} {h} {v}') from None
                time.sleep(3*(attempt+1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,future in enumerate(as_completed([pool.submit(one,j) for j in jobs]),1):
            name,status=future.result();print(i,'/',len(jobs),name,status,flush=True)
    arrays={band:np.full((len(dates),len(cells)),np.nan) for band in BANDS}
    for it,date in enumerate(dates):
        for h,v in tiles:
            z=np.load(OUT/f'{date.strftime("%Y%j")}_h{h:02d}v{v:02d}.npz');ix=z['cell_indices']
            for band in BANDS:arrays[band][it,ix]=z[band]
    ds=xr.Dataset({band:(['time','cell'],a) for band,a in arrays.items()},coords={'time':dates,'cell':np.arange(len(cells)),'latitude':('cell',cells[:,0]),'longitude':('cell',cells[:,1])},attrs={'source':'NASA MOD13A1.061 via Microsoft Planetary Computer COG mirror','sampling':'nearest native 500 m pixel at California grid cell centre; raw integer values','ndvi_nir_mir_scale':0.0001})
    ds.to_netcdf(OUT/'MOD13A1_2020_2024_samples.nc')
    print('Complete MODIS sampled archive',dict(ds.sizes),flush=True)

if __name__=='__main__':main()
