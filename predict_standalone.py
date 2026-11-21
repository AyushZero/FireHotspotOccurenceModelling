"""Apply the selected frozen method to prepared observations, without fitting."""
from pathlib import Path
import argparse,json,joblib
import xarray as xr
from research_models import drivers,NAMES
from evaluation_utils import mapped

ROOT=Path(__file__).resolve().parent

def predict(frame):
    frozen=json.loads((ROOT/'outputs/standalone/frozen_selection.json').read_text());key=frozen['primary']['id']
    bundle=joblib.load(ROOT/f'models/standalone/{key}.joblib');cm=joblib.load(ROOT/f'models/standalone/{key}_calibration.joblib')
    p=bundle['model'].predict_proba(bundle['transform'].transform(drivers(frame)))[:,1]
    return mapped(cm['kind'],cm['model'],p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,default=2024);args=parser.parse_args()
    with xr.open_dataset(ROOT/'data/standalone/features.nc') as ds:f=ds.sel(time=ds.time.dt.year==args.year).to_dataframe().reset_index()
    f=f.dropna(subset=NAMES);assert len(f),'No prepared observations for this year'
    out=f[['time','latitude','longitude']].copy();out['hotspot_probability']=predict(f)
    target=ROOT/f'outputs/standalone/predictions_{args.year}.csv';out.to_csv(target,index=False);print(target)

if __name__=='__main__':main()
