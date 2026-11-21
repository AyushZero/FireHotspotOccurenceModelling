"""Apply the frozen selected model to prepared rows; no refitting."""
from pathlib import Path
import argparse,json,joblib
from src.dataset import build_dataset
from research_models import drivers
from evaluate_research import mapped

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--year',type=int,default=2023); args=parser.parse_args()
    root=Path(__file__).resolve().parent
    key=json.loads((root/'outputs/research/frozen_selection.json').read_text())['primary']['id']
    bundle=joblib.load(root/f'models/research/{key}.joblib')
    cm=joblib.load(root/f'models/research/{key}_calibration.joblib')
    frame=build_dataset(root/'data/features');frame=frame[frame.year==args.year].copy()
    if frame.empty:raise ValueError('No prepared observations for requested year')
    raw=bundle['model'].predict_proba(bundle['transform'].transform(drivers(frame)))[:,1]
    out=frame[['time','latitude','longitude']].copy();out['hotspot_probability']=mapped(cm['kind'],cm['model'],raw)
    target=root/f'outputs/research/predictions_{args.year}.csv';out.to_csv(target,index=False);print(target)

if __name__=='__main__':main()
