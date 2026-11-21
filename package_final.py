"""Assemble one clean, self-contained final handoff; preserve research folders."""
from pathlib import Path
import shutil,json,hashlib

ROOT=Path(__file__).resolve().parent;DEST=ROOT.parent/'Wildfire_Final_Paper_2024'

def digest(path):
    checksum=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):checksum.update(chunk)
    return checksum.hexdigest()

def main():
    assert (ROOT/'final_paper.pdf').exists(),'Compile and inspect the final paper first'
    assert not DEST.exists(),'Inspect existing handoff before replacing it'
    DEST.mkdir()
    shutil.copy2(ROOT.parent/'LICENSE',DEST/'LICENSE')
    scripts=['build_standalone_data.py','standalone_models.py','train_standalone.py','evaluate_standalone.py','make_standalone_paper.py','predict_standalone.py','verify_standalone.py','research_models.py','evaluation_utils.py','fetch_modis_public.py','fetch_era5_earthmover.py','fetch_firms_archive.py','build_srtm_slope.py','check_tex_static.py','run_standalone_pipeline.ps1','standalone_paper_template.tex.j2','final_paper.tex','final_paper.pdf','requirements.txt','.gitattributes','.gitignore','config_standalone.yaml']
    for name in scripts:shutil.copy2(ROOT/name,DEST/name)
    shutil.copy2(ROOT/'export_formulas.py',DEST/'export_formulas.py')
    shutil.copy2(ROOT/'config_standalone.yaml',DEST/'config.yaml')
    (DEST/'src').mkdir()
    for name in ['__init__.py','grid.py','preprocess.py']:shutil.copy2(ROOT/'src'/name,DEST/'src'/name)
    for name in ['era5','firms','srtm','modis_public']:
        shutil.copytree(ROOT/'data/raw'/name,DEST/'data/raw'/name,ignore=shutil.ignore_patterns('chunks','*.download'))
    shutil.copytree(ROOT/'data/standalone',DEST/'data/standalone')
    (DEST/'outputs/stage3').mkdir(parents=True)
    for name in ['data_audit.json','mathematical_checks.json']:shutil.copy2(ROOT/'outputs/stage3'/name,DEST/'outputs/stage3'/name)
    shutil.copy2(ROOT/'outputs/cb_2024_us_state_500k.zip',DEST/'outputs/cb_2024_us_state_500k.zip')
    shutil.copytree(ROOT/'outputs/standalone',DEST/'outputs/standalone',ignore=shutil.ignore_patterns('*.done'))
    frozen=json.loads((ROOT/'outputs/standalone/frozen_selection.json').read_text());needed={r['id'] for r in frozen['family_winners'].values()}
    needed.update(frozen['family_winners']['ensemble']['spec']['members']);needed.update(['baseline_persistence','baseline_climatology','baseline_weather'])
    (DEST/'models/standalone').mkdir(parents=True)
    for key in needed:
        for suffix in ['.joblib','_calibration.joblib']:
            source=ROOT/'models/standalone'/f'{key}{suffix}'
            if source.exists():shutil.copy2(source,DEST/'models/standalone'/source.name)
    (DEST/'logs').mkdir()
    for source in (ROOT/'logs').glob('*.log'):
        if source.name in ['modis_public.log','era5_2024.log','firms_2024.log','build_standalone_data.py.log','train_standalone.py.log','spline_search.log','index_tree_search.log','compressed_index_search.log','evaluate_standalone.py.log','make_standalone_paper.py.log','verify_standalone.log','paper_compile.log']:shutil.copy2(source,DEST/'logs'/source.name)
    shutil.copy2(ROOT/'FINAL_README.md',DEST/'README.md')
    summary={'paper':'final_paper.pdf','training':[2021,2022],'validation':2023,'test':2024,'source_working_directory':'Wildfire_Model_Research_v3','contents':'Observed data, prepared features, complete candidate records, family-winner models and ensemble, predictions, figures, logs and reproduction code','discarded_from_handoff':'Earlier papers, obsolete feature sets and models, unused candidate model binaries, transient download chunks and caches'}
    (DEST/'HANDOFF.json').write_text(json.dumps(summary,indent=2))
    hashes={str(p.relative_to(DEST)).replace('\\','/'):digest(p) for p in DEST.rglob('*') if p.is_file()}
    (DEST/'MANIFEST.json').write_text(json.dumps(hashes,indent=2));print(DEST,flush=True)

if __name__=='__main__':main()
