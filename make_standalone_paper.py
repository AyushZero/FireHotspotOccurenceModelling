"""Render standalone proposed-method paper from frozen 2024 test results."""
from pathlib import Path
import json,joblib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve,precision_recall_curve
from jinja2 import Environment,FileSystemLoader
from build_standalone_data import load_frames
from research_models import drivers

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/standalone'
LABELS={'linear':'Regularised linear logit','smooth_additive':'Smooth additive logit','joint_index':'Joint multiplicative index','rf':'Random Forest','xgb':'XGBoost','lgbm':'LightGBM','ensemble':'Convex ensemble','persistence':'Hotspot persistence','climatology':'Cell and season climatology','weather':'Weather-only logit','constant':'Constant training rate'}

def main():
    d=json.loads((OUT/'results.json').read_text());f=json.loads((OUT/'frozen_selection.json').read_text());audit=json.loads((ROOT/'outputs/stage3/data_audit.json').read_text())
    key=d['primary'];r=d['models'][key];m=r['test'];z=np.load(OUT/'test_predictions.npz');y=z['label'];p=z[key]
    board=json.loads((OUT/'leaderboard.json').read_text())
    smooth_zero=max(a['selection']['ap'] for a in board if a['spec']['family']=='smooth_additive' and a['spec']['alpha']==0)
    smooth_positive=max(a['selection']['ap'] for a in board if a['spec']['family']=='smooth_additive' and a['spec']['alpha']>0)
    raw_joint=[a for a in board if a['spec']['family']=='joint_index' and not a['spec']['parameters'].get('history_log',False)]
    log_joint=[a for a in board if a['spec']['family']=='joint_index' and a['spec']['parameters'].get('history_log',False)]
    paired_delta=np.array([b['selection']['ap']-a['selection']['ap'] for a,b in zip(raw_joint,log_joint)])
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9})
    chosen=list(dict.fromkeys([key,f['family_winners']['smooth_additive']['id'],f['family_winners']['xgb']['id'],f['family_winners']['rf']['id'],'baseline_persistence','baseline_climatology']))
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.4),layout='constrained')
    for k in chosen:
        a=d['models'][k]['test'];name=LABELS[d['models'][k]['spec']['family']];fp,tp,_=roc_curve(y,z[k]);pr,rec,_=precision_recall_curve(y,z[k])
        axes[0].plot(fp,tp,label=f"{name}: {a['auc']:.3f}");axes[1].plot(rec,pr,label=f"{name}: {a['ap']:.3f}")
    axes[0].plot([0,1],[0,1],':',color='gray');axes[1].axhline(y.mean(),ls=':',color='gray')
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out 2024 ROC');axes[1].set(xlabel='Recall',ylabel='Precision',title='Held-out 2024 precision–recall')
    fig.savefig(OUT/'discrimination.png',dpi=300,bbox_inches='tight');plt.close(fig)
    bins=np.array_split(np.argsort(p,kind='stable'),10);bx=[float(p[ix].mean()) for ix in bins];by=[float(y[ix].mean()) for ix in bins]
    fig,ax=plt.subplots(figsize=(4.3,3.4),layout='constrained');ax.plot(bx,by,'o-');limit=max(max(bx),max(by))*1.1;ax.plot([0,limit],[0,limit],':',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed hotspot frequency');ax.grid(alpha=.2);fig.savefig(OUT/'reliability.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Conditional smooth terms are model effects, not interventions or causal effects.
    train,_,_=load_frames();x=drivers(train);median=np.median(x,axis=0);bundle=joblib.load(ROOT/f"models/standalone/{f['family_winners']['smooth_additive']['id']}.joblib")
    tf=bundle['transform'];model=bundle['model'];terms=[]
    for j in range(9):
        lo,hi=np.quantile(x[:,j],[.01,.99]);grid=np.linspace(lo,hi,100);a=np.tile(median,(100,1));a[:,j]=grid
        effect=tf.transform(a)@model.coef_;effect-=(tf.transform(median[None,:])@model.coef_).item()
        terms.append((float(np.ptp(effect)),j,grid,effect))
    names=['Temperature','NDVI','Canopy dryness','Atmospheric dryness','Soil dryness','Wind','Slope','FRP history','Count history']
    fig,axes=plt.subplots(1,3,figsize=(8.8,2.6),layout='constrained')
    for ax,(_,j,grid,effect) in zip(axes,sorted(terms,reverse=True)[:3]):
        tiny=0<np.max(np.abs(grid))<.01
        ax.plot(grid*1000 if tiny else grid,effect);ax.axhline(0,color='gray',ls=':');ax.set(xlabel=names[j]+(r' ($10^{-3}$ scaled)' if tiny else ' (scaled)'),ylabel='Centred log-odds effect');ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.savefig(OUT/'smooth_effects.png',dpi=300,bbox_inches='tight');plt.close(fig)
    rows=[]
    for k,a in d['models'].items():
        name=LABELS[a['spec']['family']];metric=a['test'];name=r'\textbf{'+name+'}' if k==key else name
        selection=f"{a['selection']['ap']:.4f}" if 'selection' in a else '---';capture=f"{100*metric['recall_top10']:.1f}" if k!='constant' else '---'
        rows.append(f"{name} & {selection} & {metric['auc']:.3f} & {metric['ap']:.4f} & {metric['brier']:.5f} & {capture} "+r'\\')
    annual=[]
    for year,a in audit['annual'].items():annual.append(f"{year} & {a['complete_rows']:,} & {a['complete_positives']:,} & {100*a['complete_positives']/a['complete_rows']:.2f} "+r'\\')
    intervals=[]
    compare=[f['family_winners'][family]['id'] for family in ['smooth_additive','rf','xgb','lgbm']]+['baseline_persistence']
    for length,values in d['bootstrap']['blocks'].items():
        for k in compare:
            if k==key:continue
            v=values[k];name=LABELS[d['models'][k]['spec']['family']]
            intervals.append(f"{length} & {name} & {v['auc_difference']:+.5f} & [{v['auc_ci'][0]:+.5f}, {v['auc_ci'][1]:+.5f}] & {v['ap_difference']:+.5f} & [{v['ap_ci'][0]:+.5f}, {v['ap_ci'][1]:+.5f}] "+r'\\')
    smooth=f['family_winners']['smooth_additive'];joint=f['family_winners']['joint_index'];ensemble=f['family_winners']['ensemble']
    w=ensemble['spec']['weights'];components=ensemble['spec']['members'];weight_text=', '.join(f"{LABELS[d['models'][c]['spec']['family']]} {weight:.3f}" for c,weight in zip(components,w))
    top=np.argsort(-p,kind='stable')[:int(np.ceil(.1*len(y)))];q=r['calibration']['kind'];best=max((k for k in d['models'] if k!='constant'),key=lambda k:d['models'][k]['test']['ap'])
    ctx={'name':LABELS[r['spec']['family']],'auc':f"{m['auc']:.3f}",'ap':f"{m['ap']:.4f}",'brier':f"{m['brier']:.5f}",'capture':f"{100*m['recall_top10']:.1f}",'precision':f"{100*m['precision_top10']:.2f}",'base_rate':f"{d['base_rate']:.4f}",'n':f"{d['n']:,}",'events':f"{d['events']:,}",'bss':f"{d['brier_skill']:.3f}",'mean':f"{m['mean']:.4f}",'ratio':f"{m['mean']/d['base_rate']:.2f}",'calibration':q,'rows':'\n'.join(rows),'annual':'\n'.join(annual),'intervals':'\n'.join(intervals),'weights':weight_text,'knots':smooth['spec']['knots'],'ridgeC':smooth['spec']['C'],'alpha':smooth['spec']['alpha'],'joint_parameters':json.dumps(joint['spec']['parameters'],sort_keys=True),'n_selected':f'{len(top):,}','captured_events':f'{int(y[top].sum()):,}','n_candidates':f['n_candidates'],'n_blends':f['n_blends'],'selection_ap':f"{r['selection']['ap']:.4f}",'smooth_auc':f"{d['models'][smooth['id']]['test']['auc']:.3f}",'smooth_ap':f"{d['models'][smooth['id']]['test']['ap']:.4f}",'joint_auc':f"{d['models'][joint['id']]['test']['auc']:.3f}",'joint_ap':f"{d['models'][joint['id']]['test']['ap']:.4f}",'best_test_name':LABELS[d['models'][best]['spec']['family']],'best_test_ap':f"{d['models'][best]['test']['ap']:.4f}",'max_bin_gap':f'{max(abs(a-b) for a,b in zip(bx,by)):.4f}'}
    ctx.update(smooth_zero_ap=f'{smooth_zero:.4f}',smooth_positive_ap=f'{smooth_positive:.4f}',history_wins=int((paired_delta>0).sum()),history_median_delta=f'{np.median(paired_delta):+.4f}',raw_index_ap=f"{max(a['selection']['ap'] for a in raw_joint):.4f}",log_index_ap=f"{max(a['selection']['ap'] for a in log_joint):.4f}",raw_brier=f"{r['raw_test']['brier']:.6f}",mapped_brier=f"{m['brier']:.6f}")
    if r['spec']['family']=='ensemble':ctx['name']='convex ensemble'
    if d['models'][best]['spec']['family']=='ensemble':ctx['best_test_name']='the convex ensemble'
    env=Environment(loader=FileSystemLoader(str(ROOT)),variable_start_string='<<',variable_end_string='>>');tex=env.get_template('standalone_paper_template.tex.j2').render(**ctx)
    assert '<<' not in tex
    # Number references by first citation, including references within grouped citations.
    keys=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',tex):
        for k in group.split(','):
            if k not in keys:keys.append(k)
    entries={m.group(1):m.group(0) for m in re.finditer(r'\\bibitem\{([^}]+)\}[^\n]+',tex)}
    first=tex.index(r'\bibitem');last=tex.index(r'\end{thebibliography}')
    tex=tex[:first]+'\n'.join(entries[k] for k in keys)+'\n'+tex[last:]
    (ROOT/'final_paper.tex').write_text(tex,encoding='utf-8');(OUT/'paper_numbers.json').write_text(json.dumps(ctx,indent=2));print('Standalone manuscript generated',flush=True)

if __name__=='__main__':main()
