"""Build figures and manuscript exclusively from frozen search/evaluation outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from jinja2 import Environment,FileSystemLoader

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs/research'

def label(spec):
    if spec['family']!='logistic':return {'rf':'Random Forest','xgb':'XGBoost','lgbm':'LightGBM'}[spec['family']]
    return {'minmax':'Min--max logistic','standard':'Standardized logistic','loghistory':'Log-history logistic','spline':'Spline additive','interactions':'Interaction logistic','multiplicative':'Multiplicative'}[spec['kind']]+(' constrained' if spec['positive'] else '')

def main():
    d=json.loads((OUT/'research_results.json').read_text()); f=json.loads((OUT/'frozen_selection.json').read_text()); board=json.loads((OUT/'leaderboard.json').read_text())
    primary=d['primary']; main=d['models'][primary]; m=main['calibrated_benchmark']; spec=main['spec']; primary_label=label(spec)
    z=np.load(OUT/'benchmark_predictions.npz'); y=z['label']; p=z[primary]
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9})
    selected=[primary,'baseline_random_forest','baseline_raw_logistic','baseline_unconstrained_additive','baseline_persistence']
    labels={primary:'Selected v2','baseline_random_forest':'Original RF','baseline_raw_logistic':'Original standardized LR','baseline_unconstrained_additive':'Original additive','baseline_persistence':'Persistence'}
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.8),layout='constrained')
    for key in selected:
        a=d['models'][key]['calibrated_benchmark']; fp,tp,_=roc_curve(y,z[key]); pr,re,_=precision_recall_curve(y,z[key])
        axes[0].plot(fp,tp,label=f"{labels[key]}: {a['auc']:.3f}")
        axes[1].plot(re,pr,label=f"{labels[key]}: {a['ap']:.3f}")
    axes[0].plot([0,1],[0,1],':',color='gray');axes[1].axhline(y.mean(),ls=':',color='gray')
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend()
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='2023 retrospective ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='2023 retrospective precision–recall')
    fig.savefig(OUT/'research_discrimination.png',dpi=300,bbox_inches='tight');plt.close(fig)
    bins=np.array_split(np.argsort(p,kind='stable'),10); bx=[p[ix].mean() for ix in bins]; by=[y[ix].mean() for ix in bins]
    fig,ax=plt.subplots(figsize=(5,3.8),layout='constrained');ax.plot(bx,by,'o-');limit=max(max(bx),max(by))*1.1;ax.plot([0,limit],[0,limit],':',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed hotspot frequency',title='Selected v2: equal-count bins');ax.grid(alpha=.2);fig.savefig(OUT/'research_reliability.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.2,3.7),layout='constrained')
    for kind,color in [('minmax','tab:blue'),('standard','tab:orange'),('loghistory','tab:green')]:
        for positive,style in [(False,'-'),(True,'--')]:
            rr=sorted([r for r in board if r['spec']['family']=='logistic' and r['spec']['kind']==kind and r['spec']['positive']==positive],key=lambda r:r['spec']['C'])
            ax.plot([r['spec']['C'] for r in rr],[r['selection']['ap'] for r in rr],style,marker='o',color=color,label=kind+(' constrained' if positive else ' free'))
    ax.set(xscale='log',xlabel='Inverse penalty strength C',ylabel='2022 selection average precision');ax.legend(ncol=2,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02));ax.grid(alpha=.2);fig.savefig(OUT/'research_constraints.png',dpi=300,bbox_inches='tight');plt.close(fig)
    rows=[]
    for key,r in sorted(d['models'].items(),key=lambda kv:kv[1]['calibrated_benchmark']['ap'],reverse=True):
        a=r['calibrated_benchmark']; name=label(r['spec']) if 'spec' in r else labels.get(key,key.replace('baseline_','Original ').replace('_',' '))
        if key==primary:name=r'\textbf{'+name+'}'
        val=f"{r['selection']['ap']:.4f}" if 'selection' in r else '---'
        rows.append(f"{name} & {val} & {a['auc']:.3f} & {a['ap']:.4f} & {a['brier']:.5f} & {100*a['recall_top10']:.1f} \\")
    # LaTeX rows need two literal terminal slashes.
    rows=[row+'\\' for row in rows]
    cirows=[]
    for length,comparisons in d['paired_full_population_bootstrap']['block_lengths'].items():
        for key,v in comparisons.items():
            cirows.append(f"{length} & {labels[key]} & {v['difference']:+.3f} & [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] & {v['ap_difference']:+.3f} & [{v['ap_ci95'][0]:+.3f}, {v['ap_ci95'][1]:+.3f}] "+r'\\')
    winner=json.dumps(spec,sort_keys=True).replace('_',r'\_').replace('{',r'\{').replace('}',r'\}')
    old=json.loads((ROOT/'outputs/review_followup.json').read_text())
    const=float(np.mean((y-old['train_base_rate'])**2)); bss=1-m['brier']/const
    counts=np.argsort(-p,kind='stable')[:int(np.ceil(.1*len(y)))]; hits=int(y[counts].sum())
    ctx={'primary_name':primary_label,'auc':f"{m['auc']:.3f}",'ap':f"{m['ap']:.4f}",'brier':f"{m['brier']:.5f}",'recall':f"{100*m['recall_top10']:.1f}",'precision':f"{100*m['precision_top10']:.2f}",'mean':f"{m['mean']:.4f}",'bss':f'{bss:.3f}','hits':f'{hits:,}','n_selected':f'{len(counts):,}','rows':'\n'.join(rows),'cirows':'\n'.join(cirows),'spec':winner,'calibration':main['calibration']['kind'],'n_candidates':len(board),'selection_n':f"{f['selection_n']:,}",'calibration_n':f"{f['calibration_n']:,}",'selection_ap':f"{main['selection']['ap']:.4f}"}
    env=Environment(loader=FileSystemLoader(str(ROOT)),variable_start_string='<<',variable_end_string='>>')
    tex=env.get_template('research_paper_template.tex.j2').render(**ctx)
    (ROOT/'research_paper.tex').write_text(tex,encoding='utf-8')
    (OUT/'paper_numbers.json').write_text(json.dumps(ctx,indent=2))
    print('Generated research figures and manuscript',flush=True)

if __name__=='__main__':main()
