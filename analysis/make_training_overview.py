"""Six-panel overview of the training histories of all experiments (thesis Fig. 4.5): python analysis/make_training_overview.py"""
import glob,os,pandas as pd,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pathlib
O=str(pathlib.Path(__file__).resolve().parents[1]/'outputs')   # experiment outputs of the repository
def h(name):
    f=glob.glob(f'{O}/experiments/{name}*/history.csv')+glob.glob(f'{O}/archive/experiments/{name}*/history.csv')
    return pd.read_csv(f[0])
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.3})
B,R='#2a78d6','#e8622c'
fig,ax=plt.subplots(2,3,figsize=(11,6.4))
a=ax[0,0]
d=pd.read_csv(f'{O}/archive/experiments/exp000_baseline_cnn_vit/cnn/history.csv'); a.plot(d.epoch,d.val_mae_N,color=B,lw=1.6,label='CNN exp000')
d=pd.read_csv(f'{O}/archive/experiments/exp000_baseline_cnn_vit/vit/history.csv'); a.plot(d.epoch,d.val_mae_N,color=R,lw=1.6,label='ViT exp000')
for e,ls in [('exp001','--'),('exp002',':'),('exp003','-.')]:
    d=h(e); a.plot(d.epoch,d.val_mae_N,ls,color='0.35',lw=1.1,label=f'CNN {e}')
a.set(title='(a) P1 strict: validation MAE (hash)',xlabel='Epoch',ylabel='Validation MAE (N)',yscale='log'); a.legend(fontsize=7)
a=ax[0,1]
for e,c,l in [('exp004',B,'CNN exp004'),('exp012',R,'ViT exp012')]:
    d=h(e); a.plot(d.epoch,d.val_mae_N,color=c,lw=1.6,label=l)
a.set(title='(b) P2 seen indenters: validation MAE',xlabel='Epoch',ylabel='Validation MAE (N)',yscale='log'); a.legend(fontsize=7)
a=ax[0,2]
grey=['exp005','exp006','exp008','exp009','exp010','exp013','exp014','exp016','exp017','exp018','exp020']
for e in grey:
    d=h(e); a.plot(d.epoch,d.val_mae_N,color='0.75',lw=.8)
for e,c,l in [('exp007','#7fb2ea','CNN exp007'),('exp021',B,'CNN exp021 (final)'),('exp022','#123f7a','CNN exp022 (heart)'),('exp011','#f2a37f','ViT exp011'),('exp023',R,'ViT exp023')]:
    d=h(e); a.plot(d.epoch,d.val_mae_N,color=c,lw=1.5,label=l)
a.plot([],[],color='0.75',lw=.8,label='other P3 variants')
a.set(title='(c) P3 calibration: validation MAE',xlabel='Epoch',ylabel='Validation MAE (N)',yscale='log'); a.legend(fontsize=6.5,ncol=1)
a=ax[1,0]
for e,c,l in [('exp004',B,'CNN exp004'),('exp007','#7fb2ea','CNN exp007'),('exp021','#123f7a','CNN exp021'),('exp012',R,'ViT exp012'),('exp023','#f2a37f','ViT exp023')]:
    d=h(e); a.plot(d.epoch,d.train_loss,color=c,lw=1.4,label=l)
a.set(title='(d) Training loss (Huber, scaled units)',xlabel='Epoch',ylabel='Training loss',yscale='log'); a.legend(fontsize=7)
a=ax[1,1]
d=h('exp024'); a.plot(d.epoch,d.train_loss,color=B,lw=1.6,label='training loss'); a.set(xlabel='Epoch',ylabel='Cross-entropy loss',title='(e) Shape classifier exp024')
a2=a.twinx(); a2.plot(d.epoch,100*d.val_acc,color=R,lw=1.6,label='validation accuracy'); a2.set_ylabel('Validation accuracy (%)'); a2.grid(False); a2.spines['top'].set_visible(False)
l1,n1=a.get_legend_handles_labels(); l2,n2=a2.get_legend_handles_labels(); a.legend(l1+l2,n1+n2,fontsize=7,loc='lower right')
a=ax[1,2]
rows=[]
for f in sorted(glob.glob(f'{O}/archive/experiments/exp0*/history.csv')+glob.glob(f'{O}/experiments/exp0*/history.csv')+glob.glob(f'{O}/archive/experiments/exp000*/*/history.csv')):
    d=pd.read_csv(f); n=f.split('/')[-2] if 'exp000' not in f else 'exp000_'+f.split('/')[-2]
    if 'sec' in d: rows.append((n[:6] if 'exp000' not in n else 'exp000'+('v' if 'vit' in n else 'c'), d.sec.median()/60, 'vit' in n))
rows=sorted([r for r in rows if r[0]!='exp024'])
a.bar(range(len(rows)),[r[1] for r in rows],color=[R if r[2] else B for r in rows])
a.set_xticks(range(len(rows))); a.set_xticklabels([r[0].replace('exp','') for r in rows],rotation=90,fontsize=6.5)
a.set(title='(f) Time per epoch',ylabel='Minutes',xlabel='Experiment')
fig.tight_layout(); os.makedirs(O+'/figures',exist_ok=True); fig.savefig(O+'/figures/training_all.png',dpi=200)
print(rows)
