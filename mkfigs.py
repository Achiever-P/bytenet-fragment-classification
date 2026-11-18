import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import sys; sys.path.insert(0,'.')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
A=json.load(open('results/aggregate.json'))
IND='#3333AA';ORG='#D9731A';GR='#8A8FA3'
order=['base','imgonly','byteonly','histonly','hist','histgate']
names={'base':'Baseline\n(byte FC + image)','imgonly':'Image\nonly','byteonly':'Byte FC\nonly','histonly':'Histogram\nonly','hist':'Image +\nhistogram','histgate':'Image + hist.\n+ gate'}
cols=[GR,GR,GR,IND,IND,IND]
fig,ax=plt.subplots(figsize=(6.4,3.3),dpi=250)
m=[A[k]['mean'] for k in order];s=[A[k]['std'] for k in order]
ax.bar(range(6),m,yerr=s,color=cols,capsize=3,width=.65)
for i,k in enumerate(order):
    ax.scatter([i]*3,A[k]['acc'],color='k',s=7,zorder=3)
    ax.text(i,m[i]+s[i]+2,f'{m[i]:.1f}',ha='center',fontsize=9)
ax.set_xticks(range(6));ax.set_xticklabels([names[k] for k in order],fontsize=8)
ax.set_ylabel('Test accuracy (%)');ax.set_ylim(0,92)
for sp in('top','right'):ax.spines[sp].set_visible(False)
plt.tight_layout();plt.savefig('figs/fig_acc.png');plt.close()
# robustness
fig,ax=plt.subplots(figsize=(6.4,3.3),dpi=250)
x=[0,5,10,20,30]
for k,c,ls,lab in [('base',GR,'-','Baseline'),('imgonly',GR,'--','Image only'),('hist',IND,'-','Image + histogram'),('histgate',ORG,'-','Image + histogram + gate'),('histonly',IND,'--','Histogram only')]:
    ax.errorbar(x,A[k]['rob_mean'],yerr=A[k]['rob_std'],color=c,ls=ls,marker='o',ms=3.5,capsize=2,label=lab,lw=1.6)
ax.set_xlabel('Corrupted bytes (%)');ax.set_ylabel('Test accuracy (%)');ax.set_ylim(45,82)
ax.legend(fontsize=8,frameon=False,loc='lower left')
for sp in('top','right'):ax.spines[sp].set_visible(False)
plt.tight_layout();plt.savefig('figs/fig_rob.png');plt.close()
# former bar
fig,ax=plt.subplots(figsize=(3.6,3.3),dpi=250)
fb=A['former_base'];fh=A['former_histgate']
ax.bar([0,1],[fb['mean'],fh['mean']],yerr=[fb['std'],fh['std']],color=[GR,IND],capsize=3,width=.6)
for i,v in enumerate([fb,fh]):ax.text(i,v['mean']+v['std']+2,f"{v['mean']:.1f}",ha='center')
ax.set_xticks([0,1]);ax.set_xticklabels(['ByteFormer','ByteFormer +\nhist. + gate'],fontsize=8);ax.set_ylim(0,92);ax.set_ylabel('Test accuracy (%)')
for sp in('top','right'):ax.spines[sp].set_visible(False)
plt.tight_layout();plt.savefig('figs/fig_former.png');plt.close()
# confusion (sum over 3 seeds hist)
cn=json.load(open('results/hist_s1/result.json'))['class_names']
cm=sum(np.array(json.load(open(f'results/hist_s{i}/result.json'))['confusion_matrix']) for i in(1,2,3)).astype(float)
cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(4.2,3.7),dpi=250)
ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
for i in range(7):
    for j in range(7):ax.text(j,i,f'{cmn[i,j]*100:.0f}',ha='center',va='center',fontsize=8,color='w' if cmn[i,j]>.5 else 'k')
ax.set_xticks(range(7));ax.set_xticklabels(cn,fontsize=8);ax.set_yticks(range(7));ax.set_yticklabels(cn,fontsize=8)
ax.set_xlabel('Predicted');ax.set_ylabel('True')
plt.tight_layout();plt.savefig('figs/fig_cm.png');plt.close()
# architecture
fig,ax=plt.subplots(figsize=(7.2,3.0),dpi=250);ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,5)
def box(x,y,w,h,t,c,tc='w',fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.1',fc=c,ec='none'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',color=tc,fontsize=fs)
def arr(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,color='#444',lw=1))
box(0,1.9,1.5,1.2,'256-byte\nfragment','#E4E6F2','k')
box(2.4,3.3,2.4,1.1,'Byte2Image\n(bit shift + n-gram)','#E4E6F2','k')
box(2.4,0.6,2.4,1.1,'Statistics\n256-bin histogram\n+ entropy (257-d)','#F6D9BF','k')
box(5.5,3.3,2.0,1.1,'Image branch\n4-stage CNN','#6B6BC8')
box(5.5,0.6,2.0,1.1,'Statistics MLP\n(position-invariant)','#D9731A')
box(8.3,1.7,1.7,1.6,'Gated fusion\nsigmoid gate\nover both','#3333AA')
box(10.5,2.0,1.4,1.0,'Softmax\n7 classes','#E4E6F2','k')
arr(1.5,2.7,2.4,3.8);arr(1.5,2.3,2.4,1.2);arr(4.8,3.85,5.5,3.85);arr(4.8,1.15,5.5,1.15)
arr(7.5,3.85,8.3,2.9);arr(7.5,1.15,8.3,2.1);arr(10,2.5,10.5,2.5)
ax.text(6,0.1,'The position-dependent byte FC branch of the paper is replaced by the statistics branch',ha='center',fontsize=8,style='italic',color='#555')
plt.tight_layout();plt.savefig('figs/fig_arch.png');plt.close()
# byte2image example
from bytenet.byte2image import byte2image
d=np.load('data/fragments.npz',allow_pickle=True);X,y=d['X'],d['y']
fig,axs=plt.subplots(1,4,figsize=(7,2.2),dpi=250)
for ax,c in zip(axs,[0,3,5,6]):
    i=int(np.where(y==c)[0][0]);im=np.array(byte2image(X[i],4),dtype=float)
    ax.imshow(im,cmap='gray',aspect='auto');ax.set_title(str(d['class_names'][c]),fontsize=9);ax.axis('off')
plt.tight_layout();plt.savefig('figs/fig_b2i.png');plt.close()
