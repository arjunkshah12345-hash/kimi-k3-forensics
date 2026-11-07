import sys, torch, numpy as np
sys.argv=['x']
import run_v1 as R
R.e3_kda()
print(R.RESULTS['E3']['A_log_lengths'], R.RESULTS['E3']['per_layer'][:3])
print(R.RESULTS['E3']['A_log_shape_note'][:2])
# embedding rows via range
sh,s,e,dt,shape=R.meta('language_model.model.embed_tokens.weight')
row=7168*2; ids=range(2000,2512)
E=R.to_tensor(R.get_range(sh,s+2000*row,s+2512*row-1),dt,[512,7168]).float()
L=1; pfx=f"{R.LM}layers.{L}.block_sparse_moe."
lnw=R.fetch(f"{R.LM}layers.{L}.post_attention_layernorm.weight").float()
Wd=R.fetch(pfx+"routed_expert_down_proj.weight").float()
z=R.rmsnorm(E,lnw)@Wd.T
zr=torch.randn(z.shape)*z.std(0,keepdim=True)
for ex in range(3):
  t={p:R.fetch(f"{pfx}experts.{ex}.{p}") for p in ["w1.weight_packed","w1.weight_scale","w3.weight_packed","w3.weight_scale","w2.weight_packed","w2.weight_scale"]}
  for lf in (True,False):
    w1=R.dequant(t["w1.weight_packed"],t["w1.weight_scale"],lf)
    w3=R.dequant(t["w3.weight_packed"],t["w3.weight_scale"],lf)
    w2=R.dequant(t["w2.weight_packed"],t["w2.weight_scale"],lf)
    y=R.situ(z@w1.T,z@w3.T)@w2.T
    print(ex,lf,'w1std',w1.std().item(),'ratio',((z@w1.T).std()/(zr@w1.T).std()).item(),'ynorm',y.norm(dim=1).mean().item(),'z norm',z.norm(dim=1).mean().item())
print('GB',R.BYTES[0]/1e9)
