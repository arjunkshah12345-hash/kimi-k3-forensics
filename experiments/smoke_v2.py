import time, torch
import run_v2 as R
R.TC["num_experts"]=4
U=torch.randn(R.V,7168).half()*0.03
R.TIME_BUDGET=time.time()-R.T0+45
R.v2_writers(U)
print(R.RESULTS["V2_writers"]["per_layer"][:2])
print(R.RESULTS["V2_writers"]["examples"][:1])
