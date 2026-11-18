import json, numpy as np
cfgs = [("ByteResNet baseline (byte FC + image)","base"),("Image branch only","imgonly"),("Byte FC branch only","byteonly"),
        ("Histogram branch only (proposed)","histonly"),("ByteResNet + histogram branch","hist"),("ByteResNet + histogram + gated fusion","histgate"),
        ("ByteFormer baseline","former_base"),("ByteFormer + histogram + gated fusion","former_histgate")]
rates = ["0.0","0.05","0.1","0.2","0.3"]; agg = {}
for lab, k in cfgs:
    R = [json.load(open(f"results/{k}_s{s}/result.json")) for s in (1,2,3)]
    acc = np.array([r["final_test_acc"] for r in R])*100
    rob = np.array([[r["robustness"][x] for x in rates] for r in R])*100
    agg[k] = dict(label=lab, acc=acc.tolist(), mean=acc.mean(), std=acc.std(ddof=1), rob_mean=rob.mean(0).tolist(), rob_std=rob.std(0, ddof=1).tolist(), params=R[0]["n_params"])
    print(f"{lab:42s} {acc.mean():5.1f} ± {acc.std(ddof=1):4.1f}  seeds={np.round(acc,1)}  params={R[0]['n_params']:,}  rob={np.round(rob.mean(0),1)}")
json.dump(agg, open("results/aggregate.json","w"), indent=1)
b = np.array(agg["base"]["acc"]); 
for k in ("hist","histgate","histonly"):
    d = np.array(agg[k]["acc"]) - b; print(k, "paired diff vs base per seed:", np.round(d,1), "mean", round(d.mean(),1))
print("test n =", R[0]["n_test"], " binomial SE at 75% ~", round(100*np.sqrt(.75*.25/R[0]["n_test"]),1))
