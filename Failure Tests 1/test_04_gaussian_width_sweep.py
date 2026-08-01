"""Test 4: detect flux-transport behaviour using a Gaussian-width sweep.

For a correct surface-brightness SIS sampler, changing source sigma changes ring
thickness but not the primary radius. A peak following r_E + sigma indicates that
integrated annular flux is being transported instead of brightness being sampled.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

FILES=["scatter_to_log_128.pt","forward_from_log_128.pt","scatter_from_log_128.pt"]

def apply(x,m):
    b,c,h,w=x.shape; y=torch.sparse.mm(m,x.reshape(b*c,h*w).T); return y.T.reshape(b,c,h,w)

def primary_peak(x):
    x=x[0,0]; n=x.shape[-1]; yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing="ij")
    r=torch.sqrt((xx-(n-1)/2)**2+(yy-(n-1)/2)**2).long(); s=torch.zeros(int(r.max())+1); c=torch.zeros_like(s)
    s.scatter_add_(0,r.flatten(),x.flatten()); c.scatter_add_(0,r.flatten(),torch.ones_like(x).flatten()); p=s/c.clamp_min(1)
    ids=[i for i in range(1,len(p)-1) if p[i]>p[i-1] and p[i]>=p[i+1]]
    return max(ids,key=lambda i:float(p[i])) if ids else int(torch.argmax(p))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mapping-dir",required=True); p.add_argument("--resolution",type=float,required=True); p.add_argument("--theta-e",type=float,required=True); p.add_argument("--sigmas",type=float,nargs="+",default=[1,2,4,6]); p.add_argument("--output",default="gaussian_width_report.json")
    a=p.parse_args(); mats=[torch.load(Path(a.mapping_dir)/f,map_location="cpu").coalesce() for f in FILES]
    n=128; yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing="ij"); rr=torch.sqrt((xx-(n-1)/2)**2+(yy-(n-1)/2)**2)
    rows=[]
    for sigma in a.sigmas:
        x=torch.exp(-0.5*(rr/sigma)**2)[None,None]
        for m in mats:x=apply(x,m)
        rows.append({"sigma":sigma,"sparse_peak":primary_peak(x),"expected_surface_brightness_peak":a.theta_e/(a.resolution/2),"flux_transport_prediction":a.theta_e/(a.resolution/2)+sigma})
    peaks=[r["sparse_peak"] for r in rows]; drift=max(peaks)-min(peaks)
    report={"results":rows,"peak_drift_pixels":drift,"passed":drift<=1}
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__":main()
