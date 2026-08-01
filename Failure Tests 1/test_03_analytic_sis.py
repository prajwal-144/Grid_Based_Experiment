"""Test 3: compare the sparse SIS forward chain with direct inverse ray shooting.

A centred Gaussian should produce one annulus near theta_E. The sparse result must
match the analytic result in radius, topology, and relative MSE.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

FILES = ["scatter_to_log_128.pt", "forward_from_log_128.pt", "scatter_from_log_128.pt"]

def apply(x, m):
    b,c,h,w=x.shape; y=torch.sparse.mm(m, x.reshape(b*c,h*w).T)
    return y.T.reshape(b,c,h,w)

def radial(x):
    x=x[0,0].cpu(); n=x.shape[-1]; yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing="ij")
    r=torch.sqrt((xx-(n-1)/2)**2+(yy-(n-1)/2)**2).long(); s=torch.zeros(int(r.max())+1); c=torch.zeros_like(s)
    s.scatter_add_(0,r.flatten(),x.flatten()); c.scatter_add_(0,r.flatten(),torch.ones_like(x).flatten()); return s/c.clamp_min(1)

def peaks(p):
    ids=[i for i in range(1,len(p)-1) if p[i]>p[i-1] and p[i]>=p[i+1]]
    return sorted(ids,key=lambda i:float(p[i]),reverse=True)[:5]

def analytic(source, theta_e, pix):
    n=source.shape[-1]; coord=(torch.arange(n)-(n-1)/2)*pix; yy,xx=torch.meshgrid(coord,coord,indexing="ij")
    rr=torch.sqrt(xx**2+yy**2).clamp_min(1e-8); bx=xx-theta_e*xx/rr; by=yy-theta_e*yy/rr
    gx=2*bx/((n-1)*pix); gy=2*by/((n-1)*pix); grid=torch.stack([gx,gy],-1)[None]
    return F.grid_sample(source,grid,mode="bilinear",padding_mode="zeros",align_corners=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mapping-dir",required=True); p.add_argument("--resolution",type=float,required=True); p.add_argument("--theta-e",type=float,required=True); p.add_argument("--sigma",type=float,default=4); p.add_argument("--output",default="analytic_sis_report.json")
    a=p.parse_args(); root=Path(a.mapping_dir); mats=[torch.load(root/f,map_location="cpu").coalesce() for f in FILES]
    n=128; yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing="ij"); rr=torch.sqrt((xx-(n-1)/2)**2+(yy-(n-1)/2)**2)
    src=torch.exp(-0.5*(rr/a.sigma)**2)[None,None]
    sparse=src
    for m in mats: sparse=apply(sparse,m)
    ref=analytic(src,a.theta_e,a.resolution/2)
    ps,pr=radial(sparse),radial(ref); mse=float(((sparse-ref)**2).mean()/ref.square().mean().clamp_min(1e-8))
    report={"expected_radius_pixels":a.theta_e/(a.resolution/2),"sparse_peaks":peaks(ps),"analytic_peaks":peaks(pr),"relative_mse":mse,"passed":abs(peaks(ps)[0]-peaks(pr)[0])<=1 and mse<0.05}
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
