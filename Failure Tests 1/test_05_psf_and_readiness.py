"""Test 5: validate FITS-PSF sampling and combine prior reports into a readiness decision.

The native FITS sampling and LR image sampling are different quantities. This test
checks angular resampling, central/edge flux, kernel support, then reads tests 1-4.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from psf import build_psf_kernel, apply_psf


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--psf-path",required=True); p.add_argument("--resolution",type=float,required=True)
    p.add_argument("--psf-source-pixscale-arcsec",type=float,required=True)
    p.add_argument("--hr-shape",type=int,default=128); p.add_argument("--reports-dir",default=".")
    p.add_argument("--output",default="readiness_report.json")
    a=p.parse_args(); hr_scale=a.resolution/2
    psf=build_psf_kernel("fits",0.16,hr_scale,path=a.psf_path,source_pixscale_arcsec=a.psf_source_pixscale_arcsec,device="cpu")
    center=torch.zeros(1,1,a.hr_shape,a.hr_shape); center[...,a.hr_shape//2,a.hr_shape//2]=1
    edge=torch.zeros_like(center); edge[...,4,4]=1
    psf_result={
        "kernel_shape":list(psf.shape),"kernel_sum":float(psf.sum()),
        "resampling_factor":a.psf_source_pixscale_arcsec/hr_scale,
        "center_delta_sum":float(apply_psf(center,psf).sum()),
        "edge_delta_sum":float(apply_psf(edge,psf).sum()),
        "kernel_larger_than_image":bool(psf.shape[-1]>a.hr_shape),
    }
    psf_result["passed"]=abs(psf_result["kernel_sum"]-1)<1e-5 and abs(psf_result["center_delta_sum"]-1)<1e-3 and psf_result["edge_delta_sum"]>0.95
    root=Path(a.reports_dir)
    report_files=["surface_brightness_report.json","analytic_sis_report.json","gaussian_width_report.json"]
    prior={}
    for name in report_files:
        path=root/name
        prior[name]=json.loads(path.read_text()) if path.exists() else {"missing":True}
    ready=(psf_result["passed"] and prior.get("surface_brightness_report.json",{}).get("raw",{}).get("passed",False)
           and prior.get("analytic_sis_report.json",{}).get("passed",False)
           and prior.get("gaussian_width_report.json",{}).get("passed",False))
    out={"psf":psf_result,"prior_reports":prior,"ready_for_training":ready,
         "decision":"PASS: regenerate matching convergence maps and train mu=0 baseline" if ready else "FAIL: do not train; fix operator or PSF boundary handling first"}
    Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
