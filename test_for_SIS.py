import torch

for filename in [
    "sis_stage3_checkpoint.pt",
]:
    checkpoint = torch.load(filename, map_location="cpu")
print(
    filename,
    "initial =", checkpoint["args"]["theta_e"],
    "final =", checkpoint["theta_e_arcsec"],
)