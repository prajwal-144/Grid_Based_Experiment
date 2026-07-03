# import os
# folder_path = 'val/axion'
#
#
# def rename_files_simultaneously(path):
#     # Get list of all files in the directory
#     files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#
#     # Filter to ensure we only pick the specific sim_ files (optional but safer)
#     target_files = [f for f in files if f.startswith('sim_')]
#
#     # Sort the files alphabetically.
#     # Since your numbers are fixed length, standard string sorting works correctly
#     # (sim_...16282 comes before sim_...16627).
#     target_files.sort()
#
#     total_files = len(target_files)
#     print(f"Found {total_files} files to rename.")
#
#     if total_files == 0:
#         print("No matching files found.")
#         return
#
#     # Rename loop
#     for index, filename in enumerate(target_files):
#         # Construct new name: sim_0, sim_1, ... sim_17999
#         new_name = f"sim_{index}"
#
#         # Preserve file extension if it exists (though your examples didn't show one)
#         # If your files have no extension, this line just returns the name as is.
#         _, ext = os.path.splitext(filename)
#         if ext:
#             new_name += ext
#
#         # Full paths
#         src = os.path.join(path, filename)
#         dst = os.path.join(path, new_name)
#
#         try:
#             os.rename(src, dst)
#             # Optional: Print progress every 1000 files to avoid flooding the console
#             if index % 1000 == 0:
#                 print(f"Renamed {index}...")
#         except Exception as e:
#             print(f"Error renaming {filename}: {e}")
#
#     print(f"Done! Renamed {total_files} files from sim_0 to sim_{total_files - 1}.")
#
#
# if __name__ == "__main__":
#     rename_files_simultaneously(folder_path)
#
# import numpy as np
#
# x = np.load(r"train\axion\sim_1666.npy", allow_pickle=True)
# print(type(x))
# print(x.dtype if hasattr(x, "dtype") else "no dtype")
# print(x.shape if hasattr(x, "shape") else "no shape")
# print(x)

import glob, os, numpy as np
folders = ["train/no_sub","train/axion","train/cdm","val/no_sub","val/axion","val/cdm"]
bad = []
for folder in folders:
    for p in glob.glob(os.path.join(folder, "sim_*.npy")):
        try:
            x = np.load(p, allow_pickle=True)
            # unwrap common wrappers
            if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
                x = x.item()
            if isinstance(x, np.ndarray) and x.dtype == object and x.size == 2:
                x = x[0]
            _ = np.asarray(x, dtype=np.float32)
        except Exception as e:
            bad.append((p, str(e)))
for b in bad:
    print("BAD:", b)
if not bad:
    print("All files look OK to load.")
