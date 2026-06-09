"""
Download the 5 benchmark models into a local models/ folder for offline use.

Uses huggingface_hub.snapshot_download — files already in the HF cache are
reused, so this is fast on a second run. Each model lands in models/<key>/.
"""
import os
from huggingface_hub import snapshot_download

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# (local_folder_name, hf_repo_id)
MODELS = [
    ("depth_anything_v2_small", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("dpt_large",               "Intel/dpt-large"),
    ("dpt_hybrid",              "Intel/dpt-hybrid-midas"),
    ("zoedepth",                "Intel/zoedepth-nyu-kitti"),
    ("glpn_nyu",                "vinvino02/glpn-nyu"),
]

# Skip duplicate weight formats to save disk (keep safetensors when present).
IGNORE = ["*.msgpack", "*.h5", "*.ot", "flax_model.*", "tf_model.*"]


def main():
    for key, repo in MODELS:
        dest = os.path.join(MODELS_DIR, key)
        print(f"\n=== {key}  <-  {repo} ===")
        path = snapshot_download(
            repo_id=repo,
            local_dir=dest,
            ignore_patterns=IGNORE,
        )
        # report size
        total = 0
        for root, _, files in os.walk(dest):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        print(f"  saved to {path}  ({total / 1e6:.0f} MB)")

    print(f"\nAll models in: {MODELS_DIR}")
    print("To use a local copy, pass the folder path as the model id, e.g.:")
    print('  pipeline("depth-estimation", model="models/depth_anything_v2_small", device=0)')


if __name__ == "__main__":
    main()
