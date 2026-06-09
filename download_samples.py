"""Download a few standard test images for depth-estimation comparison."""
import os
import requests

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
os.makedirs(INPUT_DIR, exist_ok=True)

# Public, freely-usable test images covering distinct scene types.
SAMPLES = {
    # Classic indoor scene used in many depth demos (HF example image).
    "indoor_cats.jpg": "http://images.cocodataset.org/val2017/000000039769.jpg",
    # Outdoor street scene with strong depth ordering.
    "street.jpg": "http://images.cocodataset.org/val2017/000000037777.jpg",
    # Person / portrait-style scene.
    "person.jpg": "http://images.cocodataset.org/val2017/000000000785.jpg",
    # Open landscape with far-field depth.
    "landscape.jpg": "http://images.cocodataset.org/val2017/000000001000.jpg",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (depth-compare)"}


def main():
    for name, url in SAMPLES.items():
        dest = os.path.join(INPUT_DIR, name)
        if os.path.exists(dest):
            print(f"[skip] {name} already exists")
            continue
        print(f"[get ] {name} <- {url}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"[ok  ] saved {dest} ({len(r.content)//1024} KB)")
    print(f"\nDone. Images in: {INPUT_DIR}")


if __name__ == "__main__":
    main()
