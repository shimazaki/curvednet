"""Download Hokusai ukiyo-e prints and convert to binary {-1,+1} patterns."""

import glob
import os
import urllib.request

import numpy as np
from PIL import Image

N_SIDE = 128

# Public-domain images from Wikimedia Commons (Hokusai, Fugaku Sanjurokkei)
SOURCES = {
    "great_wave": (
        "https://upload.wikimedia.org/wikipedia/commons/"
        "a/a5/Tsunami_by_hokusai_19th_century.jpg"
    ),
    "mona_lisa": (
        "https://upload.wikimedia.org/wikipedia/commons/"
        "6/6a/Mona_Lisa.jpg"
    ),
}

OUT_DIR = "data"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  cached: {dest}")
        return
    print(f"  downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Python/ukiyo-e-dl"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def to_binary(img_path: str, n: int) -> np.ndarray:
    img = Image.open(img_path).convert("L").resize((n, n), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    threshold = np.median(pixels)
    return np.where(pixels >= threshold, 1.0, -1.0)


def load_patterns(n_side: int = 128, data_dir: str = "data") -> list:
    """Load JPEGs from data_dir and convert to flat {-1,+1} arrays at any resolution.

    Only images whose stem matches a key in SOURCES are loaded.
    """
    paths = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*.jpg"))
        if os.path.splitext(os.path.basename(p))[0] in SOURCES
    )
    if not paths:
        raise FileNotFoundError(
            f"No JPEG images in {data_dir}/. Run generate_patterns.py first."
        )
    return [to_binary(p, n_side).ravel() for p in paths]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, url in SOURCES.items():
        print(f"Processing {name}...")
        ext = url.rsplit(".", 1)[-1].split("?")[0]
        raw_path = os.path.join(OUT_DIR, f"{name}.{ext}")
        download(url, raw_path)

    print("Done.")


if __name__ == "__main__":
    main()
