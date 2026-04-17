"""Download Hokusai ukiyo-e prints and convert to binary {-1,+1} patterns."""

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

OUT_DIR = "patterns"


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


def save_preview(pattern: np.ndarray, path: str) -> None:
    pixels = ((pattern + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(pixels, "L").save(path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    for name, url in SOURCES.items():
        print(f"Processing {name}...")
        ext = url.rsplit(".", 1)[-1].split("?")[0]
        raw_path = os.path.join(cache_dir, f"{name}.{ext}")
        download(url, raw_path)

        pattern = to_binary(raw_path, N_SIDE)
        npy_path = os.path.join(OUT_DIR, f"{name}.npy")
        np.save(npy_path, pattern)
        print(f"  saved: {npy_path}  shape={pattern.shape}")

        png_path = os.path.join(OUT_DIR, f"{name}.png")
        save_preview(pattern, png_path)
        print(f"  preview: {png_path}")

    print("Done.")


if __name__ == "__main__":
    main()
