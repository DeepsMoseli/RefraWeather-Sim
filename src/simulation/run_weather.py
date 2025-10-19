# run_weather.py
import argparse, glob, os, random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Dict

import cv2, numpy as np, yaml
from tqdm import tqdm

from .Weather_artifacts.hetero_fog import apply_heterogeneous_fog
from .Weather_artifacts.lens_flare import apply_flare
from .Weather_artifacts.uniform_fog import apply_uniform_fog
from ..simulation.utils import ensure_dirs, save_selection_json, safe_imread, visualize_kmap_on_white


DEFAULT_CFG = {
    "input_dir": "data/raw",
    "base_out": "data/sim/weather",
    "max_samples": 2000,
    "process_pool": True,
    "seed": 1234,
    "fog_strength_base": 0.5,
    "perlin_scales": [4, 8, 16, 32, 64, 128],
    "perlin_weights": [0.30, 0.22, 0.15, 0.11, 0.08, 0.07],
    "flare": {"default_intensity": 0.6, "default_radius_frac": 0.3, "seed": None},
    "out_dirs": {
        "uniform_fog": "uniform_fog/images_corrupted",
        "uniform_fog_kmap": "uniform_fog/alpha_maps",
        "uniform_fog_kvis": "uniform_fog/alpha_maps_viz",
        "hetero_fog": "hetero_fog/images_corrupted",
        "hetero_fog_kmap": "hetero_fog/alpha_maps",
        "hetero_fog_kvis": "hetero_fog/alpha_maps_viz",
        "flare": "flare/images_corrupted",
        "flare_kmap": "flare/alpha_maps",
        "flare_kvis": "flare/alpha_maps_viz",
    },
}


def _merge_cfg(path: str | None) -> Dict:
    if not path:
        return DEFAULT_CFG
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    cfg = DEFAULT_CFG | user
    cfg["flare"] = DEFAULT_CFG["flare"] | user.get("flare", {})
    cfg["out_dirs"] = DEFAULT_CFG["out_dirs"] | user.get("out_dirs", {})
    return cfg


def _process_one(path: str, out_dirs_abs: Dict, cfg: Dict):
    img = safe_imread(path)
    if img is None:
        return
    base = os.path.splitext(os.path.basename(path))[0]

    # Uniform fog (±5%)
    s_uni = cfg["fog_strength_base"] * random.uniform(0.95, 1.05)
    uni_img, k_uni = apply_uniform_fog(img, s_uni)
    cv2.imwrite(f"{out_dirs_abs['uniform_fog']}/{base}.png", uni_img)
    np.save(f"{out_dirs_abs['uniform_fog_kmap']}/{base}.npy", k_uni)
    cv2.imwrite(f"{out_dirs_abs['uniform_fog_kvis']}/{base}.png", visualize_kmap_on_white(k_uni))

    # Heterogeneous fog (±5%)
    s_het = cfg["fog_strength_base"] * random.uniform(0.95, 1.05)
    het_img, k_het = apply_heterogeneous_fog(img, s_het)
    cv2.imwrite(f"{out_dirs_abs['hetero_fog']}/{base}.png", het_img)
    np.save(f"{out_dirs_abs['hetero_fog_kmap']}/{base}.npy", k_het)
    cv2.imwrite(f"{out_dirs_abs['hetero_fog_kvis']}/{base}.png", visualize_kmap_on_white(k_het))

    # Flare (±5%)
    fl = cfg["flare"]
    intensity = fl["default_intensity"] * random.uniform(0.95, 1.05)
    radius = fl["default_radius_frac"] * random.uniform(0.95, 1.05)
    seed = fl["seed"] if fl["seed"] is not None else random.randint(0, 2**31 - 1)
    flare_img, flare_mask = apply_flare(img, intensity=intensity, radius_frac=radius, seed=seed)
    cv2.imwrite(f"{out_dirs_abs['flare']}/{base}.png", flare_img)
    np.save(f"{out_dirs_abs['flare_kmap']}/{base}.npy", flare_mask)
    cv2.imwrite(f"{out_dirs_abs['flare_kvis']}/{base}.png", visualize_kmap_on_white(flare_mask))


def run_pipeline(cfg: Dict):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    # resolve out dirs under base_out
    base_out = cfg["base_out"]
    out_dirs_abs = {k: os.path.join(base_out, v) if not os.path.isabs(v) else v
                    for k, v in cfg["out_dirs"].items()}
    ensure_dirs(out_dirs_abs.values())

    # sample inputs
    paths = glob.glob(os.path.join(cfg["input_dir"], "*.*"))
    sampled = random.sample(paths, min(cfg["max_samples"], len(paths)))
    save_selection_json([os.path.basename(p) for p in sampled],
                        os.path.join(base_out, "selection_photometric.json"))

    if cfg["process_pool"]:
        with ProcessPoolExecutor() as ex:
            list(tqdm(ex.map(_process_one, sampled, repeat(out_dirs_abs), repeat(cfg)),
                      total=len(sampled), desc="Generating photometric distortions"))
    else:
        for p in tqdm(sampled, desc="Generating photometric distortions"):
            _process_one(p, out_dirs_abs, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--input_dir", type=str, default=None)
    ap.add_argument("--base_out", type=str, default=None)
    args = ap.parse_args()

    cfg = _merge_cfg(args.config)
    if args.input_dir: cfg["input_dir"] = args.input_dir
    if args.base_out:  cfg["base_out"] = args.base_out
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
