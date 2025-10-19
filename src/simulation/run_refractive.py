# run_refractive.py
import argparse, glob, os, random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import cv2, numpy as np, yaml
from tqdm import tqdm

from .Refractive_distortions.divfree import apply_incompressible_distortion
# Support both file names: tpl.py (Triangular Power-Law) or tps.py
try:
    from .Refractive_distortions.tpl import apply_tpl_distortion
except Exception:
    from .Refractive_distortions.tps import apply_tpl_distortion  # type: ignore

from .Refractive_distortions.perlin import apply_perlin_distortion
from .Refractive_distortions.radial import apply_redistortion, load_first_calib
from ..simulation.utils import (
    ensure_dirs, jitter_distortion, make_checkerboard,
    save_selection_json, safe_imread
)


DEFAULT_CFG = {
    "input_dir": "data/raw",
    "base_out": "data/sim/refractive",
    "max_samples": 25000,
    "max_workers": 40,
    "seed": 1234,
    "checkerboard": {"height": 1080, "width": 1920, "tile_size": 120},
    "calibration": {
        "path": "dashcam_calib.json",
        "d_jitter_alpha": 0.01,
        "d_jitter_abs_min": 1.0e-6,
        "undistort_alpha": 0.0,
    },
    "uv_dtype": "float16",
    "out_dirs": {
        "radial": "radial/images_distorted",
        "radial_uv": "radial/uv_maps",
        "radial_checkboard": "radial/checkerboards",
        "perlin": "perlin/images_distorted",
        "perlin_uv": "perlin/uv_maps",
        "perlin_checkboard": "perlin/checkerboards",
        "tpl": "tpl/images_distorted",
        "tpl_uv": "tpl/uv_maps",
        "tpl_checkboard": "tpl/checkerboards",
        "incompressible": "divfree/images_distorted",
        "incompressible_uv": "divfree/uv_maps",
        "incompressible_checkboard": "divfree/checkerboards",
    },
    "warps": {
        "perlin":  {"amp_range": [1.0, 3.0], "len_scale_range": [10.0, 20.0],
                    "var_range": [0.5, 1.5], "seed_x": None, "seed_y": None,
                    "downscale_range": [3, 8]},
        "tpl":     {"amp_range": [1.0, 3.0], "len_scale_range": [10.0, 20.0],
                    "var_range": [0.5, 1.5], "exponent_range": [1.0, 2.0],
                    "seed_x": None, "seed_y": None, "downscale_range": [3, 8]},
        "incompressible": {"amp_range": [1.0, 3.0], "len_scale_range": [10.0, 20.0],
                           "var_range": [0.5, 1.5], "seed": None,
                           "downscale_range": [3, 8]},
    },
}


def _merge_cfg(path: str | None) -> Dict:
    if not path:
        return DEFAULT_CFG
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    cfg = DEFAULT_CFG | user
    for k in ("checkerboard", "calibration", "out_dirs", "warps"):
        cfg[k] = DEFAULT_CFG[k] | user.get(k, {})
    if "warps" in user:
        for wk in DEFAULT_CFG["warps"]:
            cfg["warps"][wk] = DEFAULT_CFG["warps"][wk] | user["warps"].get(wk, {})
    return cfg


def _rf(lo_hi: Tuple[float, float]) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1]); return random.uniform(lo, hi)


def _ri(lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1]); return random.randint(lo, hi)


def _process_one(path: str, out_dirs_abs: Dict, cfg: Dict, K_calib, D_calib, checker_color):
    und = safe_imread(path)
    if und is None:
        return
    h, w = und.shape[:2]
    base = os.path.splitext(os.path.basename(path))[0]

    # Radial
    cal = cfg["calibration"]
    D_pert = jitter_distortion(D_calib, cal["d_jitter_alpha"], cal["d_jitter_abs_min"])
    radial, uv_rad = apply_redistortion(und, K_calib, D_pert, image_size=(w, h), alpha=cal["undistort_alpha"])
    cv2.imwrite(f"{out_dirs_abs['radial']}/{base}.png", radial)
    np.save(f"{out_dirs_abs['radial_uv']}/{base}.npy", uv_rad)

    # Perlin
    wp = cfg["warps"]["perlin"]
    perlin, uv_per = apply_perlin_distortion(
        und, amp=_rf(wp["amp_range"]), len_scale=_rf(wp["len_scale_range"]), var=_rf(wp["var_range"]),
        seed_x=(wp["seed_x"] if wp["seed_x"] is not None else random.randint(0, 2**31 - 1)),
        seed_y=(wp["seed_y"] if wp["seed_y"] is not None else random.randint(0, 2**31 - 1)),
        downscale=_ri(wp["downscale_range"]),
    )
    cv2.imwrite(f"{out_dirs_abs['perlin']}/{base}.png", perlin)
    np.save(f"{out_dirs_abs['perlin_uv']}/{base}.npy", uv_per)

    # TPL / TPS
    wt = cfg["warps"]["tpl"]
    tpl, uv_tpl = apply_tpl_distortion(
        und, amp=_rf(wt["amp_range"]), len_scale=_rf(wt["len_scale_range"]), var=_rf(wt["var_range"]),
        exponent=_rf(wt["exponent_range"]) if "exponent_range" in wt else 1.5,
        seed_x=(wt["seed_x"] if wt["seed_x"] is not None else random.randint(0, 2**31 - 1)),
        seed_y=(wt["seed_y"] if wt["seed_y"] is not None else random.randint(0, 2**31 - 1)),
        downscale=_ri(wt["downscale_range"]),
    )
    cv2.imwrite(f"{out_dirs_abs['tpl']}/{base}.png", tpl)
    np.save(f"{out_dirs_abs['tpl_uv']}/{base}.npy", uv_tpl)

    # Divergence-free
    wi = cfg["warps"]["incompressible"]
    inc, uv_inc = apply_incompressible_distortion(
        und, amp=_rf(wi["amp_range"]), len_scale=_rf(wi["len_scale_range"]), var=_rf(wi["var_range"]),
        seed=(wi["seed"] if wi["seed"] is not None else random.randint(0, 2**31 - 1)),
        downscale=_ri(wi["downscale_range"]),
    )
    cv2.imwrite(f"{out_dirs_abs['incompressible']}/{base}.png", inc)
    np.save(f"{out_dirs_abs['incompressible_uv']}/{base}.npy", uv_inc)

    # Checkerboards via UVs
    def warp_checker(uv):
        mx, my = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
        return cv2.remap(checker_color, mx, my, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    cv2.imwrite(f"{out_dirs_abs['radial_checkboard']}/{base}.png", warp_checker(uv_rad))
    cv2.imwrite(f"{out_dirs_abs['perlin_checkboard']}/{base}.png", warp_checker(uv_per))
    cv2.imwrite(f"{out_dirs_abs['tpl_checkboard']}/{base}.png", warp_checker(uv_tpl))
    cv2.imwrite(f"{out_dirs_abs['incompressible_checkboard']}/{base}.png", warp_checker(uv_inc))


def run_pipeline(cfg: Dict):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    # outputs resolved under base_out
    base_out = cfg["base_out"]
    out_dirs_abs = {k: os.path.join(base_out, v) if not os.path.isabs(v) else v
                    for k, v in cfg["out_dirs"].items()}
    ensure_dirs(out_dirs_abs.values())

    # inputs
    paths = glob.glob(os.path.join(cfg["input_dir"], "*.*"))
    sampled = random.sample(paths, min(cfg["max_samples"], len(paths)))
    save_selection_json([os.path.basename(p) for p in sampled],
                        os.path.join(base_out, "selection_geometric.json"))

    # calib
    K_calib, D_calib = load_first_calib(cfg["calibration"]["path"])
    D_calib = D_calib.ravel()

    # checkerboard
    cb = cfg["checkerboard"]
    checker = make_checkerboard((cb["height"], cb["width"]), tile_size=cb["tile_size"])
    checker_color = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)

    # pool
    workers = int(cfg["max_workers"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(
            tqdm(
                ex.map(lambda p: _process_one(p, out_dirs_abs, cfg, K_calib, D_calib, checker_color), sampled),
                total=len(sampled),
                desc="Generating geometric distortions",
            )
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--input_dir", type=str, default=None)
    ap.add_argument("--base_out", type=str, default=None)
    ap.add_argument("--max_workers", type=int, default=None)
    args = ap.parse_args()

    cfg = _merge_cfg(args.config)
    if args.input_dir:   cfg["input_dir"] = args.input_dir
    if args.base_out:    cfg["base_out"] = args.base_out
    if args.max_workers is not None: cfg["max_workers"] = int(args.max_workers)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
