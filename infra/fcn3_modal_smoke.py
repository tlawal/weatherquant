"""Modal app — FCN3 inference smoke test.

Run with:
    modal run infra/fcn3_modal_smoke.py

Goal: prove FCN3 runs, fetches current GFS init, produces sensible t2m
for Atlanta. Production version (4×/day cron with HTTPS endpoint) comes
later.
"""
import modal

app = modal.App("weatherquant-fcn3-test")

# ── Container image — ABI-stable dependency stack ─────────────────────────
#
# Previous 10 attempts (May 2026) failed because of upstream ABI drift:
#   - numpy>=2 broke compiled extensions (torch_harmonics, kvikio, dgl, cupy)
#   - zarr>=3 forced numpy 2 transitively
#   - earth2studio upgraded torch 2.5→2.11 via transitive deps
#   - physicsnemo 1.1+ uses wp.context.Device that warp-lang 1.13 doesn't expose
#
# Fix: accept numpy>=2/zarr>=3.1.0 (earth2studio 0.13+ requires it),
#      use physicsnemo>=1.1.1 (makani 0.2 requires it),
#      warp-lang 1.10-1.12 (avoids 1.13 wp.context.Device breakage),
#      torch_harmonics>=0.9.0 (makani 0.2 requirement).
#
# Weights come from HuggingFace (nvidia/fourcastnet3) — no NGC auth needed.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
    )
    .apt_install("git", "build-essential", "libeccodes-dev")
    # Strategy: earth2studio 0.13+ requires zarr>=3.1.0 which forces numpy>=2.
    # We accept this. torch_harmonics>=0.9.0 is required by makani 0.2.
    # Force CUDA extension build for torch_harmonics so FCN3 runs at GPU speed.
    .pip_install(
        "numpy>=2.0,<3.0",
        "zarr>=3.1.0",
        "warp-lang>=1.10.0,<1.13",
        "nvidia-physicsnemo>=1.1.1",
        "makani @ git+https://github.com/NVIDIA/makani.git",
        "earth2studio>=0.13.0",
        "xarray",
        "h5netcdf",
        "netCDF4",
        "huggingface-hub",
        "moviepy",
        "Pillow",
    )
    # Build torch_harmonics CUDA extensions separately with the env var set.
    # This MUST be a separate build step so the env var is active during compile.
    .run_commands(
        "FORCE_CUDA_EXTENSION=1 pip install --no-build-isolation 'torch-harmonics>=0.9.0'"
    )
)


@app.function(
    image=image,
    gpu="H100",                    # FCN3 needs a >=60 GB GPU for the smoke run
    timeout=900,                   # 15 min cold-start budget
)
def run_fcn3_atlanta():
    """Run one FCN3 forecast for Atlanta from current GFS init.

    Returns the projected t2m series at Atlanta lat/lon over the 10-day
    forecast horizon, plus the daily-high estimate.
    """
    from datetime import datetime, timedelta, timezone

    # ── Diagnostic preamble: surface exactly what pip resolved ─────────────
    import importlib.metadata as md
    print("=" * 60)
    print("DIAGNOSTIC PREAMBLE")
    print("=" * 60)
    for pkg in ("earth2studio", "physicsnemo", "warp-lang", "torch", "makani"):
        try:
            v = md.version(pkg)
            print(f"  {pkg}: {v}")
        except md.PackageNotFoundError:
            print(f"  {pkg}: NOT INSTALLED")
    try:
        from earth2studio.models import px as _px
        models = sorted(x for x in dir(_px) if not x.startswith("_"))
        print(f"  earth2studio.models.px exports: {models}")
        print(f"  FCN3 in exports: {'FCN3' in models}")
    except Exception as e:
        print(f"  earth2studio.models.px: import failed — {e}")
    print("=" * 60)

    # Detailed optional-dependency diagnostics.
    print("\n--- Optional dependency check ---")
    try:
        import torch_harmonics
        print(f"  torch_harmonics: {torch_harmonics.__version__}")
    except Exception as e:
        print(f"  torch_harmonics: FAILED — {e}")
    try:
        from makani.models.model_package import load_model_package
        print(f"  makani.load_model_package: OK")
    except Exception as e:
        print(f"  makani.load_model_package: FAILED — {e}")
    try:
        import warp as wp
        print(f"  warp: OK (version check skipped)")
    except Exception as e:
        print(f"  warp: FAILED — {e}")
    try:
        import physicsnemo
        print(f"  physicsnemo: OK")
    except Exception as e:
        print(f"  physicsnemo: FAILED — {e}")
    print("--- End dependency check ---\n")

    # Earth2Studio imports (only available inside the container).
    from earth2studio.models.px import FCN3
    from earth2studio.data import GFS
    from earth2studio.run import deterministic
    import numpy as np

    # 1. Try to identify which optional dependency is failing.
    print("\n--- Probing FCN3 optional deps ---")
    try:
        from earth2studio.models.px.fcn3 import FCN3 as _FCN3
        print("  earth2studio FCN3 module import: OK")
    except Exception as e:
        print(f"  earth2studio FCN3 module import: FAILED — {e}")
    try:
        package = FCN3.load_default_package()
        print(f"  FCN3.load_default_package: OK — {package}")
    except Exception as e:
        print(f"  FCN3.load_default_package: FAILED — {e}")
        return {"error": str(e), "stage": "load_default_package"}

    try:
        model = FCN3.load_model(package)
        print(f"  FCN3.load_model: OK — params={sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  FCN3.load_model: FAILED — {e}")
        return {"error": str(e), "stage": "load_model"}

    # 2. Fetch the most recent GFS init.
    print("fetching GFS initial conditions…")
    data_source = GFS()

    # 3. Run inference for 10-day forecast (40 steps × 6h).
    init_time = datetime.now(timezone.utc) - timedelta(hours=12)
    init_time = init_time.replace(
        hour=(init_time.hour // 6) * 6, minute=0, second=0, microsecond=0,
    )
    print(f"running inference from init={init_time.isoformat()}")

    forecast = deterministic(
        time=[init_time],
        nsteps=40,
        prognostic=model,
        data=data_source,
    )

    # 4. Extract Atlanta point (KATL: 33.6367, -84.4281)
    atl_lat, atl_lon = 33.6367, -84.4281
    if atl_lon < 0:
        atl_lon += 360  # FCN3 uses 0-360 longitude
    atl_t2m = forecast["t2m"].sel(
        lat=atl_lat, lon=atl_lon, method="nearest",
    )
    atl_t2m_f = (atl_t2m.values - 273.15) * 9 / 5 + 32

    daily_high_f = float(np.max(atl_t2m_f))

    print(f"\n=== Atlanta t2m forecast (FCN3) ===")
    print(f"init time:        {init_time.isoformat()}")
    print(f"forecast horizon: 10 days × 6h steps")
    print(f"projected high:   {daily_high_f:.1f}°F")
    print(f"first 24h trace:  {[round(float(v), 1) for v in atl_t2m_f[:5]]}")

    return {
        "init_time": init_time.isoformat(),
        "atlanta_daily_high_f": daily_high_f,
        "first_5_steps_f": [float(v) for v in atl_t2m_f[:5]],
    }


@app.local_entrypoint()
def main():
    """Trigger a single inference run from the local machine."""
    result = run_fcn3_atlanta.remote()
    print(f"\nResult returned to local machine: {result}")
