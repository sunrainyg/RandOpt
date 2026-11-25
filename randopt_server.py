#!/usr/bin/env python3
"""
vLLM OpenAI-compatible server extended with weight perturbation endpoints
via WorkerExtension.

  POST /perturb       {"seed": int, "sigma": float, "negate": bool}
  POST /restore       {"seed": int, "sigma": float, "negate": bool}
  POST /store_base    {}   — snapshot current weights as the reset target
  POST /reset         {}   — restore weights to the last /store_base snapshot

/store_base + /reset is the preferred alternative to /restore: no seed needed,
always exact (no floating-point add/subtract), safe even if perturbation params
are not remembered.  Call /store_base once after the server is ready, then use
/perturb … /reset for each evaluation cycle.

All standard vLLM API server arguments are accepted and forwarded unchanged.

Example:
  python3 randopt_server.py \\
    --model Qwen/Qwen2.5-32B-Instruct \\
    --worker-extension-cls utils.worker_extn.WorkerExtension \\
    --tensor-parallel-size 4 \\
    --max-model-len 16384 \\
    --max-num-seqs 64
"""

import os
import shutil
import sys

_RANDOPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _install_utils_next_to_vllm():
    """Make `utils.worker_extn` importable in vLLM worker processes.

    Newer vLLM forces 'spawn' for workers AND launches the engine-core / worker
    processes with a filtered environment, so neither the launcher's sys.path
    nor a runtime PYTHONPATH reaches them -- resolving
    --worker-extension-cls utils.worker_extn.WorkerExtension then fails with
    ModuleNotFoundError. vLLM's own install directory is always on every
    worker's import path (that is how they import vllm), so copying the package
    there is reliable. This runs in the API-server process before any engine or
    worker process is spawned.
    """
    if _RANDOPT_DIR not in sys.path:
        sys.path.insert(0, _RANDOPT_DIR)

    # Copy once, from the main API-server process, before any engine/worker is
    # spawned. Spawned processes re-run this module as __mp_main__; they only
    # need /randopt on sys.path (above) and must not race on the copy.
    if __name__ == "__main__":
        import vllm

        vllm_site_dir = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
        target_utils = os.path.join(vllm_site_dir, "utils")
        print(f"[randopt] copying {_RANDOPT_DIR}/utils -> {target_utils}", flush=True)
        shutil.copytree(os.path.join(_RANDOPT_DIR, "utils"), target_utils, dirs_exist_ok=True)

    import importlib

    importlib.invalidate_caches()
    import utils.worker_extn  # noqa: F401  verify resolution in this process

    print(
        f"[randopt] pid={os.getpid()} {__name__}: "
        f"utils.worker_extn -> {utils.worker_extn.__file__}",
        flush=True,
    )


_install_utils_next_to_vllm()

import uvloop
from fastapi import FastAPI, Request
from pydantic import BaseModel

import vllm.entrypoints.openai.api_server as _api_server
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser


class PerturbRequest(BaseModel):
    seed: int
    sigma: float
    negate: bool = False


class RestoreRequest(BaseModel):
    seed: int
    sigma: float
    negate: bool = False


_original_build_app = _api_server.build_app


# Forward *args/**kwargs: vLLM's build_app signature varies across versions
# (nightly: build_app(args, supported_tasks=None, model_config=None)).
def _build_app_with_randopt_routes(*args, **kwargs) -> FastAPI:
    app = _original_build_app(*args, **kwargs)

    @app.post("/perturb")
    async def perturb(body: PerturbRequest, raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc(
            "perturb_self_weights",
            args=(body.seed, body.sigma, body.negate),
        )
        return {"status": "ok", "seed": body.seed, "sigma": body.sigma}

    @app.post("/restore")
    async def restore(body: RestoreRequest, raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc(
            "restore_self_weights",
            args=(body.seed, body.sigma, body.negate),
        )
        return {"status": "ok", "seed": body.seed, "sigma": body.sigma}

    @app.post("/store_base")
    async def store_base(raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc("store_base_weights", args=())
        return {"status": "ok"}

    @app.post("/reset")
    async def reset(raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc("reset_to_base_weights", args=())
        return {"status": "ok"}

    return app


_api_server.build_app = _build_app_with_randopt_routes


if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with /perturb, /restore, /store_base and /reset endpoints."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
