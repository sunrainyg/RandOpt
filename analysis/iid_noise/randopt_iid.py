import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import ray

import core.engine
import randopt
from analysis.iid_noise.iid_worker_extn import IIDNcclLLM

core.engine.RandOptNcclLLM = IIDNcclLLM

if __name__ == "__main__":
    args = randopt.parse_args()
    args.noise_mode = "iid"  # recorded in args.json
    pythonpath = os.pathsep.join(filter(None, [REPO, os.environ.get("PYTHONPATH")]))
    ray.init(address="auto" if os.environ.get("RAY_ADDRESS") else "local", ignore_reinit_error=True,
             runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
    randopt.main(args)
