import argparse, os, sys, time

import ray
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--local_root", default="/tmp/randopt_models")
    a = ap.parse_args()
    t0 = time.time()
    snap = snapshot_download(a.model_name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.py"])
    print(f"shared snapshot: {snap} ({time.time()-t0:.0f}s)", flush=True)
    local = os.path.join(a.local_root, a.model_name.replace("/", "__"))

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as S

    @ray.remote(num_cpus=0)
    def copy(src, dst):
        import shutil, socket, subprocess, time as _t
        t = _t.time()
        if not os.path.exists(f"{dst}/.complete"):
            tmp = dst + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp, symlinks=False)
            os.rename(tmp, dst)
            open(f"{dst}/.complete", "w").write(src)
        n = subprocess.run(f"du -sh {dst} | cut -f1", shell=True, capture_output=True, text=True).stdout.strip()
        return socket.gethostname()[:8], n, round(_t.time() - t)

    nodes = [n for n in ray.nodes() if n["Alive"]]
    res = ray.get([copy.options(scheduling_strategy=S(node_id=n["NodeID"], soft=False)).remote(snap, local) for n in nodes])
    slow = [r for r in res if r[2] > 5]
    print(f"staged on {len(res)} nodes -> {local} ; sizes {set(r[1] for r in res)} ; copies>5s: {slow[:5]}", flush=True)
    print(local)


if __name__ == "__main__":
    main()
