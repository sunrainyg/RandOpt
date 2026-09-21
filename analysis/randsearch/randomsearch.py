import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); os.chdir(REPO)

import ray
from transformers import AutoTokenizer
from vllm import SamplingParams

from core.engine import launch_engines, cleanup_engines
from data_handlers import get_dataset_handler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="countdown", help="one dataset, or a comma list recorded back-to-back on the same engines (out_dir/<dataset>/)")
    p.add_argument("--model_name", default="allenai/Olmo-3-7B-Instruct")
    p.add_argument("--model_local_path", default=None)
    p.add_argument("--precision", default="bfloat16")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--n_select", type=int, default=200)
    p.add_argument("--n_test", type=int, default=None, help="cap on test prompts (default: all)")
    p.add_argument("--population_size", type=int, required=True, help="number of perturbed members")
    p.add_argument("--sigma_values", default="0.0001,0.0003,0.001,0.002")
    p.add_argument("--temps", default="0.6,0.8,1.0,1.2")
    p.add_argument("--n_base_samples", type=int, default=16)
    p.add_argument("--n_test_samples", type=int, default=2)
    p.add_argument("--n_select_samples", type=int, default=8)
    p.add_argument("--num_engines", type=int, required=True)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--engine_batch", type=int, default=None, help="engines initialized per batch (default: half the fleet)")
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def git_hash(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def extract(handler, text, data):
    if handler.name == "countdown":
        formula = handler.extract_answer(text)
        if "**" in formula or len(formula) > 300:
            return "", False
    import signal
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(5)
    try:
        return _extract(handler, text, data)
    except _Timeout:
        return "", False
    finally:
        signal.alarm(0)


def _extract(handler, text, data):
    if handler.name == "countdown":
        ans, ok, _ = handler.extract_answer_for_voting(text, numbers=data.get("numbers"))
        ans = ans if ok else ""
    elif hasattr(handler, "extract_answer_for_voting"):
        ans = handler.extract_answer_for_voting(text) or ""
    else:
        ans = handler.extract_answer(text) or ""
    if not ans:
        return "", False
    if hasattr(handler, "is_voted_answer_correct"):
        ok = handler.is_voted_answer_correct(ans, data["ground_truth"])
    else:
        ok = handler.is_answer_correct(handler.format_answer_for_check(ans), data["ground_truth"])
    return ans, bool(ok)


@ray.remote(num_cpus=1)
def _score_remote(dataset, texts, datas):
    import os as _os, sys as _sys
    _sys.path.insert(0, REPO); _os.chdir(REPO)
    from data_handlers import get_dataset_handler as _g
    h = _g(dataset)
    out = []
    for t, d in zip(texts, datas):
        try:
            out.append(extract(h, t, d))
        except Exception:  # noqa: BLE001
            out.append(("", False))
    return out


def score_texts(dataset, texts, datas, chunk=64):
    refs = [_score_remote.remote(dataset, texts[i:i + chunk], datas[i:i + chunk]) for i in range(0, len(texts), chunk)]
    res = []
    for r in ray.get(refs):
        res.extend(r)
    return res


def chunks(seq, n):
    k, m = divmod(len(seq), n)
    out, s = [], 0
    for i in range(n):
        e = s + k + (1 if i < m else 0)
        out.append(seq[s:e]); s = e
    return out


def main():
    a = parse_args()
    datasets = [x for x in a.dataset.split(",") if x]
    multi = len(datasets) > 1
    sigmas = [float(s) for s in a.sigma_values.split(",")]
    temps = [float(t) for t in a.temps.split(",")] if a.temps else []
    tok = AutoTokenizer.from_pretrained(a.model_name)
    is_instruct = any(x in a.model_name.lower() for x in ["instruct", "chat", "it"])

    def fmt(msgs):
        if is_instruct and tok.chat_template:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in msgs) + "\n"

    rt_env = {"py_executable": sys.executable,
              "env_vars": {"PYTHONPATH": REPO, "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1",
                           "HF_HOME": os.environ.get("HF_HOME", ""), "TOKENIZERS_PARALLELISM": "false"}}
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, runtime_env=rt_env)
    engines, pgs = launch_engines(a.num_engines, a.model_local_path or a.model_name, precision=a.precision, tensor_parallel_size=1,
                                  gpu_memory_utilization=a.gpu_memory_utilization,
                                  batch_size=a.engine_batch or max(1, a.num_engines // 2))
    E = len(engines)
    try:
        for ds in datasets:
            out = f"{a.out_dir}/{ds}" if multi else a.out_dir
            print(f"\n===== dataset {ds} -> {out} =====", flush=True)
            record_dataset(a, ds, out, engines, E, fmt, sigmas, temps)
        print("DONE", flush=True)
    finally:
        cleanup_engines(engines, pgs)


def record_dataset(a, dataset, out_dir, engines, E, fmt, sigmas, temps):
    os.makedirs(f"{out_dir}/members", exist_ok=True)
    handler = get_dataset_handler(dataset)
    max_tokens = a.max_tokens or handler.default_max_tokens
    train_path, test_path = handler.default_train_path, handler.default_test_path
    if train_path == test_path:
        allx = handler.load_data(train_path, split="train", max_samples=None)
        select, test = allx[:a.n_select], allx[a.n_select:]
    else:
        select = handler.load_data(train_path, split="train", max_samples=a.n_select)
        test = handler.load_data(test_path, split="test", max_samples=None)
    if a.n_test:
        test = test[:a.n_test]
    print(f"prompts: select={len(select)} test={len(test)} max_tokens={max_tokens} temps={temps}", flush=True)
    P = {"select": [fmt(d["messages"]) for d in select], "test": [fmt(d["messages"]) for d in test]}
    all_prompts = P["select"] + P["test"]; all_data = select + test; nS = len(select)

    rng = np.random.default_rng(seed=a.global_seed)
    seeds = rng.choice(2**31, size=a.population_size, replace=False).tolist()
    msig = rng.choice(sigmas, size=a.population_size).tolist()
    members = [(int(s), float(g)) for s, g in zip(seeds, msig)]
    args = dict(vars(a)); args["dataset"] = dataset
    meta = dict(args=args, randopt_commit=git_hash(REPO), script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
                n_select=nS, n_test=len(test), members=members, temps=temps, max_tokens=max_tokens,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
    greedy = SamplingParams(temperature=0.0, seed=a.global_seed, max_tokens=max_tokens)

    def sp_list(T, n, seed, idxs):
        return [SamplingParams(temperature=T, seed=(int(seed) * 1_000_003 + int(j) * 7919 + 1) % (2**31), max_tokens=max_tokens, n=n) for j in idxs]

    def record(outs, datas):
        res = score_texts(dataset, [o.outputs[0].text for o in outs], list(datas))
        ans = np.array([r[0] for r in res], dtype=object); ok = np.array([r[1] for r in res], dtype=bool)
        tr = np.array([o.outputs[0].finish_reason == "length" for o in outs], dtype=bool)
        return ans, ok, tr

    def record_multi(outs, datas):
        n = len(outs[0].outputs)
        texts = [c.text for o in outs for c in o.outputs]
        res = score_texts(dataset, texts, [d for d in datas for _ in range(n)])
        ok = np.array([r[1] for r in res], dtype=bool).reshape(len(outs), n)
        ans = np.array([r[0] for r in res], dtype=object).reshape(len(outs), n)
        return ans, ok

    def distributed(prompts, sps):
        parts = chunks(list(range(len(prompts))), E)
        futs = [engines[i].generate.remote([prompts[j] for j in parts[i]], sps if not isinstance(sps, list) else [sps[j] for j in parts[i]], use_tqdm=False)
                for i in range(E) if parts[i]]
        flat = [o for part in ray.get(futs) for o in part]; order = [j for part in parts for j in part]
        out = [None] * len(prompts)
        for j, o in zip(order, flat):
            out[j] = o
        return out

    if not os.path.exists(f"{out_dir}/base_greedy.npz"):
        t0 = time.time()
        outs = distributed(all_prompts, greedy)
        ans, ok, tr = record(outs, all_data)
        np.savez(f"{out_dir}/base_greedy.npz", answers=ans, correct=ok, truncated=tr, n_select=nS)
        print(f"[base greedy] select={ok[:nS].mean():.4f} test={ok[nS:].mean():.4f} truncated(test)={tr[nS:].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    for T in temps:
        f = f"{out_dir}/base_T{T}.npz"
        if os.path.exists(f):
            continue
        t0 = time.time()
        outs = distributed(all_prompts, sp_list(T, a.n_base_samples, a.global_seed, range(len(all_prompts))))
        ans, ok = record_multi(outs, all_data)
        np.savez(f, answers=ans, correct=ok, n_select=nS)
        print(f"[base T={T}] expected select={ok[:nS].mean():.4f} test={ok[nS:].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    todo = [i for i in range(len(members)) if not os.path.exists(f"{out_dir}/members/{i}.npz")]
    print(f"[members] {len(todo)} to run on {E} engines (dynamic dispatch: an engine takes the next member as soon as it finishes)", flush=True)
    t_all = time.time(); CH = 512
    queue_ = list(todo); active = {}; scoring = []; n_saved = 0; t_last = time.time()

    def start(e, i):
        # actor tasks on one engine run in submission order, so apply -> generates -> reset needs no waiting here
        engines[e].collective_rpc.remote("apply_perturbation", args=(members[i][0], members[i][1]))
        futs = {"greedy": engines[e].generate.remote(all_prompts, greedy, use_tqdm=False)}
        for T in temps:
            if a.n_select_samples > 0:
                futs[f"select_T{T}"] = engines[e].generate.remote(P["select"], sp_list(T, a.n_select_samples, members[i][0] + 1, range(nS)), use_tqdm=False)
            if a.n_test_samples > 0:
                futs[f"test_T{T}"] = engines[e].generate.remote(P["test"], sp_list(T, a.n_test_samples, members[i][0], range(len(test))), use_tqdm=False)
        engines[e].collective_rpc.remote("reset_to_base_weights", args=())
        active[e] = (i, futs)

    def submit_scoring(i, res):
        jobs = [("greedy", 1, [o.outputs[0].text for o in res["greedy"]], all_data)]
        for T in temps:
            if f"select_T{T}" in res:
                jobs.append((f"select_T{T}", a.n_select_samples, [c.text for o in res[f"select_T{T}"] for c in o.outputs], [d for d in select for _ in range(a.n_select_samples)]))
            if f"test_T{T}" in res:
                jobs.append((f"test_T{T}", a.n_test_samples, [c.text for o in res[f"test_T{T}"] for c in o.outputs], [d for d in test for _ in range(a.n_test_samples)]))
        refs = []
        for j, (key, n, texts, datas) in enumerate(jobs):
            for c0 in range(0, len(texts), CH):
                refs.append((j, c0, _score_remote.remote(dataset, texts[c0:c0 + CH], datas[c0:c0 + CH])))
        trunc = np.array([o.outputs[0].finish_reason == "length" for o in res["greedy"]], dtype=bool)
        return dict(i=i, jobs=[(k, n) for k, n, _, _ in jobs], refs=refs, trunc=trunc)

    def finish_scoring(S):
        got = ray.get([r for _, _, r in S["refs"]])
        scored = {}
        for (j, c0, _), res in zip(S["refs"], got):
            scored.setdefault(j, []).append((c0, res))
        out = {}
        for j, (key, n) in enumerate(S["jobs"]):
            res = [x for _, part in sorted(scored[j]) for x in part]
            out[key] = (res, np.array([r[1] for r in res], dtype=bool), n)
        res, ok, _ = out["greedy"]
        extra = {k: v[1].reshape(-1, v[2]) for k, v in out.items() if k != "greedy"}
        i = S["i"]
        np.savez(f"{out_dir}/members/{i}.npz", seed=members[i][0], sigma=members[i][1], answers=np.array([r[0] for r in res], dtype=object),
                 correct=ok, truncated=S["trunc"], n_select=nS, **extra)
        return ok

    for e in range(E):
        if queue_:
            start(e, queue_.pop(0))
    while active or scoring:
        progressed = False
        for e in list(active):
            i, futs = active[e]
            ready, _ = ray.wait(list(futs.values()), num_returns=len(futs), timeout=0)
            if len(ready) == len(futs):
                res = {k: ray.get(v) for k, v in futs.items()}
                scoring.append(submit_scoring(i, res)); del res
                del active[e]; progressed = True
                if queue_:
                    start(e, queue_.pop(0))
        for S in list(scoring):
            ready, _ = ray.wait([r for _, _, r in S["refs"]], num_returns=len(S["refs"]), timeout=0)
            if len(ready) == len(S["refs"]):
                finish_scoring(S); scoring.remove(S); n_saved += 1; progressed = True
                if n_saved % 40 == 0 or n_saved == len(todo):
                    print(f"[members] {n_saved}/{len(todo)} saved, {len(active)} generating, {len(queue_)} queued ({(time.time()-t_all)/60:.1f} min)", flush=True)
        if not progressed:
            time.sleep(2)
        if time.time() - t_last > 600:
            t_last = time.time(); print(f"[members] heartbeat: {n_saved}/{len(todo)} saved, {len(active)} generating, {len(scoring)} scoring ({(time.time()-t_all)/60:.1f} min)", flush=True)
    acc = [np.load(f"{out_dir}/members/{i}.npz", allow_pickle=True)["correct"] for i in todo]
    if acc:
        print(f"[members] done: mean greedy select={np.mean([c[:nS].mean() for c in acc]):.4f} test={np.mean([c[nS:].mean() for c in acc]):.4f} (total {(time.time()-t_all)/60:.1f} min)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
