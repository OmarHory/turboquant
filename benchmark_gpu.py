"""
TurboQuant GPU Benchmark via RunPod

Spins up a GPU pod, runs the benchmark, prints results, and terminates the pod.
No lingering workers, no surprise charges.

Usage:
    # Add your keys to .env (copy from .env.example)
    cp .env.example .env

    # Run (defaults to RTX 4090, cheapest option)
    python benchmark_gpu.py

    # Or pick a GPU
    python benchmark_gpu.py --gpu a100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

API_BASE = "https://rest.runpod.io/v1"

GPU_MAP = {
    "4090": "NVIDIA GeForce RTX 4090",
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "l4": "NVIDIA L4",
    "a40": "NVIDIA A40",
}

BENCHMARK_SCRIPT = r'''
import time, math, gc, json, sys, os
import torch
import numpy as np
from scipy.stats import norm

print("=== TurboQuant GPU Benchmark (CUDA) ===", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"CUDA: {torch.version.cuda}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)

# ── TurboQuant Core (GPU-optimized) ──────────────────────────────────
# All matmul ops run as CUDA kernels via PyTorch.
# Key optimization: cache dequantized history, only dequantize new tokens.

def _lloyd_max_gaussian(num_levels, sigma=1.0, max_iter=200):
    k = num_levels
    centroids = np.array([sigma * norm.ppf((2*i+1)/(2*k)) for i in range(k)])
    for _ in range(max_iter):
        boundaries = np.empty(k+1)
        boundaries[0], boundaries[k] = -np.inf, np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
        new_c = np.empty(k)
        for i in range(k):
            lo, hi = boundaries[i], boundaries[i+1]
            lo_c, hi_c = max(lo, -6*sigma), min(hi, 6*sigma)
            num = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
            den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
            new_c[i] = num/den if den > 1e-15 else (lo_c+hi_c)/2.0
        if np.allclose(centroids, new_c, atol=1e-12):
            break
        centroids = new_c
    boundaries = np.empty(k+1)
    boundaries[0], boundaries[k] = -np.inf, np.inf
    for i in range(1, k):
        boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
    return centroids, boundaries


class TurboQuantMSE:
    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        d = head_dim
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R)); ds[ds==0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()
        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2**bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()
        self.head_dim = head_dim

    @torch.no_grad()
    def quantize(self, x):
        """x: (..., D) raw vectors. Returns (idx, norms)."""
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(x.shape), norms.squeeze(-1)

    @torch.no_grad()
    def dequantize(self, idx, norms):
        """idx: (..., D) uint8 indices, norms: (...) float32. Returns (..., D) float32."""
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(idx.shape)


# ── KV Cache (optimized: incremental dequantize) ─────────────────────

from transformers.cache_utils import DynamicCache, DynamicLayer

class TQLayer(DynamicLayer):
    def __init__(self, hd, bw, dev):
        super().__init__()
        self._tq = TurboQuantMSE(bw, hd, dev)
        self._hd, self._bw = hd, bw
        self._idx_k, self._norms_k, self._shapes_k = [], [], []
        self._idx_v, self._norms_v, self._shapes_v = [], [], []
        self._ck = self._cv = None

    def lazy_initialization(self, ks, vs):
        self.dtype, self.device, self.is_initialized = ks.dtype, ks.device, True

    def _dq_one(self, idx, norms, shape):
        return self._tq.dequantize(idx, norms).reshape(shape).to(self.dtype)

    def update(self, ks, vs, cache_kwargs=None):
        if not self.is_initialized: self.lazy_initialization(ks, vs)

        ki, kn = self._tq.quantize(ks)
        self._idx_k.append(ki); self._norms_k.append(kn); self._shapes_k.append(ks.shape)
        vi, vn = self._tq.quantize(vs)
        self._idx_v.append(vi); self._norms_v.append(vn); self._shapes_v.append(vs.shape)

        new_k = self._dq_one(ki, kn, ks.shape)
        new_v = self._dq_one(vi, vn, vs.shape)

        if self._ck is None:
            self._ck, self._cv = new_k, new_v
        else:
            self._ck = torch.cat([self._ck, new_k], dim=-2)
            self._cv = torch.cat([self._cv, new_v], dim=-2)
        return self._ck, self._cv

    def get_seq_length(self):
        return sum(s[-2] for s in self._shapes_k) if self._shapes_k else 0
    def get_max_cache_shape(self): return -1
    def mem_bits(self):
        t = 0
        for idx, norms in zip(self._idx_k + self._idx_v, self._norms_k + self._norms_v):
            t += idx.numel() * self._bw + norms.numel() * 32
        return t

    @property
    def keys(self): return self._ck if self._ck is not None else torch.tensor([])
    @keys.setter
    def keys(self, v): pass
    @property
    def values(self): return self._cv if self._cv is not None else torch.tensor([])
    @values.setter
    def values(self, v): pass


class TQCache(DynamicCache):
    def __init__(self, hd, bw, nl, dev):
        super().__init__()
        self.layers = [TQLayer(hd, bw, dev) for _ in range(nl)]
    def mem_bits(self):
        return sum(l.mem_bits() for l in self.layers)


def bl_kv_mem(cache):
    t = 0
    for l in cache.layers:
        if hasattr(l, 'keys') and hasattr(l.keys, 'numel'):
            k, v = l.keys, l.values
            if k.numel()>0: t += k.numel()*k.element_size()
            if v.numel()>0: t += v.numel()*v.element_size()
    return t


# ── Benchmark ────────────────────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.environ.get("HF_TOKEN", None)
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)

print(f"\nLoading {MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True, token=hf_token
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hd = model.config.hidden_size // model.config.num_attention_heads
nl = model.config.num_hidden_layers
nh = model.config.num_key_value_heads
nparams = round(sum(p.numel() for p in model.parameters())/1e9, 2)
print(f"Model loaded: {nparams}B params, {nl} layers, {nh} heads, head_dim={hd}", flush=True)

results = {
    "gpu": gpu_name, "model": MODEL,
    "model_config": {"layers": nl, "heads": nh, "head_dim": hd, "params_B": nparams}
}

prompts = [
    ("Factual QA", "What is the capital of France? Answer in one sentence."),
    ("Reasoning", "If a train travels at 60 mph for 2.5 hours, how far does it go?"),
    ("Code Generation", "Write a Python function to compute factorial recursively."),
]

def generate(msgs, cache=None):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    torch.cuda.synchronize(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        kw = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  max_new_tokens=60, do_sample=False, use_cache=True, return_dict_in_generate=True)
        if cache is not None: kw["past_key_values"] = cache
        out = model.generate(**kw)
    torch.cuda.synchronize()
    dt = time.time() - t0
    gi = out.sequences[0][inputs["input_ids"].shape[1]:]
    kv = bl_kv_mem(out.past_key_values) if cache is None else (out.past_key_values.mem_bits()//8)
    return {
        "text": tokenizer.decode(gi, skip_special_tokens=True),
        "tokens": int(gi.shape[0]), "time": round(dt,3),
        "tps": round(int(gi.shape[0])/dt, 1) if dt>0 else 0,
        "kv_memory": kv,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated()/1e6, 1),
    }

for cfg, label, cfn in [
    ("baseline", "Baseline FP16", lambda: None),
    ("tq4", "TurboQuant 4-bit", lambda: TQCache(hd,4,nl,device)),
    ("tq3", "TurboQuant 3-bit", lambda: TQCache(hd,3,nl,device)),
]:
    print(f"\nRunning {label}...", flush=True)
    results[cfg] = []
    for name, content in prompts:
        r = generate([{"role":"user","content":content}], cfn())
        r["prompt"] = name
        results[cfg].append(r)
        print(f"  {name}: {r['tps']} tok/s, {r['tokens']} tokens", flush=True)
        gc.collect(); torch.cuda.empty_cache()

# ── Quantized Attention (skip full dequantize for Q @ K^T) ────────────
#
# Standard:    K_hat = centroids[idx] @ Pi  ->  Q @ K_hat^T
# Quantized:   Q_rot = Q @ Pi^T  ->  Q_rot @ centroids[idx]^T * norms
# The win: reads uint8 indices from HBM instead of float16 keys.

class QuantizedAttention:
    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device
        self.scale = 1.0 / math.sqrt(head_dim)
        d = head_dim
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R)); ds[ds==0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device)
        self.Pi_T = self.Pi.T.contiguous()
        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2**bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device)
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device)

    @torch.no_grad()
    def quantize_keys(self, K):
        flat = K.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi_T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(K.shape), norms.squeeze(-1).view(K.shape[:-1])

    @torch.no_grad()
    def dequantize(self, idx, norms):
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(idx.shape)

    @torch.no_grad()
    def quantized_attention_scores(self, Qf, K_idx, K_norms, dtype=torch.float16):
        Q_rot = (Qf.float() @ self.Pi_T).to(dtype)
        C_K = self.centroids[K_idx.long()].to(dtype)
        raw = torch.matmul(Q_rot, C_K.transpose(-2, -1))
        return raw * K_norms.unsqueeze(-2).to(dtype) * self.scale

print("\n=== Quantized Attention Speedup Benchmark ===", flush=True)
WARMUP, ITERS = 20, 500
d = hd
speedup = {}

for sl in [512, 1024, 2048, 4096, 8192, 16384]:
    Qf = torch.randn(1, nh, 1, d, dtype=torch.float16, device=device)
    Kf = torch.randn(1, nh, sl, d, dtype=torch.float16, device=device)

    qa = QuantizedAttention(4, d, device)
    K_idx, K_norms = qa.quantize_keys(Kf)

    # Baseline: standard Q @ K^T
    for _ in range(WARMUP): _ = torch.matmul(Qf, Kf.transpose(-2,-1))
    t0e = torch.cuda.Event(enable_timing=True)
    t1e = torch.cuda.Event(enable_timing=True)
    t0e.record()
    for _ in range(ITERS): _ = torch.matmul(Qf, Kf.transpose(-2,-1))
    t1e.record(); torch.cuda.synchronize()
    baseline_ms = t0e.elapsed_time(t1e) / ITERS

    # Old: dequantize then matmul
    for _ in range(WARMUP):
        Kd = qa.dequantize(K_idx, K_norms).reshape(1,nh,sl,d).half()
        _ = torch.matmul(Qf, Kd.transpose(-2,-1))
    t0e.record()
    for _ in range(ITERS):
        Kd = qa.dequantize(K_idx, K_norms).reshape(1,nh,sl,d).half()
        _ = torch.matmul(Qf, Kd.transpose(-2,-1))
    t1e.record(); torch.cuda.synchronize()
    dequant_ms = t0e.elapsed_time(t1e) / ITERS

    # New: quantized attention (no full dequantize)
    for _ in range(WARMUP): _ = qa.quantized_attention_scores(Qf, K_idx, K_norms)
    t0e.record()
    for _ in range(ITERS): _ = qa.quantized_attention_scores(Qf, K_idx, K_norms)
    t1e.record(); torch.cuda.synchronize()
    qattn_ms = t0e.elapsed_time(t1e) / ITERS

    entry = {
        "baseline_ms": round(baseline_ms, 4),
        "dequant_then_matmul_ms": round(dequant_ms, 4),
        "quantized_attn_ms": round(qattn_ms, 4),
        "speedup_vs_baseline": round(baseline_ms / qattn_ms, 2) if qattn_ms > 0 else 0,
        "speedup_vs_dequant": round(dequant_ms / qattn_ms, 2) if qattn_ms > 0 else 0,
    }
    speedup[sl] = entry
    print(f"  seq_len={sl}: baseline={baseline_ms:.3f}ms  dequant+mm={dequant_ms:.3f}ms  "
          f"quant_attn={qattn_ms:.3f}ms  speedup_vs_base={entry['speedup_vs_baseline']:.2f}x  "
          f"speedup_vs_dequant={entry['speedup_vs_dequant']:.2f}x", flush=True)

results["attention_speedup"] = speedup

with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n=== RESULTS_JSON_START ===", flush=True)
print(json.dumps(results, indent=2, default=str), flush=True)
print("=== RESULTS_JSON_END ===", flush=True)
print("BENCHMARK_COMPLETE", flush=True)
'''


def api(method, path, data=None):
    key = os.environ["RUNPOD_API_KEY"]
    url = f"{API_BASE}{path}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = getattr(requests, method)(url, headers=headers, json=data, timeout=30)
    if r.status_code >= 400:
        print(f"API error {r.status_code}: {r.text}")
        sys.exit(1)
    return r


def get_ssh_pubkey():
    for name in ["id_rsa.pub", "id_ed25519.pub", "id_ecdsa.pub"]:
        p = Path.home() / ".ssh" / name
        if p.exists():
            return p.read_text().strip()
    return None


def create_pod(gpu_type: str, hf_token: str):
    pubkey = get_ssh_pubkey()
    if not pubkey:
        print("ERROR: No SSH public key found in ~/.ssh/")
        print("  Run: ssh-keygen -t ed25519")
        sys.exit(1)

    data = {
        "name": "tq-bench",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeIds": [gpu_type],
        "gpuCount": 1,
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pubkey},
    }
    if hf_token:
        data["env"]["HF_TOKEN"] = hf_token
    print(f"  Creating pod with {gpu_type}...")
    r = api("post", "/pods", data)
    pod = r.json()
    pod_id = pod.get("id")
    print(f"  Pod created: {pod_id}")
    return pod_id


def wait_for_pod(pod_id: str, timeout=300):
    print(f"  Waiting for pod to be ready (timeout {timeout}s)...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        r = api("get", f"/pods/{pod_id}")
        pod = r.json()

        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings", {})

        if public_ip and port_mappings and "22" in port_mappings:
            ssh_addr = f"{public_ip}:{port_mappings['22']}"
            print(f" ready! ({int(time.time()-start)}s)")
            print(f"  SSH: {ssh_addr}")
            return ssh_addr

        print(".", end="", flush=True)
        time.sleep(5)
    print(" TIMEOUT!")
    return None


def terminate_pod(pod_id: str):
    print(f"  Terminating pod {pod_id}...")
    try:
        api("delete", f"/pods/{pod_id}")
        print("  Pod terminated.")
    except Exception as e:
        print(f"  Warning: could not terminate pod: {e}")
        print(f"  MANUALLY TERMINATE at: https://www.runpod.io/console/pods")


def run_benchmark(ssh_addr: str, hf_token: str):
    """Upload and run the benchmark script via SSH."""
    import subprocess
    import base64

    host, port = ssh_addr.rsplit(":", 1)

    key_file = None
    for name in ["id_rsa", "id_ed25519", "id_ecdsa"]:
        p = Path.home() / ".ssh" / name
        if p.exists():
            key_file = str(p)
            break

    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "IdentitiesOnly=yes", "-i", key_file]
    ssh_base = ["ssh"] + ssh_opts + ["-p", port, f"root@{host}"]
    scp_base = ["scp"] + ssh_opts + ["-P", port]

    print("  Uploading benchmark script...")
    encoded = base64.b64encode(BENCHMARK_SCRIPT.encode()).decode()
    upload_cmd = ssh_base + [f"echo '{encoded}' | base64 -d > /workspace/bench.py"]
    ret = subprocess.run(upload_cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  Upload error: {ret.stderr}")
        return None
    print("  Upload complete.")

    print("  Installing dependencies...")
    install = ssh_base + ["pip install -q transformers accelerate sentencepiece protobuf scipy numpy 2>&1 | tail -3"]
    subprocess.run(install)

    env_prefix = f"HF_TOKEN={hf_token} " if hf_token else ""
    print("  Running benchmark (this takes 2-4 minutes)...")
    print("  " + "=" * 60)
    run_cmd = ssh_base + [f"{env_prefix}python /workspace/bench.py"]
    ret = subprocess.run(run_cmd)

    if ret.returncode != 0:
        print("  Benchmark script returned non-zero exit code.")

    print("  Downloading results...")
    results_local = Path(__file__).parent / "benchmark_results.json"
    scp_cmd = scp_base + [f"root@{host}:/workspace/results.json", str(results_local)]
    subprocess.run(scp_cmd, capture_output=True)

    if results_local.exists():
        return json.loads(results_local.read_text())
    return None


def print_results(results: dict):
    print(f"\n{'=' * 85}")
    print(f"  TurboQuant GPU Benchmark Results")
    print(f"  GPU: {results['gpu']}")
    print(f"  Model: {results['model']} ({results['model_config']['params_B']}B params)")
    cfg = results['model_config']
    print(f"  Config: {cfg['layers']} layers, {cfg['heads']} heads, head_dim={cfg['head_dim']}")
    print(f"{'=' * 85}\n")

    def avg(k, f):
        items = results.get(k, [])
        return sum(r[f] for r in items) / len(items) if items else 0

    def fmt(b):
        if b < 1024: return f"{b} B"
        if b < 1024**2: return f"{b/1024:.1f} KB"
        return f"{b/1024**2:.2f} MB"

    col = 22
    hdr = f"{'Metric':<28} | {'Baseline (FP16)':>{col}} | {'TurboQuant 4-bit':>{col}} | {'TurboQuant 3-bit':>{col}}"
    print(hdr)
    print("-" * len(hdr))

    bm, t4, t3 = avg("baseline","kv_memory"), avg("tq4","kv_memory"), avg("tq3","kv_memory")
    print(f"{'Avg KV Cache Memory':<28} | {fmt(bm):>{col}} | {fmt(t4):>{col}} | {fmt(t3):>{col}}")
    if bm > 0:
        r4 = bm/t4 if t4>0 else float('inf')
        r3 = bm/t3 if t3>0 else float('inf')
        print(f"{'Compression Ratio':<28} | {'1.0x':>{col}} | {f'{r4:.1f}x':>{col}} | {f'{r3:.1f}x':>{col}}")
    print(f"{'Avg Tokens/sec':<28} | {avg('baseline','tps'):>{col}.1f} | {avg('tq4','tps'):>{col}.1f} | {avg('tq3','tps'):>{col}.1f}")
    print(f"{'Avg Gen Time (s)':<28} | {avg('baseline','time'):>{col}.2f} | {avg('tq4','time'):>{col}.2f} | {avg('tq3','time'):>{col}.2f}")
    print(f"{'Avg Peak GPU (MB)':<28} | {avg('baseline','peak_gpu_mb'):>{col}.0f} | {avg('tq4','peak_gpu_mb'):>{col}.0f} | {avg('tq3','peak_gpu_mb'):>{col}.0f}")

    print(f"\n{'=' * 85}")
    print(f"  GENERATION COMPARISON")
    print(f"{'=' * 85}")
    for i, bl in enumerate(results.get("baseline", [])):
        print(f"\n--- {bl['prompt']} ---")
        for key, label in [("baseline","Baseline FP16"),("tq4","TQ 4-bit"),("tq3","TQ 3-bit")]:
            items = results.get(key, [])
            if i < len(items):
                t = items[i]["text"].strip()[:200]
                print(f"  [{label}] ({items[i]['tokens']} tok, {items[i]['tps']} tok/s): {t}")

    if "attention_speedup" in results:
        print(f"\n{'=' * 85}")
        print(f"  QUANTIZED ATTENTION SPEEDUP (GPU)")
        print(f"{'=' * 85}")
        hdr = f"  {'SeqLen':>8} | {'Baseline':>10} | {'Dequant+MM':>12} | {'Quant Attn':>12} | {'vs Base':>8} | {'vs Dequant':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for sl, d in sorted(results["attention_speedup"].items(), key=lambda x: int(x[0])):
            bms = d.get('baseline_ms', d.get('baseline_ms', 0))
            dms = d.get('dequant_then_matmul_ms', d.get('tq4_ms', 0))
            qms = d.get('quantized_attn_ms', dms)
            svb = d.get('speedup_vs_baseline', d.get('ratio', 0))
            svd = d.get('speedup_vs_dequant', 0)
            print(f"  {sl:>8} | {bms:>9.3f}ms | {dms:>11.3f}ms | {qms:>11.3f}ms | {svb:>7.2f}x | {svd:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant GPU Benchmark (RunPod)")
    parser.add_argument("--gpu", default="4090", choices=list(GPU_MAP.keys()),
                        help="GPU type (default: 4090)")
    args = parser.parse_args()
    gpu_type = GPU_MAP[args.gpu]

    if not os.environ.get("RUNPOD_API_KEY"):
        print("Error: RUNPOD_API_KEY not found.")
        print("  1. cp .env.example .env")
        print("  2. Add your key from https://www.runpod.io/console/user/settings")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"{'=' * 60}")
    print(f"  TurboQuant GPU Benchmark")
    print(f"  GPU: {args.gpu} ({gpu_type})")
    print(f"  Model: SmolLM2-1.7B-Instruct")
    print(f"{'=' * 60}\n")

    pod_id = None
    try:
        pod_id = create_pod(gpu_type, hf_token)
        ssh_addr = wait_for_pod(pod_id)
        if not ssh_addr:
            print("ERROR: Pod never became ready. Terminating.")
            terminate_pod(pod_id)
            sys.exit(1)

        time.sleep(10)

        results = run_benchmark(ssh_addr, hf_token)
        if results:
            print_results(results)
            print(f"\n  Results saved to benchmark_results.json")
        else:
            print("  No results returned. Check logs above.")

    finally:
        if pod_id:
            terminate_pod(pod_id)
            print("\n  Pod terminated. No lingering charges.")

    # Cleanup temp file
    tmp = Path(__file__).parent / "_gpu_bench_remote.py"
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
