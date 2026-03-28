"""
TurboQuant GPU Benchmark via RunPod

Spins up a GPU pod, runs the benchmark, prints results, and terminates the pod.
No lingering workers, no surprise charges.

Usage:
    python -m benchmarks.gpu                          # SmolLM2 on A40
    python -m benchmarks.gpu --model llama-8b         # Llama-3.1-8B-Instruct on A40
    python -m benchmarks.gpu --model mistral-7b       # Mistral-7B-Instruct-v0.3 on A40
    python -m benchmarks.gpu --gpu a100               # A100 GPU
    python -m benchmarks.gpu --model llama-8b --gpu a100  # Llama-8B on A100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def load_dotenv():
    for env_path in [Path(__file__).parent.parent / ".env", Path(__file__).parent / ".env"]:
        if not env_path.exists():
            continue
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

MODEL_MAP = {
    "smollm": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "short": "SmolLM2-1.7B",
        "min_gpu_gb": 8,
    },
    "llama-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "Llama-3.1-8B",
        "min_gpu_gb": 24,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "Mistral-7B",
        "min_gpu_gb": 24,
    },
}

BENCHMARK_SCRIPT = r'''
import time, math, gc, json, sys, os
import torch
import numpy as np
from scipy.stats import norm

MODEL = os.environ.get("BENCH_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
print(f"=== TurboQuant GPU Benchmark (CUDA) ===", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"CUDA: {torch.version.cuda}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"Model: {MODEL}", flush=True)


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
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(x.shape), norms.squeeze(-1)

    @torch.no_grad()
    def dequantize(self, idx, norms):
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(idx.shape)


from transformers.cache_utils import DynamicCache, DynamicLayer

class TQLayer(DynamicLayer):
    def __init__(self, hd, bw, dev, num_outlier_ch=0, outlier_bw=0):
        super().__init__()
        self._bw = bw
        self._hd = hd
        self._outlier_ch = num_outlier_ch
        self._outlier_bw = outlier_bw

        regular_dim = hd - num_outlier_ch if num_outlier_ch > 0 and outlier_bw > bw else hd
        self._regular_dim = regular_dim
        self._outlier_dim = num_outlier_ch if num_outlier_ch > 0 and outlier_bw > bw else 0

        self._tq = TurboQuantMSE(bw, regular_dim, dev)
        self._tq_out = TurboQuantMSE(outlier_bw, num_outlier_ch, dev, rotation_seed=43) if self._outlier_dim > 0 else None

        self._key_data, self._val_data = [], []
        self._ck = self._cv = None
        self._channel_mask = None

    def lazy_initialization(self, ks, vs):
        self.dtype, self.device, self.is_initialized = ks.dtype, ks.device, True
        if self._tq_out is not None and self._channel_mask is None:
            rms = ks.float().pow(2).mean(dim=(0,1,2)).sqrt()
            _, top = rms.topk(min(self._outlier_ch, rms.shape[0]))
            self._channel_mask = torch.zeros(rms.shape[0], dtype=torch.bool, device=ks.device)
            self._channel_mask[top] = True

    def _quant(self, x):
        shape = x.shape
        if self._tq_out is not None and self._channel_mask is not None:
            reg_mask = ~self._channel_mask
            xf = x.float()
            r = xf[..., reg_mask].reshape(-1, self._regular_dim)
            rn = r.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            ri = self._tq.quantize(r / rn)[0]
            o = xf[..., self._channel_mask].reshape(-1, self._outlier_dim)
            on = o.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            oi = self._tq_out.quantize(o / on)[0]
            return {'ri': ri, 'rn': rn.squeeze(-1), 'oi': oi, 'on': on.squeeze(-1), 's': shape}
        flat = x.float().reshape(-1, self._hd)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        idx, _ = self._tq.quantize(flat / norms)
        return {'idx': idx, 'norms': norms.squeeze(-1), 's': shape}

    def _dequant_one(self, d):
        shape = d['s']
        if 'ri' in d:
            r_hat = self._tq.dequantize(d['ri'], d['rn']).reshape(shape[0], shape[1], shape[2], self._regular_dim)
            o_hat = self._tq_out.dequantize(d['oi'], d['on']).reshape(shape[0], shape[1], shape[2], self._outlier_dim)
            out = torch.zeros(shape, dtype=torch.float32, device=self.device)
            out[..., ~self._channel_mask] = r_hat
            out[..., self._channel_mask] = o_hat
            return out.to(self.dtype)
        return self._tq.dequantize(d['idx'], d['norms']).reshape(shape).to(self.dtype)

    def update(self, ks, vs, cache_kwargs=None):
        if not self.is_initialized: self.lazy_initialization(ks, vs)
        kd = self._quant(ks); self._key_data.append(kd)
        vd = self._quant(vs); self._val_data.append(vd)
        nk = self._dequant_one(kd)
        nv = self._dequant_one(vd)
        if self._ck is None:
            self._ck, self._cv = nk, nv
        else:
            self._ck = torch.cat([self._ck, nk], dim=-2)
            self._cv = torch.cat([self._cv, nv], dim=-2)
        return self._ck, self._cv

    def get_seq_length(self):
        return sum(d['s'][-2] for d in self._key_data) if self._key_data else 0
    def get_max_cache_shape(self): return -1
    def mem_bits(self):
        t = 0
        for d in self._key_data + self._val_data:
            if 'ri' in d:
                t += d['ri'].numel() * self._bw + d['oi'].numel() * self._outlier_bw
                t += (d['rn'].numel() + d['on'].numel()) * 32
            else:
                t += d['idx'].numel() * self._bw + d['norms'].numel() * 32
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
    def __init__(self, hd, bw, nl, dev, num_outlier_ch=0, outlier_bw=0):
        super().__init__()
        self.layers = [TQLayer(hd, bw, dev, num_outlier_ch, outlier_bw) for _ in range(nl)]
    def mem_bits(self):
        return sum(l.mem_bits() for l in self.layers)
    def eff_bits(self):
        layer = self.layers[0]
        if layer._outlier_dim > 0:
            hd = layer._hd
            return (layer._regular_dim * layer._bw + layer._outlier_dim * layer._outlier_bw) / hd
        return float(layer._bw)


def bl_kv_mem(cache):
    t = 0
    for l in cache.layers:
        if hasattr(l, 'keys') and hasattr(l.keys, 'numel'):
            k, v = l.keys, l.values
            if k.numel()>0: t += k.numel()*k.element_size()
            if v.numel()>0: t += v.numel()*v.element_size()
    return t


from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.environ.get("HF_TOKEN", None)
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
print(f"Model loaded: {nparams}B params, {nl} layers, {nh} KV heads, head_dim={hd}", flush=True)

results = {
    "gpu": gpu_name, "model": MODEL,
    "model_config": {"layers": nl, "heads": nh, "head_dim": hd, "params_B": nparams}
}

prompts = [
    ("Factual QA", "What is the capital of France? Answer in one sentence."),
    ("Reasoning", "If a train travels at 60 mph for 2.5 hours, how far does it go? Show your reasoning."),
    ("Code Generation", "Write a Python function to compute factorial recursively."),
    ("Long Generation", "Write a detailed explanation of how neural networks learn, covering backpropagation, gradient descent, learning rate, loss functions, and regularization techniques. Include concrete examples and explain the intuition behind each concept."),
]

def generate(msgs, cache=None, max_new_tokens=100):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    torch.cuda.synchronize(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        kw = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, return_dict_in_generate=True)
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

outlier_ch = 32 if hd == 128 else (16 if hd == 64 else 0)

configs = [
    ("baseline", "Baseline FP16", lambda: None),
    ("tq4", "TurboQuant 4-bit", lambda: TQCache(hd, 4, nl, device)),
    ("tq3.5", "TurboQuant 3.5-bit (outlier)", lambda: TQCache(hd, 3, nl, device, outlier_ch, 4)),
    ("tq3", "TurboQuant 3-bit", lambda: TQCache(hd, 3, nl, device)),
    ("tq2.5", "TurboQuant 2.5-bit (outlier)", lambda: TQCache(hd, 2, nl, device, outlier_ch, 3)),
]

for cfg_key, label, cfn in configs:
    print(f"\nRunning {label}...", flush=True)
    results[cfg_key] = []
    for name, content in prompts:
        try:
            max_tok = 512 if name == "Long Generation" else 100
            r = generate([{"role":"user","content":content}], cfn(), max_new_tokens=max_tok)
            r["prompt"] = name
            results[cfg_key].append(r)
            print(f"  {name}: {r['tps']} tok/s, KV={r['kv_memory']} bytes", flush=True)
        except Exception as e:
            print(f"  {name}: ERROR — {e}", flush=True)
            results[cfg_key].append({"prompt": name, "error": str(e)})
        gc.collect(); torch.cuda.empty_cache()


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

    for _ in range(WARMUP): _ = torch.matmul(Qf, Kf.transpose(-2,-1))
    t0e = torch.cuda.Event(enable_timing=True)
    t1e = torch.cuda.Event(enable_timing=True)
    t0e.record()
    for _ in range(ITERS): _ = torch.matmul(Qf, Kf.transpose(-2,-1))
    t1e.record(); torch.cuda.synchronize()
    baseline_ms = t0e.elapsed_time(t1e) / ITERS

    for _ in range(WARMUP):
        Kd = qa.dequantize(K_idx, K_norms).reshape(1,nh,sl,d).half()
        _ = torch.matmul(Qf, Kd.transpose(-2,-1))
    t0e.record()
    for _ in range(ITERS):
        Kd = qa.dequantize(K_idx, K_norms).reshape(1,nh,sl,d).half()
        _ = torch.matmul(Qf, Kd.transpose(-2,-1))
    t1e.record(); torch.cuda.synchronize()
    dequant_ms = t0e.elapsed_time(t1e) / ITERS

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
    del Qf, Kf, K_idx, K_norms; gc.collect(); torch.cuda.empty_cache()

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
        "containerDiskInGb": 30,
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


def run_benchmark(ssh_addr: str, hf_token: str, model_hf_id: str):
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
                "-o", "LogLevel=ERROR", "-o", "IdentitiesOnly=yes", "-i", key_file,
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
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

    env_parts = [f"BENCH_MODEL={model_hf_id}"]
    if hf_token:
        env_parts.append(f"HF_TOKEN={hf_token}")
    env_prefix = " ".join(env_parts) + " "

    print("  Running benchmark (this takes 5-30 minutes for 7-8B models)...")
    print("  " + "=" * 60)

    subprocess.run(ssh_base + [
        f"rm -f /workspace/results.json /workspace/bench.done && "
        f"nohup bash -c '{env_prefix}python /workspace/bench.py > /workspace/bench.log 2>&1; "
        f"echo $? > /workspace/bench.done' &"
    ], capture_output=True)

    poll_interval = 30
    max_wait = 1800
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        check = subprocess.run(ssh_base + ["cat /workspace/bench.done 2>/dev/null"],
                               capture_output=True, text=True)
        if check.returncode == 0 and check.stdout.strip():
            exit_code = check.stdout.strip()
            print(f"  Benchmark finished (exit code {exit_code}) after ~{waited}s")
            break
        if waited % 60 == 0:
            tail = subprocess.run(ssh_base + ["tail -3 /workspace/bench.log 2>/dev/null"],
                                  capture_output=True, text=True)
            if tail.stdout.strip():
                for line in tail.stdout.strip().splitlines():
                    print(f"  [remote] {line}")
    else:
        print(f"  WARNING: Benchmark did not finish within {max_wait}s")

    log = subprocess.run(ssh_base + ["cat /workspace/bench.log 2>/dev/null"],
                         capture_output=True, text=True)
    if log.stdout.strip():
        print(log.stdout)

    print("  Downloading results...")
    results_local = Path(__file__).parent.parent / "results" / "benchmark_results.json"
    results_local.parent.mkdir(parents=True, exist_ok=True)
    scp_cmd = scp_base + [f"root@{host}:/workspace/results.json", str(results_local)]
    subprocess.run(scp_cmd, capture_output=True)

    if results_local.exists():
        return json.loads(results_local.read_text())
    return None


def print_results(results: dict):
    print(f"\n{'=' * 95}")
    print(f"  TurboQuant GPU Benchmark Results")
    print(f"  GPU: {results['gpu']}")
    print(f"  Model: {results['model']} ({results['model_config']['params_B']}B params)")
    cfg = results['model_config']
    print(f"  Config: {cfg['layers']} layers, {cfg['heads']} KV heads, head_dim={cfg['head_dim']}")
    print(f"{'=' * 95}\n")

    def avg(k, f):
        items = [r for r in results.get(k, []) if "error" not in r]
        return sum(r[f] for r in items) / len(items) if items else 0

    def fmt(b):
        if b < 1024: return f"{b} B"
        if b < 1024**2: return f"{b/1024:.1f} KB"
        return f"{b/1024**2:.2f} MB"

    config_keys = [k for k in ["baseline", "tq4", "tq3.5", "tq3", "tq2.5"] if k in results]

    col = 18
    labels = {"baseline": "Baseline FP16", "tq4": "TQ 4-bit", "tq3.5": "TQ 3.5-bit",
              "tq3": "TQ 3-bit", "tq2.5": "TQ 2.5-bit"}
    hdr = f"{'Metric':<22} | " + " | ".join(f"{labels.get(k,k):>{col}}" for k in config_keys)
    print(hdr)
    print("-" * len(hdr))

    bm = avg("baseline", "kv_memory")
    mem_row = f"{'Avg KV Cache Memory':<22} | "
    mem_row += " | ".join(f"{fmt(avg(k, 'kv_memory')):>{col}}" for k in config_keys)
    print(mem_row)

    if bm > 0:
        ratio_row = f"{'Compression Ratio':<22} | "
        for k in config_keys:
            m = avg(k, "kv_memory")
            r = bm / m if m > 0 else float('inf')
            ratio_row += f"{f'{r:.1f}x':>{col}} | "
        print(ratio_row.rstrip(" | "))

    tps_row = f"{'Avg Tokens/sec':<22} | "
    tps_row += " | ".join(f"{avg(k, 'tps'):>{col}.1f}" for k in config_keys)
    print(tps_row)

    peak_row = f"{'Avg Peak GPU (MB)':<22} | "
    peak_row += " | ".join(f"{avg(k, 'peak_gpu_mb'):>{col}.0f}" for k in config_keys)
    print(peak_row)

    print(f"\n{'=' * 95}")
    print(f"  GENERATION COMPARISON")
    print(f"{'=' * 95}")
    baseline_items = results.get("baseline", [])
    for i, bl in enumerate(baseline_items):
        if "error" in bl:
            continue
        print(f"\n--- {bl['prompt']} ---")
        for key in config_keys:
            items = results.get(key, [])
            if i < len(items) and "error" not in items[i]:
                t = items[i]["text"].strip()[:200]
                print(f"  [{labels.get(key,key)}] ({items[i]['tokens']} tok, {items[i]['tps']} tok/s): {t}")

    if "attention_speedup" in results:
        print(f"\n{'=' * 95}")
        print(f"  QUANTIZED ATTENTION SPEEDUP (GPU)")
        print(f"{'=' * 95}")
        hdr = f"  {'SeqLen':>8} | {'Baseline':>10} | {'Dequant+MM':>12} | {'Quant Attn':>12} | {'vs Base':>8} | {'vs Dequant':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for sl, d in sorted(results["attention_speedup"].items(), key=lambda x: int(x[0])):
            bms = d.get('baseline_ms', 0)
            dms = d.get('dequant_then_matmul_ms', 0)
            qms = d.get('quantized_attn_ms', dms)
            svb = d.get('speedup_vs_baseline', 0)
            svd = d.get('speedup_vs_dequant', 0)
            print(f"  {sl:>8} | {bms:>9.3f}ms | {dms:>11.3f}ms | {qms:>11.3f}ms | {svb:>7.2f}x | {svd:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant GPU Benchmark (RunPod)")
    parser.add_argument("--gpu", default="a40", choices=list(GPU_MAP.keys()),
                        help="GPU type (default: a40)")
    parser.add_argument("--model", default="smollm", choices=list(MODEL_MAP.keys()),
                        help="Model to benchmark (default: smollm)")
    args = parser.parse_args()

    gpu_type = GPU_MAP[args.gpu]
    model_info = MODEL_MAP[args.model]

    if not os.environ.get("RUNPOD_API_KEY"):
        print("Error: RUNPOD_API_KEY not found.")
        print("  1. cp .env.example .env")
        print("  2. Add your key from https://www.runpod.io/console/user/settings")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"{'=' * 60}")
    print(f"  TurboQuant GPU Benchmark")
    print(f"  GPU: {args.gpu} ({gpu_type})")
    print(f"  Model: {model_info['short']} ({model_info['hf_id']})")
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

        r = run_benchmark(ssh_addr, hf_token, model_info["hf_id"])
        if r:
            print_results(r)

            model_slug = model_info["short"].lower().replace(".", "").replace("-", "_")
            gpu_slug = args.gpu.replace("-", "_")
            results_dir = Path(__file__).parent.parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            out_path = results_dir / f"{gpu_slug}_{model_slug}.json"
            out_path.write_text(json.dumps(r, indent=2, default=str))
            print(f"\n  Results saved to {out_path.relative_to(Path(__file__).parent.parent)}")
        else:
            print("  No results returned. Check logs above.")

    finally:
        if pod_id:
            terminate_pod(pod_id)
            print("\n  Pod terminated. No lingering charges.")


if __name__ == "__main__":
    main()
