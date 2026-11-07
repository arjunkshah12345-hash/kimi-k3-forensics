"""Kimi K3 weight forensics: weight-only experiments on a 2.8T open-weight MoE.

Pulls individual tensors out of moonshotai/Kimi-K3 via HTTP range requests
(no full 1.56 TB download) and runs four experiments:

  E1  under-trained ("glitch") token detection from embeddings + unembeddings
  E2  router atlas: aux-loss-free balancing biases as a fossil record of
      training-time load imbalance, embedding-level routing, router geometry
  E3  Kimi Delta Attention memory horizons from A_log / dt_bias
  E4  run every routed expert of three layers on real token probes:
      functional redundancy between experts + expert "logit lens"
"""
import base64
import collections
import concurrent.futures as cf
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import requests
import torch

T0 = time.time()
TIME_BUDGET = 9.5 * 3600
REPO = "moonshotai/Kimi-K3"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LM = "language_model.model."
EPS = 1e-5
RESULTS: dict = {"device": DEV, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "k3-forensics/0.1"
_lock = threading.Lock()
BYTES = [0]


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def save_results():
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)


# ----------------------------------------------------------------- fetching
_cdn: dict = {}


def cdn_url(shard, refresh=False):
    with _lock:
        if shard in _cdn and not refresh:
            return _cdn[shard]
    r = SESSION.head(BASE + shard, allow_redirects=False, timeout=60)
    url = r.headers.get("Location", BASE + shard) if r.status_code in (301, 302, 303, 307, 308) else BASE + shard
    if url.startswith("/"):
        url = "https://huggingface.co" + url
    with _lock:
        _cdn[shard] = url
    return url


def get_range(shard, start, end):
    """Inclusive byte range [start, end]."""
    for attempt in range(8):
        try:
            r = SESSION.get(cdn_url(shard, refresh=attempt > 0),
                            headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                with _lock:
                    BYTES[0] += len(r.content)
                return r.content
            log("range retry", shard, r.status_code, len(r.content))
        except Exception as e:  # noqa: BLE001
            log("range error", shard, repr(e)[:200])
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed range {shard} {start}-{end}")


INDEX = json.loads(SESSION.get(BASE + "model.safetensors.index.json", timeout=120).content)["weight_map"]
CONFIG = json.loads(SESSION.get(BASE + "config.json", timeout=120).content)
TC = CONFIG["text_config"]
_headers: dict = {}


def header(shard):
    if shard not in _headers:
        n = struct.unpack("<Q", get_range(shard, 0, 7))[0]
        _headers[shard] = (8 + n, json.loads(get_range(shard, 8, 7 + n)))
    return _headers[shard]


DT = {"BF16": (np.uint16, torch.bfloat16), "F32": (np.float32, torch.float32),
      "U8": (np.uint8, torch.uint8), "F16": (np.float16, torch.float16)}


def meta(name):
    shard = INDEX[name]
    base, h = header(shard)
    m = h[name]
    return shard, base + m["data_offsets"][0], base + m["data_offsets"][1], m["dtype"], m["shape"]


def to_tensor(buf, dtype, shape):
    npdt, tdt = DT[dtype]
    arr = np.frombuffer(buf, dtype=npdt).reshape(shape)
    t = torch.from_numpy(arr.copy())
    if dtype == "BF16":
        t = t.view(torch.int16).view(torch.bfloat16)
    return t


def fetch(name, chunk=64 << 20, workers=16):
    shard, s, e, dtype, shape = meta(name)
    if e - s <= chunk:
        return to_tensor(get_range(shard, s, e - 1), dtype, shape)
    buf = bytearray(e - s)
    spans = [(o, min(o + chunk, e)) for o in range(s, e, chunk)]

    def one(span):
        a, b = span
        buf[a - s:b - s] = get_range(shard, a, b - 1)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, spans))
    return to_tensor(bytes(buf), dtype, shape)


# ----------------------------------------------------------------- tokenizer
def load_vocab():
    raw = SESSION.get(BASE + "tiktoken.model", timeout=120).content.decode()
    vocab = {}
    for line in raw.splitlines():
        if line.strip():
            b64, rank = line.split()
            vocab[int(rank)] = base64.b64decode(b64)
    tcfg = json.loads(SESSION.get(BASE + "tokenizer_config.json", timeout=120).content)
    named = {int(k): v["content"] for k, v in tcfg.get("added_tokens_decoder", {}).items()}
    return vocab, named


VOCAB, NAMED = load_vocab()
N_BPE = len(VOCAB)
V = TC["vocab_size"]


def tok_str(i):
    if i in VOCAB:
        return VOCAB[i].decode("utf-8", errors="backslashreplace")
    return NAMED.get(i, f"<|reserved_token_{i}|>")


RESERVED = [i for i in range(N_BPE, V) if i not in NAMED]
log(f"vocab bpe={N_BPE} total={V} named_special={len(NAMED)} reserved={len(RESERVED)}")


def rmsnorm(x, w=None):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return x * w.float() if w is not None else x


# =================================================================== E1
def e1_glitch():
    log("E1: fetching embed_tokens + lm_head")
    out = {}
    mats = {}
    for key, name in (("embed", "language_model.model.embed_tokens.weight"),
                      ("unembed", "language_model.lm_head.weight")):
        M = fetch(name).to(DEV).float()
        norms = M.norm(dim=1)
        mu = M.mean(0)
        ref = torch.tensor(RESERVED, device=DEV)
        d = M[ref].mean(0) - mu
        d = d / d.norm()
        score = M @ d - mu @ d  # projection on "untrained direction"
        ref_score = score[ref]
        thr = torch.quantile(ref_score, 0.05).item()
        bpe_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
        bpe_mask[:N_BPE] = True
        cand = torch.nonzero(bpe_mask & (score >= thr)).flatten()
        out[key] = {
            "norm_mean": norms[:N_BPE].mean().item(), "norm_std": norms[:N_BPE].std().item(),
            "reserved_norm_mean": norms[ref].mean().item(),
            "reserved_pairwise_cos_mean": torch.nn.functional.normalize(M[ref], dim=1).matmul(
                torch.nn.functional.normalize(M[ref], dim=1).T).mean().item(),
            "threshold": thr, "n_candidates": int(cand.numel()),
        }
        np.save(f"{OUT}/{key}_norms.npy", norms.cpu().numpy().astype(np.float32))
        np.save(f"{OUT}/{key}_untrained_score.npy", score.cpu().numpy().astype(np.float32))
        mats[key] = M.half()
        del M, norms, score
        torch.cuda.empty_cache()
    E, U = mats["embed"], mats["unembed"]
    tie = torch.cat([torch.nn.functional.cosine_similarity(E[i:i + 8192].float(), U[i:i + 8192].float(), dim=1)
                     for i in range(0, N_BPE, 8192)])
    out["embed_unembed_cos_mean"] = tie.mean().item()
    # combined candidates: flagged by unembedding indicator (Land & Bartolo 2024 style)
    su = torch.from_numpy(np.load(f"{OUT}/unembed_untrained_score.npy"))
    se = torch.from_numpy(np.load(f"{OUT}/embed_untrained_score.npy"))
    order = torch.argsort(su[:N_BPE], descending=True)
    top = []
    for i in order[:400].tolist():
        top.append({"id": i, "tok": tok_str(i), "unembed_score": float(su[i]), "embed_score": float(se[i])})
    with open(f"{OUT}/glitch_candidates.json", "w") as f:
        json.dump(top, f, ensure_ascii=False, indent=0)
    RESULTS["E1"] = out
    save_results()
    log("E1 done", json.dumps(out)[:600])
    return E, U


# =================================================================== E2
def e2_router(E, U):
    log("E2: router atlas")
    moe_layers = sorted({int(k.split(".")[3]) for k in INDEX if k.endswith("block_sparse_moe.gate.weight")})
    Ehat = torch.cat([rmsnorm(E[s:s + 16384]).half() for s in range(0, N_BPE, 16384)])  # [N_BPE, 7168]
    biases, loads, eff_rank, gate_rows = [], [], [], {}
    top16_save = {}
    corr = []
    for L in moe_layers:
        W = fetch(f"{LM}layers.{L}.block_sparse_moe.gate.weight").to(DEV).float()
        b = fetch(f"{LM}layers.{L}.block_sparse_moe.gate.e_score_correction_bias").to(DEV).float()
        lnw = fetch(f"{LM}layers.{L}.post_attention_layernorm.weight").to(DEV).float()
        Wn = (W * lnw).half()
        cnt = torch.zeros(W.shape[0], device=DEV)
        keep = L in (1, 2, 3, 46, 91)
        tops = []
        for s in range(0, N_BPE, 16384):
            logits = Ehat[s:s + 16384] @ Wn.T
            sc = torch.sigmoid(logits.float()) + b
            idx = sc.topk(TC["num_experts_per_token"], dim=1).indices
            cnt += torch.bincount(idx.flatten(), minlength=W.shape[0]).float()
            if keep:
                tops.append(idx.short().cpu())
        if keep:
            top16_save[L] = torch.cat(tops).numpy()
            np.save(f"{OUT}/route_emb_L{L}_top16.npy", top16_save[L])
        S = torch.linalg.svdvals(W)
        p = S / S.sum()
        eff_rank.append(float(torch.exp(-(p * p.log()).sum())))
        biases.append(b.cpu().numpy())
        loads.append(cnt.cpu().numpy())
        lc = torch.log1p(cnt)
        corr.append(float(torch.corrcoef(torch.stack([lc, b]))[0, 1]))
        gate_rows[L] = torch.nn.functional.normalize(W, dim=1).half()
        log(f"  L{L} bias[{b.min():.3f},{b.max():.3f}] std={b.std():.4f} effrank={eff_rank[-1]:.1f} "
            f"corr(log load,bias)={corr[-1]:.3f} zero-load={int((cnt == 0).sum())}")
    B = np.stack(biases)
    np.save(f"{OUT}/router_bias.npy", B)
    np.save(f"{OUT}/router_emb_load.npy", np.stack(loads))
    # cross-layer: does expert i at layer L "line up" with expert i at L+1, or with any expert?
    same_idx, best_match, null = [], [], []
    g = torch.Generator(device=DEV).manual_seed(0)
    for a, bL in zip(moe_layers[:-1], moe_layers[1:]):
        C = gate_rows[a].float() @ gate_rows[bL].float().T
        same_idx.append(float(C.diag().mean()))
        best_match.append(float(C.abs().max(1).values.mean()))
    R = torch.nn.functional.normalize(torch.randn(896, 7168, device=DEV, generator=g), dim=1)
    R2 = torch.nn.functional.normalize(torch.randn(896, 7168, device=DEV, generator=g), dim=1)
    null_best = float((R @ R2.T).abs().max(1).values.mean())
    # all-pairs layer similarity of router subspaces (mean squared cos of principal angles, top-64)
    bases = {}
    for L in moe_layers:
        _, _, Vh = torch.linalg.svd(gate_rows[L].float(), full_matrices=False)
        bases[L] = Vh[:64]
    n = len(moe_layers)
    sub = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(moe_layers):
        for j, c in enumerate(moe_layers):
            sub[i, j] = float((bases[a] @ bases[c].T).pow(2).sum() / 64)
    np.save(f"{OUT}/router_subspace_overlap.npy", sub)
    # router rows through the unembedding: which vocab directions does each layer's router read?
    RESULTS["E2"] = {
        "moe_layers": moe_layers,
        "bias_std_per_layer": B.std(1).tolist(),
        "bias_range_per_layer": (B.max(1) - B.min(1)).tolist(),
        "bias_mean_per_layer": B.mean(1).tolist(),
        "eff_rank_per_layer": eff_rank,
        "corr_logload_bias_per_layer": corr,
        "adjacent_same_index_cos": same_idx,
        "adjacent_best_match_abs_cos": best_match,
        "null_best_match_abs_cos": null_best,
        "random_subspace_overlap_expected": 64 / 7168,
    }
    save_results()
    del Ehat
    torch.cuda.empty_cache()
    return moe_layers


# =================================================================== E3
def e3_kda():
    log("E3: KDA memory horizons")
    kda = sorted({int(k.split(".")[3]) for k in INDEX if k.endswith("self_attn.A_log")})
    lb = TC["linear_attn_config"].get("gate_lower_bound", -5.0)
    H = TC["linear_attn_config"]["num_heads"]
    D = TC["linear_attn_config"]["head_dim"]
    per_layer, hl_all, alog_tail = [], [], []
    for L in kda:
        A = fetch(f"{LM}layers.{L}.self_attn.A_log").float()
        dt = fetch(f"{LM}layers.{L}.self_attn.dt_bias").float().view(H, D)
        alog_tail.append({"layer": L, "len": int(A.numel()),
                          "tail_beyond_heads": A[H:].tolist()[:8], "head_vals_head": A[:4].tolist()})
        Ah = A[:H].view(H, 1)
        g = lb * torch.sigmoid(Ah.exp() * dt)  # resting log-decay per step (input-dependent part = 0)
        hl = math.log(2) / (-g).clamp_min(1e-12)  # half-life in tokens
        hl_all.append(hl.numpy().astype(np.float32))
        per_layer.append({"layer": L, "median_hl": float(hl.median()),
                          "frac_hl_gt_1k": float((hl > 1000).float().mean()),
                          "frac_hl_lt_2": float((hl < 2).float().mean()),
                          "max_hl": float(hl.max())})
    np.save(f"{OUT}/kda_halflife.npy", np.stack(hl_all))
    RESULTS["E3"] = {"kda_layers": kda, "per_layer": per_layer, "A_log_shape_note": alog_tail[:3],
                     "A_log_lengths": sorted({a["len"] for a in alog_tail})}
    save_results()
    log("E3 done; A_log lengths", RESULTS["E3"]["A_log_lengths"])


# =================================================================== E3b
def e3b_attnres(U):
    log("E3b: attention-residual queries through the unembedding")
    fn = fetch(f"{LM}norm.weight").to(DEV).float()
    rows = []
    for L in range(TC["num_hidden_layers"]):
        for kind in ("self_attention_res_proj", "mlp_res_proj"):
            name = f"{LM}layers.{L}.{kind}.weight"
            if name in INDEX:
                rows.append((L, kind, fetch(name).to(DEV).float().flatten()))
    Q = torch.stack([r[2] for r in rows])
    Qn = torch.nn.functional.normalize(Q, dim=1)
    sim = (Qn @ Qn.T).cpu().numpy()
    np.save(f"{OUT}/attnres_query_sim.npy", sim)
    lens = []
    for (L, kind, q) in rows[::4]:
        lg = (U[:N_BPE] @ (q * fn).half()).float()
        lens.append({"layer": L, "kind": kind, "top": [tok_str(i) for i in lg.topk(12).indices.tolist()],
                     "norm": float(q.norm())})
    RESULTS["E3b"] = {"n": len(rows), "norms": [float(r[2].norm()) for r in rows], "lens": lens}
    save_results()


# =================================================================== E4
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant(packed, scale, low_first=True):
    """MXFP4: packed uint8 [out, in/2], e8m0 scale uint8 [out, in/32] -> fp16 [out, in]."""
    p = packed.to(DEV).long()
    lo, hi = p & 0xF, p >> 4
    codes = torch.stack([lo, hi] if low_first else [hi, lo], dim=-1).flatten(1)
    lut = FP4.to(DEV)
    w = lut[codes]
    s = torch.exp2(scale.to(DEV).float() - 127.0)
    w = (w.view(w.shape[0], -1, 32) * s.unsqueeze(-1)).view(w.shape[0], -1)
    return w


def situ(gate, up, beta=TC.get("activation_situ_beta", 4.0), lbeta=TC.get("activation_situ_linear_beta")):
    gate, up = gate.float(), up.float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lbeta is not None:
        up = lbeta * torch.tanh(up / lbeta)
    return a * up


def expert_ranges(L):
    names = []
    for e in range(TC["num_experts"]):
        for w in ("w1", "w2", "w3"):
            for part in ("weight_packed", "weight_scale"):
                names.append(f"{LM}layers.{L}.block_sparse_moe.experts.{e}.{w}.{part}")
    ms = sorted((meta(n) + (n,) for n in names), key=lambda m: m[1])
    shard = ms[0][0]
    assert all(m[0] == shard for m in ms), "expert tensors span shards"
    chunks, cur, cur_s = [], [], None
    for m in ms:
        if cur and (m[2] - cur_s > (96 << 20) or m[1] != cur[-1][2]):
            chunks.append(cur)
            cur = []
        if not cur:
            cur_s = m[1]
        cur.append(m)
    if cur:
        chunks.append(cur)
    return shard, chunks


def e4_experts(L, E, U, probe_ids, low_first=None):
    log(f"E4: layer {L} — running all experts")
    pfx = f"{LM}layers.{L}.block_sparse_moe."
    lnw = fetch(f"{LM}layers.{L}.post_attention_layernorm.weight").to(DEV).float()
    Wd = fetch(pfx + "routed_expert_down_proj.weight").to(DEV).half()  # [3584, 7168]
    Wu = fetch(pfx + "routed_expert_up_proj.weight").to(DEV).float()  # [7168, 3584]
    rn = fetch(pfx + "routed_expert_norm.weight").to(DEV).float()
    W = fetch(pfx + "gate.weight").to(DEV).float()
    b = fetch(pfx + "gate.e_score_correction_bias").to(DEV).float()
    fn = fetch(f"{LM}norm.weight").to(DEV).float()
    h = rmsnorm(E[probe_ids].float(), lnw)  # [P, 7168]
    z = (h.half() @ Wd.T).float()  # latent probes [P, 3584]
    sc = torch.sigmoid(h @ W.T)
    route = (sc + b).topk(TC["num_experts_per_token"], dim=1).indices  # [P, 16]
    g = torch.Generator(device=DEV).manual_seed(1)
    zr = torch.randn(z.shape, device=DEV, generator=g) * z.std(0, keepdim=True)  # isotropic-ish null
    proj = torch.randn(z.shape[1], 256, device=DEV, generator=g) / 16.0
    P = z.shape[0]
    nE = TC["num_experts"]
    Ys = torch.empty(nE, P, 256, dtype=torch.float16)
    stats = []
    shard, chunks = expert_ranges(L)
    total_bytes = sum(c[-1][2] - c[0][1] for c in chunks)
    log(f"  {len(chunks)} chunks, {total_bytes / 1e9:.2f} GB from {shard}")
    pending: dict = collections.defaultdict(dict)
    order_votes = []

    def dl(chunk):
        s, e = chunk[0][1], chunk[-1][2]
        buf = get_range(shard, s, e - 1)
        out = {}
        for m in chunk:
            out[m[5]] = to_tensor(buf[m[1] - s:m[2] - s], m[3], m[4])
        return out

    done = 0
    t_layer = time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        futs = collections.deque()
        it = iter(chunks)
        for c in it:
            futs.append(ex.submit(dl, c))
            if len(futs) >= 24:
                break
        while futs:
            res = futs.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                futs.append(ex.submit(dl, nxt))
            for name, t in res.items():
                parts = name.split(".")
                e = int(parts[parts.index("experts") + 1])
                pending[e][parts[-2] + "." + parts[-1]] = t
            for e in [k for k, v in pending.items() if len(v) == 6]:
                t = pending.pop(e)
                if low_first is None and e < 8:
                    for lf in (True, False):
                        w1 = dequant(t["w1.weight_packed"], t["w1.weight_scale"], lf)
                        r = ((z @ w1.T).std() / (zr @ w1.T).std()).item()
                        order_votes.append((e, lf, r))
                    if e == 7:
                        rt = np.mean([v[2] for v in order_votes if v[1]])
                        rf = np.mean([v[2] for v in order_votes if not v[1]])
                        low_first = bool(rt >= rf)
                        log(f"  nibble order: low_first ratio={rt:.4f} high_first ratio={rf:.4f} -> low_first={low_first}")
                lf = True if low_first is None else low_first
                w1 = dequant(t["w1.weight_packed"], t["w1.weight_scale"], lf)
                w3 = dequant(t["w3.weight_packed"], t["w3.weight_scale"], lf)
                w2 = dequant(t["w2.weight_packed"], t["w2.weight_scale"], lf)
                a = situ(z @ w1.T, z @ w3.T)
                y = a @ w2.T  # [P, 3584]
                Ys[e] = (y @ proj).half().cpu()
                mine = (route == e).any(1)
                n_mine = int(mine.sum())
                ym = y[mine].mean(0) if n_mine >= 4 else y.mean(0)
                hid = Wu @ (rmsnorm(ym.unsqueeze(0), rn)[0])
                lg = (U[:N_BPE] @ rmsnorm(hid.unsqueeze(0), fn)[0].half()).float()
                sc_codes = t["w1.weight_scale"].float()
                stats.append({
                    "e": e, "n_probe_routed": n_mine,
                    "out_norm_mean": float(y.norm(dim=1).mean()),
                    "out_norm_routed": float(y[mine].norm(dim=1).mean()) if n_mine else None,
                    "w1_fro": float(w1.float().norm()), "w2_fro": float(w2.float().norm()),
                    "scale_exp_mean": float(sc_codes.mean() - 127), "scale_exp_std": float(sc_codes.std()),
                    "promotes": lg.topk(10).indices.tolist(),
                    "in_tokens": [int(i) for i in probe_ids[mine][:12].tolist()],
                })
                done += 1
                if done % 128 == 0:
                    rate = BYTES[0] / max(time.time() - T0, 1) / 1e6
                    log(f"  L{L}: {done}/{nE} experts  avg dl {rate:.0f} MB/s")
            if time.time() - T0 > TIME_BUDGET:
                log("time budget hit inside E4")
                break
    log(f"  layer {L} experts done in {time.time() - t_layer:.0f}s")
    # functional similarity between experts on shared probes
    n = len(stats)
    ids = [s["e"] for s in stats]
    Y = Ys[ids].float()
    Y = Y - Y.mean(1, keepdim=True)
    Yf = torch.nn.functional.normalize(Y.reshape(n, -1), dim=1).to(DEV).half()
    Sim = (Yf @ Yf.T).float().cpu().numpy()
    np.fill_diagonal(Sim, np.nan)
    np.save(f"{OUT}/expert_sim_L{L}.npy", Sim.astype(np.float32))
    with open(f"{OUT}/expert_stats_L{L}.json", "w") as f:
        json.dump(stats, f)
    nn_sim = np.nanmax(Sim, 1)
    RESULTS.setdefault("E4", {})[str(L)] = {
        "n_experts": n, "nibble_low_first": low_first, "order_votes": order_votes,
        "nn_sim_mean": float(np.nanmean(nn_sim)), "nn_sim_p95": float(np.nanpercentile(nn_sim, 95)),
        "pairs_gt_0.9": int(np.nansum(Sim > 0.9) // 2), "pairs_gt_0.7": int(np.nansum(Sim > 0.7) // 2),
        "offdiag_mean": float(np.nanmean(Sim)),
        "probe_routed_counts_zero": int(sum(1 for s in stats if s["n_probe_routed"] == 0)),
    }
    save_results()
    log("E4", L, json.dumps({k: v for k, v in RESULTS["E4"][str(L)].items() if k != "order_votes"}))
    return low_first



def v2_glitch():
    log("V2-A: nearest-neighbour glitch detection")
    out = {}
    mats = {}
    for key, name in (("embed", "language_model.model.embed_tokens.weight"),
                      ("unembed", "language_model.lm_head.weight")):
        M = fetch(name).to(DEV).half()
        Mn = torch.nn.functional.normalize(M.float(), dim=1).half() if False else torch.cat(
            [torch.nn.functional.normalize(M[i:i + 16384].float(), dim=1).half() for i in range(0, V, 16384)])
        nn = torch.empty(V, device=DEV)
        nn_id = torch.empty(V, dtype=torch.long, device=DEV)
        for i in range(0, V, 4096):
            S = Mn[i:i + 4096] @ Mn.T
            S[torch.arange(S.shape[0]), torch.arange(i, i + S.shape[0])] = -2
            m = S.max(1)
            nn[i:i + 4096] = m.values.float()
            nn_id[i:i + 4096] = m.indices
            del S
        x = M.float() if False else None
        kurt = torch.empty(V, device=DEV)
        for i in range(0, V, 16384):
            r = M[i:i + 16384].float()
            r = r - r.mean(1, keepdim=True)
            kurt[i:i + 16384] = (r.pow(4).mean(1) / r.pow(2).mean(1).pow(2))
        ref = torch.tensor(RESERVED, device=DEV)
        thr = torch.quantile(nn[ref], 0.99).item()
        bpe = torch.arange(N_BPE, device=DEV)
        cand = bpe[nn[:N_BPE] <= thr]
        np.save(f"{OUT}/{key}_nncos.npy", nn.cpu().numpy())
        np.save(f"{OUT}/{key}_nnid.npy", nn_id.cpu().numpy().astype(np.int32))
        np.save(f"{OUT}/{key}_kurtosis.npy", kurt.cpu().numpy())
        out[key] = {"reserved_nncos_p50": float(nn[ref].median()), "reserved_nncos_p99": thr,
                    "bpe_nncos_p50": float(nn[:N_BPE].median()), "bpe_nncos_p01": float(torch.quantile(nn[:N_BPE].float()[torch.randperm(N_BPE, device=DEV)[:100000]], 0.01)),
                    "reserved_kurtosis_mean": float(kurt[ref].mean()), "bpe_kurtosis_median": float(kurt[:N_BPE].median()),
                    "n_untrained_bpe": int(cand.numel()),
                    "untrained": [{"id": int(i), "tok": tok_str(int(i)), "nncos": float(nn[i])} for i in cand[torch.argsort(nn[cand])].tolist()[:3000]]}
        mats[key] = M
        del Mn
        torch.cuda.empty_cache()
        log(f"  {key}: reserved nn p50={out[key]['reserved_nncos_p50']:.3f} p99={thr:.3f} bpe p50={out[key]['bpe_nncos_p50']:.3f} -> {cand.numel()} untrained BPE")
    RESULTS["V2_glitch"] = out
    save_results()
    return mats["embed"], mats["unembed"]


def v2_bias(E):
    log("V2-B: router load with vs without balancing bias")
    moe_layers = sorted({int(k.split(".")[3]) for k in INDEX if k.endswith("block_sparse_moe.gate.weight")})
    Ehat = torch.cat([rmsnorm(E[s:s + 16384]).half() for s in range(0, N_BPE, 16384)])
    rows = []
    for L in moe_layers:
        W = fetch(f"{LM}layers.{L}.block_sparse_moe.gate.weight").to(DEV).float()
        b = fetch(f"{LM}layers.{L}.block_sparse_moe.gate.e_score_correction_bias").to(DEV).float()
        lnw = fetch(f"{LM}layers.{L}.post_attention_layernorm.weight").to(DEV).float()
        Wn = (W * lnw).half()
        c0 = torch.zeros(W.shape[0], device=DEV)
        c1 = torch.zeros(W.shape[0], device=DEV)
        for s in range(0, N_BPE, 16384):
            sc = torch.sigmoid((Ehat[s:s + 16384] @ Wn.T).float())
            c0 += torch.bincount(sc.topk(16, dim=1).indices.flatten(), minlength=W.shape[0]).float()
            c1 += torch.bincount((sc + b).topk(16, dim=1).indices.flatten(), minlength=W.shape[0]).float()
        cv = lambda c: float(c.std() / c.mean())
        rows.append({"layer": L, "cv_nobias": cv(c0), "cv_bias": cv(c1),
                     "corr_logload_nobias_bias": float(torch.corrcoef(torch.stack([torch.log1p(c0), b]))[0, 1]),
                     "zero_nobias": int((c0 == 0).sum()), "zero_bias": int((c1 == 0).sum())})
        if L in (1, 2, 3):
            np.save(f"{OUT}/load_nobias_L{L}.npy", c0.cpu().numpy())
    RESULTS["V2_bias"] = rows
    save_results()
    log("  ", rows[:3])
    del Ehat
    torch.cuda.empty_cache()


def lexical_core(s):
    import unicodedata
    c = "".join(ch for ch in s.lower() if unicodedata.category(ch)[0] in "LN")
    return c


def v2_writers(U):
    log("V2-C: expert write-direction lexicon across all layers")
    moe_layers = sorted({int(k.split(".")[3]) for k in INDEX if k.endswith("block_sparse_moe.gate.weight")})
    fn = fetch(f"{LM}norm.weight").to(DEV).float()
    Uh = (U[:N_BPE].float() * 1).half() if False else U[:N_BPE]
    cores = [lexical_core(tok_str(i)) for i in range(N_BPE)]
    per_layer = []
    lex_examples = []
    lf = True
    for L in moe_layers:
        if time.time() - T0 > TIME_BUDGET:
            log("time budget hit")
            break
        pfx = f"{LM}layers.{L}.block_sparse_moe."
        Wu = fetch(pfx + "routed_expert_up_proj.weight").to(DEV).float()
        rn = fetch(pfx + "routed_expert_norm.weight").to(DEV).float()
        A = Wu * rn
        G = A.T @ A
        names = []
        for e in range(TC["num_experts"]):
            for part in ("weight_packed", "weight_scale"):
                names.append(f"{pfx}experts.{e}.w2.{part}")
        ms = [meta(n) + (n,) for n in names]
        shard = ms[0][0]
        pairs = [(ms[2 * e], ms[2 * e + 1]) for e in range(TC["num_experts"])]

        def dl(pr):
            a, b2 = pr
            if a[2] == b2[1]:
                buf = get_range(shard, a[1], b2[2] - 1)
                return (to_tensor(buf[:a[2] - a[1]], a[3], a[4]), to_tensor(buf[a[2] - a[1]:], b2[3], b2[4]))
            return (to_tensor(get_range(shard, a[1], a[2] - 1), a[3], a[4]),
                    to_tensor(get_range(shard, b2[1], b2[2] - 1), b2[3], b2[4]))

        dirs = torch.empty(TC["num_experts"], 7168, device=DEV)
        svals = torch.empty(TC["num_experts"], device=DEV)
        with cf.ThreadPoolExecutor(24) as ex:
            for e, (pk, sc) in enumerate(ex.map(dl, pairs)):
                w2 = dequant(pk, sc, lf)  # [3584, 3072]
                v = torch.randn(w2.shape[1], device=DEV)
                for _ in range(25):
                    v = w2.T @ (G @ (w2 @ v))
                    v = v / v.norm()
                u = A @ (w2 @ v)
                svals[e] = u.norm()
                dirs[e] = u / u.norm()
        lg = (Uh @ (dirs * fn).half().T).float()  # [N_BPE, 896]
        top_pos = lg.topk(10, dim=0).indices.T.tolist()
        top_neg = (-lg).topk(10, dim=0).indices.T.tolist()
        lex = []
        for e in range(TC["num_experts"]):
            best = 0
            for tops in (top_pos[e], top_neg[e]):
                cs = [cores[i] for i in tops if cores[i]]
                if not cs:
                    continue
                cnt = collections.Counter(cs).most_common(1)[0][1]
                best = max(best, cnt / 10)
            lex.append(best)
        lex = np.array(lex)
        per_layer.append({"layer": L, "lex_mean": float(lex.mean()), "frac_lex_ge_0.5": float((lex >= 0.5).mean()),
                          "sval_mean": float(svals.mean())})
        order = np.argsort(-lex)[:6]
        for e in order:
            lex_examples.append({"layer": L, "e": int(e), "lex": float(lex[e]),
                                 "pos": [tok_str(i) for i in top_pos[e]], "neg": [tok_str(i) for i in top_neg[e]]})
        np.save(f"{OUT}/writer_lex_L{L}.npy", lex)
        json.dump({"pos": top_pos, "neg": top_neg}, open(f"{OUT}/writer_tops_L{L}.json", "w"))
        log(f"  L{L}: lex_mean={lex.mean():.3f} frac>=0.5={(lex >= 0.5).mean():.3f}  "
            f"{BYTES[0] / max(time.time() - T0, 1) / 1e6:.0f} MB/s  ex: {[tok_str(i) for i in top_pos[int(order[0])][:6]]}")
        RESULTS["V2_writers"] = {"per_layer": per_layer, "examples": lex_examples}
        save_results()


def main():
    E, U = v2_glitch()
    v2_bias(E)
    v2_writers(U)
    RESULTS["downloaded_GB"] = BYTES[0] / 1e9
    RESULTS["elapsed_s"] = time.time() - T0
    save_results()
    log("ALL DONE", RESULTS["downloaded_GB"], "GB")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        RESULTS["fatal"] = traceback.format_exc()
        save_results()
        raise
