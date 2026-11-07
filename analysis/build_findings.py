"""Turn raw Kaggle outputs (results/v1, results/v2) into one findings.json.

Usage: python3 analysis/build_findings.py [out_path]
Default out_path: site/src/data/findings.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "v1"
V2 = ROOT / "results" / "v2"


def r(x, n=3):
    return round(float(x), n)


def main(out_path: Path) -> None:
    r1 = json.loads((V1 / "results.json").read_text())
    r2 = json.loads((V2 / "results.json").read_text())

    # 01 expert write directions: share of word-family ("spelling") experts per layer
    writers = [{"layer": p["layer"], "share": r(p["frac_lex_ge_0.5"]), "lex_mean": r(p["lex_mean"])}
               for p in r2["V2_writers"]["per_layer"]]
    picks = [(56, 743), (56, 368), (84, 757), (84, 752), (84, 699), (72, 457), (80, 519),
             (40, 184), (40, 146), (89, 825), (90, 33), (66, 479), (24, 562), (85, None)]
    examples = r2["V2_writers"]["examples"]
    gallery = []
    for layer, expert in picks:
        for x in examples:
            if x["layer"] == layer and (expert is None or x["e"] == expert):
                gallery.append({"layer": layer, "expert": x["e"], "promotes": x["pos"][:8], "opposite": x["neg"][:5]})
                break
    early = np.mean([w["share"] for w in writers if w["layer"] <= 47])
    late = np.mean([w["share"] for w in writers if 82 <= w["layer"] <= 91])
    peak = max(writers, key=lambda w: w["share"])

    # 02 untrained tokens (nearest-neighbour cosine on input embeddings)
    g = r2["V2_glitch"]
    untrained = [{"id": u["id"], "token": u["tok"], "nn_cos": r(u["nncos"])} for u in g["embed"]["untrained"]]
    glitch = {k: {kk: (r(vv) if isinstance(vv, float) else vv) for kk, vv in g[k].items() if kk != "untrained"}
              for k in ("embed", "unembed")}

    # 03 KDA memory horizons
    kda = [{"layer": p["layer"], "median_half_life": r(p["median_hl"], 1), "share_over_1k": r(p["frac_hl_gt_1k"])}
           for p in r1["E3"]["per_layer"]]
    hl = np.load(V1 / "kda_halflife.npy").ravel()

    # 04 expert functional similarity
    esim = {}
    for L in (1, 46, 91):
        S = np.load(V1 / f"expert_sim_L{L}.npy")
        nn = np.nanmax(S, 1)
        esim[str(L)] = {"max_pair": r(np.nanmax(nn)), "mean_nearest": r(np.nanmean(nn), 4),
                        "p95_nearest": r(np.nanpercentile(nn, 95), 4)}
    nibble = r1["E4"]["1"]["order_votes"]

    # 05 balancing bias
    B = np.load(V1 / "router_bias.npy")
    layers = r1["E2"]["moe_layers"]
    bias = [{"layer": L, "min": r(B[i].min()), "max": r(B[i].max()), "std": r(B[i].std(), 4)} for i, L in enumerate(layers)]

    # 06 attention-residual block structure
    A = np.load(V1 / "attnres_query_sim.npy")
    n = A.shape[0]
    blk = np.arange(n) // 2 // 12
    off = ~np.eye(n, dtype=bool)
    same = A[off & (blk[:, None] == blk[None, :])].mean()
    diff = A[blk[:, None] != blk[None, :]].mean()

    out = {
        "model": {"repo": "moonshotai/Kimi-K3", "params": "2.8T", "active": "104B", "layers": 93,
                  "experts": 896, "top_k": 16, "kda_layers": 69, "mla_layers": 24, "vocab": 163840,
                  "checkpoint_bytes": 1560860324864},
        "run": {"downloaded_gb": r(r1["downloaded_GB"] + r2["downloaded_GB"], 1),
                "gpu_minutes": r((r1["elapsed_s"] + r2["elapsed_s"]) / 60, 1), "gpu": r1["gpu"]},
        "writers": {"per_layer": writers, "gallery": gallery, "early_share": r(early), "late_share": r(late),
                    "peak": peak, "experts_scanned": 92 * 896},
        "untrained": {"tokens": untrained, "stats": glitch, "embed_unembed_cos": r(r1["E1"]["embed_unembed_cos_mean"], 4)},
        "kda": {"per_layer": kda, "share_over_1k": r((hl > 1000).mean()), "median_all": r(np.median(hl), 1)},
        "experts": {"similarity": esim, "nibble_votes": nibble},
        "bias": {"per_layer": bias, "no_bias_check": r2["V2_bias"][:3]},
        "attn_res": {"within_block": r(same), "across_block": r(diff), "block_size": 12},
        "a_log": {"stored": 128, "heads": 96, "tail_all_zero": True},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"spelling experts: early {early:.3f}, late {late:.3f}, peak L{peak['layer']} {peak['share']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site" / "src" / "data" / "findings.json")
