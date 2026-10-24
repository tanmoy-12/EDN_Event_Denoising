from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import edn_model as _M
from edn_model import (EDNConfig, MultiScaleDensity, baf_baseline_score,  # noqa: F401
                       band_offsets, denoise_loss, physical_knn)


# ==============================================================================
# 0. CONFIG
# ==============================================================================
@dataclass
class AblConfig(EDNConfig):
    """EDNConfig plus the ablation switches. Defaults == the published model."""

    # A1 -- "rate": log(count / expected-under-uniform-BA-at-this-window's-rate)
    #       "raw" : log(count), the same counts with no denominator.
    density_norm: str = "rate"

    # A2 -- "physical": candidates from a fixed PHYSICAL time window (tau).
    #       "index"   : candidates from a fixed EVENT-INDEX band.
    graph_mode: str = "physical"
    # Index half-widths, in events. Chosen so that on clean ED24 at the training
    # window the band spans roughly the same physical extent as tau_fine (2 ms)
    # and tau_coarse (20 ms): the model's own docstring measures a +-512-event
    # band at 4.03 ms with no injected noise, so ~256 and ~2560 events match the
    # two taus at THAT rate. They stop matching at any other rate, which is
    # precisely the property A2 exists to expose.
    idx_band_fine: int = 256
    idx_band_coarse: int = 2560

    # A3 -- window-level conditioner + FiLM modulation of every block.
    #   use_film=False       removes the conditioner MLP and the FiLM
    #                        projections outright (-0.091 M parameters), so the
    #                        ablation also removes capacity.
    #   film_input="zeros"   keeps every one of those weights and feeds the
    #                        conditioner a zero vector instead of the window
    #                        statistics. The parameter count is then IDENTICAL
    #                        to the full model and the FiLM scale/shift become
    #                        learned constants, so only the window INFORMATION
    #                        is removed. This is the capacity-controlled form
    #                        of A3; use_film=False is the capacity-confounded
    #                        one, and both are kept so the two can be compared.
    use_film: bool = True
    film_input: str = "stats"

    def __post_init__(self):
        super().__post_init__()
        if self.density_norm not in ("rate", "raw"):
            raise ValueError(f"density_norm must be rate|raw, got {self.density_norm!r}")
        if self.graph_mode not in ("physical", "index"):
            raise ValueError(f"graph_mode must be physical|index, got {self.graph_mode!r}")
        if getattr(self, "film_input", "stats") not in ("stats", "zeros"):
            raise ValueError(f"film_input must be stats|zeros, "
                             f"got {self.film_input!r}")


# ==============================================================================
# 1. A2 -- EVENT-INDEX NEIGHBOURHOOD
# ==============================================================================
def index_knn(coords_scaled: torch.Tensor, half_width: int,
              offsets: torch.Tensor, k: int, n_strata: int) -> torch.Tensor:
    """k nearest neighbours, candidates drawn from a fixed EVENT-INDEX band.

    Deliberately identical to physical_knn in every respect except the rule
    that picks the candidate range: same dense band, same number of stratified
    samples, same scaled-distance metric, same k. Only `lo`/`hi` change, from
    two searchsorted lookups on t to index arithmetic. Any difference in the
    trained model is therefore attributable to the candidate rule and not to
    the candidate budget.
    """
    B, N, _ = coords_scaled.shape
    dev = coords_scaled.device
    idx = torch.arange(N, device=dev).view(1, N)

    lo = (idx - half_width).clamp(0, N - 1)
    hi = (idx + half_width).clamp(0, N - 1)

    u = torch.linspace(0.0, 1.0, n_strata, device=dev).view(1, 1, n_strata)
    span = (hi - lo).unsqueeze(-1).float()
    strat = (lo.unsqueeze(-1) + (u * span).round().long()).expand(B, -1, -1)

    band = (idx.unsqueeze(-1) + offsets.view(1, 1, -1)).expand(B, -1, -1)
    cand = torch.cat([band, strat], dim=-1)
    valid = (cand >= 0) & (cand <= N - 1) & (cand != idx.unsqueeze(-1))
    cand = cand.clamp(0, N - 1).contiguous()

    gathered = _M._gather_points(coords_scaled, cand)
    d2 = ((gathered - coords_scaled.unsqueeze(2)) ** 2).sum(-1)
    d2 = d2.masked_fill(~valid, float("inf"))

    sel = torch.topk(d2, min(k, cand.shape[-1]), dim=-1, largest=False).indices
    return torch.gather(cand, 2, sel)


# ==============================================================================
# 2. A1 -- EXACT DE-NORMALISATION OF THE DENSITY DESCRIPTOR
# ==============================================================================
class _Denorm:
    """Recovers log(count + 0.5) from the descriptor's normalised output.

    Returns a (B, 1, C) additive correction aligned to the descriptor's channel
    layout, which is, in order:
        per density scale (4):  log(n_all/E), log(n_dist/E), log(n_same/0.5E),
                                same-polarity fraction      <- a ratio, untouched
        per motion scale (2):   log(best/E_m), log(best/zero) <- ratio, untouched
        pixel (3):              log(n_px/E_px), log(dt/E_dt), hot flag <- flag untouched
    """

    def __init__(self, density: MultiScaleDensity):
        self.d = density
        self.cfg = density.cfg

    def __call__(self, dur: torch.Tensor, N: int) -> torch.Tensor:
        cfg, d = self.cfg, self.d
        W, H = _M.SENSOR_W, _M.SENSOR_H
        dur = dur.clamp_min(1.0)
        B = dur.shape[0]
        cols: List[torch.Tensor] = []
        zero = torch.zeros(B, 1, device=dur.device, dtype=dur.dtype)

        rt_all = d._extents(dur, [s[1] for s in cfg.density_scales],
                            cfg.density_time_fracs)
        for si, (r, _) in enumerate(cfg.density_scales):
            rt = rt_all[si].squeeze(-1)                       # (B,)
            E = ((N - 1) * (r * r * rt) / (W * H * dur)).view(B, 1)
            cols += [torch.log(E + 0.5), torch.log(E + 0.5),
                     torch.log(0.5 * E + 0.5), zero]

        if d.n_motion:
            mrt_all = d._extents(dur, [s[1] for s in cfg.motion_scales],
                                 cfg.motion_time_fracs)
            for mi, (r, _) in enumerate(cfg.motion_scales):
                rt = mrt_all[mi].squeeze(-1)
                E = ((N - 1) * (r * r * rt) / (W * H * dur)).view(B, 1)
                cols += [torch.log(E + 0.5), zero]

        if cfg.use_pixel_features:
            E_px = torch.full((B, 1), float(N) / (W * H),
                              device=dur.device, dtype=dur.dtype)
            E_dt = (dur * W * H / N).view(B, 1)
            cols += [torch.log(E_px + 0.5), torch.log(E_dt.clamp_min(1.0)), zero]

        return torch.cat(cols, dim=-1).unsqueeze(1)           # (B,1,C)


# ==============================================================================
# 3. THE ABLATED MODEL
# ==============================================================================
class EDNetAbl(_M.EDNet):
    """EDNet with the three switches. Default config == EDNet, exactly."""

    def __init__(self, cfg: AblConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self._denorm = _Denorm(self.density) if cfg.density_norm == "raw" else None
        if not getattr(cfg, "use_film", True):
            # Remove rather than bypass, so the parameter count reports the
            # removal honestly instead of carrying dead weights.
            del self.cond
            del self.film

    def forward(self, feat: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x_px, y_px, t_us, pol, dur = self._physical(feat, meta)
        N = feat.shape[1]

        dens, ladder = self.density(x_px, y_px, t_us, pol, dur)
        if self._denorm is not None:                                   # A1
            dens = dens + self._denorm(dur, N)

        h = torch.cat([feat[..., :3], self.pol_embed((pol > 0.5).long()), dens],
                      dim=-1)
        x = self.stem(h)

        use_film = getattr(cfg, "use_film", True)
        if use_film:                                                   # A3
            rate = meta[:, 1] / (dur / 1e6)
            gstat = torch.cat([
                torch.log10(dur.unsqueeze(1)),
                torch.log10(rate.clamp_min(1.0).unsqueeze(1)),
                dens.mean(dim=1), dens.amax(dim=1), ladder,
            ], dim=-1)
            if getattr(cfg, "film_input", "stats") == "zeros":        # A3b
                # Same tensor shape, same dtype, same device -- every weight in
                # self.cond and self.film still exists and still trains. What
                # the conditioner no longer receives is the window's duration,
                # rate and density summary, so the FiLM scale and shift become
                # constants learned from the bias terms alone.
                gstat = torch.zeros_like(gstat)
            cond = self.cond(gstat)

        tau_f, tau_c, sig_f, sig_c = self._graph_scales(dur)
        s_xy = cfg.sigma_xy_px
        c_fine = torch.stack([x_px / s_xy, y_px / s_xy, t_us / sig_f], dim=-1)
        c_coarse = torch.stack([x_px / s_xy, y_px / s_xy, t_us / sig_c], dim=-1)

        if getattr(cfg, "graph_mode", "physical") == "index":          # A2
            nbr_fine = index_knn(c_fine, cfg.idx_band_fine,
                                 self._offsets, cfg.k, cfg.n_strata)
            nbr_coarse = index_knn(c_coarse, cfg.idx_band_coarse,
                                   self._offsets, cfg.k, cfg.n_strata)
        else:
            nbr_fine = physical_knn(c_fine, t_us, tau_f,
                                    self._offsets, cfg.k, cfg.n_strata)
            nbr_coarse = physical_knn(c_coarse, t_us, tau_c,
                                      self._offsets, cfg.k, cfg.n_strata)

        for i, blk in enumerate(self.blocks):
            if use_film:
                scale, shift = self.film[i](cond).chunk(2, dim=-1)
            else:
                z = x.new_zeros(x.shape[0], cfg.hidden_dim)
                scale, shift = z, z
            args = (x, c_fine, c_coarse, pol, nbr_fine, nbr_coarse, (scale, shift))
            if cfg.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, *args, use_reentrant=False)
            else:
                x = blk(*args)

        return self.head(self.norm(x)).squeeze(-1)


# ==============================================================================
# 4. VARIANT REGISTRY
# ==============================================================================
# name -> (config overrides, training-command overrides, one-line description)
VARIANTS = {
    "full":      ({}, {},
                  "published model (control)"),
    "a1_rawdens": ({"density_norm": "raw"}, {},
                   "rate-normalised density -> raw local counts"),
    "a2_idxgraph": ({"graph_mode": "index"}, {},
                    "physical-time neighbourhood -> event-index band, same k"),
    "a3_nofilm": ({"use_film": False}, {},
                  "no window-level conditioner, no FiLM (-0.091 M params)"),
    "a3b_constfilm": ({"film_input": "zeros"}, {},
                      "conditioner and FiLM kept, fed zeros: removes the "
                      "window information at identical parameter count"),
    "a4_nomotion": ({"motion_scales": [], "motion_time_fracs": []}, {},
                    "no motion-adaptive (sheared-frame) channels"),
    "a5_nopixel": ({"use_pixel_features": False}, {},
                   "no pixel-recurrence / hot-pixel channels"),
    "a6_noaug":  ({}, {"shot_noise_prob": 0.0},
                  "no synthetic shot-noise augmentation (data-side only)"),
}


def build_config(variant: str, **base) -> AblConfig:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; "
                         f"choose from {', '.join(VARIANTS)}")
    over, _, _ = VARIANTS[variant]
    return AblConfig(**{**base, **over})


def config_from_checkpoint(ck, seq_len: int = 4096) -> AblConfig:
    """Like edn_model.config_from_checkpoint but preserves the ablation fields.

    USE THIS IN THE EVAL SCRIPTS. The base version builds an EDNConfig, which
    has no graph_mode / density_norm / use_film, so an A2 checkpoint would load
    without error -- the parameter set is identical -- and then be evaluated
    with the PHYSICAL graph it was never trained with. That is a silently wrong
    number, not a crash.
    """
    saved = ck.get("cfg", None)
    sd = ck.get("model", {}) or {}
    cfg = AblConfig(seq_len=seq_len)

    if saved is not None:
        inst = getattr(saved, "__dict__", {})
        for f in cfg.__dataclass_fields__:
            if f in inst:
                setattr(cfg, f, inst[f])
        if "scale_mode" not in inst:
            cfg.scale_mode = "absolute"

    # Cross-check against the weights, which cannot lie about what was trained.
    if "cond.0.weight" not in sd and getattr(cfg, "use_film", True):
        raise SystemExit(
            "checkpoint has no conditioner weights but the config says "
            "use_film=True.\nThis is an A3 checkpoint being loaded as a full "
            "model; fix the config or the checkpoint.")
    if "cond.0.weight" in sd and not getattr(cfg, "use_film", True):
        raise SystemExit(
            "checkpoint HAS conditioner weights but the config says "
            "use_film=False.")
    if "cond.0.weight" in sd and getattr(cfg, "film_input", "stats") not in (
            "stats", "zeros"):
        raise SystemExit(f"unknown film_input {cfg.film_input!r} in checkpoint")
    w = sd.get("stem.0.weight", None)
    if w is not None:
        want = 3 + 4 + MultiScaleDensity(cfg).out_dim
        got = int(w.shape[1])
        if want != got:
            raise SystemExit(
                f"checkpoint stem expects {got} input channels but this config "
                f"builds {want}; the density configuration does not match.")
    cfg.__post_init__()
    return cfg


# Re-exported under the names the existing scripts import, so an eval script
# needs ONE line changed:  from edn_model import ...  ->  from edn_ablate import ...
EDNet = EDNetAbl
EDNConfigAbl = AblConfig


def print_variants(N: int = 10240):
    print(f"{'variant':14s} {'params(M)':>10s} {'dens ch':>8s} {'stem':>5s}   description")
    for name in VARIANTS:
        cfg = build_config(name, seq_len=N)
        m = EDNetAbl(cfg)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:14s} {n:10.3f} {m.density.out_dim:8d} "
              f"{m.stem[0].in_features:5d}   {VARIANTS[name][2]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ablation model definitions")
    ap.add_argument("--self_check", action="store_true",
                    help="assert the default config reproduces EDNet exactly")
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()

    if args.variants:
        print_variants()

    if args.self_check:
        torch.manual_seed(0)
        B, N = 2, 2048
        feat = torch.rand(B, N, 4) * 2 - 1
        feat[..., 2] = torch.sort(feat[..., 2], dim=1).values
        feat[..., 3] = (torch.rand(B, N) > 0.5).float()
        meta = torch.tensor([[33000.0, N], [2554.0, N]])

        torch.manual_seed(1); base = _M.EDNet(EDNConfig(seq_len=N)).eval()
        torch.manual_seed(1); abl = EDNetAbl(AblConfig(seq_len=N)).eval()
        with torch.no_grad():
            a, b = base(feat, meta), abl(feat, meta)
        d = (a - b).abs().max().item()
        print(f"  default AblConfig vs EDNet: max |delta logit| = {d:.3e}  "
              f"{'IDENTICAL' if d == 0.0 else 'DIFFERS -- investigate'}")

        # A1 must equal log(count+0.5) exactly.
        torch.manual_seed(1)
        raw = EDNetAbl(AblConfig(seq_len=N, density_norm="raw")).eval()
        x_px, y_px, t_us, pol, dur = raw._physical(feat, meta)
        dens, _ = raw.density(x_px, y_px, t_us, pol, dur)
        corr = raw._denorm(dur, N)
        # channel 0 of scale 0: log(n_all+0.5); recover n_all and check >= 0
        n_all = torch.exp(dens[..., 0] + corr[..., 0]) - 0.5
        print(f"  A1 recovered raw count: min {n_all.min():.4f} "
              f"max {n_all.max():.1f}  (min must be >= -1e-4)")
        assert n_all.min() > -1e-4, "de-normalisation is wrong"
        print("  A1 de-normalisation exact.")
