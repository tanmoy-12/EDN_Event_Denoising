from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

SENSOR_W, SENSOR_H = 346, 260  # override via ed24_data.set_sensor_geometry


# ==============================================================================
# 0. CONFIG
# ==============================================================================
@dataclass
class EDNConfig:
    seq_len: int = 4096

    # ---- how temporal extents are chosen -------------------------------
    # "relative": every temporal extent is a fraction of the window duration,
    #             floored at min_cell_us.  Rungs stay distinct at any rate.
    # "absolute": the previous behaviour -- fixed us constants, clamped to the
    #             window duration.  Kept for the ablation.
    # MEASURED: "relative" costs -0.029 / -0.046 mean AUC on DND21 hotel-bar /
    # driving against an unchanged ED24 validation score, and makes the sweep
    # 10-13x more erratic (mean |second difference| 0.0037 -> 0.0373).  The
    # absolute extents are load-bearing: a contrast edge has a characteristic
    # correlation time set by sensor dynamics and shot noise does not, so a bin
    # that travels with the window can no longer measure it.  Rate-invariance
    # belongs in the NORMALISATION of the counts, not in the scales at which
    # they are taken.  "relative" is kept only as that ablation.
    scale_mode: str = "absolute"
    min_cell_us: float = 25.0

    # ---- density descriptor ----
    # Spatial radii are sensor-physical and stay absolute: a pixel is a pixel.
    # Temporal extents come from density_time_fracs in relative mode and from
    # the second element of density_scales in absolute mode.  The two lists
    # describe the SAME ladder -- the fractions are what the absolute ladder
    # works out to at ED24's own window length (~33 ms at 2.0 V):
    #     250/33000 ~ 1/128    1000/33000 ~ 1/32
    #    5000/33000 ~ 1/8     20000/33000 ~ 1/2
    density_scales: List[Tuple[float, float]] = field(default_factory=lambda: [
        (2.0, 250.0), (2.0, 1000.0), (4.0, 1000.0),
        (4.0, 5000.0), (8.0, 5000.0), (8.0, 20000.0),
    ])
    density_time_fracs: List[float] = field(default_factory=lambda: [
        1 / 128, 1 / 32, 1 / 32, 1 / 8, 1 / 8, 1 / 2,
    ])
    hash_size: int = 1 << 18   # see the collision note in MultiScaleDensity
    use_pixel_features: bool = True
    # Number of half-cell-offset grids the binned counts are averaged over.  A
    # single grid loses any neighbour falling on the far side of a cell
    # boundary.  Measured single-channel AUC on ED24 Bicycle_01_2.0:
    #   (8px, 5ms):  1 grid 0.798 | 2 grids 0.842 | 4 grids 0.855
    #   (3px, 5ms):  1 grid 0.703 | 2 grids 0.760 | 4 grids 0.787
    density_grid_shifts: int = 4

    # ---- motion-adaptive density ----
    motion_scales: List[Tuple[float, float]] = field(default_factory=lambda: [
        (4.0, 5000.0), (8.0, 20000.0),
    ])
    motion_time_fracs: List[float] = field(default_factory=lambda: [1 / 8, 1 / 2])
    # Absolute mode: px per millisecond.  Relative mode: px of displacement
    # across the WHOLE window, which is the same bank expressed in units that
    # do not shrink with the window.  +-0.5 px/ms over ED24's 33 ms window is
    # +-16.5 px/window, hence the bank below; over a 2.6 ms driving window the
    # old bank displaced only +-1.3 px, less than one cell at r=4 or r=8, so
    # the sheared frames were duplicates of the stationary one and the whole
    # motion branch silently switched itself off at exactly the noise rates
    # where driving needs it.
    velocity_bank_px_per_ms: List[float] = field(
        default_factory=lambda: [-0.5, 0.0, 0.5])
    velocity_bank_px_per_window: List[float] = field(
        default_factory=lambda: [-16.0, 0.0, 16.0])
    motion_grid_shifts: int = 2

    # ---- neighbourhood graph ----
    k: int = 12
    band_dense: int = 16          # always-included immediate index neighbours
    # Stratified samples from the +-tau window.  64, not 48: at N=10240 a fixed
    # physical tau contains ~2.5x the events it did at N=4096, so a fixed
    # stratum count samples it 2.5x more thinly.  This holds the candidate
    # density roughly constant as the window grows.
    n_strata: int = 64
    # Absolute mode: fixed us.  Relative mode: fraction of the window duration,
    # floored.  2000/33000 ~ 1/16 and 20000/33000 ~ 1/2, so again the relative
    # values are what the absolute ones meant on ED24.
    knn_tau_fine_us: float = 2000.0
    knn_tau_coarse_us: float = 20000.0
    # tau_coarse is 1/4, not the 1/2 the absolute value implied.  Measured,
    # the absolute pair covered 2*tau/dur = 121% of the window at ED24's own
    # 33 ms and up to 999% on driving -- the "coarse" branch drew its 48 strata
    # from the ENTIRE window at every operating point in the paper, a 1.2%
    # uniform sample that is local in nothing.  At 1/4 the branch covers half
    # the window, stays 4x coarser than the fine branch, and samples it ~4x
    # more densely.
    knn_tau_fine_frac: float = 1 / 16
    knn_tau_coarse_frac: float = 1 / 4
    knn_tau_fine_min_us: float = 150.0
    knn_tau_coarse_min_us: float = 600.0
    sigma_xy_px: float = 3.0
    sigma_t_fine_us: float = 1000.0
    sigma_t_coarse_us: float = 20000.0
    # Relative mode ties sigma_t to tau at the ratio the absolute pair implied
    # (1000/2000 and 20000/20000), so the temporal term of the distance metric
    # keeps a constant weight against the spatial term at every rate.
    sigma_t_fine_ratio: float = 0.5
    sigma_t_coarse_ratio: float = 1.0

    # ---- backbone ----
    hidden_dim: int = 96
    num_blocks: int = 4
    rel_dim: int = 32
    ffn_expansion: int = 2
    dropout: float = 0.0
    use_checkpoint: bool = False

    # ---- loss ----
    label_smoothing: float = 0.01
    max_pos_weight: float = 4.0
    min_pos_weight: float = 0.25

    def __post_init__(self):
        if self.scale_mode not in ("relative", "absolute"):
            raise ValueError(f"scale_mode must be relative|absolute, "
                             f"got {self.scale_mode!r}")
        if len(self.density_time_fracs) != len(self.density_scales):
            raise ValueError("density_time_fracs must match density_scales")
        if len(self.motion_time_fracs) != len(self.motion_scales):
            raise ValueError("motion_time_fracs must match motion_scales")


# ==============================================================================
# 1. SCALE-FREE DENSITY DESCRIPTOR (+ per-pixel channels)
# ==============================================================================
class MultiScaleDensity(nn.Module):
    """Non-learned, per-event, noise-rate-invariant local statistics.

    Per spatiotemporal scale (4 channels):
        log(observed / expected)                       -- all events in the cell
        log(distinct-pixel support / expected)         -- repeats excluded
        log(same-polarity count / expected)
        same-polarity fraction, centred
    Per motion scale (2 channels):
        log(best sheared-frame count / expected)
        log(best sheared-frame count / stationary count)
    Plus, if cfg.use_pixel_features, 3 channels shared across scales:
        log(same-pixel count / expected)
        log(dt to nearest same-pixel event / expected interval)
        a hot-pixel indicator

    "expected" is always the count that would be seen if every event in this
    window were uniform background activity at this window's own measured rate.
    That is what makes the descriptors invariant to the noise rate: doubling
    the rate doubles numerator and denominator together for a noise event,
    while a structured event stays anomalous.
    """

    _P1, _P2, _P3, _P4 = 73856093, 19349663, 83492791, 2654435761

    _SHIFTS = {
        1: [(0.0, 0.0, 0.0)],
        2: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        4: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)],
    }

    def __init__(self, cfg: EDNConfig):
        super().__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.density_scales)
        self.n_motion = len(cfg.motion_scales)
        self.out_dim = (4 * self.n_scales + 2 * self.n_motion
                        + (3 if cfg.use_pixel_features else 0))

    # ---- temporal extents -------------------------------------------------
    def _extents(self, dur: torch.Tensor, nominal: List[float],
                 fracs: List[float]) -> torch.Tensor:
        """(B,) durations -> (S, B, 1) effective temporal cell extents.

        Relative mode floors at cfg.min_cell_us rather than clamping at the
        window length, so two nominally different rungs can never become the
        same rung.  Absolute mode clamps at the window length, which is the
        behaviour that produced the duplicates.
        """
        cfg = self.cfg
        B = dur.shape[0]
        if cfg.scale_mode == "relative":
            fr = torch.as_tensor(fracs, device=dur.device, dtype=dur.dtype)
            rt = (fr.view(-1, 1) * dur.view(1, B)).clamp_min(cfg.min_cell_us)
        else:
            nom = torch.as_tensor(nominal, device=dur.device, dtype=dur.dtype)
            rt = torch.minimum(nom.view(-1, 1), dur.view(1, B))
        return rt.clamp_min(1.0).unsqueeze(-1)          # (S, B, 1)

    @staticmethod
    def _counts(keys: torch.Tensor, weights: torch.Tensor, table: int) -> torch.Tensor:
        """keys (B,N) int64 in [0,table), weights (B,N) -> per-event bucket total."""
        B, N = keys.shape
        acc = torch.zeros(B, table, device=keys.device, dtype=weights.dtype)
        acc.scatter_add_(1, keys, weights)
        return acc.gather(1, keys)

    @staticmethod
    def _same_pixel_dt(pid: torch.Tensor, t_us: torch.Tensor) -> torch.Tensor:
        """(B,N) -> time to the nearest OTHER event at the same pixel.

        Events are time-ordered, so a stable sort by pixel id leaves each
        pixel's events in time order and the gaps fall out of a single diff.
        O(N log N), fully batched, no python loop over pixels.
        """
        big = torch.full_like(t_us, 1e9)
        order = torch.argsort(pid, dim=1, stable=True)
        pid_s = torch.gather(pid, 1, order)
        t_s = torch.gather(t_us, 1, order)

        gap = t_s[:, 1:] - t_s[:, :-1]
        same = pid_s[:, 1:] == pid_s[:, :-1]
        gap = torch.where(same, gap, torch.full_like(gap, 1e9))

        prev = big.clone(); prev[:, 1:] = gap      # gap to the earlier event
        nxt = big.clone(); nxt[:, :-1] = gap       # gap to the later event
        md_s = torch.minimum(prev, nxt)

        out = torch.empty_like(md_s)
        out.scatter_(1, order, md_s)
        return out

    def forward(self, x_px, y_px, t_us, pol, dur_us):
        """All (B,N) except dur_us (B,).
        -> (feats (B,N,out_dim), ladder position (B,n_scales)).

        The second return used to be a per-scale saturation FLAG -- 1 where the
        rung had clamped to the window.  In relative mode nothing clamps, so it
        carries log10(extent / duration) instead: where each rung actually sits
        in the window, which is strictly more information in the same width and
        is what the FiLM conditioner needs in either mode.

        Three corrections matter most at the fine scales, where counts are
        small and a constant offset dominates:

        (a) SELF-COUNT REMOVED.  Every event falls in its own cell.  Measured
            on ED24 at (2 px, 250 us) the mean cell count is 1.0347, of which
            the event itself is 1.0 -- so an uncorrected count is 97% constant.
        (b) HASH-COLLISION BIAS REMOVED.  With N events in a table of size T
            each event collects (N-1)/T spurious counts in expectation.  At the
            old T = 2^16 with N = 4096 that is 0.0625, nearly twice the real
            signal at the finest scale.  T is now 2^18 and the residual
            expectation is subtracted analytically.
        (c) REPETITION SEPARATED FROM SUPPORT.  A hot pixel firing ten times
            and a moving edge lighting ten neighbouring pixels give the same
            cell count and opposite labels, so (cell, pixel) is hashed too and
            support from OTHER pixels is reported separately.
        """
        B, N = x_px.shape
        cfg = self.cfg
        table = cfg.hash_size
        ones = torch.ones_like(x_px)
        dur = dur_us.clamp_min(1.0)
        shifts = self._SHIFTS[cfg.density_grid_shifts]
        pid = (y_px.long() * SENSOR_W + x_px.long()).clamp_(
            0, SENSOR_W * SENSOR_H - 1)
        collide = (N - 1) / float(table)      # expected spurious count per grid

        rt_all = self._extents(dur, [s[1] for s in cfg.density_scales],
                               cfg.density_time_fracs)

        feats, pos = [], []
        for si, (r, _) in enumerate(cfg.density_scales):
            rt_eff = rt_all[si]                                   # (B,1)
            pos.append(torch.log10(rt_eff.squeeze(1) / dur))

            n_all = torch.zeros_like(x_px)
            n_on = torch.zeros_like(x_px)
            n_pix = torch.zeros_like(x_px)
            for (sx, sy, st) in shifts:
                cx = torch.div(x_px + sx * r, r, rounding_mode="floor").long()
                cy = torch.div(y_px + sy * r, r, rounding_mode="floor").long()
                ct = torch.div(t_us + st * rt_eff, rt_eff, rounding_mode="floor").long()
                key = ((cx * self._P1) ^ (cy * self._P2) ^ (ct * self._P3))
                key = key.remainder(table)
                n_all = n_all + self._counts(key, ones, table)
                n_on = n_on + self._counts(key, pol, table)
                # same cell AND same pixel -> repetition rather than support
                key2 = (key * self._P4 ^ pid * self._P2).remainder(table)
                n_pix = n_pix + self._counts(key2, ones, table)
            inv = 1.0 / len(shifts)
            n_all = (n_all * inv - 1.0 - collide).clamp_min(0.0)   # (a), (b)
            n_pix = (n_pix * inv - 1.0 - collide).clamp_min(0.0)
            n_on = n_on * inv
            n_same = torch.where(pol > 0.5, n_on, n_all + 1.0 + collide - n_on)
            n_same = (n_same - 1.0).clamp_min(0.0)
            n_dist = (n_all - n_pix).clamp_min(0.0)                # (c)

            cell_volume_frac = (r * r * rt_eff.squeeze(1)) / (SENSOR_W * SENSOR_H * dur)
            expected = ((N - 1) * cell_volume_frac).unsqueeze(1)

            feats.append(torch.log((n_all + 0.5) / (expected + 0.5)))
            feats.append(torch.log((n_dist + 0.5) / (expected + 0.5)))
            feats.append(torch.log((n_same + 0.5) / (0.5 * expected + 0.5)))
            feats.append(n_same / (n_all + 1e-3) - 0.5)

        # ---- motion-adaptive support ----
        #
        # A real event sits on a moving edge, so its neighbours are displaced
        # in (x,y) in proportion to elapsed time; an axis-aligned cell captures
        # them only if the motion is slow relative to the cell size, which on a
        # fast driving sequence it is not.
        #
        # The standard remedy is motion compensation, but estimating a motion
        # field from the event stream is circular exactly where it is needed:
        # the estimate degrades as noise rises.  Instead the coordinates are
        # sheared by each velocity in a small FIXED bank and the per-event
        # maximum count is taken.  Nothing is estimated, nothing has to
        # converge, and under heavy noise the behaviour degrades to the
        # isotropic count rather than to a confidently wrong estimate.
        #
        # Measured on ED24 Bicycle_01 at 2.0V, single channel, (4 px, 5 ms):
        #     isotropic       0.7996 clean / 0.7259 at 10 Hz/px injected
        #     9-velocity max  0.8352        / 0.7611     (+0.036 / +0.035)
        #     25-velocity max 0.8333        / 0.7632     (no better than 9)
        # A max-minus-mean "contrast" variant was worse than the max alone
        # (0.7976 / 0.6894), so only the max and its margin over the
        # stationary frame are emitted.  That sweep densified a FIXED range; it
        # says nothing about the range itself, which is what relative mode
        # changes.
        if self.n_motion:
            mshifts = self._SHIFTS[cfg.motion_grid_shifts]
            relative = cfg.scale_mode == "relative"
            bank = (cfg.velocity_bank_px_per_window if relative
                    else cfg.velocity_bank_px_per_ms)
            # displacement multiplier: px per unit of `clock`
            clock = (t_us / dur.view(B, 1)) if relative else (t_us / 1000.0)
            mrt_all = self._extents(dur, [s[1] for s in cfg.motion_scales],
                                    cfg.motion_time_fracs)
            for mi, (r, _) in enumerate(cfg.motion_scales):
                rt_eff = mrt_all[mi]                              # (B,1)
                best = None
                zero = None
                for vx in bank:
                    for vy in bank:
                        xs = x_px - vx * clock
                        ys = y_px - vy * clock
                        c = torch.zeros_like(x_px)
                        for (sx, sy, st) in mshifts:
                            cx = torch.div(xs + sx * r, r, rounding_mode="floor").long()
                            cy = torch.div(ys + sy * r, r, rounding_mode="floor").long()
                            ct = torch.div(t_us + st * rt_eff, rt_eff,
                                           rounding_mode="floor").long()
                            key = ((cx * self._P1) ^ (cy * self._P2)
                                   ^ (ct * self._P3)).remainder(table)
                            c = c + self._counts(key, ones, table)
                        c = (c / len(mshifts) - 1.0 - collide).clamp_min(0.0)
                        best = c if best is None else torch.maximum(best, c)
                        if vx == 0.0 and vy == 0.0:
                            zero = c
                cell_volume_frac = (r * r * rt_eff.squeeze(1)) / (SENSOR_W * SENSOR_H * dur)
                expected = ((N - 1) * cell_volume_frac).unsqueeze(1)
                feats.append(torch.log((best + 0.5) / (expected + 0.5)))
                feats.append(torch.log((best + 0.5) / (zero + 0.5)))

        # ---- per-pixel (hot-pixel) channels ----
        #
        # ED24's noise is measured DAVIS346 background activity, which includes
        # hot pixels: on Bicycle_01_2.0 the top 50 pixels hold 10.3% of all
        # noise events but 0.5% of all signal events, and one pixel fires 284
        # times in a second against a signal maximum of 26.  A purely
        # neighbourhood-density model cannot separate that, since any small box
        # around a hot pixel is dominated by its own repeats.
        #
        # The relationship is NOT monotone -- a logistic probe on the raw
        # same-pixel count scores AUC 0.44 as a noise score, because ordinary
        # edge events also repeat -- which is exactly why these belong as input
        # channels to a network rather than as another hand-set threshold.  In
        # a linear probe they lift held-out AUC from 0.897 (density only) to
        # 0.915.
        if cfg.use_pixel_features:
            n_px = self._counts(pid, ones, SENSOR_W * SENSOR_H)
            exp_px = torch.full_like(n_px, float(N) / (SENSOR_W * SENSOR_H))
            feats.append(torch.log((n_px + 0.5) / (exp_px + 0.5)))

            dt_px = self._same_pixel_dt(pid, t_us)
            exp_dt = (dur * SENSOR_W * SENSOR_H / N).unsqueeze(1)
            feats.append(torch.log(dt_px.clamp_min(1.0) / exp_dt.clamp_min(1.0)))

            feats.append((n_px >= n_px.amax(dim=1, keepdim=True) * 0.5).float() - 0.5)

        return torch.stack(feats, dim=-1), torch.stack(pos, dim=-1)


# ==============================================================================
# 2. NEIGHBOURHOOD GRAPH OVER A PHYSICAL TIME WINDOW
# ==============================================================================
def band_offsets(cfg: EDNConfig, device) -> torch.Tensor:
    """The always-included immediate index neighbours (a small dense band).

    These are cheap and, being adjacent in time, are genuine candidates at any
    rate.  Everything beyond them comes from a physical-time window.
    """
    pos = torch.arange(1, cfg.band_dense + 1)
    return torch.cat([-pos.flip(0), pos]).to(device)


def _gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B,N,D), idx (B,N,K) -> (B,N,K,D) without an N x N expand."""
    B, N, D = x.shape
    K = idx.shape[-1]
    flat = idx.reshape(B, N * K)
    return torch.gather(x, 1, flat.unsqueeze(-1).expand(B, N * K, D)).reshape(B, N, K, D)


def physical_knn(coords_scaled: torch.Tensor, t_us: torch.Tensor,
                 tau_us, offsets: torch.Tensor, k: int,
                 n_strata: int) -> torch.Tensor:
    """k nearest neighbours, candidates drawn from a PHYSICAL time window.

    `tau_us` is a float or a (B,1) tensor -- per-window in relative mode, since
    a constant tau is a different fraction of every window (see the module
    docstring: 20 ms against a 2.6 ms window is the whole window).

    WHY NOT AN INDEX BAND
    ---------------------
    Picking candidates at fixed offsets in the event index makes the
    neighbourhood's physical extent depend on the event rate -- the one
    variable the architecture is supposed to be invariant to.  Measured on ED24
    with synthetic noise injected, a +-512-event band spans:

        0 Hz/px injected  ->  4.03 ms        5 Hz/px  ->  0.93 ms
        1 Hz/px           ->  2.50 ms       10 Hz/px  ->  0.52 ms
                                            20 Hz/px  ->  0.40 ms

    a tenfold contraction across the sweep.  The graph the attention runs on
    was therefore the last rate-dependent component of the model, sitting
    directly underneath the claim that the model is rate-invariant.

    HOW
    ---
    Events are sorted in t, so the index range covering [t_i - tau, t_i + tau]
    is two searchsorted lookups.  `n_strata` stratified samples are spread
    across that range, unioned with the dense index band, and the k nearest by
    scaled distance are taken.  The candidate set spans a constant physical
    extent at any rate; the sampling density inside it varies instead, which is
    the harmless half of the trade -- provided tau is a sane fraction of the
    window.  When it is not, the strata spread over everything and the draw
    stops being local at all.
    """
    B, N, _ = coords_scaled.shape
    dev = coords_scaled.device
    idx = torch.arange(N, device=dev).view(1, N)
    if not torch.is_tensor(tau_us):
        tau_us = torch.as_tensor(float(tau_us), device=dev,
                                 dtype=t_us.dtype).view(1, 1)

    lo = torch.searchsorted(t_us.contiguous(), (t_us - tau_us).contiguous(),
                            right=False)
    hi = torch.searchsorted(t_us.contiguous(), (t_us + tau_us).contiguous(),
                            right=True) - 1
    hi = torch.maximum(hi, lo)

    u = torch.linspace(0.0, 1.0, n_strata, device=dev).view(1, 1, n_strata)
    span = (hi - lo).unsqueeze(-1).float()
    strat = lo.unsqueeze(-1) + (u * span).round().long()

    band = (idx.unsqueeze(-1) + offsets.view(1, 1, -1)).expand(B, -1, -1)
    cand = torch.cat([band, strat], dim=-1)
    valid = (cand >= 0) & (cand <= N - 1) & (cand != idx.unsqueeze(-1))
    cand = cand.clamp(0, N - 1).expand(B, -1, -1).contiguous()

    gathered = _gather_points(coords_scaled, cand)
    d2 = ((gathered - coords_scaled.unsqueeze(2)) ** 2).sum(-1)
    d2 = d2.masked_fill(~valid, float("inf"))

    sel = torch.topk(d2, min(k, cand.shape[-1]), dim=-1, largest=False).indices
    return torch.gather(cand, 2, sel)


# ==============================================================================
# 3. RELATION-GATED NEIGHBOURHOOD ATTENTION
# ==============================================================================
class NeighborAttention(nn.Module):
    """Aggregate over k neighbours with scalar weights predicted from the
    pairwise relation (dx, dy, dt, distance, polarity agreement)."""

    def __init__(self, dim: int, rel_dim: int, dropout: float = 0.0):
        super().__init__()
        self.rel = nn.Sequential(nn.Linear(5, rel_dim), nn.GELU(),
                                 nn.Linear(rel_dim, rel_dim), nn.GELU())
        self.q = nn.Linear(dim, rel_dim)
        self.kk = nn.Linear(dim, rel_dim)
        # A single Linear here made the attention weights INDEPENDENT of the
        # query: W(rel + k - q) = W(rel + k) - Wq, and -Wq is constant along
        # the neighbour axis, which softmax cancels.  Verified numerically --
        # replacing q with unrelated random values moved the weights by 3e-7,
        # and dA/dW_q was ~5e-7.  A hidden nonlinearity makes the interaction
        # real.
        self.score = nn.Sequential(nn.Linear(rel_dim, rel_dim), nn.GELU(),
                                   nn.Linear(rel_dim, 1))
        self.v = nn.Linear(dim, dim)
        self.rel_to_v = nn.Linear(rel_dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, coords, pol, nbr):
        nb_coords = _gather_points(coords, nbr)
        nb_pol = _gather_points(pol.unsqueeze(-1), nbr)
        delta = nb_coords - coords.unsqueeze(2)
        dist = delta.norm(dim=-1, keepdim=True)
        agree = 1.0 - (nb_pol - pol.unsqueeze(-1).unsqueeze(2)).abs()
        rel = self.rel(torch.cat([delta, dist, agree], dim=-1))

        nb_k = _gather_points(self.kk(x), nbr)
        logits = self.score(rel + nb_k - self.q(x).unsqueeze(2)).squeeze(-1)
        alpha = torch.softmax(logits, dim=-1).unsqueeze(-1)

        nb_v = _gather_points(self.v(x), nbr) + self.rel_to_v(rel)
        return self.out(self.drop((alpha * nb_v).sum(dim=2)))


class DualScaleBlock(nn.Module):
    """Fine + coarse timescale neighbourhood attention, gated fusion,
    FiLM-modulated FFN."""

    def __init__(self, cfg: EDNConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.norm1 = nn.LayerNorm(d)
        self.fine = NeighborAttention(d, cfg.rel_dim, cfg.dropout)
        self.coarse = NeighborAttention(d, cfg.rel_dim, cfg.dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * cfg.ffn_expansion), nn.GELU(),
                                 nn.Linear(d * cfg.ffn_expansion, d))

    def forward(self, x, c_fine, c_coarse, pol, nbr_fine, nbr_coarse, film):
        h = self.norm1(x)
        a = self.fine(h, c_fine, pol, nbr_fine)
        b = self.coarse(h, c_coarse, pol, nbr_coarse)
        g = self.gate(torch.cat([a, b], dim=-1))
        x = x + g * a + (1.0 - g) * b

        scale, shift = film
        h = self.norm2(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x + self.ffn(h)


# ==============================================================================
# 4. THE MODEL
# ==============================================================================
class EDNet(nn.Module):
    def __init__(self, cfg: EDNConfig):
        super().__init__()
        self.cfg = cfg
        self.density = MultiScaleDensity(cfg)

        self.pol_embed = nn.Embedding(2, 4)
        in_dim = 3 + 4 + self.density.out_dim
        d = cfg.hidden_dim
        self.stem = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))

        # Global conditioner -> FiLM per block.  Inputs: log duration, log event
        # rate, pooled density statistics, and where each density rung sits in
        # this window, so the network is told the geometry of its own feature
        # ladder instead of having to infer it.
        cond_in = 2 + 2 * self.density.out_dim + self.density.n_scales
        self.cond = nn.Sequential(nn.Linear(cond_in, d), nn.GELU(),
                                  nn.Linear(d, d), nn.GELU())
        self.film = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(cfg.num_blocks)])
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.blocks = nn.ModuleList([DualScaleBlock(cfg) for _ in range(cfg.num_blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.register_buffer("_offsets", band_offsets(cfg, "cpu"), persistent=False)

    def _physical(self, feat, meta):
        x_px = (feat[..., 0] + 1.0) * 0.5 * (SENSOR_W - 1)
        y_px = (feat[..., 1] + 1.0) * 0.5 * (SENSOR_H - 1)
        dur = meta[:, 0].clamp_min(1.0)
        t_us = (feat[..., 2] + 1.0) * 0.5 * dur.unsqueeze(1)
        return x_px, y_px, t_us, feat[..., 3], dur

    def _graph_scales(self, dur: torch.Tensor):
        """(B,) -> (tau_fine, tau_coarse, sigma_t_fine, sigma_t_coarse).

        Each is (B,1) in relative mode and a broadcastable scalar tensor in
        absolute mode.  sigma_t tracks tau so that the temporal term of the
        distance metric keeps a constant weight against the spatial term; with
        a fixed sigma_t and a shrinking window the metric silently becomes
        purely spatial, which is what it had become on driving.
        """
        cfg = self.cfg
        B = dur.shape[0]
        if cfg.scale_mode == "relative":
            tf = (cfg.knn_tau_fine_frac * dur).clamp_min(
                cfg.knn_tau_fine_min_us).view(B, 1)
            tc = (cfg.knn_tau_coarse_frac * dur).clamp_min(
                cfg.knn_tau_coarse_min_us).view(B, 1)
        else:
            tf = torch.full((B, 1), float(cfg.knn_tau_fine_us),
                            device=dur.device, dtype=dur.dtype)
            tc = torch.full((B, 1), float(cfg.knn_tau_coarse_us),
                            device=dur.device, dtype=dur.dtype)
        if cfg.scale_mode == "relative":
            sf = (tf * cfg.sigma_t_fine_ratio).clamp_min(1.0)
            sc = (tc * cfg.sigma_t_coarse_ratio).clamp_min(1.0)
        else:
            sf = torch.full_like(tf, float(cfg.sigma_t_fine_us))
            sc = torch.full_like(tc, float(cfg.sigma_t_coarse_us))
        return tf, tc, sf, sc

    def forward(self, feat: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """feat (B,N,4) = [x_n, y_n, t_n, polarity]; meta (B,2) = [dur_us, n]
        -> (B,N) logits of P(noise)."""
        cfg = self.cfg
        x_px, y_px, t_us, pol, dur = self._physical(feat, meta)

        dens, ladder = self.density(x_px, y_px, t_us, pol, dur)

        h = torch.cat([feat[..., :3], self.pol_embed((pol > 0.5).long()), dens], dim=-1)
        x = self.stem(h)

        rate = meta[:, 1] / (dur / 1e6)
        gstat = torch.cat([
            torch.log10(dur.unsqueeze(1)),
            torch.log10(rate.clamp_min(1.0).unsqueeze(1)),
            dens.mean(dim=1), dens.amax(dim=1), ladder,
        ], dim=-1)
        cond = self.cond(gstat)

        tau_f, tau_c, sig_f, sig_c = self._graph_scales(dur)
        s_xy = cfg.sigma_xy_px
        c_fine = torch.stack([x_px / s_xy, y_px / s_xy, t_us / sig_f], dim=-1)
        c_coarse = torch.stack([x_px / s_xy, y_px / s_xy, t_us / sig_c], dim=-1)
        nbr_fine = physical_knn(c_fine, t_us, tau_f,
                                self._offsets, cfg.k, cfg.n_strata)
        nbr_coarse = physical_knn(c_coarse, t_us, tau_c,
                                  self._offsets, cfg.k, cfg.n_strata)

        for i, blk in enumerate(self.blocks):
            scale, shift = self.film[i](cond).chunk(2, dim=-1)
            args = (x, c_fine, c_coarse, pol, nbr_fine, nbr_coarse, (scale, shift))
            if cfg.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, *args, use_reentrant=False)
            else:
                x = blk(*args)

        return self.head(self.norm(x)).squeeze(-1)


# ==============================================================================
# 5. LOSS
# ==============================================================================
def denoise_loss(logits: torch.Tensor, label: torch.Tensor,
                 cfg: EDNConfig) -> torch.Tensor:
    """Weighted BCE with a per-batch positive weight.

    Focal loss with a fixed alpha is the wrong tool here: across ED24's 21
    voltage levels the noise fraction sweeps from well under 1% to most of the
    stream, so the minority class switches sides mid-dataset and no single
    alpha is right at both ends.  A pos_weight recomputed from the batch's own
    balance tracks that automatically.  Mild label smoothing keeps logits
    unsaturated, which matters because AUC is computed on the continuous score.
    """
    import torch.nn.functional as F
    pos = label.sum()
    neg = label.numel() - pos
    w = (neg / pos.clamp_min(1.0)).clamp(cfg.min_pos_weight, cfg.max_pos_weight)
    eps = cfg.label_smoothing
    target = label * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)


# ==============================================================================
# 6. CLASSICAL BASELINE (unlearned) -- keep it in every eval table
# ==============================================================================
# Pinned to absolute mode: this is the published BAF, a fixed (3 px, 5 ms)
# filter.  Making it window-relative would silently change the baseline every
# table in the paper is measured against.
_BAF_CFG = EDNConfig(scale_mode="absolute",
                     density_scales=[(3.0, 5000.0)], density_time_fracs=[1 / 8],
                     motion_scales=[], motion_time_fracs=[],
                     use_pixel_features=False)
_BAF_MOD = MultiScaleDensity(_BAF_CFG)


@torch.no_grad()
def baf_baseline_score(feat: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """BAF-style score, HIGH means noise.  Same convention as EDNet, so it
    drops into the same AUC call.  If the learned model cannot beat this, the
    problem is the pipeline, not the architecture."""
    x_px = (feat[..., 0] + 1.0) * 0.5 * (SENSOR_W - 1)
    y_px = (feat[..., 1] + 1.0) * 0.5 * (SENSOR_H - 1)
    dur = meta[:, 0].clamp_min(1.0)
    t_us = (feat[..., 2] + 1.0) * 0.5 * dur.unsqueeze(1)
    d, _ = _BAF_MOD.to(feat.device)(x_px, y_px, t_us, feat[..., 3], dur)
    return -d[..., 0]


# ==============================================================================
# 7. CHECKPOINT COMPATIBILITY
# ==============================================================================
def config_from_checkpoint(ck, seq_len: int = 4096) -> EDNConfig:
    """-> an EDNConfig matching the architecture actually stored in `ck`.

    A pickled dataclass restores exactly the __dict__ it was saved with, so an
    instance written by an older revision simply lacks the fields that revision
    did not have, and getattr on it silently returns today's class default.
    Fields this class no longer declares are ignored.  Anything that changes
    the parameter set is then cross-checked against the WEIGHTS, which cannot
    lie about what was trained.

    scale_mode is the one field where today's default is WRONG for an older
    checkpoint: a checkpoint saved before the field existed was trained with
    absolute extents, and silently giving it the new default would change what
    every one of its features means without changing a single weight.  So its
    absence is read as "absolute", not as the class default.
    """
    saved = ck.get("cfg", None)
    sd = ck.get("model", {}) or {}
    cfg = EDNConfig(seq_len=seq_len)

    if saved is not None:
        inst = getattr(saved, "__dict__", {})
        for f in cfg.__dataclass_fields__:
            if f in inst:
                setattr(cfg, f, inst[f])
        if "scale_mode" not in inst:
            cfg.scale_mode = "absolute"
        dropped = [f for f in ("use_mctr", "use_global", "causal_blocks",
                               "n_latents", "n_global_layers")
                   if f in inst and inst[f] not in (False, 0, None)]
        if dropped:
            raise SystemExit(
                f"\nThis checkpoint was trained with modules this build no "
                f"longer contains: {', '.join(dropped)}.\n"
                f"Use the revision of edn_model.py that defines them, or "
                f"retrain.\n")

    for pre, name in (("mctr.", "MCTR"), ("globalctx", "GlobalContext")):
        if any(k.startswith(pre) for k in sd):
            raise SystemExit(
                f"\nThis checkpoint contains {name} weights, which this build "
                f"does not define.\nUse the revision of edn_model.py that "
                f"does, or retrain.\n")

    # The stem's input width pins down how many density channels the
    # checkpoint was built with, catching a motion_scales or
    # use_pixel_features mismatch before it becomes a confusing error.
    w = sd.get("stem.0.weight", None)
    if w is not None:
        want = 3 + 4 + MultiScaleDensity(cfg).out_dim
        got = int(w.shape[1])
        if want != got:
            raise SystemExit(
                f"checkpoint stem expects {got} input channels but this config "
                f"builds {want}. The density configuration (motion_scales, "
                f"density_scales, use_pixel_features) does not match the "
                f"checkpoint. Retrain, or set those fields to the values used "
                f"for this checkpoint.")
    return cfg


if __name__ == "__main__":
    cfg = EDNConfig()
    m = EDNet(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"EDNet: {n/1e6:.3f} M parameters  (scale_mode={cfg.scale_mode})")
    print(f"  density channels {m.density.out_dim}  "
          f"stem input width {m.stem[0].in_features}")
    B, N = 2, 4096
    feat = torch.rand(B, N, 4) * 2 - 1
    feat[..., 2] = torch.sort(feat[..., 2], dim=1).values
    feat[..., 3] = (torch.rand(B, N) > 0.5).float()
    meta = torch.tensor([[33000.0, N], [2554.0, N]])
    with torch.no_grad():
        out = m(feat, meta)
    print(f"  forward OK: {tuple(out.shape)}  "
          f"logit range [{out.min():.3f}, {out.max():.3f}]")
