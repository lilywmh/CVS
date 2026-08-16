#!/usr/bin/env python3
"""APIM for indistinguishable dyads, without statsmodels.

Model (one per behavior b):
    own_outcome ~ z(own_b) + z(partner_b)

Non-independence within dyad is handled two ways, both reported:
  * cluster-robust (CR1) standard errors clustered on pair_id, t with df = G-1
  * Freedman-Lane permutation that shuffles reduced-model residuals in
    intact dyad blocks (preserves the two-member structure)

BH-FDR is applied across the whole family of actor+partner tests per outcome.
"""
import numpy as np, pandas as pd, argparse, json
from pathlib import Path

def z(v):
    v = np.asarray(v, float); return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)

def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b

def cr1_se(X, y, b, groups):
    n, k = X.shape; G = len(np.unique(groups))
    u = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g; Xg = X[m]; ug = u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.diag(c * XtXi @ meat @ XtXi)), G

def fl_perm(X, y, groups, j, nperm, rng):
    """Freedman-Lane p for coefficient j, permuting residuals in dyad blocks."""
    keep = [c for c in range(X.shape[1]) if c != j]
    Xr = X[:, keep]
    br = ols(Xr, y); fit = Xr @ br; res = y - fit
    b_obs = ols(X, y)[j]
    gs = np.unique(groups)
    idx = [np.where(groups == g)[0] for g in gs]
    null = np.empty(nperm)
    for i in range(nperm):
        order = rng.permutation(len(gs))
        rs = np.concatenate([res[idx[o]] for o in order])
        pos = np.concatenate(idx)
        yp = np.empty_like(y); yp[pos] = fit[pos] + rs
        null[i] = ols(X, yp)[j]
    return (np.sum(np.abs(null) >= abs(b_obs)) + 1) / (nperm + 1), b_obs

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        rank = n - r
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--outcome', default='partner_eval')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--nperm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    BEH = ['n_turns','speak_time','mean_turn_dur','ttr','q_rate','i_rate','we_rate','bc_rate']
    rng = np.random.default_rng(a.seed)
    rows = []
    for b in BEH:
        sub = d[[b, 'partner_' + b, a.outcome, 'pair_id']].dropna()
        y = z(sub[a.outcome].values)
        X = np.column_stack([np.ones(len(sub)), z(sub[b]), z(sub['partner_' + b])])
        g = sub.pair_id.values
        beta = ols(X, y); se, G = cr1_se(X, y, beta, g); df = G - 1
        for j, lab in [(1, 'actor'), (2, 'partner')]:
            t = beta[j] / se[j]
            pperm, _ = fl_perm(X, y, g, j, a.nperm, rng)
            crit = 2.069 if df == 23 else 2.0   # t_.975(23)
            rows.append(dict(behavior=b, effect=lab, beta=beta[j], se_cr1=se[j],
                             ci_lo=beta[j] - crit * se[j], ci_hi=beta[j] + crit * se[j],
                             t=t, p_perm=pperm, n=len(sub), n_dyads=G))
    R = pd.DataFrame(rows)
    R['q_perm_BH'] = bh(R.p_perm.values)
    R = R.round(4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False)
    print(f'outcome = {a.outcome} | n = {R.n.iloc[0]} 人 / {R.n_dyads.iloc[0]} 对 | '
          f'{a.nperm} 次置换 | seed {a.seed}')
    print(R[['behavior','effect','beta','ci_lo','ci_hi','p_perm','q_perm_BH']]
          .to_string(index=False))

if __name__ == '__main__':
    main()
