"""LSTM / GRU walk-forward with the same information sets as the tree models."""
import sys, os, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from experiment import build, OUT

torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, nf, cell="LSTM", h=32):
        super().__init__()
        R = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = R(nf, h, batch_first=True)
        self.drop = nn.Dropout(0.2); self.head = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


def windows(X, L):
    # X: (n, f) -> (n, L, f) where window i ends at row i (rows < L-1 padded by edge)
    pad = np.repeat(X[:1], L - 1, axis=0)
    Xp = np.vstack([pad, X])
    idx = np.arange(L)[None, :] + np.arange(len(X))[:, None]
    return Xp[idx]


def fit_predict(Xtr, ytr, Xte, cell, seed, epochs=30, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    nv = int(len(Xtr) * 0.15)
    Xa, ya, Xv, yv = Xtr[:-nv], ytr[:-nv], Xtr[-nv:], ytr[-nv:]
    net = Net(Xtr.shape[2], cell); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    ta, tya = torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya, dtype=torch.float32)
    tv, tyv = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(ta))
        for i in range(0, len(ta), 128):
            b = perm[i:i+128]; opt.zero_grad()
            l = lossf(net(ta[b]), tya[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = lossf(net(tv), tyv).item()
        if vl < best - 1e-4: best, bstate, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))).numpy()


def run(market, targets_, sets, cells=("LSTM", "GRU"), has_fb=True, nyt_mode="preopen", tag="",
        L=20, block=3, seeds=(0, 1, 2), first_test_year=None):
    df, tech, glob_, sent = build(market, has_fb=has_fb, nyt_mode=nyt_mode)
    lagged = df[tech + glob_].shift(1).add_suffix("_lag1"); df = df.join(lagged)
    tl = [c + "_lag1" for c in tech]; gl = [c + "_lag1" for c in glob_]
    fty = first_test_year or (2009 if market == "IN" else 1998)
    for tg in targets_:
        for fs in sets:
            if tg == "up_today":
                feats = {"T": tl, "T+S": tl + sent, "T+G": tl + gl, "T+G+S": tl + gl + sent}[fs]
            else:
                feats = {"T": tech, "T+S": tech + sent, "T+G": tech + glob_, "T+G+S": tech + glob_ + sent}[fs]
            d = df[feats + [tg]].copy()
            d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            years = sorted(d.index.year.unique()); last = years[-1]
            for cell in cells:
                t0 = time.time(); preds = []
                for y0 in range(fty, last + 1, block):
                    trm = (d.index.year < y0); tem = (d.index.year >= y0) & (d.index.year < y0 + block)
                    mu = d.loc[trm, feats].mean(); sd = d.loc[trm, feats].std().replace(0, 1)
                    X = ((d[feats] - mu) / sd).clip(-6, 6).values.astype(np.float32)
                    W = windows(X, L); y = d[tg].values
                    tri = np.where(trm & ~np.isnan(y))[0][:-5]; tei = np.where(tem & ~np.isnan(y))[0]
                    if len(tei) == 0: continue
                    p = np.mean([fit_predict(W[tri], y[tri], W[tei], cell, s) for s in seeds], axis=0)
                    preds.append(pd.DataFrame({"y": y[tei], "p": p, "maj": float(y[tri].mean() > .5),
                                               "base_rate": y[tri].mean()}, index=d.index[tei]))
                o = pd.concat(preds); key = f"{market}{tag}__{tg}__{cell}__{fs}"
                o.to_parquet(f"{OUT}/{key}.parquet")
                acc = ((o.p > .5) == (o.y > .5)).mean()
                print(f"{key:55s} n={len(o)} acc={acc:.4f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(**json.loads(sys.argv[1]))
