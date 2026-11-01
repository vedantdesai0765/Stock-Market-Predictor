"""
Company-targeted (entity-level) sentiment model.

Fine-tunes FinBERT (ProsusAI/finbert) as a sentence-pair classifier:
    input  = [CLS] company name [SEP] headline [SEP]
    output = sentiment of the headline TOWARDS that company (negative / neutral / positive)
Training data: SEntFiN 1.0 (Sinha et al., 2022; MIT licence) - 10,753 Economic Times
headlines with 14,404 entity-level labels. Split 80/10/10 by headline (no headline in two splits).
"""
import ast, hashlib, json, os, time, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

torch.set_num_threads(2); torch.manual_seed(0)
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, FINBERT
SRC = str(RAW / "sentfin" / "SEntFiN.csv"); BASE = str(FINBERT); OUT = str(RAW / "finbert_entity")
LAB = {"negative": 0, "neutral": 1, "positive": 2}

def load_pairs():
    d = pd.read_csv(SRC); rows = []
    for _, r in d.iterrows():
        try: dec = ast.literal_eval(r.Decisions)
        except Exception: continue
        for ent, lab in dec.items():
            if lab in LAB: rows.append((int(r["S No."]), r.Title, ent, LAB[lab], len(dec)))
    p = pd.DataFrame(rows, columns=["hid", "headline", "entity", "y", "n_ent"])
    h = p.headline.str.lower().map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 10)
    p["split"] = np.where(h < 8, "train", np.where(h == 8, "val", "test"))
    return p

def batches(tok, df, bs, shuffle):
    idx = np.arange(len(df))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = df.iloc[idx[i:i+bs]]
        x = tok(b.entity.tolist(), b.headline.tolist(), padding=True, truncation=True, max_length=64, return_tensors="pt")
        yield x, torch.tensor(b.y.values)

def evaluate(model, tok, df, bs=64):
    model.eval(); P = []
    with torch.inference_mode():
        for x, _ in batches(tok, df, bs, False):
            P.append(model(**x).logits.softmax(-1).numpy())
    return np.vstack(P)

def metrics(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    yh = p.argmax(1); return {"acc": round(accuracy_score(y, yh), 4), "macro_f1": round(f1_score(y, yh, average="macro"), 4), "n": int(len(y))}

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    p = load_pairs(); tr, va, te = [p[p.split == s].reset_index(drop=True) for s in ["train", "val", "test"]]
    print("pairs", len(p), "train/val/test", len(tr), len(va), len(te), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    # FinBERT's own labels are positive/negative/neutral; we re-initialise a 3-way head in our order
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3, ignore_mismatched_sizes=True)
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}; model.config.label2id = LAB
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    bs = 32; steps = epochs * int(np.ceil(len(tr) / bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    t0 = time.time(); best = -1
    for ep in range(epochs):
        model.train(); k = 0
        for x, y in batches(tok, tr, bs, True):
            loss = torch.nn.functional.cross_entropy(model(**x).logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); k += 1
            if k % 50 == 0: print(f"ep{ep} step {k} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        m = metrics(va.y.values, evaluate(model, tok, va)); print("val", ep, m, flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]; model.save_pretrained(OUT); tok.save_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT)
    pt = evaluate(model, tok, te); res = {"test_entity_model": metrics(te.y.values, pt)}
    multi = te.n_ent.values > 1
    res["test_multi_entity_headlines"] = metrics(te.y.values[multi], pt[multi])
    # baseline: plain (untargeted) FinBERT on the headline alone
    base = AutoModelForSequenceClassification.from_pretrained(BASE).eval()
    P = []
    with torch.inference_mode():
        for i in range(0, len(te), 64):
            x = tok(te.headline.iloc[i:i+64].tolist(), padding=True, truncation=True, max_length=64, return_tensors="pt")
            P.append(base(**x).logits.softmax(-1).numpy())
    P = np.vstack(P)[:, [1, 2, 0]]           # finbert order pos,neg,neu -> neg,neu,pos
    res["test_plain_finbert"] = metrics(te.y.values, P); res["test_plain_finbert_multi_entity"] = metrics(te.y.values[multi], P[multi])
    json.dump(res, open(f"{OUT}/eval.json", "w"), indent=1); print(json.dumps(res, indent=1), flush=True)
