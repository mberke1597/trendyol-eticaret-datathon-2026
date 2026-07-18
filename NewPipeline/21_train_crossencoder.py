"""
Stage 21 — fine-tune the tyroberta CROSS-ENCODER relevance classifier (center piece).

For each of the N_FOLDS GroupKFold(term_id) folds:
  - train tyroberta + a fresh 2-class sequence-classification head on that fold's
    training rows (query [SEP] item_text, label in {0,1}, class-weighted loss),
  - predict P(relevant) on the held-out fold -> OOF prediction for those rows.

The concatenation of all folds' held-out predictions is a leakage-free OOF
`ce_relevance_prob` for the ENTIRE labeled train set: every row is scored by a
model that never saw its term_id. That column is:
  (1) a strong GBDT feature (join in 40_merge_features.py), and
  (2) an honest 4th stacking member alongside lgb/xgb/cat (41_train_ensemble.py),
  (3) optionally a standalone argmax classification submission (your "classify,
      don't threshold" request) — see 50_make_submission.py.

Each fold's fitted model is saved so 22_score_crossencoder.py can score the 3.36M
submission pairs (averaged over the 5 folds).

Why cross-encoder > the existing bi-encoder cosine: query and item are read
TOGETHER with full token-level attention, so "kırmızı elbise" vs "kırmızı
ayakkabı" is separable — the single-vector cosine cannot see that interaction.
And unlike zero-shot Qwen, it is trained on THIS competition's labels, so it does
not carry the zero-shot weakness that lost on the leaderboard.

Run (per fold, resumable):   python 21_train_crossencoder.py            # all folds
                             python 21_train_crossencoder.py --fold 0   # one fold
GPU: T4 fine (fp16). ~1 epoch over ~1.07M train rows/fold.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    CE_MODEL, CE_MAX_LEN, CE_EPOCHS, CE_BATCH_SIZE, CE_EVAL_BATCH_SIZE, CE_LR,
    CE_WARMUP_RATIO, CE_FP16, CE_POS_WEIGHT, CE_COL, CE_TRAIN_PARQUET, MODEL_DIR,
    N_FOLDS, RANDOM_SEED, fold_parquet,
)
from data import load_catalog, build_item_text, query_series

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PairDataset(Dataset):
    def __init__(self, queries, item_texts, labels, tokenizer, max_len):
        self.q = queries; self.it = item_texts; self.y = labels
        self.tok = tokenizer; self.max_len = max_len

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        enc = self.tok(self.q[i], self.it[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(int(self.y[i]), dtype=torch.long)
        return item


def _assemble_text(pairs, qmap, imap):
    q = qmap.reindex(pairs["term_id"].values).fillna("").tolist()
    it = imap.reindex(pairs["item_id"].values).fillna("").tolist()
    return q, it


def train_one_fold(fold, folds_df, qmap, imap, args):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    ckpt_dir = MODEL_DIR / f"ce_fold{fold}"
    oof_path = MODEL_DIR / f"ce_oof_fold{fold}.parquet"
    if oof_path.exists() and ckpt_dir.exists() and not args.overwrite:
        print(f"[21][fold {fold}] already done ({oof_path}) -- skip (use --overwrite to redo)")
        return pd.read_parquet(oof_path)

    tr = folds_df[folds_df.fold != fold].reset_index(drop=True)
    va = folds_df[folds_df.fold == fold].reset_index(drop=True)
    print(f"[21][fold {fold}] train={len(tr):,} val={len(va):,}")

    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=2).to(DEVICE)

    q_tr, it_tr = _assemble_text(tr, qmap, imap)
    q_va, it_va = _assemble_text(va, qmap, imap)
    ds_tr = PairDataset(q_tr, it_tr, tr["label"].values, tok, CE_MAX_LEN)
    ds_va = PairDataset(q_va, it_va, None, tok, CE_MAX_LEN)
    dl_tr = DataLoader(ds_tr, batch_size=CE_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=CE_EVAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=CE_LR)
    total_steps = len(dl_tr) * CE_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(CE_WARMUP_RATIO * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=CE_FP16 and DEVICE == "cuda")
    class_w = torch.tensor([1.0, CE_POS_WEIGHT], device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    model.train()
    t0 = time.time()
    for epoch in range(CE_EPOCHS):
        for step, batch in enumerate(dl_tr):
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CE_FP16 and DEVICE == "cuda"):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if step % 200 == 0:
                print(f"  [fold {fold}] ep{epoch} step {step}/{len(dl_tr)} "
                      f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

    # OOF predict
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dl_va:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=CE_FP16 and DEVICE == "cuda"):
                logit = model(**batch).logits
            probs.append(torch.softmax(logit.float(), dim=1)[:, 1].cpu().numpy())
    va[CE_COL] = np.concatenate(probs)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir); tok.save_pretrained(ckpt_dir)
    out = va[["term_id", "item_id", "label", CE_COL]]
    out.to_parquet(oof_path, index=False)
    try:
        from sklearn.metrics import roc_auc_score
        print(f"[21][fold {fold}] OOF AUC={roc_auc_score(va['label'], va[CE_COL]):.4f} "
              f"({time.time()-t0:.0f}s) -> {ckpt_dir}")
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1, help="train a single fold (default: all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    folds_df = pd.read_parquet(fold_parquet())
    items, terms = load_catalog()
    qmap = query_series(terms)
    imap = build_item_text(items)

    folds = [args.fold] if args.fold >= 0 else list(range(N_FOLDS))
    for f in folds:
        train_one_fold(f, folds_df, qmap, imap, args)

    # Assemble the full leakage-free OOF whenever all 5 per-fold files exist
    # (works whether folds were trained here in one run or one-at-a-time across
    # several sessions).
    fold_files = [MODEL_DIR / f"ce_oof_fold{f}.parquet" for f in range(N_FOLDS)]
    if all(p.exists() for p in fold_files):
        full = pd.concat([pd.read_parquet(p) for p in fold_files], ignore_index=True)
        full[["term_id", "item_id", CE_COL]].to_parquet(CE_TRAIN_PARQUET, index=False)
        try:
            from sklearn.metrics import roc_auc_score
            print(f"[21] FULL OOF AUC={roc_auc_score(full['label'], full[CE_COL]):.4f}")
        except Exception:
            pass
        print(f"[21] assembled OOF {full.shape} -> {CE_TRAIN_PARQUET}")
    else:
        done = [f for f in range(N_FOLDS) if fold_files[f].exists()]
        print(f"[21] folds done so far: {done}. Run the remaining folds to assemble the full OOF.")


if __name__ == "__main__":
    main()
