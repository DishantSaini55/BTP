"""Leakage-free, from-scratch E1 runner for the frozen GP6 split."""
from __future__ import annotations

import argparse, csv, hashlib, json, random, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from .context_TF import ContextIndividualTF
    from .dynamic_fusion_TF import FusionContextIndividualTF
    from .individual_TF import IndividualTF
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from context_TF import ContextIndividualTF
    from dynamic_fusion_TF import FusionContextIndividualTF
    from individual_TF import IndividualTF


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"
SPLIT = RESULTS / "splits" / "gp6_from_scratch_v1"
SEEDS = list(range(20260928, 20260938))
NAMES = (
    "MTN (baseline)", "PACE-Net without SATIE (SAGE + MASC)",
    "PACE-Net (SAGE + MASC + SATIE)", "PACE-Net with ROUTE",
    "PACE-Net with ROUTE and SATIE",
)
KW = dict(N=3, d_model=32, d_ff=128, h=4, dropout=.1)
MAX_EPOCHS, PATIENCE = 50, 10


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def model_for(name: str) -> torch.nn.Module:
    if name == NAMES[0]: return IndividualTF(15, 4, 4, 4, **KW)
    if name == NAMES[1]: return ContextIndividualTF(15, 4, 4, 4, **KW)
    if name == NAMES[2]:
        model = ContextIndividualTF(15, 4, 4, 4, **KW); return make_c4_encoder_self_attention_shared(model)
    if name == NAMES[3]: return FusionContextIndividualTF(15, 4, 4, 4, **KW)
    if name == NAMES[4]:
        model = FusionContextIndividualTF(15, 4, 4, 4, **KW); return make_c4_encoder_self_attention_shared(model)
    raise ValueError(name)


def count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
def slug(name): return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_").lower()
def digest(tensor): return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_provenance():
    """Record the exact source revision and local files used for a run."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    files = ("individual_TF.py", "context_TF.py", "dynamic_fusion_TF.py", "c4_encoder_shared.py", Path(__file__).name)
    hashes = {}
    for file in files:
        path = Path(file) if isinstance(file, Path) else Path(__file__).parent / file
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"git_head": revision, "git_dirty": bool(status), "git_status": status, "source_sha256": hashes}


def mean_loss(model, loader, loss):
    """Dataset mean MSE, not an unweighted mean of differently sized batches."""
    total, count = 0.0, 0
    for batch in loader:
        value = loss(forward(model, batch), batch["pred_target"])
        batch_size = batch["pred_target"].shape[0]
        total += value.item() * batch_size
        count += batch_size
    return total / count


class WindowDataset(Dataset):
    def __init__(self, samples, stats):
        self.samples, self.stats = samples, stats
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]; mean, std, ms, ss = self.stats
        boxes = (sample["bbox"] - mean) / std
        speed = (sample["obd_speed"] - ms) / ss
        return {
            "boxes": torch.from_numpy(boxes[:15].astype(np.float32)),
            "speed": torch.from_numpy(speed.astype(np.float32)),
            "ego_flow": torch.from_numpy(sample["ego_op_flow"]),
            "ped_flow": torch.from_numpy(sample["ped_op_flow"]),
            "pred_target": torch.from_numpy(boxes[15:].astype(np.float32)),
            "scene_context": torch.from_numpy(sample["scene_context"][0]),
            "observed_occlusion": torch.from_numpy(sample["occlusion"][:15]),
            "sample_id": sample["id"], "video": sample["video"], "pid": sample["pid"],
            "frame_start": sample["frame_start"], "frame_end": sample["frame_end"],
        }


def load_data():
    import pickle
    path = SPLIT / "flow_splits.pkl"
    if not path.exists(): raise FileNotFoundError("Flow cache is required before a smoke test or E1 run.")
    with path.open("rb") as handle: split = pickle.load(handle)
    all_boxes = np.concatenate([sample["bbox"] for sample in split["train"]])
    all_speed = np.concatenate([sample["obd_speed"] for sample in split["train"]])
    stats = (all_boxes.mean(0), np.maximum(all_boxes.std(0), 1e-6), all_speed.mean(0), np.maximum(all_speed.std(0), 1e-6))
    return {role: WindowDataset(samples, stats) for role, samples in split.items()}, stats


def decoder_attention(model):
    base = model if isinstance(model, IndividualTF) else model.base
    return [layer.src_attn.attn.detach().cpu().numpy() for layer in base.model.decoder.layers]


def forward(model, batch): return model.forward_batch(batch)


def smoke(datasets):
    out = RESULTS / "smoke" / "from_scratch_gp6_smoke.json"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = next(iter(DataLoader(datasets["train"], batch_size=2, shuffle=False)))
    expected = dict(zip(NAMES, (133700, 139142, 130694, 154007, 145559)))
    checks = {}
    for name in NAMES:
        set_seed(SEEDS[0]); model = model_for(name); model.eval()
        with torch.no_grad(): prediction = forward(model, batch)
        item = {"parameters": count(model), "expected_parameters": expected[name], "shape": list(prediction.shape), "finite": bool(torch.isfinite(prediction).all())}
        if name in (NAMES[1], NAMES[2]):
            original = model._context_gate
            model._context_gate = lambda token, boxes: torch.zeros(token.shape[0], dtype=token.dtype, device=token.device)
            with torch.no_grad():
                gated_zero = forward(model, batch)
                masked_attention = decoder_attention(model)
                motion = model.base.forward_batch(batch)
            model._context_gate = original
            item["g0_equals_motion_only"] = bool(torch.equal(gated_zero, motion))
            item["g0_context_token_absent"] = bool(all(values.shape[-1] == 15 for values in masked_attention))
        if name in (NAMES[2], NAMES[4]): item["satie"] = verify_c4_sharing(model)
        checks[name] = item
        if item["parameters"] != expected[name] or not item["finite"]: raise AssertionError(name)
        if "g0_equals_motion_only" in item and not item["g0_equals_motion_only"]: raise AssertionError(f"g=0 mismatch: {name}")
        if "g0_context_token_absent" in item and not item["g0_context_token_absent"]: raise AssertionError(f"g=0 context token present: {name}")
    meta = json.loads((SPLIT / "split_metadata.json").read_text())
    payload = {"checks": checks, "test_sample_id_sha256": meta["test_sample_id_sha256"]}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def evaluate(model, loader, stats):
    mean, std, _, _ = stats; rows=[]; predictions=[]; targets=[]; x_last=[]; gates=[]; logits=[]; attentions=[[] for _ in range(3)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = forward(model, batch).cpu(); target = batch["pred_target"].cpu()
            raw_prediction, raw_target = prediction.numpy() * std + mean, target.numpy() * std + mean
            predictions.append(raw_prediction); targets.append(raw_target); x_last.append(batch["boxes"][:, -1].numpy())
            if hasattr(model, "last_context_gate") and model.last_context_gate is not None:
                gates.append(model.last_context_gate.cpu().numpy()); logits.append(model.last_context_gate_logits.cpu().numpy())
            elif hasattr(model, "fusion"):
                gates.append(model.fusion.last_gate.cpu().numpy()); logits.append(model.fusion.last_gate_logits.cpu().numpy())
            for layer, values in enumerate(decoder_attention(model)): attentions[layer].append(values)
            for index in range(prediction.shape[0]):
                centre_p=(raw_prediction[index,:,0:2]+raw_prediction[index,:,2:4])/2; centre_t=(raw_target[index,:,0:2]+raw_target[index,:,2:4])/2
                rows.append({"sample_id":batch["sample_id"][index],"video":batch["video"][index],"pid":batch["pid"][index],"frame_start":int(batch["frame_start"][index]),"frame_end":int(batch["frame_end"][index]),"mse":float(np.mean((raw_prediction[index]-raw_target[index])**2)),"cmse":float(np.mean((centre_p-centre_t)**2)),"cfmse":float(np.mean((centre_p[-1]-centre_t[-1])**2))})
    # All-closed SAGE batches take the exact motion-only path (15 memory
    # tokens); batches containing an admitted sample have 16.  Preserve every
    # attention weight by padding only the absent context-token column with
    # NaN, and record the native key count for every test sample.
    key_counts = [chunk.shape[-1] for chunk in attentions[0]]
    if any(chunk.shape[-1] != keys for layer in attentions for chunk, keys in zip(layer, key_counts)):
        raise AssertionError("Decoder layers disagreed about their memory-token count")
    max_keys = max(key_counts)
    def padded_attention(chunks):
        return np.concatenate([
            np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, max_keys - chunk.shape[-1])),
                   mode="constant", constant_values=np.nan)
            for chunk in chunks
        ])
    arrays={"predictions":np.concatenate(predictions),"targets":np.concatenate(targets),"x_t":np.concatenate(x_last),"attention_memory_tokens":np.concatenate([np.full(chunk.shape[0], keys, dtype=np.int8) for chunk, keys in zip(attentions[0], key_counts)]),"attention_layer_0":padded_attention(attentions[0]),"attention_layer_1":padded_attention(attentions[1]),"attention_layer_2":padded_attention(attentions[2])}
    if gates: arrays["gate"] = np.concatenate(gates); arrays["gate_logits"] = np.concatenate(logits)
    metrics={key:float(np.mean([row[key] for row in rows])) for key in ("mse","cmse","cfmse")}
    return metrics, rows, arrays


def run_one(name, seed, datasets, stats):
    run = RESULTS / "runs" / "E1" / slug(name) / str(seed)
    config_path = RESULTS / "config" / f"E1_{slug(name)}_{seed}.json"
    if run.exists() or config_path.exists(): raise FileExistsError(f"Refusing to overwrite {run}")
    run.mkdir(parents=True); set_seed(seed); model=model_for(name); optimiser=torch.optim.Adam(model.parameters(), lr=1e-3); loss=torch.nn.MSELoss()
    tensors=[{"name":key,"shape":list(value.shape),"sha256":digest(value)} for key,value in model.state_dict().items()]
    config={"model":name,"seed":seed,"initialization":"from scratch; no official checkpoint loaded","loaded_keys":[],"initial_state":tensors,"parameters":count(model),"protocol":{"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"min_delta":0,"optimizer":"Adam","lr":1e-3,"batch_size":8,"loss":"MSE"},"versions":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"opencv":cv2.__version__},"provenance":source_provenance()}
    config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    train=DataLoader(datasets["train"],batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed)); valid=DataLoader(datasets["validation"],batch_size=8,shuffle=False); history=[]; best=float("inf"); best_state=None; stale=0
    started=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        for batch in train: optimiser.zero_grad(); value=loss(forward(model,batch),batch["pred_target"]); value.backward(); optimiser.step()
        model.eval()
        with torch.no_grad(): validation=mean_loss(model, valid, loss)
        history.append(validation)
        if validation < best: best, best_state, stale = validation, {key:value.detach().clone() for key,value in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= PATIENCE: break
    model.load_state_dict(best_state); checkpoint=run/"checkpoint.pth"; torch.save(model.state_dict(),checkpoint)
    metrics, rows, arrays=evaluate(model,DataLoader(datasets["test"],batch_size=8,shuffle=False),stats)
    metrics.update({"parameters":count(model),"checkpoint_bytes":checkpoint.stat().st_size,"best_validation_mse":best,"epochs_completed":len(history),"stopping_reason":"early stopping" if len(history) < MAX_EPOCHS else "maximum epochs","training_seconds":time.perf_counter()-started,"test_sample_id_sha256":json.loads((SPLIT/"split_metadata.json").read_text())["test_sample_id_sha256"]})
    if "gate" in arrays:
        if arrays["gate"].ndim == 1:
            metrics["admission_rate"] = float(np.mean(arrays["gate"]))
        else:
            metrics["route_identity_rate"] = float(np.mean(np.argmax(arrays["gate"], axis=1) == 0))
            metrics["route_additive_rate"] = float(np.mean(np.argmax(arrays["gate"], axis=1) == 1))
            metrics["route_cross_attention_rate"] = float(np.mean(np.argmax(arrays["gate"], axis=1) == 2))
    (run/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); np.savez_compressed(run/"predictions.npz",**arrays)
    with (run/"per_sample.csv").open("x",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (run/"train_log.txt").write_text("\n".join(map(str,history))+"\n",encoding="utf-8")
    print(json.dumps({"model":name,"seed":seed,"metrics":metrics},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("smoke","run")); parser.add_argument("--models",nargs="*",default=NAMES); parser.add_argument("--seeds",type=int,nargs="*",default=SEEDS); args=parser.parse_args()
    datasets,stats=load_data()
    if args.action=="smoke": smoke(datasets)
    else:
        for name in args.models:
            if name not in NAMES: raise ValueError(name)
            for seed in args.seeds: run_one(name,seed,datasets,stats)
if __name__=="__main__": main()
