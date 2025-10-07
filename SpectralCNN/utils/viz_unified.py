from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------- Datasource adapters ----------


class SpectrumSource:
    """Minimal interface."""

    def __len__(self) -> int: ...
    def get_by_index(self, idx: int) -> Dict: ...
    def find_indices_by_name(self, name: str) -> List[int]: ...
    def label_of(self, idx: int) -> str: ...


class SyntheticDatasetSource(SpectrumSource):
    """
    Wraps your synthetic *raw* dataset (the one with .get_raw(i) that returns dicts).
    Pass a name_fn for mapping a raw sample to a lowercase display name.
    """

    def __init__(self, raw_ds, name_fn: Optional[Callable[[Dict], str]] = None):
        self.ds = raw_ds
        self._name_fn = name_fn

    def __len__(self):
        return len(self.ds)

    def _name_from_sample(self, s: Dict) -> str:
        if self._name_fn:
            return self._name_fn(s)
        # best-effort fallbacks
        for k in ("label", "labels", "name", "name_cleaned"):
            if k in s and isinstance(s[k], str):
                return s[k].lower()
        if "labels_idx" in s:
            # if your LabelMapper is accessible, you can inject a name_fn that uses it
            return f"idx_{int(s['labels_idx'])}"
        return "unknown"

    def get_by_index(self, idx: int) -> Dict:
        s = self.ds.get_raw(idx)
        if isinstance(s, list):
            s = s[0]
        spec = s["spectra"]
        spec = (
            spec.cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
        )
        return {
            "spectra": spec,
            "name": self._name_from_sample(s),
            "idx": idx,
            "raw": s,
        }

    def find_indices_by_name(self, name: str) -> List[int]:
        target = name.strip().lower()
        out = []
        for i in range(len(self.ds)):
            s = self.ds.get_raw(i)
            if isinstance(s, list):
                s = s[0]
            if self._name_from_sample(s) == target:
                out.append(i)
        return out

    def label_of(self, idx: int) -> str:
        return self.get_by_index(idx)["name"]


class RealParquetSource(SpectrumSource):
    """Parquet adapter with simple alias matching."""

    def __init__(
        self,
        df: pd.DataFrame,
        spec_col="spectrum",
        name_col="name_cleaned",
        alt_name_col="itemName",
        alias_map: Optional[Dict[str, Iterable[str]]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.spec_col = spec_col
        self.name_col = name_col
        self.alt_name_col = alt_name_col

        # lowercased alias map
        self.alias_map = None
        if alias_map:
            self.alias_map = {
                str(k).lower(): {str(a).lower() for a in v}
                for k, v in alias_map.items()
            }

    def __len__(self):
        return len(self.df)

    def get_by_index(self, idx: int) -> Dict:
        row = self.df.iloc[int(idx)]
        spec = np.asarray(row[self.spec_col])
        name = str(
            row.get(self.name_col) or row.get(self.alt_name_col) or "unknown"
        ).lower()
        return {"spectra": spec, "name": name, "idx": int(idx), "raw": row.to_dict()}

    def _alias_candidates(self, name: str) -> set:
        q = str(name).strip().lower()
        cands = {q}
        if not self.alias_map:
            return cands
        # direct: key -> aliases
        cands |= self.alias_map.get(q, set())
        # reverse: alias -> key
        for canon, aliases in self.alias_map.items():
            if q == canon or q in aliases:
                cands |= aliases | {canon}
        return cands

    def find_indices_by_name(self, name: str) -> List[int]:
        cands = self._alias_candidates(name)
        prim = self.df[self.name_col].astype(str).str.lower()
        m = prim.isin(cands)
        if self.alt_name_col in self.df.columns:
            alt = self.df[self.alt_name_col].astype(str).str.lower()
            m |= alt.isin(cands)
        return list(np.flatnonzero(m.values))

    def label_of(self, idx: int) -> str:
        row = self.df.iloc[int(idx)]
        return str(
            row.get(self.name_col) or row.get(self.alt_name_col) or "unknown"
        ).lower()


# ---------- Pipeline tapping ----------


def flatten_pipeline(pipeline) -> List:
    return (
        list(pipeline.required_pre)
        + list(pipeline.optional_transforms)
        + list(pipeline.required_post)
    )


def run_pipeline_with_taps(
    sample_dict: Dict,
    pipeline,
    taps: Optional[Iterable[str]] = None,
    is_real_data: bool = False,
    until_stage: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns spectra at checkpoints keyed by:
      - 'raw' (input)
      - class names that match taps (after each such transform)
      - 'final' (after full pipeline)
    Example taps: ['AddNoise', 'BaselineCorrection', 'NormIT', 'NormSqrt']
    """
    out = {}
    batch = dict(sample_dict)
    spec = batch["spectra"]
    batch["spectra"] = (
        spec.clone() if isinstance(spec, torch.Tensor) else np.array(spec, copy=True)
    )

    out["raw"] = (
        batch["spectra"].cpu().numpy()
        if isinstance(batch["spectra"], torch.Tensor)
        else np.asarray(batch["spectra"])
    )

    steps = flatten_pipeline(pipeline)
    taps = set(taps or [])
    for t in steps:
        batch = t(batch, is_real_data=is_real_data)  # put it through pipeline step
        tname = t.__class__.__name__
        if tname in taps:
            arr = batch["spectra"]
            out[tname] = (
                arr.detach().cpu().numpy()
                if isinstance(arr, torch.Tensor)
                else np.asarray(arr)
            )
        if until_stage and tname == until_stage:
            arr = batch["spectra"]
            out["final"] = (
                arr.detach().cpu().numpy()
                if isinstance(arr, torch.Tensor)
                else np.asarray(arr)
            )
            return out

    arr = batch["spectra"]
    out["final"] = (
        arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    )
    return out


# ---------- Plot helpers ----------
def find_mix_index_by_classes(mix_ds, class_idx_a: int, class_idx_b: int) -> int:
    """
    Return the first index in `mix_ds` whose soft label mixes {A,B}.
    Raises ValueError if none found.
    """
    for i, s in enumerate(mix_ds.mixed_samples):
        soft = s["soft_labels"]
        if torch.is_tensor(soft):
            soft = soft.cpu().numpy()
        nz = set(np.flatnonzero(soft > 0.0).tolist())
        if {class_idx_a, class_idx_b}.issubset(nz):
            return i
    raise ValueError(f"No mix found for classes {class_idx_a} & {class_idx_b}")


def _get_axis(ax=None, figsize=(8, 4)):
    if ax is not None:
        return ax
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return ax


def _to_keV_axis(n: int, emax: float = 20.0) -> np.ndarray:
    return np.linspace(0.0, emax, n)


def plot_single(
    source: SpectrumSource,
    idx_or_name: Union[int, str],
    pipeline=None,
    stage: str = "final",  # 'raw', any tapped class name, or 'final'
    taps: Iterable[str] = ("BaselineCorrection", "NormIT", "NormSqrt"),
    is_real: bool = False,
    ax=None,
    logy: bool = True,
    keV: bool = True,
    title: Optional[str] = None,
):
    # resolve index
    if isinstance(idx_or_name, int):
        idx = idx_or_name
    else:
        hits = source.find_indices_by_name(idx_or_name)
        if not hits:
            raise ValueError(f"No match for name '{idx_or_name}'")
        idx = hits[0]

    # print(f"Source: {source}")
    # print(f"Pipeline: {pipeline}")
    # print(f"Index: {idx}")
    # print(f"Stage: {stage}")

    item = source.get_by_index(idx)
    spec0 = item["spectra"]

    if pipeline is None or stage == "raw":
        y = spec0
        stage_name = "raw"
    else:
        taps_local = set(taps or ())
        taps_local.add(stage)  # <-- ensure we capture the requested stage
        taps_out = run_pipeline_with_taps(
            {
                "spectra": spec0,
                **(
                    {k: v for k, v in item["raw"].items()}
                    if isinstance(item["raw"], dict)
                    else {}
                ),
            },
            pipeline,
            taps=taps_local,
            is_real_data=is_real,
            until_stage=stage,  # <-- stop here to avoid later transforms needing soft_labels
        )
        if stage not in taps_out:
            # fall back to whatever we stopped at
            y = taps_out["final"]
        else:
            y = taps_out[stage]
        stage_name = stage

    x = _to_keV_axis(len(y)) if keV else np.arange(len(y))
    ax = _get_axis(ax)
    ax.plot(x, y, drawstyle="steps-mid")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)" if keV else "Channel")
    ax.set_ylabel("Intensity")
    class_idx = None
    if isinstance(item.get("raw"), dict) and "labels_idx" in item["raw"]:
        class_idx = int(item["raw"]["labels_idx"])
    title_str = (
        title
        or f"{item['name']}  (data_idx {idx}"
        + (f", class_idx {class_idx}" if class_idx is not None else "")
        + f", {stage_name})"
    )
    ax.set_title(title_str)
    return ax


def plot_pre_post(
    source: SpectrumSource,
    idx_or_name: Union[int, str],
    pipeline,
    before: str,  # class name (e.g., 'BaselineCorrection')
    after: str,  # class name (e.g., 'NormSqrt')
    is_real: bool = False,
    figsize=(12, 4),
    logy: bool = True,
    keV=True,
):
    # run taps
    if isinstance(idx_or_name, int):
        idx = idx_or_name
    else:
        hits = source.find_indices_by_name(idx_or_name)
        if not hits:
            raise ValueError(f"No match for name '{idx_or_name}'")
        idx = hits[0]

    item = source.get_by_index(idx)
    taps_out = run_pipeline_with_taps(
        {
            "spectra": item["spectra"],
            **({} if not isinstance(item["raw"], dict) else item["raw"]),
        },
        pipeline,
        taps=[before, after],
        is_real_data=is_real,
    )

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for ax, stage in zip(axs, [before, after]):
        y = taps_out.get(
            stage, taps_out["final"] if stage == "final" else taps_out["raw"]
        )
        x = _to_keV_axis(len(y)) if keV else np.arange(len(y))
        ax.plot(x, y, drawstyle="steps-mid")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{item['name']} — {stage}")
        ax.set_xlabel("Energy (keV)" if keV else "Channel")
        ax.set_ylabel("Intensity")
    plt.tight_layout()
    return axs


def plot_comparison(
    mineral_name: str,
    synth_src: SpectrumSource,
    real_src: SpectrumSource,
    synth_pipeline=None,
    real_pipeline=None,
    synth_stage: str = "final",
    real_stage: str = "final",
    taps: Iterable[str] = ("BaselineCorrection", "NormIT", "NormSqrt"),
    figsize=(10, 4),
    keV=True,
    logy=True,
    normalize: bool = True,
):
    # resolve an index for each
    s_hits = synth_src.find_indices_by_name(mineral_name)
    r_hits = real_src.find_indices_by_name(mineral_name)
    if not s_hits:
        raise ValueError(f"Synthetic has no '{mineral_name}'.")
    if not r_hits:
        raise ValueError(f"Real has no '{mineral_name}'.")

    synth = synth_src.get_by_index(s_hits[0])
    real = real_src.get_by_index(r_hits[0])

    def stagey(item, pipe, stage, is_real):
        if pipe is None or stage == "raw":
            return item["spectra"]
        taps_out = run_pipeline_with_taps(
            {
                "spectra": item["spectra"],
                **({} if not isinstance(item["raw"], dict) else item["raw"]),
            },
            pipe,
            taps=taps,
            is_real_data=is_real,
        )
        return taps_out[stage] if stage in taps_out else taps_out["final"]

    ys = stagey(synth, synth_pipeline, synth_stage, is_real=False)
    yr = stagey(real, real_pipeline, real_stage, is_real=True)

    # normalize
    if normalize:
        if ys.max() > 0:
            ys = ys / ys.max()
        if yr.max() > 0:
            yr = yr / yr.max()

    x = _to_keV_axis(len(ys)) if keV else np.arange(len(ys))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, ys, drawstyle="steps-mid", label=f"Synth ({synth_stage})")
    ax.plot(x, yr, drawstyle="steps-mid", label=f"Real ({real_stage})", alpha=0.8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)" if keV else "Channel")
    ax.set_ylabel("Intensity (norm)" if normalize else "Intensity")
    ax.set_title(f"{mineral_name.capitalize()} — Synthetic vs Real")
    ax.legend()
    return ax
