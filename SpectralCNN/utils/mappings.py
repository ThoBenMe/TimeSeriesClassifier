import json
import pandas as pd
import ast
import numpy as np


class LabelMapper:
    def __init__(self, mapping_file: str):
        with open(mapping_file, "r") as f:
            mappings = json.load(f)

        self.mineral_mapping = mappings.get("mineral_mapping", {})
        self.num_mapping = mappings.get("numeral_mapping", {})
        self.alias_map = mappings.get("alias_map", {})  # may be missing; default {}
        self.filepath = mapping_file

        self.num_to_name = {v: k for k, v in self.num_mapping.items()}

    def get_aliases(self, name: str) -> list[str]:
        """Return aliases for a class name, falling back to [name]."""
        return self.alias_map.get(name, [name])

    def get_num_classes(self) -> int:
        """Returns the number of classes."""
        return len(self.num_mapping)

    def get_classnames(self) -> list[str]:
        """Returns the class names as list."""
        return list(self.num_to_name.values())

    def get_classname_by_idx(self, idx: int) -> str | None:
        """Returns the class name for a given index."""
        return self.num_to_name.get(idx)

    @staticmethod
    def _ensure_list(x):
        """
        Coerce Similars-like field into a flat list[str].
        - Accepts list/tuple/set/ndarray/Series (recursively flattens)
        - Accepts stringified list (e.g. "['A', 'B']") via ast.literal_eval
        - Accepts comma-separated strings ("A, B")
        - Filters None/NaN items
        """

        def _is_nan(v):
            # Treat only true NaN scalars as NaN
            try:
                # shortcut: None and empty strings are not NaN here, handle separately
                return isinstance(v, float) and np.isnan(v)
            except Exception:
                return False

        def _flatten(v):
            if v is None:
                return
            # numpy/pandas containers
            if isinstance(v, (np.ndarray, pd.Series)):
                for item in list(v):
                    yield from _flatten(item)
                return
            # python containers
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    yield from _flatten(item)
                return
            # scalar/string
            yield v

        # string cases
        if isinstance(x, str):
            s = x.strip()
            # try list literal first
            if (s.startswith("[") and s.endswith("]")) or (
                s.startswith("(") and s.endswith(")")
            ):
                try:
                    parsed = ast.literal_eval(s)
                    return LabelMapper._ensure_list(parsed)
                except Exception:
                    pass
            # fallback: comma-separated string
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts

        # non-string: flatten and stringify
        out = []
        for item in _flatten(x):
            if item is None:
                continue
            if _is_nan(item):
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    def get_idx_by_name_or_alias(self, q: str) -> int | None:
        ql = str(q).strip().lower()
        name2idx = {k.lower(): v for k, v in self.num_mapping.items()}
        if ql in name2idx:
            return name2idx[ql]
        for canon, aliases in self.alias_map.items():
            al = [str(a).lower() for a in aliases]
            if ql == canon.lower() or ql in al:
                return name2idx.get(canon.lower())
        return None

    @staticmethod
    def _dedup_preserve(seq):
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def create_mappings_from_dataframe(
        dataframe: pd.DataFrame,
        mappings_filepath: str = "./mappings.json",
        create_new_mappings: bool = False,  # kept for API compat; not used here
    ) -> dict:
        """
        Build mappings from the *selected* minerals only.
        alias_map[name] = [name] + Similars (if any), deduped, order-preserving.
        """
        df = dataframe.copy()
        assert "Name" in df.columns, "DataFrame must have a 'Name' column."

        # Coerce Similars to lists
        if "Similars" not in df.columns:
            df["Similars"] = [[] for _ in range(len(df))]
        else:
            df["Similars"] = df["Similars"].apply(LabelMapper._ensure_list)

        # Keep the current order of selected minerals
        names = [str(x) for x in df["Name"].tolist()]

        # 1) mineral_mapping: identity mapping for selected classes
        mineral_mapping = {n: n for n in names}

        # 2) numeral_mapping: index by current order (+ UNK)
        numeral_mapping = {n: i for i, n in enumerate(names)}
        numeral_mapping["UNK"] = len(numeral_mapping)

        # 3) alias_map: per-row [Name] + Similars
        alias_map = {}
        for _, row in df.iterrows():
            name = str(row["Name"])
            similars = LabelMapper._ensure_list(row.get("Similars", []))
            aliases = LabelMapper._dedup_preserve([name] + similars)
            alias_map[name] = aliases

        mappings = {
            "mineral_mapping": mineral_mapping,
            "numeral_mapping": numeral_mapping,
            "alias_map": alias_map,
        }

        with open(mappings_filepath, "w") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        return mappings
