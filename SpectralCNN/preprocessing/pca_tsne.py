import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from thesis.configs.config_reader import load_config
from thesis.utils.mappings import LabelMapper
from matplotlib.cm import get_cmap
import math
import pandas as pd
import plotly.express as px
from pathlib import Path

config = load_config("../configs/config.yml")


class DataInspector:
    def __init__(self, hdf5_path: str, dataset_key: str = "spectra"):
        self.hdf5_path = hdf5_path
        self.dataset_key = dataset_key
        self.data = None
        self.pca_scores = None
        self.tsne_emb = None

    def load_data(self, max_samples: int = None):
        """
        Load data from the HDF5 file.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            raw = f[self.dataset_key][:]
        if max_samples is not None:
            raw = raw[:max_samples]
        self.data = raw.astype(np.float32)
        return self.data

    def compute_pca(self, n_components: int = 3) -> np.ndarray:
        """
        Compute PCA on the loaded data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.pca = PCA(n_components=n_components)
        self.pca_scores = self.pca.fit_transform(self.data)
        print(f"PCA kept", self.pca.n_components_, "components.")
        return self.pca_scores

    def plot_pca(
        self,
        dims: tuple[int, int] = (0, 1),
        labels: np.ndarray = None,
        class_names: list = None,
        figsize: tuple[int, int] = (8, 6),
    ):
        """Scatter plot of two PCs."""
        assert labels is None or len(labels) == len(self.pca_scores)

        x = self.pca_scores[:, dims[0]]
        y = self.pca_scores[:, dims[1]]

        fig, ax = plt.subplots(figsize=figsize)

        if labels is not None and class_names is not None:
            unique = np.unique(labels)
            # choose a qualitative colormap with enough distinct colors
            cmap = get_cmap("tab20", len(unique))
            for i, cls in enumerate(unique):
                mask = labels == cls
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=cmap(i),
                    label=class_names[cls],
                    s=15,
                    alpha=0.8,
                )

            # decide columns (e.g. ~15 entries per column)
            per_col = 50
            ncol = math.ceil(len(unique) / per_col)

            # shrink axes to 75% width (reserve 25% for legend)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            # place legend outside
            leg = ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                fontsize="small",
                ncol=ncol,
                title="Classes",
            )

            # IMPORTANT: reserve space on the right so the legend is never cut off
            plt.tight_layout(rect=[0, 0, 0.75, 1])

        else:
            ax.scatter(x, y, s=5, alpha=0.7, color="gray")
            plt.tight_layout()

        ax.set_xlabel(f"PC {dims[0]+1}")
        ax.set_ylabel(f"PC {dims[1]+1}")
        ax.set_title(f"PCA Scatter (PC{dims[0]+1} vs PC{dims[1]+1})")
        plt.tight_layout()
        plt.show()

    def plot_interactive_pca(
        self,
        pca_scores: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        title: str = "Interactive PCA",
        width: int = 1200,
        height: int = 800,
        opacity: float = 0.8,
    ):
        """Plot PCA scores interactively using Plotly."""
        if self.pca_scores is None:
            raise RuntimeError("Call compute_pca() first")

        df = pd.DataFrame(
            {
                "PC1": self.pca_scores[:, 0],
                "PC2": self.pca_scores[:, 1],
                "ClassIdx": labels,
                "Mineral": [class_names[i] for i in labels],
            }
        )

        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="Mineral",
            hover_name="Mineral",
            title=title,
            width=width,
            height=height,
            opacity=opacity,
        )
        fig.show()

    def plot_interactive_pca_3d(self, labels, class_names):
        if self.pca_scores is None or self.pca_scores.shape[1] < 3:
            raise RuntimeError("Run compute_pca(n_components=3) first")
        import pandas as pd, plotly.express as px

        df = pd.DataFrame(
            {
                "PC1": self.pca_scores[:, 0],
                "PC2": self.pca_scores[:, 1],
                "PC3": self.pca_scores[:, 2],
                "Mineral": [class_names[i] for i in labels],
            }
        )
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Mineral",
            hover_name="Mineral",
            title="3D PCA",
        )
        fig.update_traces(marker=dict(size=3))
        fig.show()

    def compute_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
    ) -> np.ndarray:
        data_in = self.pca_scores if self.pca_scores is not None else self.data
        if data_in is None:
            raise RuntimeError("Call load_data() or compute_pca() first")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
        )
        self.tsne_emb = tsne.fit_transform(data_in)
        return self.tsne_emb

    def plot_tsne(
        self,
        labels: np.ndarray,
        class_names: list[str],
        figsize: tuple[int, int] = (8, 6),
        per_col: int = 15,
        title: str = "Classes",
    ):
        """
        Scatter plot of self.tsne_emb with a multi‐column legend neatly placed
        outside on the right—just like PCA’s.
        """
        if self.tsne_emb is None:
            raise RuntimeError("Call compute_tsne() first")

        x = self.tsne_emb[:, 0]
        y = self.tsne_emb[:, 1]
        unique = np.unique(labels)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = get_cmap("tab20", len(unique))

        # plot each class separately
        for i, cls in enumerate(unique):
            mask = labels == cls
            ax.scatter(
                x[mask],
                y[mask],
                color=cmap(i),
                label=class_names[cls],
                s=10,
                alpha=0.8,
            )

        # decide how many legend columns
        ncol = math.ceil(len(unique) / per_col)

        # shrink axes width to 75% of the figure, leaving 25% for legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

        # place legend outside on the right
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            fontsize="small",
            ncol=ncol,
            title="Classes",
            frameon=True,
        )

        # reserve the left 75% of the canvas for the plot area
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.set_title("t-SNE Projection")
        plt.show()

    def plot_interactive_tsne(
        self,
        labels: np.ndarray,
        class_names: list[str],
        title: str = "Interactive t-SNE",
        width: int = 1600,
        height: int = 1200,
        opacity: float = 0.8,
    ):
        """
        Plot t-SNE scores interactively using Plotly.
        """
        if self.tsne_emb is None:
            raise RuntimeError("Call compute_tsne() first")

        df = pd.DataFrame(
            {
                "t-SNE1": self.tsne_emb[:, 0],
                "t-SNE2": self.tsne_emb[:, 1],
                "ClassIdx": labels,
                "Mineral": [class_names[i] for i in labels],
                "Composition": ...,
            }
        )

        fig = px.scatter(
            df,
            x="t-SNE1",
            y="t-SNE2",
            color="Mineral",
            hover_name="Mineral",
            title=title,
            width=width,
            height=height,
            opacity=opacity,
        )
        fig.show()

    def plot_interactive_tsne_3d(
        self, labels, class_names, title: str = "Interactive t-SNE 3D"
    ):
        if self.tsne_emb is None or self.tsne_emb.shape[1] < 3:
            raise RuntimeError("Run compute_tsne(n_components=3) first")
        import pandas as pd, plotly.express as px

        df = pd.DataFrame(
            {
                "Dim1": self.tsne_emb[:, 0],
                "Dim2": self.tsne_emb[:, 1],
                "Dim3": self.tsne_emb[:, 2],
                "Mineral": [class_names[i] for i in labels],
            }
        )
        fig = px.scatter_3d(
            df,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            color="Mineral",
            hover_name="Mineral",
            title=title,
        )
        fig.update_traces(marker=dict(size=3))
        fig.show()


def main():
    # ─── Configuration ─────────────────────────────────────────────────────────────
    BASE = Path("../data")
    DATASET_NAME = config["DATA"]["DATASET_KWARGS"][0]["ds_kwargs"].get("dataset_name")
    HDF5_FILE = config["DATA"]["DATASET_KWARGS"][0]["ds_kwargs"].get("h5_file_name")
    H5 = BASE / DATASET_NAME / HDF5_FILE
    # H5 = BASE / "spectra/spectra_112_samples_nooxides_pertubated_1.hdf5"
    INSTR = config["DATA"]["DATASET_KWARGS"][0]["ds_kwargs"].get("data_names")[0]
    DATA_KEY = f"/{INSTR}/spectra"
    LABELS_KEY = f"/{INSTR}/labels_idx"
    LABEL_MAPPER = BASE / DATASET_NAME / "mappings/label_mapping.json"

    MAX_SAMPLES = None
    PCA_COMPONENTS = 20
    TSNE_PERP = 20.0
    SWEEP_PERPS = [10, 20, 30, 50]
    IS_INTERACTIVE = True
    USE_SWEEP = False  # if True, override TSNE_PERP with SWEEP_PERPS
    USE_3D = False
    FIGSIZE_PCA = (1200, 800) if IS_INTERACTIVE else (16, 12)
    FIGSIZE_TSNE = (1200, 800) if IS_INTERACTIVE else (16, 12)
    LEGEND_COLSIZE = 50  # ~items per column in legend

    # ─── Load labels ────────────────────────────────────────────────────────────────
    with h5py.File(H5, "r") as f:
        labels = f[LABELS_KEY][:]

    # ─── Instantiate & run PCA ────────────────────────────────────────────────────
    inspector = DataInspector(str(H5), dataset_key=DATA_KEY)
    inspector.load_data(max_samples=MAX_SAMPLES)
    inspector.compute_pca(n_components=PCA_COMPONENTS)
    N_SAMPLES = len(inspector.data)

    # ─── Plot PCA ─────────────────────────────────────────────────────────────────
    label_mapper = LabelMapper(LABEL_MAPPER)
    class_names = label_mapper.get_classnames()[:-1]  # drop UNK if needed
    num_classes = label_mapper.get_num_classes() - 1
    print(class_names)
    print(f"Number of classes: {num_classes}")

    if USE_3D:
        inspector.plot_interactive_pca_3d(
            labels=labels,
            class_names=class_names,
        )
    else:
        if IS_INTERACTIVE:
            inspector.plot_interactive_pca(
                labels=labels,
                class_names=class_names,
                title="PCA (PC1 vs PC2)",
                width=FIGSIZE_PCA[0],
                height=FIGSIZE_PCA[1],
                pca_scores=inspector.pca_scores,
                opacity=0.8,
            )
        else:
            inspector.plot_pca(
                dims=(0, 1), labels=labels, class_names=class_names, figsize=FIGSIZE_PCA
            )

    # ─── Prepare TSNE perplexities ────────────────────────────────────────────────
    perps = SWEEP_PERPS if USE_SWEEP else [TSNE_PERP]
    print(f"N SAMPLES: {N_SAMPLES}")
    print(f"TSNE PERPLEXITIES: {perps}")

    if N_SAMPLES <= perps[-1]:
        perps = [(N_SAMPLES - 1) / 3]

    for perp in perps:
        title = f"{'Interactive' if IS_INTERACTIVE else 'Static'} t-SNE (perp={perp})"
        if USE_3D:
            inspector.compute_tsne(n_components=3, perplexity=perp)
            inspector.plot_interactive_tsne_3d(
                labels=labels,
                class_names=class_names,
                title=title,
            )
        else:
            inspector.compute_tsne(perplexity=perp)
            if IS_INTERACTIVE:
                inspector.plot_interactive_tsne(
                    labels=labels,
                    class_names=class_names,
                    title=title,
                    width=FIGSIZE_TSNE[0],
                    height=FIGSIZE_TSNE[1],
                )
            else:
                inspector.plot_tsne(
                    labels=labels,
                    class_names=class_names,
                    per_col=LEGEND_COLSIZE,
                    figsize=FIGSIZE_TSNE,
                )


if __name__ == "__main__":
    main()
