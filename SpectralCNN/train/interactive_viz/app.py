from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
import json
import io
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.colors as mcolors


app = FastAPI()

# --- Configuration ---
DATA_DIR = Path("../segmentation_output")  # Use Path object directly

# Global storage for loaded data
loaded_data = {}
label_info = {}
H, W = 0, 0  # Image dimensions


def create_colormap_and_norm(class_names, present_ids, unk_idx):
    all_relevant_original_ids = sorted(list(set(present_ids)))

    original_id_to_palette_idx = {}
    palette_idx_counter = 0

    if unk_idx != -1:
        original_id_to_palette_idx[unk_idx] = 0
        palette_idx_counter = 1

    for original_id in all_relevant_original_ids:
        if original_id not in original_id_to_palette_idx:
            if palette_idx_counter > 255:
                print(
                    f"Warning: More than 256 unique class IDs, some will share colors or be omitted from palette."
                )
                break
            original_id_to_palette_idx[original_id] = palette_idx_counter
            palette_idx_counter += 1

    num_colors_to_generate = max(palette_idx_counter, 1)
    base_cmap = plt.get_cmap("tab20", num_colors_to_generate)

    pil_palette_rgb = [0] * 256 * 3  # RGB, default to black

    for i in range(palette_idx_counter):
        r, g, b, a = base_cmap(i % base_cmap.N)

        if 0 <= i < 256:  # Ensure we don't go out of bounds for the 256-entry palette
            pil_palette_rgb[i * 3 + 0] = int(r * 255)
            pil_palette_rgb[i * 3 + 1] = int(g * 255)
            pil_palette_rgb[i * 3 + 2] = int(b * 255)

    return pil_palette_rgb, original_id_to_palette_idx  # Return RGB palette


@app.on_event("startup")
async def load_all_data():
    """Loads all necessary data into memory when the FastAPI app starts."""
    global loaded_data, label_info, H, W

    print(f"Loading data from {DATA_DIR}...")

    if not DATA_DIR.exists():
        raise RuntimeError(
            f"Data directory '{DATA_DIR}' does not exist. Please run the model with SegmentationMapCallback first."
        )

    try:
        loaded_data["original_image"] = np.load(DATA_DIR / "original_image.npy")
        loaded_data["predicted_seg_map"] = np.load(DATA_DIR / "segmentation_map.npy")
        loaded_data["ground_truth_map"] = np.load(DATA_DIR / "ground_truth_map.npy")
        loaded_data["topk_segmaps"] = np.load(DATA_DIR / "topk_segmaps.npy")
        loaded_data["topk_probs"] = np.load(DATA_DIR / "topk_probs.npy")
        loaded_data["all_spectra"] = np.load(DATA_DIR / "all_spectra.npy")
        loaded_data["all_class_probs"] = np.load(DATA_DIR / "all_class_probs.npy")

        with open(DATA_DIR / "label_info.json", "r") as f:
            label_info = json.load(f)

        H, W = loaded_data["original_image"].shape
        print(f"Data loaded successfully. Image dimensions: {H}x{W}")

    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing data file: {e}. Please ensure SegmentationMapCallback has saved all required files."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/image/{image_type}.png")
async def get_image(image_type: str):
    valid_image_types = [
        "original",
        "predicted_seg_map_top1",
        "predicted_seg_map_top2",
        "predicted_seg_map_top3",
        "ground_truth_map",
        "uncertainty_map",
    ]
    if image_type not in valid_image_types:
        raise HTTPException(
            status_code=404,
            detail=f"Image type '{image_type}' not found. Valid types are: {', '.join(valid_image_types)}",
        )

    img_array = None

    if image_type == "original":
        img_array = loaded_data["original_image"]

        if img_array.dtype != np.uint8:
            img_array = (
                (img_array - img_array.min())
                / (img_array.max() - img_array.min())
                * 255
            ).astype(np.uint8)

        max_display_width = 600
        current_h, current_w = img_array.shape

        if current_w > max_display_width:
            scale_factor = max_display_width / current_w
            new_w = max_display_width
            new_h = int(current_h * scale_factor)
            img = Image.fromarray(img_array, mode="L").resize(
                (new_w, new_h), Image.LANCZOS
            )
        else:
            img = Image.fromarray(img_array, mode="L")

    elif image_type == "uncertainty_map":
        if "all_class_probs" not in loaded_data:
            raise HTTPException(
                status_code=500,
                detail="Full probabilities not loaded for uncertainty map calculation.",
            )

        uncertainty_map_flat = np.array(
            [
                entropy(p + 1e-9) if not np.all(np.isnan(p)) else np.nan
                for p in loaded_data["all_class_probs"].reshape(
                    -1, loaded_data["all_class_probs"].shape[-1]
                )
            ]
        ).reshape(H, W)

        min_val = np.nanmin(uncertainty_map_flat)
        max_val = np.nanmax(uncertainty_map_flat)
        if max_val == min_val or np.isnan(min_val) or np.isnan(max_val):
            uncertainty_map_display = np.full(
                uncertainty_map_flat.shape, 127, dtype=np.uint8
            )
        else:
            uncertainty_map_display = (
                (uncertainty_map_flat - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
        uncertainty_map_display = np.nan_to_num(uncertainty_map_display, nan=0).astype(
            np.uint8
        )

        img = Image.fromarray(uncertainty_map_display, mode="L")

    else:  # segmentation maps handler (predicted, ground_truth, top-k)
        if image_type == "predicted_seg_map_top1":
            img_array = loaded_data["predicted_seg_map"]
        elif image_type == "predicted_seg_map_top2":
            if loaded_data["topk_segmaps"].shape[2] < 2:
                raise HTTPException(
                    status_code=404, detail="Top 2 predictions not available (k < 2)."
                )
            img_array = loaded_data["topk_segmaps"][:, :, 1]
        elif image_type == "predicted_seg_map_top3":
            if loaded_data["topk_segmaps"].shape[2] < 3:
                raise HTTPException(
                    status_code=404, detail="Top 3 predictions not available (k < 3)."
                )
            img_array = loaded_data["topk_segmaps"][:, :, 2]
        elif image_type == "ground_truth_map":
            img_array = loaded_data["ground_truth_map"]

        if img_array is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve image data for type: {image_type}. Data might be missing or key is incorrect.",
            )

        current_present_original_ids = np.unique(img_array[img_array != -1]).tolist()

        unk_idx = label_info.get("unk_idx", -1)

        pil_palette_rgb, original_id_to_palette_idx = create_colormap_and_norm(
            label_info["class_names"], current_present_original_ids, unk_idx
        )

        default_palette_idx_for_unmapped = original_id_to_palette_idx.get(unk_idx, 0)
        img_for_palette_indices = np.full(
            img_array.shape, default_palette_idx_for_unmapped, dtype=np.uint8
        )

        for original_id, palette_idx in original_id_to_palette_idx.items():
            if original_id != -1:
                img_for_palette_indices[img_array == original_id] = palette_idx

        img_for_palette_indices[img_array == -1] = default_palette_idx_for_unmapped

        img = Image.fromarray(img_for_palette_indices, mode="P")
        img.putpalette(pil_palette_rgb)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.get("/pixel_data/{image_type}/{x}/{y}")
async def get_pixel_data(image_type: str, x: int, y: int):
    if not (0 <= x < W and 0 <= y < H):
        raise HTTPException(status_code=400, detail="Coordinates out of bounds.")

    if (
        "all_spectra" not in loaded_data
        or "topk_segmaps" not in loaded_data
        or "topk_probs" not in loaded_data
        or "ground_truth_map" not in loaded_data
        or "predicted_seg_map" not in loaded_data  # Ensure predicted_seg_map is loaded
    ):
        raise HTTPException(
            status_code=500, detail="Required data for pixel information not loaded."
        )

    spectrum = loaded_data["all_spectra"][y, x, :].tolist()
    class_names = label_info.get("class_names", [])
    predictions = []

    if image_type == "ground_truth_map":
        gt_label_idx = loaded_data["ground_truth_map"][y, x]
        if gt_label_idx != -1:  # Assuming -1 is the unk/no-data value
            gt_label_name = (
                class_names[gt_label_idx]
                if 0 <= gt_label_idx < len(class_names)
                else f"Unknown ({gt_label_idx})"
            )
            predictions.append(
                {
                    "rank": 1,
                    "class_name": gt_label_name,
                    "probability": 1.0,  # Ground truth is a certainty
                }
            )
        else:
            predictions.append(
                {"rank": 1, "class_name": "No Ground Truth Data", "probability": 0.0}
            )
    elif image_type in [
        "predicted_seg_map_top1",
        "predicted_seg_map_top2",
        "predicted_seg_map_top3",
    ]:
        k_limit = 3  # Always show top 3
        actual_k = min(loaded_data["topk_segmaps"].shape[2], k_limit)

        top_k_preds_indices = loaded_data["topk_segmaps"][y, x, :actual_k].tolist()
        top_k_probs = loaded_data["topk_probs"][y, x, :actual_k].tolist()

        for i in range(len(top_k_preds_indices)):
            pred_idx = top_k_preds_indices[i]
            pred_prob = top_k_probs[i]
            pred_name = (
                class_names[pred_idx]
                if 0 <= pred_idx < len(class_names)
                else f"Unknown ({pred_idx})"
            )
            if not np.isnan(pred_prob):
                predictions.append(
                    {
                        "rank": i + 1,
                        "class_name": pred_name,
                        "probability": float(pred_prob),
                    }
                )
        if not predictions:  # If no valid predictions were added (e.g., all NaN or -1)
            predictions.append(
                {"rank": 1, "class_name": "No Prediction Data", "probability": 0.0}
            )

    else:
        top1_pred_idx = loaded_data["predicted_seg_map"][y, x]
        top1_pred_prob = loaded_data["topk_probs"][y, x, 0]
        if top1_pred_idx != -1 and not np.isnan(top1_pred_prob):
            top1_pred_name = (
                class_names[top1_pred_idx]
                if 0 <= top1_pred_idx < len(class_names)
                else f"Unknown ({top1_pred_idx})"
            )
            predictions.append(
                {
                    "rank": 1,
                    "class_name": top1_pred_name,
                    "probability": float(top1_pred_prob),
                }
            )
        else:
            predictions.append(
                {"rank": 1, "class_name": "No Prediction", "probability": 0.0}
            )

    return {"x": x, "y": y, "spectrum": spectrum, "predictions": predictions}


app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
