import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import pandas as pd
from collections import OrderedDict

from thesis.configs.config_reader import load_config
from thesis.utils import tools
from thesis.utils import mappings as mp
import thesis.utils.plotting as plt
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate_perturbed_compositions(
    comp: list,
    perturbation_std: float = 0.05,
    num_variants: int = 2,
    max_variants: int = 10,
) -> list:
    pass


def select_materials(
    all_materials: pd.DataFrame,
    num_samples: int,
    must_haves: list = None,
) -> pd.DataFrame:
    if not must_haves:
        return all_materials.sample(n=num_samples, random_state=42).reset_index(
            drop=True
        )

    must_haves = list(OrderedDict.fromkeys(must_haves))  # ensure unique must-haves
    num_must_haves = len(must_haves)
    logger.info(f"Must-haves: {must_haves} (count: {num_must_haves})")

    if num_must_haves >= num_samples:
        logger.warning(
            f"Must-haves list is larger than num_samples (MH: {num_must_haves}, NS: {num_samples}). Continuing with must-haves only."
        )
        selected_names = must_haves[: num_samples + 1]
        selected_materials = all_materials.loc[
            all_materials["Name"].isin(selected_names)
        ]
        return selected_materials.reset_index(drop=True)

    # select must-haves
    selected_materials = all_materials.loc[
        all_materials["Name"].isin(must_haves)
    ].copy()
    num_additional = num_samples - len(selected_materials)
    pool = all_materials[~all_materials["Name"].isin(must_haves)]
    add_samples = pool.sample(n=num_additional, random_state=42)
    selected_materials = pd.concat([selected_materials, add_samples], ignore_index=True)
    return selected_materials.reset_index(drop=True)


def run_pipeline(
    instrument_path: str,
    all_materials: pd.DataFrame,
    output_root: Path,
    num_samples: int | float,
    perturbation_factor: float,
    num_perturbation_variants: int,
    max_variants: int,
    variant_equimolars: int,
    variant_step: int,
    use_oxides: bool,
    seed: np.random.RandomState | int,
    deactivate_formulas: bool = False,
    must_haves: list = None,
) -> None:

    # ----------------------
    # Select initial materials
    # ----------------------
    selected_materials: pd.DataFrame = select_materials(
        all_materials=all_materials,
        num_samples=num_samples,
        must_haves=must_haves,
    )
    logger.info(f"Selected {len(selected_materials)} materials for dataset generation.")
    selected_materials = selected_materials.copy()

    # ensure comp is a string, not list
    selected_materials = selected_materials[
        ~selected_materials["Composition"].isna()
    ].copy()
    selected_materials["Composition"] = selected_materials["Composition"].astype(str)

    # ----------------------
    # Prepare output dirs
    # ----------------------
    ds_name_parts = [
        f"ds_{num_samples}materials",
        (
            f"perturb{int(100*perturbation_factor)}percent"
            if num_perturbation_variants > 0
            else None
        ),
        (
            f"variants{num_perturbation_variants}"
            if num_perturbation_variants > 0
            else None
        ),
        "noformulas" if deactivate_formulas else None,
        "oxides" if use_oxides else None,
    ]
    ds_name = "_".join([p for p in ds_name_parts if p])
    dataset_dir = output_root / ds_name
    csv_dir = dataset_dir / "csv"
    mappings_dir = dataset_dir / "mappings"
    spectra_dir = dataset_dir / "spectra"
    for d in (csv_dir, mappings_dir, spectra_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created dataset directories under {str(dataset_dir)}")

    # ------------------------
    # Spectrum generator setup
    # ------------------------
    variant_gen: tools.FormulaVariantGenerator = tools.FormulaVariantGenerator(
        max_variants=max_variants,
        equimolar_count=variant_equimolars,
        step=variant_step,
        seed=seed,
    )

    all_compounds: list[tools.Compound] = []
    bboxes: list[tuple[float, float, float, float]] = []  # keep track of boundary
    count: int = 0  # store current num of compounds

    # -----------------------------------
    # Process materials (with replacement)
    # -----------------------------------
    def process_materials(material_df: pd.DataFrame):
        nonlocal count
        skipped = set()

        for _, row in material_df.iterrows():
            name = row["Name"]
            formula = str(row.get("Chemical Formula", "")).strip() or ""
            san = filter_formula(formula)

            # Skip Mendelevium
            if "Md" in formula:
                logger.warning(f"Skipping {name} (contains Mendelevium).")
                skipped.add(name)
                continue

            # Prepare formula variants
            formula_variants: list[str] = []
            if not deactivate_formulas and san:
                try:
                    if "(" in san:
                        formula_variants = variant_gen.generate(san)
                    else:
                        formula_variants = [san]
                except Exception as e:
                    logger.warning(
                        f"Failed to parse formula for {name} ('{formula}'): {e}. Skipping formulas."
                    )
                    formula_variants = []

            # --- Prepare composition variants (main + row Variants)
            main_comp = row["Composition"]  # ensured string earlier
            variants = row.get("Variants", [])
            variants = [v for v in variants]
            comps = [main_comp] + variants

            # --- Optional perturbed composition strings (as percent strings)
            if num_perturbation_variants > 0:
                logger.info(
                    f"Generating perturbed compositions for {name} with factor {perturbation_factor}"
                )
                try:
                    extras_dicts = generate_perturbed_compositions(
                        comp=comps,
                        perturbation_std=perturbation_factor,
                        num_variants=num_perturbation_variants,
                    )
                    extras_str = [tools.dict_to_cmp_string(d) for d in extras_dicts]
                    comps.extend(extras_str)
                except Exception as e:
                    logger.warning(f"Perturbation generation failed for {name}: {e}")

            # --- final list
            prepared = formula_variants + comps
            if not prepared:
                logger.warning(
                    f"No valid compositions for {name} (formula: {formula!r}). Skipping."
                )

            start = count
            added = 0
            added_formulas = 0

            # Generate compounds
            for i, entry in enumerate(prepared):
                if "Md" in entry:
                    logger.warning(f"Skipping {entry} - Mendelevium not supported.")
                    continue
                try:
                    comp_obj = tools.create_compound_list(
                        formulas=[entry],
                        use_random_compounds=False,
                        num_random_compounds=0,
                    )[0]
                    all_compounds.append(comp_obj)
                    count += 1
                    added += 1
                    if i < len(formula_variants):
                        added_formulas += 1
                except Exception as e:
                    logger.error(f"Failed to create Compound for {entry}: {e}")

            bboxes.append(
                (start, start + added, name, added_formulas, added - added_formulas)
            )
        return skipped

    # Process selected materials
    skipped_names = process_materials(selected_materials)

    # replace skipped materials
    while skipped_names:
        logger.info(f"{len(skipped_names)} materials skipped. Drawing replacements...")
        used_names = set(selected_materials["Name"])
        available_pool = all_materials[~all_materials["Name"].isin(used_names)]

        if available_pool.empty:
            logger.error("No more materials available for replacement.")
            break

        replacements = available_pool.sample(
            n=len(skipped_names), replace=False, random_state=42
        )
        selected_materials = pd.concat(
            [selected_materials, replacements], ignore_index=True
        )
        skipped_names = process_materials(replacements)

    # Final sanity check
    assert (
        len(set(selected_materials["Name"])) >= num_samples
    ), f"Not enough valid materials. Expected {num_samples}, got {len(set(selected_materials['Name']))}."

    # ----------------------
    # Final mappings + save CSV
    # ----------------------
    materials_csv = csv_dir / "materials.csv"
    selected_materials.to_csv(materials_csv, index=False, encoding="utf-8")
    logger.info(f"Saved selected materials to {materials_csv!r}")

    # build mappings
    mappings_file = mappings_dir / "label_mapping.json"
    mappings = mp.LabelMapper.create_mappings_from_dataframe(
        selected_materials, mappings_filepath=str(mappings_file)
    )
    logger.info(f"Wrote label mappings to {mappings_file}.")

    # ----------------------
    # Generate spectra
    # ----------------------
    if not all_compounds:
        logger.error("No compounds generated. Aborting.")
        return

    spectrometer: PSF = tools.load_spectrometer(instrument_path)
    spectrometer.calculate_spectra(all_compounds)
    logger.info("Spectra generation complete.")

    # Update labels
    tools.update_labels(spectrometer, selected_materials, mappings_file, bboxes)
    logger.info("Labels updated on spectrum container.")

    # Save to HDF5
    output_h5 = spectra_dir / "spectra.h5"
    spectrometer.spectrum_container.to_h5(str(output_h5), mode="w")
    logger.info(f"Spectra saved to {output_h5}.")


# main function to run the pipeline
if __name__ == "__main__":
    parser = tools.build_arg_parser()
    args = parser.parse_args()

    config = load_config("../configs/config.yml")["UTILS"]
    instrument_path = os.path.join(config["DATA_ROOT_PATH"], config["INSTRUMENT_PATH"])
    materials_source = os.path.join(
        config["DATA_ROOT_PATH"], config["materialS_INPUT_PARQUET"]
    )
    output_root = Path(config["DATA_ROOT_PATH"])
    ##################################################################

    # -- load all materials from CSV
    # all_materials = pd.read_csv(materials_source, encoding="utf-8")
    all_materials = pd.read_parquet(materials_source, engine="pyarrow")
    logger.info(f"Loaded all materials. Total available samples: {len(all_materials)}.")
    logger.info(f"Using Formulas: {not args.deactivate_formulas}")

    must_haves = [
        "X",
        "Y",
        "Z",
    ]  # materials that must be present
    seed = np.random.seed(42)  # set seed for reproducibility

    # # prepare kwargs for the pipeline
    pipeline_kwargs = {
        "instrument_path": instrument_path,
        "all_materials": all_materials,
        "output_root": output_root,
        "num_samples": args.num_samples or config["NUM_SAMPLES"],
        "perturbation_factor": args.num_perturbation_variants
        and args.num_perturbation_variants > 0
        and float(args.num_perturbation_variants) / config.get("NUM_SAMPLES", 1)
        or config.get("PERTURBATION_FACTOR", 0.00),
        "num_perturbation_variants": args.num_perturbation_variants
        or config.get("NUM_PERTURBATION_VARIANTS", 5),
        "max_variants": args.max_formula_variants or config.get("MAX_VARIANTS", 10),
        "variant_equimolars": args.variant_equimolars
        or config.get("VARIANT_EQUIMOLARS", 1),
        "variant_step": args.variant_step or config.get("VARIANT_STEP", 1),
        "use_oxides": args.use_oxides or config.get("USE_OXIDES", False),
        "deactivate_formulas": args.deactivate_formulas
        or config.get("DEACTIVATE_FORMULAS", False),
        "must_haves": must_haves,
        "seed": seed,
    }

    # -- to start the main logic
    run_pipeline(**pipeline_kwargs)
    logger.info("Pipeline finished.")
