import numpy as np
import pandas as pd
from pathlib import Path
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img


def run_first_level(sub_id, save_pth, input_pth):
    """Run first-level GLM for switch vs stay contrast and save the effect size map.

    Parameters
    ----------
    sub_id : str
        Subject identifier, e.g. "sub-001".
    save_pth : str or Path
        Directory where the z-map will be saved.
    input_pth : str or Path
        Root path to the BIDS dataset containing sub-XXX/func/ directories.
    """
    input_pth = Path(input_pth)
    save_pth = Path(save_pth)
    save_pth.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    bold_file = input_pth / sub_id / "func" / f"{sub_id}_task-skewedGamblingTask_bold.nii.gz"
    events_file = input_pth / sub_id / "func" / f"{sub_id}_task-skewedGamblingTask_events.tsv"

    events_all = pd.read_csv(events_file, sep="\t")

    # --- Filter to choice events and compute switch/stay ---
    choice_events = events_all[events_all["trial_period"] == "choice"].copy()
    choice_events = choice_events.reset_index(drop=True)

    choice_events["prev_choice"] = choice_events["choice"].shift(1)
    choice_events["is_switch"] = (
        choice_events["choice"] != choice_events["prev_choice"]
    ).astype(int)

    # Drop trial 1 (no previous choice)
    choice_events = choice_events.iloc[1:].reset_index(drop=True)

    # --- Build GLM events ---
    # 1. Switch/Stay events
    choice_events["condition"] = (
        choice_events["trial_type"] + "_" +
        choice_events["is_switch"].map({1: "switch", 0: "stay"})
    )
    switch_stay_events = choice_events[["onset"]].copy()
    switch_stay_events["onset"] = switch_stay_events["onset"] - 0.1
    switch_stay_events["duration"] = 4.1
    switch_stay_events["trial_type"] = choice_events["condition"].values

    # 2. Response Time events
    all_choice = events_all[events_all["trial_period"] == "choice"].copy()
    all_choice["rt"] = pd.to_numeric(all_choice["response_time"], errors="coerce")
    rt_valid = all_choice[all_choice["rt"] > 0].copy()
    rt_events = rt_valid[["onset"]].copy()
    rt_events["duration"] = rt_valid["rt"].values
    rt_events["trial_type"] = "response_time"

    # 3. Valuation events (prechoice period)
    prechoice = events_all[events_all["trial_period"] == "prechoice"].copy()
    val_events = prechoice[["onset"]].copy()
    val_events["duration"] = 0.1
    val_events["trial_type"] = prechoice["trial_type"].values

    # Combine
    glm_events = pd.concat(
        [switch_stay_events, rt_events, val_events], ignore_index=True
    )

    # --- Design matrix ---
    t_r = 2
    n_scans = 442
    frame_times = np.arange(n_scans) * t_r

    design_matrix = make_first_level_design_matrix(
        frame_times,
        glm_events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model="glover",
    )

    # --- Fit model ---
    fmri_glm = FirstLevelModel(t_r=t_r, noise_model="ar1", signal_scaling=0)
    fmri_glm = fmri_glm.fit(str(bold_file), design_matrices=design_matrix)

    # --- Switch > Stay contrast ---
    dm_columns = fmri_glm.design_matrices_[0].columns.tolist()
    contrast_values = []
    for col in dm_columns:
        if col.endswith("_switch"):
            contrast_values.append(1 / 3)
        elif col.endswith("_stay"):
            contrast_values.append(-1 / 3)
        else:
            contrast_values.append(0)

    contrast_switch_vs_stay = np.array(contrast_values)

    effect_map = fmri_glm.compute_contrast(
        contrast_switch_vs_stay,
        stat_type="t",
        output_type="effect_size",
    )

    # --- Resample to MNI152 2mm space ---
    template = load_mni152_template(resolution=2)
    effect_map = resample_to_img(effect_map, template, interpolation="continuous")

    # --- Save ---
    output_path = save_pth / f"{sub_id}_switch-vs-stay_effect.nii.gz"
    effect_map.to_filename(str(output_path))
    print(f"[{sub_id}] Effect map saved to: {output_path}")


if __name__ == "__main__":
    import sys

    idx = int(sys.argv[1])

    all_subjects = [
        f"sub-{i:03d}" for i in range(1, 76)
    ]

    input_pth = "/oak/stanford/groups/russpold/data/ds006105"

    sub_id = all_subjects[idx]
    save_pth = Path(__file__).resolve().parent / "results"

    run_first_level(sub_id, save_pth, input_pth)
