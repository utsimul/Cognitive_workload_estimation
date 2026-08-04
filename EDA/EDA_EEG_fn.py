import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu


class_order = [
    "easy_arithmetic",
    "hard_arithmetic",
    "relaxation"
]



def read_file(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        return data

def epsilon_squared(H, n, k):
    """
    Effect size for Kruskal-Wallis test.
    """
    return (H - k + 1) / (n - k)

def rank_biserial_from_u(U, n1, n2):
    """
    Rank-biserial correlation from Mann-Whitney U.
    """
    return (2 * U) / (n1 * n2) - 1

def feature_family(feature):

    if feature in band_features:
        return "band"

    elif feature in avg_band_features:
        return "average_band"

    elif feature in ratio_features:
        return "ratio"

    elif feature in asymmetry_features:
        return "asymmetry"

    elif feature in wavelet_features:
        return "wavelet"

    else:
        return "other"


# task_paths = {
#     "easy_arithmetic":
#     "/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/ES140/Lab/Features/arithmetix_easy/EEG_features_30_sec.pickle",

#     "hard_arithmetic":
#     "/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/ES140/Lab/Features/arithmetix_hard/EEG_features_30_sec.pickle",

#     "relaxation":
#     "/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/ES140/Lab/Features/relaxation_video/EEG_features_30_sec.pickle",

    
# }


def EDA_per_patient(patient_id, type, time_window):

    task_paths = {
        "easy_arithmetic":
        f"/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/{patient_id}/{type}/Features/arithmetix_easy/EEG_features_{time_window}_sec.pickle",

        "hard_arithmetic":
        f"/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/{patient_id}/{type}/Features/arithmetix_hard/EEG_features_{time_window}_sec.pickle",

        "relaxation":
        f"/Users/krisha/Projs/Neuro-brain-states/Datasets/EPIStress/{patient_id}/{type}/Features/relaxation_video/EEG_features_{time_window}_sec.pickle",

    }
    all_dfs = []

    for task, path in task_paths.items():

        df = read_file(path)

        df["task_label"] = task

        all_dfs.append(df)


    EEG_all = pd.concat(all_dfs, ignore_index=True)

    print(EEG_all.shape)
    EEG_all.head()



    le = LabelEncoder()
    EEG_all["task_id"] = le.fit_transform(EEG_all["task_label"])

    print(dict(zip(le.classes_, le.transform(le.classes_))))

    X = EEG_all.drop(columns=["task_label", "task_id"])
    y = EEG_all["task_id"]


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    features = [
        "avg_δ",
        "avg_θ",
        "avg_α",
        "avg_β",
        "avg_γ"
    ]

    global band_features, avg_band_features, ratio_features, asymmetry_features, wavelet_features

    band_features = [
        c for c in EEG_all.columns
        if any(c.startswith(f"{band}_") for band in ["δ", "θ", "α", "β", "γ"])
    ]

    avg_band_features = [
        c for c in EEG_all.columns
        if c in ["avg_δ", "avg_θ", "avg_α", "avg_β", "avg_γ"]
    ]

    ratio_features = [
        c for c in EEG_all.columns
        if c.startswith("rat_")
    ]

    asymmetry_features = [
        c for c in EEG_all.columns
        if "asy" in c.lower()
    ]

    wavelet_features = [
        c for c in EEG_all.columns
        if any(x in c for x in ["_a5", "_d3", "_d4", "_d5"])
    ]

    

    # ## Kruskal Wallis

    feature_columns = [
        c for c in EEG_all.columns
        if c not in ["task_label", "task_id"]
    ]

    results = []

    for feature in feature_columns:

        groups = [
            EEG_all.loc[
                EEG_all["task_label"] == task,
                feature
            ].dropna()
            for task in class_order
        ]

        
        if any(len(group) < 2 for group in groups):
            continue

        try:
            statistic, p_value = kruskal(*groups)

            results.append({
                "feature": feature,
                "H_statistic": statistic,
                "p_value": p_value
            })

        except ValueError:
            continue

    stats_results = pd.DataFrame(results)

    stats_results = stats_results.sort_values("p_value")

    # stats_results.head(20)


    reject, pvals_corrected, _, _ = multipletests(
        stats_results["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    stats_results["p_fdr"] = pvals_corrected
    stats_results["significant"] = reject

    stats_results = stats_results.sort_values("p_fdr")

    # stats_results.head(30)

    # print(
    #     f"Significant features after FDR correction: "
    #     f"{stats_results['significant'].sum()} / "
    #     f"{len(stats_results)}"
    # )

    significant_features = stats_results.loc[
        stats_results["significant"],
        "feature"
    ].tolist()

    # print(significant_features)

    top_n = 15

    top_features = stats_results.head(top_n)["feature"]

    # print(class_order)

    effect_results = []

    k = len(class_order)

    for _, row in stats_results.iterrows():

        feature = row["feature"]

        values = EEG_all[feature].dropna()

        n = len(values)

        if n <= k:
            continue

        effect = epsilon_squared(
            row["H_statistic"],
            n,
            k
        )

        effect_results.append({
            "feature": feature,
            "epsilon_squared": effect
        })

    effect_results = pd.DataFrame(effect_results)

    stats_results = stats_results.merge(
        effect_results,
        on="feature",
        how="left"
    )

    stats_results = stats_results.sort_values(
        "epsilon_squared",
        ascending=False
    )

    # stats_results.head(30)

    top_effect_features = stats_results.head(20)



    pairwise_comparisons = [
        ("easy_arithmetic", "hard_arithmetic"),
        ("easy_arithmetic", "relaxation"),
        ("hard_arithmetic", "relaxation")
    ]

    # %%
    pairwise_results = []

    for feature in feature_columns:

        for class_a, class_b in pairwise_comparisons:

            x = EEG_all.loc[
                EEG_all["task_label"] == class_a,
                feature
            ].dropna()

            y = EEG_all.loc[
                EEG_all["task_label"] == class_b,
                feature
            ].dropna()

            if len(x) < 2 or len(y) < 2:
                continue

            try:
                statistic, p_value = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

                pairwise_results.append({
                    "feature": feature,
                    "class_a": class_a,
                    "class_b": class_b,
                    "U_statistic": statistic,
                    "p_value": p_value
                })

            except ValueError:
                continue

    pairwise_results = pd.DataFrame(pairwise_results)

    reject, pvals_corrected, _, _ = multipletests(
        pairwise_results["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    pairwise_results["p_fdr"] = pvals_corrected
    pairwise_results["significant"] = reject

    pairwise_results = pairwise_results.sort_values("p_fdr")

    # pairwise_results.head(30)

    pairwise_effects = []

    for _, row in pairwise_results.iterrows():

        feature = row["feature"]
        class_a = row["class_a"]
        class_b = row["class_b"]

        x = EEG_all.loc[
            EEG_all["task_label"] == class_a,
            feature
        ].dropna()

        y = EEG_all.loc[
            EEG_all["task_label"] == class_b,
            feature
        ].dropna()

        effect = rank_biserial_from_u(
            row["U_statistic"],
            len(x),
            len(y)
        )

        pairwise_effects.append(effect)

    pairwise_results["rank_biserial"] = pairwise_effects

    easy_hard = pairwise_results[
        (
            (pairwise_results["class_a"] == "easy_arithmetic") &
            (pairwise_results["class_b"] == "hard_arithmetic")
        )
    ]

    easy_hard = easy_hard.sort_values("p_fdr")


    easy_relax = pairwise_results[
        (
            (pairwise_results["class_a"] == "easy_arithmetic") &
            (pairwise_results["class_b"] == "relaxation")
        )
    ]

    easy_relax = easy_relax.sort_values("p_fdr")

    # easy_relax.head(20)

    hard_relax = pairwise_results[
        (
            (pairwise_results["class_a"] == "hard_arithmetic") &
            (pairwise_results["class_b"] == "relaxation")
        )
    ]

    hard_relax = hard_relax.sort_values("p_fdr")

    # hard_relax.head(20)


    pairwise_summary = pairwise_results.pivot_table(
        index="feature",
        columns=["class_a", "class_b"],
        values="p_fdr"
    )

    # pairwise_summary.head()
    pairwise_pvalues = pairwise_results.copy()

    pairwise_pvalues["comparison"] = (
        pairwise_pvalues["class_a"]
        + " vs "
        + pairwise_pvalues["class_b"]
    )

    pairwise_pvalues = pairwise_pvalues.pivot(
        index="feature",
        columns="comparison",
        values="p_fdr"
    )

    # pairwise_pvalues.head(20)

    corr_avg = EEG_all[
        avg_band_features
    ].corr()



    if len(ratio_features) > 1:

        corr_ratio = EEG_all[
            ratio_features
        ].corr()


    corr_wavelet = EEG_all[
        wavelet_features
    ].corr()

    numeric_features = EEG_all[
        feature_columns
    ]

    corr_matrix = numeric_features.corr().abs()

    upper = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    high_corr_pairs = (
        upper
        .stack()
        .reset_index()
    )

    high_corr_pairs.columns = [
        "feature_1",
        "feature_2",
        "absolute_correlation"
    ]

    high_corr_pairs = high_corr_pairs.sort_values(
        "absolute_correlation",
        ascending=False
    )

    high_corr_pairs.head(50)

    high_corr_90 = high_corr_pairs[
        high_corr_pairs["absolute_correlation"] >= 0.90
    ]

    # print(
    #     f"Number of feature pairs with |r| >= 0.90: "
    #     f"{len(high_corr_90)}"
    # )

    high_corr_90.head(50)

    high_corr_95 = high_corr_pairs[
        high_corr_pairs["absolute_correlation"] >= 0.95
    ]

    # print(
    #     f"Number of feature pairs with |r| >= 0.95: "
    #     f"{len(high_corr_95)}"
    # )


    high_corr_90 = high_corr_90.copy()

    high_corr_90["family_1"] = (
        high_corr_90["feature_1"]
        .apply(feature_family)
    )

    high_corr_90["family_2"] = (
        high_corr_90["feature_2"]
        .apply(feature_family)
    )

    high_corr_90.head(30)

    wavelet_stat_names = [
        "mean",
        "median",
        "variance",
        "std",
        "skew",
        "kurtosis",
        "rms",
        "energy"
    ]

    for stat in wavelet_stat_names:

        matching = [
            c for c in wavelet_features
            if stat in c.lower()
        ]

        # print(f"{stat}: {len(matching)} features")

    wavelet_stat_features = [
        c for c in wavelet_features
        if any(stat in c.lower() for stat in wavelet_stat_names)
    ]

    corr_wavelet_stats = EEG_all[
        wavelet_stat_features
    ].corr()

    correlation_threshold = 0.90

    redundancy_count = (
        corr_matrix >= correlation_threshold
    ).sum(axis=1) - 1

    redundancy_table = pd.DataFrame({
        "feature": redundancy_count.index,
        "high_corr_count": redundancy_count.values
    })

    redundancy_table = redundancy_table.sort_values(
        "high_corr_count",
        ascending=False
    )

    redundancy_table.head(30)

    feature_analysis = stats_results.copy()

    feature_analysis["family"] = (
        feature_analysis["feature"]
        .apply(feature_family)
    )

    feature_analysis = feature_analysis[
        [
            "feature",
            "family",
            "H_statistic",
            "p_value",
            "p_fdr",
            "epsilon_squared",
            "significant"
        ]
    ]

    feature_analysis = feature_analysis.sort_values(
        ["significant", "epsilon_squared"],
        ascending=[False, False]
    )

    # feature_analysis.head(30)


    patient_data = EEG_all.copy()
    patient_data["patient_id"] = patient_id


    csv_path = Path(
        "/Users/krisha/Projs/Neuro-brain-states/env/EDA/EEG_all_patients.csv"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)


    if csv_path.exists():
        patient_data.to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        patient_data.to_csv(
            csv_path,
            mode="w",
            header=True,
            index=False
        )

    print(f"Added {len(patient_data)} recordings for {patient_id}")
    print(f"CSV: {csv_path}")


patients = ["134","140", "141", "208", "247", "257", "260","313","362","404","437","446","491"]



type = ["Situ","Lab","Lab", "Lab","Situ","Lab","Lab","Lab","Lab","Lab","lab"]

for patient_id, patient_type in zip(patients, type):

    if patient_type == "Situ":
        continue
    else:
        print(f"Processing patient: {patient_id} ({patient_type})")
        EDA_per_patient("ES"+patient_id, "Lab", 30)