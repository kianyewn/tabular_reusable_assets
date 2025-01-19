import numpy as np


def calculate_feat_psi_table(
    left_df, right_df, feat, left_df_name="A", right_df_name="B", eps=np.finfo(float).eps, rounding=True
):
    """Return the table used to calculate the psi to inspect individual bin psi"""
    A = left_df_name
    B = right_df_name
    left_vc = left_df[feat].value_counts(dropna=False, normalize=True).rename(A).reset_index()
    right_vc = right_df[feat].value_counts(dropna=False, normalize=True).rename(B).reset_index()
    vc = left_vc.merge(right_vc, on=[feat], how="outer").fillna(0)
    vc["var"] = feat
    vc = vc.sort_values(by=feat).rename(columns={feat: "bin"})

    vc[f"{A}-{B}"] = vc[A] - vc[B]
    vc[f"ln({A}/{B})"] = np.log(vc[A] / (vc[B] + eps))
    vc["psi"] = vc[f"{A}-{B}"] * vc[f"ln({A}/{B})"]
    vc["sum_psi"] = vc["psi"].sum()

    columns = ["var", "bin"] + [col for col in vc.columns if col not in ["var", "bin"]]
    if rounding:
        vc["psi"] = round(vc["psi"], 5)
    return vc[columns]


def calculate_feat_psi(left_df, right_df, feat, left_df_name="A", right_df_name="B", eps=np.finfo(float).eps):
    """Returns only two columns for report"""
    A = left_df_name
    B = right_df_name
    left_vc = left_df[feat].value_counts(dropna=False, normalize=True).rename(A).reset_index()
    right_vc = right_df[feat].value_counts(dropna=False, normalize=True).rename(B).reset_index()
    vc = left_vc.merge(right_vc, on=[feat], how="outer").fillna(0)
    vc["var"] = feat
    vc = vc.sort_values(by=feat).rename(columns={feat: "bin"})

    vc[f"{A}-{B}"] = vc[A] - vc[B]
    vc[f"ln({A}/{B})"] = np.log(vc[A] / (vc[B] + eps))
    vc["psi"] = vc[f"{A}-{B}"] * vc[f"ln({A}/{B})"]
    vc["sum_psi"] = vc["psi"].sum()

    return vc[["var", "sum_psi"]].drop_duplicates()
