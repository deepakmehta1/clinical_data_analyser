# core/hcc_relevance_check.py

import pandas as pd


def check_hcc_relevance(conditions: list, hcc_codes_path: str):
    """
    Check if the extracted conditions match HCC relevant codes.

    Args:
        conditions (list): List of extracted conditions.
        hcc_codes_path (str): Path to the HCC relevant codes CSV.

    Returns:
        list: List of relevant conditions with corresponding HCC codes.
    """
    hcc_codes_df = pd.read_csv(hcc_codes_path)

    relevant_conditions = []
    for condition in conditions:
        # Check if condition matches any entry in the HCC codes CSV
        matches = hcc_codes_df[
            hcc_codes_df["condition"].str.contains(condition, case=False, na=False)
        ]
        if not matches.empty:
            relevant_conditions.append(
                {"condition": condition, "hcc_codes": matches["code"].tolist()}
            )

    return relevant_conditions
