"""
EFB335 Portfolio Construction: Diversification via Eigenvector Centrality
===========================================================================

Reads monthly stock returns from an Excel workbook, builds a correlation
matrix, ranks stocks by eigenvector centrality in the correlation network,
and selects the 7 most peripheral (least central) stocks as diversification
candidates. Cross-checks the ranking against a minimum spanning tree (MST).

Run end-to-end with:
    python portfolio_centrality.py "path/to/workbook.xlsx"

If no path is given, it looks for the workbook one directory above this
script (see DEFAULT_DATA_FILE below).
"""

import argparse
import os
import sys
import datetime

import numpy as np
import pandas as pd
import openpyxl
from scipy.sparse.csgraph import minimum_spanning_tree

# ---------------------------------------------------------------------------
# Configuration: describes the fixed layout of the "Monthly_Returns" sheet.
# ---------------------------------------------------------------------------
SHEET_NAME = "Monthly_Returns"
HEADER_ROW = 3          # row containing stock names
FIRST_STOCK_COL = 2     # column B
DATA_START_ROW = 8      # 31/07/2020
DATA_END_ROW = 43       # 30/06/2023
EXPECTED_MONTHS = DATA_END_ROW - DATA_START_ROW + 1  # 36

# Known/likely benchmark or corrupted-header labels to exclude from the
# stock universe (see exclude_benchmark_columns below).
BENCHMARK_NAMES = {"ASX200I"}

DEFAULT_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "EFB335 S2 2026 Group Project Data.xlsx",
)


class DataLayoutError(Exception):
    """Raised when the workbook doesn't match the expected sheet layout."""


# ---------------------------------------------------------------------------
# Stage 1: Data loading
# ---------------------------------------------------------------------------
def load_monthly_returns(file_path: str) -> pd.DataFrame:
    """
    Load the Monthly_Returns sheet into a DataFrame of raw values
    (stocks as columns, 36 months as rows), using cached formula values.

    Raises a clear, non-traceback error if the file or sheet layout is wrong.
    """
    print(f"[1/6] Loading data from: {file_path}")

    if not os.path.isfile(file_path):
        raise DataLayoutError(
            f"Data file not found: '{file_path}'.\n"
            "Check the path/filename and try again."
        )

    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
    except Exception as exc:
        raise DataLayoutError(
            f"Could not open '{file_path}' as an Excel workbook ({exc})."
        )

    if SHEET_NAME not in wb.sheetnames:
        raise DataLayoutError(
            f"Sheet '{SHEET_NAME}' not found. Sheets present: {wb.sheetnames}"
        )
    ws = wb[SHEET_NAME]

    if ws.max_column < FIRST_STOCK_COL or ws.max_row < DATA_END_ROW:
        raise DataLayoutError(
            f"Sheet '{SHEET_NAME}' is smaller than expected "
            f"(found {ws.max_row} rows x {ws.max_column} cols; "
            f"need at least row {DATA_END_ROW}, column {FIRST_STOCK_COL})."
        )

    headers = [
        ws.cell(row=HEADER_ROW, column=c).value
        for c in range(FIRST_STOCK_COL, ws.max_column + 1)
    ]
    if all(h is None for h in headers):
        raise DataLayoutError(
            f"Row {HEADER_ROW} has no stock names starting at column "
            f"{FIRST_STOCK_COL}. Check the sheet layout hasn't changed."
        )

    data = {}
    for offset, header in enumerate(headers):
        col = FIRST_STOCK_COL + offset
        column_values = [
            ws.cell(row=r, column=col).value
            for r in range(DATA_START_ROW, DATA_END_ROW + 1)
        ]
        # Keep raw header (including None/time objects) as the column key;
        # duplicates disambiguated by column letter to avoid silent overwrite.
        key = header if header is not None else f"<blank col {col}>"
        if key in data:
            key = f"{key} (col {col})"
        data[key] = column_values

    df = pd.DataFrame(data)
    print(f"      Loaded {df.shape[1]} raw columns x {df.shape[0]} months "
          f"(rows {DATA_START_ROW}-{DATA_END_ROW}).")
    return df, headers


# ---------------------------------------------------------------------------
# Stage 1b: Data quality gate
# ---------------------------------------------------------------------------
def apply_data_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns (stocks) with a complete, numeric 36-month series.
    Prints how many were kept/excluded and lists excluded names.
    """
    print("[2/6] Applying data quality gate (complete numeric series only)...")

    def is_complete_numeric(series: pd.Series) -> bool:
        # NOTE: pandas silently converts a Python None inside a float column
        # to NaN when the DataFrame is built, and NaN is itself a `float`
        # instance -- so a plain isinstance(v, float) check would wrongly
        # treat missing values as valid data. pd.isna() catches both None
        # and NaN explicitly.
        if len(series) != EXPECTED_MONTHS:
            return False
        for v in series:
            if isinstance(v, (str, datetime.date, datetime.time)):
                return False
            if pd.isna(v):
                return False
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                continue
            return False
        return True

    kept_cols = [c for c in df.columns if is_complete_numeric(df[c])]
    excluded_cols = [c for c in df.columns if c not in kept_cols]

    print(f"      Kept:     {len(kept_cols)} stocks with complete {EXPECTED_MONTHS}-month series")
    print(f"      Excluded: {len(excluded_cols)} stocks")
    if excluded_cols:
        print("      Excluded stock names:")
        for name in excluded_cols:
            print(f"        - {name}")

    clean_df = df[kept_cols].astype(float)
    return clean_df


# ---------------------------------------------------------------------------
# Stage 1c: Exclude benchmark/index columns
# ---------------------------------------------------------------------------
def exclude_benchmark_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any benchmark/index column, e.g. one literally named 'ASX200I', or
    one whose header was corrupted into a time-like value ('00:00:00').
    Warns if either case is found.
    """
    print("[3/6] Checking for benchmark/index columns to exclude...")

    to_drop = []
    for col in df.columns:
        name_str = str(col).strip().upper()
        if name_str in BENCHMARK_NAMES:
            print(f"      WARNING: excluding benchmark column '{col}' "
                  f"(matched known benchmark name).")
            to_drop.append(col)
        elif name_str in {"00:00:00", "0:00:00"} or "<blank col" in str(col):
            print(f"      WARNING: excluding column '{col}' "
                  f"(header appears corrupted, not a stock name).")
            to_drop.append(col)

    if not to_drop:
        print("      No benchmark/corrupted-header columns found.")

    result = df.drop(columns=to_drop)
    print(f"      {result.shape[1]} eligible stocks remain for analysis.")
    return result


# ---------------------------------------------------------------------------
# Stage 2: Correlation matrix
# ---------------------------------------------------------------------------
def build_correlation_matrix(returns: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Compute the Pearson correlation matrix across eligible stocks and save it."""
    print("[4/6] Building Pearson correlation matrix...")

    corr = returns.corr()
    corr.to_csv(out_path)

    print(f"      Correlation matrix: {corr.shape[0]} x {corr.shape[1]} stocks")
    print(f"      Saved to: {out_path}")
    return corr


# ---------------------------------------------------------------------------
# Stage 3: Eigenvector centrality
# ---------------------------------------------------------------------------
def compute_eigenvector_centrality(corr: pd.DataFrame, out_path: str, top_n: int = 7) -> pd.Series:
    """
    Compute eigenvector centrality on the |correlation| network (diagonal
    zeroed), normalized to sum to 1. Saves an ascending-ranked CSV (most
    peripheral first) and prints the top_n lowest-centrality stocks.
    """
    print("[5/6] Computing eigenvector centrality...")

    tickers = corr.columns
    adj = corr.abs().to_numpy()
    np.fill_diagonal(adj, 0.0)

    # Largest eigenvalue's eigenvector (adjacency is symmetric -> use eigh).
    eigenvalues, eigenvectors = np.linalg.eigh(adj)
    top_vector = eigenvectors[:, -1]
    # Sign convention: eigenvector centrality should be non-negative.
    top_vector = np.abs(top_vector)
    centrality = top_vector / top_vector.sum()

    centrality_series = pd.Series(centrality, index=tickers, name="centrality")
    centrality_series = centrality_series.sort_values(ascending=True)
    centrality_series.to_csv(out_path, header=True, index_label="stock")

    print(f"      Saved ranked centrality (ascending) to: {out_path}")
    print(f"      Top {top_n} most peripheral (lowest centrality) stocks:")
    for name, score in centrality_series.head(top_n).items():
        print(f"        {name:<40s} {score:.6f}")

    return centrality_series


# ---------------------------------------------------------------------------
# Stage 4: Minimum spanning tree cross-check
# ---------------------------------------------------------------------------
def compute_mst_degree(corr: pd.DataFrame, out_path: str) -> pd.Series:
    """
    Convert correlations to Mantegna distance (d = sqrt(2*(1-rho))), build
    the minimum spanning tree, and return each stock's degree in the tree.
    Saves a ranked (descending) CSV of MST degree.
    """
    print("[6/6] Cross-checking with minimum spanning tree (MST)...")

    tickers = corr.columns
    distance = np.sqrt(2.0 * (1.0 - corr.to_numpy()))
    np.fill_diagonal(distance, 0.0)

    mst = minimum_spanning_tree(distance)
    mst_dense = mst.toarray()
    # MST edges are directed by the sparse solver; symmetrize to count degree.
    adjacency = (mst_dense > 0) | (mst_dense.T > 0)
    degree = adjacency.sum(axis=1)

    degree_series = pd.Series(degree, index=tickers, name="mst_degree")
    degree_series = degree_series.sort_values(ascending=False)
    degree_series.to_csv(out_path, header=True, index_label="stock")

    print(f"      Saved MST degree ranking to: {out_path}")
    return degree_series


def validate_against_mst(centrality: pd.Series, mst_degree: pd.Series, top_n: int = 7) -> None:
    """
    Print how many of the top_n lowest-centrality stocks are MST leaf nodes
    (degree 1) -- a validation check that peripheral-by-centrality stocks
    are also peripheral-by-tree-structure.
    """
    top_peripheral = centrality.head(top_n).index
    leaf_nodes = set(mst_degree[mst_degree == 1].index)

    confirmed = [name for name in top_peripheral if name in leaf_nodes]

    print(f"      Top {top_n} eigenvector-centrality stocks and their MST degree:")
    for name in top_peripheral:
        deg = mst_degree.get(name, "n/a")
        flag = "leaf (confirmed)" if name in leaf_nodes else f"degree {deg}"
        print(f"        {name:<40s} {flag}")

    print(f"      >>> {len(confirmed)} of {top_n} confirmed by MST (leaf nodes) <<<")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank stocks by eigenvector centrality to pick peripheral "
                     "(diversifying) candidates from a monthly-returns workbook."
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        default=DEFAULT_DATA_FILE,
        help="Path to the Excel workbook (default: looks one directory above "
             "this script for 'EFB335 S2 2026 Group Project Data.xlsx').",
    )
    parser.add_argument("--top-n", type=int, default=7,
                         help="Number of most peripheral stocks to report (default: 7).")
    args = parser.parse_args()

    print("=" * 70)
    print("EFB335 Portfolio Construction: Eigenvector Centrality Pipeline")
    print("=" * 70)

    try:
        raw_df, _headers = load_monthly_returns(args.data_file)
        clean_df = apply_data_quality_gate(raw_df)
        eligible_df = exclude_benchmark_columns(clean_df)
        corr = build_correlation_matrix(eligible_df, "correlation_matrix.csv")
        centrality = compute_eigenvector_centrality(
            corr, "eigenvector_centrality_ranked.csv", top_n=args.top_n
        )
        mst_degree = compute_mst_degree(corr, "mst_degree_ranked.csv")
        validate_against_mst(centrality, mst_degree, top_n=args.top_n)
    except DataLayoutError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
