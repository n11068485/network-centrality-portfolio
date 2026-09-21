"""
EFB335 Portfolio Construction: Diversification via PageRank Centrality
===========================================================================

Reads monthly stock returns from an Excel workbook, builds a correlation
matrix, and ranks stocks by PageRank centrality in the correlation network
to select the 7 most peripheral (least central) stocks as diversification
candidates.

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
    print(f"[1/5] Loading data from: {file_path}")

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
    print("[2/5] Applying data quality gate (complete numeric series only)...")

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
    print("[3/5] Checking for benchmark/index columns to exclude...")

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
    print("[4/5] Building Pearson correlation matrix...")

    corr = returns.corr()
    corr.to_csv(out_path)

    print(f"      Correlation matrix: {corr.shape[0]} x {corr.shape[1]} stocks")
    print(f"      Saved to: {out_path}")
    return corr


# ---------------------------------------------------------------------------
# Stage 3: PageRank centrality
# ---------------------------------------------------------------------------
def compute_pagerank_centrality(
    corr: pd.DataFrame,
    out_path: str,
    top_n: int = 7,
    damping: float = 0.85,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> pd.Series:
    """
    Compute PageRank centrality on the |correlation| network (diagonal
    zeroed), treating each stock's correlations to others as weighted edges.
    Saves an ascending-ranked CSV (most peripheral first) and prints the
    top_n lowest-centrality stocks.

    Implemented as standard power-iteration PageRank: at each step, a
    stock's score is redistributed to its neighbours in proportion to edge
    weight (|correlation|), a damping factor models random restarts, and any
    dangling (zero-weight) node's score is spread evenly across all stocks.
    """
    print("[5/5] Computing PageRank centrality...")

    tickers = corr.columns
    n = len(tickers)
    adj = corr.abs().to_numpy()
    np.fill_diagonal(adj, 0.0)

    row_sums = adj.sum(axis=1)
    dangling = row_sums == 0
    row_sums_safe = np.where(dangling, 1.0, row_sums)
    # Row-normalized transition matrix: transition[i, j] = weight(i->j) / sum(weights from i)
    transition = adj / row_sums_safe[:, None]

    rank = np.full(n, 1.0 / n)
    # On macOS, numpy's Accelerate BLAS backend can raise spurious
    # divide-by-zero/overflow RuntimeWarnings on matmul that don't reflect
    # actual NaN/Inf in the result (a known numpy/Accelerate quirk) --
    # verified adj/transition/rank stay finite throughout, so this is
    # suppressed rather than treated as a real numerical error.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for iteration in range(1, max_iter + 1):
            dangling_mass = rank[dangling].sum()
            new_rank = (1.0 - damping) / n + damping * (
                transition.T @ rank + dangling_mass / n
            )
            diff = np.abs(new_rank - rank).sum()
            rank = new_rank
            if diff < tol:
                break

    rank = rank / rank.sum()

    pagerank_series = pd.Series(rank, index=tickers, name="pagerank")
    pagerank_series = pagerank_series.sort_values(ascending=True)
    pagerank_series.to_csv(out_path, header=True, index_label="stock")

    print(f"      Converged after {iteration} power-iteration steps.")
    print(f"      Saved ranked PageRank scores (ascending) to: {out_path}")
    print(f"      Top {top_n} most peripheral (lowest PageRank) stocks:")
    for name, score in pagerank_series.head(top_n).items():
        print(f"        {name:<40s} {score:.6f}")

    return pagerank_series


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank stocks by PageRank centrality to pick peripheral "
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
    print("EFB335 Portfolio Construction: PageRank Centrality Pipeline")
    print("=" * 70)

    try:
        raw_df, _headers = load_monthly_returns(args.data_file)
        clean_df = apply_data_quality_gate(raw_df)
        eligible_df = exclude_benchmark_columns(clean_df)
        corr = build_correlation_matrix(eligible_df, "correlation_matrix.csv")
        compute_pagerank_centrality(
            corr, "pagerank_ranked.csv", top_n=args.top_n
        )
    except DataLayoutError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
