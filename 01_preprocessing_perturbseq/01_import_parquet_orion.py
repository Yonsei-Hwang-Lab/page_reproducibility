from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

import anndata as ad
from anndata.experimental import concat_on_disk
import scanpy as sc

# ----------------------------
# config
# ----------------------------
REPO_DIR = Path("/nas/project1/PAGE_PJH/01.rawdata/01.perturb-seq/xatlas_orion")   # downloaded HF repo root
OUT_DIR = Path("/nas/project1/PAGE_PJH/01.rawdata/01.perturb-seq/")
TMP_DIR = OUT_DIR / "_tmp"

# Lower this if RAM is tight, raise it if you have more RAM.
# 1000~5000 is recommended
ROWS_PER_CHUNK = 5000

# lzf: faster; gzip: smaller but slower
H5AD_COMPRESSION = "lzf"

OBS_COLS = [
    "cell_barcode",
    "sample",
    "num_features",
    "guide_target",
    "gene_target",
    "n_genes_by_counts",
    "total_counts",
    "total_counts_mt",
    "pct_counts_mt",
    "pass_guide_filter",
]
ALL_COLS = ["gene_token_id", "gene_expression", *OBS_COLS]


def natural_key(path: Path):
    m = re.search(r"Batch(\d+)", path.stem)
    return (path.stem.split("Batch")[0], int(m.group(1)) if m else -1)


def load_var(repo_dir: Path) -> pd.DataFrame:
    """
    Build adata.var from metadata/gene_metadata.parquet.
    Column order is gene_token_id order.
    """
    gene_meta = pq.read_table(repo_dir / "metadata" / "gene_metadata.parquet").to_pandas()
    gene_meta = gene_meta.sort_values("gene_token_id").reset_index(drop=True)

    # The released metadata uses gene_token_id as the token for non-zero expression columns.
    # This code assumes the ids are contiguous 0..n_vars-1.
    expected = np.arange(len(gene_meta), dtype=np.int64)
    actual = gene_meta["gene_token_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(
            "gene_token_id가 0..n_vars-1 연속 정수가 아닙니다. "
            "이 경우 token->column 매핑 로직으로 바꿔야 합니다."
        )

    var = gene_meta.set_index("ensembl_id")[["gene_name", "gene_token_id"]]
    return var


def recordbatch_to_adata(rb, var: pd.DataFrame) -> ad.AnnData:
    """
    Convert one pyarrow RecordBatch to AnnData with sparse CSR X.
    """
    cols = {name: rb.column(i).to_pylist() for i, name in enumerate(rb.schema.names)}

    gene_lists = cols.pop("gene_token_id")
    expr_lists = cols.pop("gene_expression")
    n_obs = len(gene_lists)
    n_vars = var.shape[0]

    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    data_parts = []
    index_parts = []
    nnz = 0

    for i, (genes, expr) in enumerate(zip(gene_lists, expr_lists)):
        genes = np.asarray(genes, dtype=np.int32)
        expr = np.asarray(expr, dtype=np.int32)

        data_parts.append(expr)
        index_parts.append(genes)

        nnz += len(expr)
        indptr[i + 1] = nnz

    if nnz == 0:
        data = np.array([], dtype=np.int32)
        indices = np.array([], dtype=np.int32)
    else:
        data = np.concatenate(data_parts)
        indices = np.concatenate(index_parts)

    X = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_obs, n_vars),
        dtype=np.int32,
    )

    obs = pd.DataFrame(cols).set_index("cell_barcode")

    adata = ad.AnnData(X=X, obs=obs, var=var.copy())
    return adata


def parquet_to_h5ad(parquet_path: Path, out_h5ad: Path, var: pd.DataFrame) -> None:
    """
    Convert one parquet file to one h5ad using chunked reading + on-disk concat.
    """
    pf = pq.ParquetFile(parquet_path)
    chunk_paths = []

    for i, rb in enumerate(pf.iter_batches(batch_size=ROWS_PER_CHUNK, columns=ALL_COLS)):
        chunk_path = TMP_DIR / f"{parquet_path.stem}.chunk{i:05d}.h5ad"

        adata = recordbatch_to_adata(rb, var)
        adata.write_h5ad(chunk_path, compression=H5AD_COMPRESSION)
        chunk_paths.append(chunk_path)

        del adata

    concat_on_disk(
        chunk_paths,
        out_h5ad,
        axis=0,
        join="inner",
        max_loaded_elems=50_000_000,  # tune if needed
    )

    for p in chunk_paths:
        p.unlink()


def build_split(split_prefix: str, out_name: str, var: pd.DataFrame) -> Path:
    """
    Build one final split-level h5ad, e.g. HCT116 -> hct116.h5ad
    """
    parquet_files = sorted(
        (REPO_DIR / "data").glob(f"{split_prefix}_Batch*.parquet"),
        key=natural_key,
    )
    if not parquet_files:
        raise FileNotFoundError(f"'{split_prefix}_Batch*.parquet' file is missing")

    batch_h5ads = []

    for parquet_path in parquet_files:
        batch_h5ad = TMP_DIR / f"{parquet_path.stem}.h5ad"
        print(f"[convert] {parquet_path.name} -> {batch_h5ad.name}")
        parquet_to_h5ad(parquet_path, batch_h5ad, var)
        batch_h5ads.append(batch_h5ad)

    out_path = OUT_DIR / out_name
    print(f"[concat] {split_prefix} -> {out_path}")
    concat_on_disk(
        batch_h5ads,
        out_path,
        axis=0,
        join="inner",
        max_loaded_elems=50_000_000,
    )

    for p in batch_h5ads:
        p.unlink()

    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    var = load_var(REPO_DIR)

    # final outputs
    hct116_h5ad = build_split("HCT116", "hct116.h5ad", var)
    hek293t_h5ad = build_split("HEK293T", "hek293t.h5ad", var)

    print(f"wrote: {hct116_h5ad}")
    print(f"wrote: {hek293t_h5ad}")

    # scanpy import (backed mode)
    adata_hct116 = sc.read_h5ad(hct116_h5ad, backed="r")
    adata_hek293t = sc.read_h5ad(hek293t_h5ad, backed="r")

    print(adata_hct116)
    print(adata_hek293t)

    # done with backed handles
    adata_hct116.file.close()
    adata_hek293t.file.close()


if __name__ == "__main__":
    main()