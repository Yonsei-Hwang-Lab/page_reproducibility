#!/usr/bin/env python3
"""Select data-compatible Fig. 5 easy/hard gene pools and validation holdouts."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = Path('/nas/project1/PAGE_PJH/01.rawdata/01.perturb-seq/replogle_k562_gwps.h5ad')
METRICS = Path('/nas/project1/PAGE_HSim/PAGE-codebase-HS/09_distribution_comparison_approach/page_reproducibility/03_prior_perturbseq_compare/intermediate')
KERNEL = Path('/nas/project1/PAGE_HSim/out/02.prior_perturbseq/exported_kernels/prior_gene_kernels/StringDB.gene_x_gene.csv.gz')
SEEDS = (42, 43, 44, 45, 46)
VAL_FRACTION = 0.1375  # Actual coefficient in data/replogle22k562gwps/get_data.py.


def read_column(group, name):
    column = group[name]
    if isinstance(column, h5py.Group):
        categories = column['categories'].asstr()[:]
        codes = column['codes'][:]
        if np.any(codes < 0):
            raise ValueError(f'Missing values in {name}')
        return categories[codes]
    return column.asstr()[:]


def raw_metadata(path, min_genes=200):
    """Read metadata only; X in this input is raw integer counts in a dense array."""
    with h5py.File(path, 'r') as f:
        conditions = read_column(f['obs'], 'perturbation')
        conditions = np.where(conditions == 'ctrl', 'control', conditions)
        genes = read_column(f['var'], f['var'].attrs['_index'])
        keep = f['obs']['ngenes'][:] >= min_genes
        if len(set(genes)) != len(genes):
            raise ValueError('Raw feature gene symbols must be unique')
        if f['X'].shape != (len(conditions), len(genes)):
            raise ValueError('Expected dense cells-by-genes X matching obs/var')
    return conditions, genes, keep


def draw_test_sets(eligible_genes, test_size, seeds):
    genes = sorted(eligible_genes)
    return [np.random.default_rng(seed).choice(genes, test_size, replace=False).tolist()
            for seed in seeds]


def select_folds(similarity, prior_scores, eligible_genes, train_size=400,
                 test_size=100, seeds=SEEDS, n_bins=100):
    """train_size is the selected pool BEFORE the 13.75% validation holdout."""
    eligible = sorted(set(eligible_genes))
    n_val = int(train_size * VAL_FRACTION)
    if not 0 < n_val < train_size or test_size < 5:
        raise ValueError('Need nonempty train/validation and at least five test genes')
    if len(eligible) < 2 * train_size + test_size:
        raise ValueError('Not enough eligible genes for disjoint easy, hard and test sets')
    folds = {}
    for i, (seed, test) in enumerate(zip(seeds, draw_test_sets(eligible, test_size, seeds))):
        candidates = [g for g in eligible if g not in set(test)]
        values = similarity.loc[candidates, test].to_numpy()
        scores = np.sort(values, axis=1)[:, -5:].mean(axis=1)
        d = pd.DataFrame({'prior_agreement': prior_scores.loc[candidates],
                          'similarity': scores}, index=candidates)
        if not np.isfinite(d.to_numpy()).all():
            raise ValueError('Non-finite prior agreement or StringDB similarity')
        d['bin'] = pd.qcut(d.prior_agreement, n_bins, duplicates='drop')
        pairs = []
        for _, group in d.groupby('bin', observed=True):
            group = group.sort_values('similarity', kind='stable')
            for j in range(len(group) // 2):
                low, high = group.iloc[j], group.iloc[-j - 1]
                pairs.append((low.name, high.name, float(high.similarity - low.similarity)))
        if len(pairs) < train_size:
            raise ValueError(f'Seed {seed}: only {len(pairs)} matched pairs; need {train_size}')
        pairs = sorted(pairs, key=lambda pair: pair[2], reverse=True)[:train_size]
        easy = [p[1] for p in pairs]
        hard = [p[0] for p in pairs]
        # Hold out the SAME pair positions in both arms, preserving the matching.
        order = np.random.RandomState(seed).permutation(train_size)
        val_positions, train_positions = order[:n_val], order[n_val:]
        folds[str(seed)] = {
            'fold_index': i, 'test': test,
            'selected_easy': easy, 'selected_hard': hard,
            'train_easy': [easy[j] for j in train_positions],
            'val_easy': [easy[j] for j in val_positions],
            'train_hard': [hard[j] for j in train_positions],
            'val_hard': [hard[j] for j in val_positions],
        }
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-h5ad', type=Path, default=RAW)
    parser.add_argument('--metrics-dir', type=Path, default=METRICS)
    parser.add_argument('--stringdb-kernel', type=Path, default=KERNEL)
    parser.add_argument('--output', type=Path, default=HERE / 'data/sets.json')
    parser.add_argument('--train-size', type=int, default=400, help='Selected genes per arm, including validation')
    parser.add_argument('--test-size', type=int, default=100)
    parser.add_argument('--min-cells', type=int, default=12)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f'{args.output} exists; use --force to replace it')
    if args.min_cells < 1:
        parser.error('--min-cells must be positive')

    conditions, features, keep = raw_metadata(args.raw_h5ad)
    counts = pd.Series(conditions[keep]).value_counts()
    if counts.get('control', 0) < 10:
        raise ValueError('Too few QC-passing controls for five train/val/test splits')
    local = pd.read_csv(args.metrics_dir / 'gene_local_overlap_allpair.csv', index_col='gene')
    global_ = pd.read_csv(args.metrics_dir / 'gene_global_overlap_allpair.csv', index_col='gene')
    prior = ((local + global_) / 2).mean(axis=1)
    available = pd.read_csv(args.metrics_dir / 'K562_GWPS_pertspace_gene_local_overlap_allpair.csv', usecols=['gene'])
    kernel_genes = pd.read_csv(args.stringdb_kernel, nrows=0).columns[1:]
    eligible = (set(counts[counts >= args.min_cells].index) & set(features)
                & set(available.gene) & set(prior[np.isfinite(prior)].index) & set(kernel_genes))
    # This experiment is single-gene; PRESAGE interprets underscores as combinations.
    eligible = sorted(g for g in eligible if g != 'control' and '_' not in g and '+' not in g)
    tests = draw_test_sets(eligible, args.test_size, SEEDS)
    needed_columns = {'gene'} | {g for test in tests for g in test}
    print(f'{len(eligible)} eligible genes; reading {len(needed_columns) - 1} StringDB columns', flush=True)
    similarity = pd.read_csv(args.stringdb_kernel, index_col='gene',
                             usecols=lambda name: name in needed_columns)
    folds = select_folds(similarity, prior, eligible, args.train_size, args.test_size)
    selected = sorted({g for fold in folds.values()
                       for key in ('selected_easy', 'selected_hard', 'test') for g in fold[key]})
    stat = args.raw_h5ad.stat()
    payload = {
        'schema_version': 1,
        'raw_h5ad': str(args.raw_h5ad.resolve()),
        'raw_size_bytes': stat.st_size, 'raw_mtime_ns': stat.st_mtime_ns,
        'selection': {
            'train_pool_size': args.train_size, 'test_size': args.test_size,
            'seeds': list(SEEDS), 'similarity': 'StringDB', 'similarity_top_k': 5,
            'prior_agreement_bins': 100, 'validation_fraction_of_non_test': VAL_FRACTION,
            'min_genes_per_cell': 200, 'min_cells_per_condition': args.min_cells,
            'eligible_gene_count': len(eligible), 'selected_gene_count': len(selected),
            'metrics_dir': str(args.metrics_dir.resolve()),
            'stringdb_kernel': str(args.stringdb_kernel.resolve()),
        },
        'cell_counts_after_qc': {g: int(counts[g]) for g in selected + ['control']},
        'folds': folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Wrote {args.output}; {len(folds)} folds per arm, '
          f'{len(folds[str(SEEDS[0])]["train_easy"])} train / '
          f'{len(folds[str(SEEDS[0])]["val_easy"])} validation / {args.test_size} test')


if __name__ == '__main__':
    main()