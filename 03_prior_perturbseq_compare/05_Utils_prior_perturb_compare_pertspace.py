import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import Dict, Tuple, List
import warnings
import random
warnings.filterwarnings('ignore')

class LocalNeighborAnalysis:

    def get_top_k_neighbors(K: np.ndarray, k: int = 50, 
                           exclude_self: bool = True) -> np.ndarray:
        n = K.shape[0]
        neighbors = np.zeros((n, k), dtype=int)
        
        for i in range(n):
            if exclude_self:
                # Set self-similarity to -inf to exclude it
                similarities = K[i, :].copy()
                similarities[i] = -np.inf
            else:
                similarities = K[i, :]
            
            # Get top-k indices
            neighbors[i, :] = np.argsort(similarities)[::-1][:k]
        
        return neighbors
    
    def neighbor_overlap(neighbors1: np.ndarray, neighbors2: np.ndarray) -> np.ndarray:
        n = neighbors1.shape[0]
        k = neighbors1.shape[1]
        overlaps = np.zeros(n)
        
        for i in range(n):
            set1 = set(neighbors1[i, :])
            set2 = set(neighbors2[i, :])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            overlaps[i] = intersection / union if union > 0 else 0
        
        null_overlaps = np.zeros(1000)
        
        for i in range(1000):
            set1 = set(neighbors1[random.choice(range(n)), :])
            set2 = set(neighbors2[random.choice(range(n)), :])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            null_overlaps[i] = intersection / union if union > 0 else 0
        
        return (overlaps-np.mean(null_overlaps))/np.std(null_overlaps)
    
    def rank_correlation_per_gene(K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
        n = K1.shape[0]
        correlations = np.zeros(n)
        null_correlations = np.zeros(n)
        
        for i in range(n):
            # Get similarity vectors (excluding self)
            sim1 = np.delete(K1[i, :], i)
            sim2 = np.delete(K2[i, :], i)

            # Compute Spearman correlation
            correlations[i] = spearmanr(sim1, sim2)[0]
        
        null_correlations = np.zeros(1000)
        
        for i in range(1000):
            sim1_ridx = random.choice(range(n))
            sim2_ridx = random.choice(range(n))
            sim1 = np.delete(K1[sim1_ridx, :], sim1_ridx)
            sim2 = np.delete(K2[sim2_ridx, :], sim2_ridx)
                        
            null_correlations[i] = spearmanr(sim1, sim2)[0]
        
        return (correlations-np.mean(null_correlations))/np.std(null_correlations)
    
    def weighted_neighbor_preservation(K1: np.ndarray, K2: np.ndarray, 
                                      k: int = 10) -> np.ndarray:
        n = K1.shape[0]
        scores = np.zeros(n)
        
        neighbors1 = LocalNeighborAnalysis.get_top_k_neighbors(K1, k)
        neighbors2 = LocalNeighborAnalysis.get_top_k_neighbors(K2, k)
        
        for i in range(n):
            # Get neighbors and their ranks in K1
            nb1 = neighbors1[i, :]
            nb2_set = set(neighbors2[i, :])
            
            # Weights decay with rank (1/rank)
            weights = 1.0 / np.arange(1, k + 1)
            
            # Score: weighted sum of preserved neighbors
            preserved = np.array([1.0 if nb in nb2_set else 0.0 for nb in nb1])
            scores[i] = np.sum(weights * preserved) / np.sum(weights)
        
        return scores
    
    def compute_local_metrics(kernels: Dict[str, pd.DataFrame], 
                             k: int = 10) -> pd.DataFrame:
        results = []
        keys = list(kernels.keys())
        
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                if i >= j:
                    continue
                                
                K1 = kernels[key1].values
                K2 = kernels[key2].values
                
                # Compute metrics
                neighbors1 = LocalNeighborAnalysis.get_top_k_neighbors(K1, k)
                neighbors2 = LocalNeighborAnalysis.get_top_k_neighbors(K2, k)
                
                overlap = LocalNeighborAnalysis.neighbor_overlap(neighbors1, neighbors2)
                rank_corr = LocalNeighborAnalysis.rank_correlation_per_gene(K1, K2)
                
                results.append({
                    'Source1': key1,
                    'Source2': key2,
                    'Mean_Neighbor_Overlap': overlap.mean(),
                    'Std_Neighbor_Overlap': overlap.std(),
                    'Mean_Rank_Correlation': rank_corr.mean(),
                    'Std_Rank_Correlation': rank_corr.std(),
                })
        
        return pd.DataFrame(results)


class GeneLevelConfidenceScore:
    """
    Assign confidence scores to genes based on cross-source consistency
    """
    
    def neighborhood_consistency_score(kernels: Dict[str, pd.DataFrame],
                                      ps_kernel: pd.DataFrame, 
                                      k: int = 10) -> pd.Series:
        
        keys = list(kernels.keys())
        n_sources = len(keys)
        gene_names = kernels[keys[0]].index
        n_genes = len(gene_names)
        
        # Store all pairwise overlaps for each gene
        all_overlaps = np.zeros((n_genes, n_sources))
        all_rank_corr = np.zeros((n_genes, n_sources))
        
        pair_idx = 0
        column_names = [] #[JUNHA]
        for i, key1 in enumerate(keys):
            
            K1 = kernels[key1].values
            K2 = ps_kernel.values
            
            neighbors1 = LocalNeighborAnalysis.get_top_k_neighbors(K1, k)
            neighbors2 = LocalNeighborAnalysis.get_top_k_neighbors(K2, k)
            
            overlap = LocalNeighborAnalysis.neighbor_overlap(neighbors1, neighbors2)
            rank_corr = LocalNeighborAnalysis.rank_correlation_per_gene(K1, K2)
            all_overlaps[:, pair_idx] = overlap
            all_rank_corr[:, pair_idx] = rank_corr
            pair_idx += 1
            column_names.append(f'{key1}') #[JUNHA]

        return pd.DataFrame(all_overlaps, index=gene_names, columns=column_names), pd.DataFrame(all_rank_corr, index=gene_names, columns=column_names) #[JUNHA]
        
        #return pd.Series(scores, index=gene_names, name='Neighborhood_Consistency')
    
    
    def integrated_confidence_score(kernels: Dict[str, pd.DataFrame],
                                   ps_kernel: pd.DataFrame, 
                                   k: int = 10) -> pd.DataFrame:
        # Compute individual metrics
        neighborhood_cons, rank_corr_cons = GeneLevelConfidenceScore.neighborhood_consistency_score(
            kernels, ps_kernel, k=k
        )

        return neighborhood_cons, rank_corr_cons # [JUNHA]
        
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
intersection_genes = np.load('/nas/project1/PAGE_PJH/01.rawdata/00.data_gene_symbols/intersection.npy')

print('Loading perturb-seq data...')
perturbseq_data = {
        'K562_GWPS': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/replogle_k562_gwps_min10cells_effectsize.csv.gz', index_col=0),
        'K562_Essential': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/replogle_k562_essential_min10cells_effectsize.csv.gz', index_col=0),
        'RPE1_Essential': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/replogle_rpe1_min10cells_effectsize.csv.gz', index_col=0),
        'HepG2_Essential': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/nadig_hepg2_min10cells_effectsize.csv.gz', index_col=0),
		'Jurkat_Essential': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/nadig_jurkat_min10cells_effectsize.csv.gz', index_col=0),
        'RPE1_Southard': pd.read_csv('/nas/project1/PAGE_PJH/02.processed/southard_rpe1_filtered_effectsize.csv.gz', index_col=0)
    }

for perturb_seq_name in ['K562_GWPS', 'K562_Essential','RPE1_Essential','HepG2_Essential','Jurkat_Essential','RPE1_Southard']:

    print(f"================== \n Analyzing {perturb_seq_name} \n==================")
    prior_perturbseq_intersection_genes = list(set(perturbseq_data[perturb_seq_name].index.tolist()).intersection(set(intersection_genes)))
    df_pert = perturbseq_data[perturb_seq_name].loc[prior_perturbseq_intersection_genes]
    K_ps = cosine_similarity(df_pert)
    K_ps = pd.DataFrame(K_ps, index=df_pert.index, columns=df_pert.index)

    '''
    01. marbach et al., CML; spearmanr
    '''
    from sklearn.metrics.pairwise import cosine_similarity

    df = pd.read_csv('/nas/project1/PAGE_PJH/01.rawdata/03.regulation/marbach_network/chronic_myelogenous_leukemia_cml_cell_line.txt.gz', sep='\t', header=None)
    df.columns = ['tf','gene','weight']
    df = df.pivot(index='gene', columns='tf', values='weight')
    df = df.loc[prior_perturbseq_intersection_genes]
    df = df.fillna(0)

    K_marbach_cml = pd.DataFrame(spearmanr(df.values.T).statistic, index=df.index, columns=df.index)

    '''
    02. UnTangled: ensembl converted to gene symbols ; spearmanr
    '''

    df = pd.read_csv('/nas/project1/PAGE_PJH/01.rawdata/05.comprehensive/multiXcan_gene_trait_assoc.txt.gz', sep=' ', header=0)
    with open('/home/jhpark/page-study/01.preprocessing/intermediate/UnTangled_geneset.txt','r') as f:
        genes = f.read().splitlines() 
    df.columns=genes
    df = df.T.loc[prior_perturbseq_intersection_genes]
    df = df.fillna(0)
    K_untangled = spearmanr(df.values.T).statistic
    K_untangled = pd.DataFrame(K_untangled, index=df.index, columns=df.index)

    '''
    03. msigdb hallmark; cosine
    '''
    with open('/nas/project1/PAGE_PJH/01.rawdata/04.function/msigdb_c5.all.v2023.2.Hs.symbols.pkl', 'rb') as f:
        df = pickle.load(f)
    df = df.loc[prior_perturbseq_intersection_genes]
    K_msigdb = cosine_similarity(df)
    K_msigdb = pd.DataFrame(K_msigdb, index=df.index, columns=df.index)

    '''
    04. stringdb medium; cosine
    '''
    with open('/nas/project1/PAGE_PJH/01.rawdata/05.comprehensive/stringdb.human.medium.pkl', 'rb') as f:
        df = pickle.load(f)

    df = df.loc[prior_perturbseq_intersection_genes]
    K_stringdb_medium = cosine_similarity(df)
    K_stringdb_medium = pd.DataFrame(K_stringdb_medium, index=df.index, columns=df.index)

    '''
    05. depmap; cosine
    '''
    with open('/nas/project1/PAGE_PJH/01.rawdata/04.function/depmap_broad_24q2.pkl', 'rb') as f:
        df = pickle.load(f)
    df = df.loc[prior_perturbseq_intersection_genes]
    K_depmap = cosine_similarity(df)
    K_depmap = pd.DataFrame(K_depmap, index=df.index, columns=df.index)

    '''
    06. periscope; spearmanr
    '''
    df = pd.read_csv('/nas/project1/PAGE_PJH/01.rawdata/04.function/periscope_a549_profile.csv.gz')
    df = df.set_index('Metadata_Foci_Barcode_MatchedTo_GeneCode')
    df = df.loc[prior_perturbseq_intersection_genes]
    K_periscope = spearmanr(df.T).statistic
    K_periscope = pd.DataFrame(K_periscope, index=df.index, columns=df.index)

    '''
    07. scgpt all available; cosine
    '''
    df = pd.read_csv('/nas/project1/PAGE_PJH/01.rawdata/02.expression/scgpt_embedding.csv')
    df = df.set_index('Unnamed: 0')
    df = df.loc[prior_perturbseq_intersection_genes]
    K_scgpt = cosine_similarity(df)
    K_scgpt = pd.DataFrame(K_scgpt, index=df.index, columns=df.index)

    '''
    08. genept; cosine with subtracted mean
    '''
    with open('/nas/project1/PAGE_PJH/01.rawdata/05.comprehensive/genept_ada.pkl', 'rb') as f:
        df = pickle.load(f)
    df = df-df.mean(axis=0)
    df = df.loc[prior_perturbseq_intersection_genes]
    K_genept = cosine_similarity(df)
    K_genept = pd.DataFrame(K_genept, index=df.index, columns=df.index)

    kernels = {
        'Marbach': K_marbach_cml,
        'MsigDB': K_msigdb,
        'DepMap': K_depmap,
        'StringDB': K_stringdb_medium,
        'Periscope': K_periscope,
        'Untangled': K_untangled,
        'scGPT': K_scgpt,
        'genePT': K_genept,
    }

    print('Kernels are computed!')

    import os
    output_dir='/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("RUNNING COMPREHENSIVE PRIOR KNOWLEDGE - PERTURBSEQ ANALYSIS")
    print("="*80)

    print("\n Computing gene-level confidence scores...")
    confidence_local_df, confidence_global_df = GeneLevelConfidenceScore.integrated_confidence_score(kernels, K_ps, k=100)
    confidence_local_df.to_csv(f'{output_dir}/260226_{perturb_seq_name}_pertspace_gene_local_overlap_allpair.csv')
    confidence_global_df.to_csv(f'{output_dir}/260226_{perturb_seq_name}_pertspace_gene_global_overlap_allpair.csv')

    print(f"Saved to {output_dir}/260226_{perturb_seq_name}_pertspace_gene_local_overlap_allpair.csv")
    print(f"Saved to {output_dir}/260226_{perturb_seq_name}_pertspace_gene_global_overlap_allpair.csv")