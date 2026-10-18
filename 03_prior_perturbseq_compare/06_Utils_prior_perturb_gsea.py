from gseapy import dotplot, barplot
import gseapy as gp
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

gene_level_local_metrics = {}
gene_level_global_metrics = {}
analysis_targets = ['K562_GWPS','K562_Essential','RPE1_Essential','HepG2_Essential','Jurkat_Essential', 'RPE1_Southard']
for perturb_seq_name in analysis_targets: 
    gene_level_local_metrics[perturb_seq_name] = pd.read_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{perturb_seq_name}_pertspace_gene_local_overlap_allpair.csv').set_index('gene')
    gene_level_global_metrics[perturb_seq_name] = pd.read_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{perturb_seq_name}_pertspace_gene_global_overlap_allpair.csv').set_index('gene')

for perturb_seq_name in analysis_targets: 
    local_df = gene_level_local_metrics[perturb_seq_name]
    global_df = gene_level_global_metrics[perturb_seq_name]
    gene_agreement_score = (global_df+local_df)/2
    all_enriched_terms = []
    all_gsea_results = []
    for col in gene_agreement_score.columns.tolist():
        #corum_library = gp.get_library(name=['CORUM','GWAS_Catalog_2025','KEGG_2026','GO_Biological_Process_2023'], organism='Human')
        #library = gp.get_library(name='GWAS_Catalog_2025', organism='Human')
        #rnk = gene_agreement_score.mean(axis=1)
        rnk = gene_agreement_score[col]
        rnk = rnk[rnk!=0]
        try:
            res = gp.prerank(rnk=rnk, gene_sets=['CORUM','GWAS_Catalog_2025','GO_Biological_Process_2023'], seed=42, min_size=10, max_size=500)
            tmp = res.res2d[res.res2d["FDR q-val"]<0.01].sort_values("NES")[::-1]
            tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/pertspace_{perturb_seq_name}_{col}.csv')
        except:
            print('Insufficient # of Genes with agreement')
            tmp = pd.DataFrame(None, columns=['Name','Term','ES','NES','NOM p-val','FDR q-val','FWER p-val','Tag %','Gene %','Lead_genes'])
            tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/pertspace_{perturb_seq_name}_{col}.csv')

gene_level_local_metrics = {}
gene_level_global_metrics = {}
analysis_targets = ['K562_GWPS','K562_Essential','RPE1_Essential','HepG2_Essential','Jurkat_Essential', 'RPE1_Southard']
for perturb_seq_name in analysis_targets: 
    gene_level_local_metrics[perturb_seq_name] = pd.read_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{perturb_seq_name}_mrnaspace_gene_local_overlap_allpair.csv').set_index('gene')
    gene_level_global_metrics[perturb_seq_name] = pd.read_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{perturb_seq_name}_mrnaspace_gene_global_overlap_allpair.csv').set_index('gene')

for perturb_seq_name in analysis_targets: 
    local_df = gene_level_local_metrics[perturb_seq_name]
    global_df = gene_level_global_metrics[perturb_seq_name]
    gene_agreement_score = (global_df+local_df)/2
    all_enriched_terms = []
    all_gsea_results = []
    for col in gene_agreement_score.columns.tolist():
        #corum_library = gp.get_library(name=['CORUM','GWAS_Catalog_2025','KEGG_2026','GO_Biological_Process_2023'], organism='Human')
        #library = gp.get_library(name='GWAS_Catalog_2025', organism='Human')
        #rnk = gene_agreement_score.mean(axis=1)
        rnk = gene_agreement_score[col]
        rnk = rnk[rnk!=0]
        try:
            res = gp.prerank(rnk=rnk, gene_sets=['CORUM','GWAS_Catalog_2025','GO_Biological_Process_2023'], seed=42, min_size=10, max_size=500)
            tmp = res.res2d[res.res2d["FDR q-val"]<0.01].sort_values("NES")[::-1]
            tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/mrnaspace_{perturb_seq_name}_{col}.csv')
        except:
            print('Insufficient # of Genes with agreement')
            tmp = pd.DataFrame(None, columns=['Name','Term','ES','NES','NOM p-val','FDR q-val','FWER p-val','Tag %','Gene %','Lead_genes'])
            tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/mrnaspace_{perturb_seq_name}_{col}.csv')