import gseapy as gp
import pandas as pd

gene_level_global_metrics = pd.read_csv('/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/gene_global_overlap_allpair.csv').set_index('gene')
gene_level_local_metrics = pd.read_csv('/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/gene_local_overlap_allpair.csv').set_index('gene')

gene_agreement_score = (gene_level_global_metrics+gene_level_local_metrics)/2

all_enriched_terms = []
all_gsea_results = []
for col in gene_agreement_score.columns.tolist():
    
    rnk = gene_agreement_score[col]
    rnk = rnk[rnk!=0]

    try:
        res = gp.prerank(rnk=rnk, gene_sets=['CORUM','GWAS_Catalog_2025','GO_Biological_Process_2023'], seed=42, min_size=10, max_size=500)
        tmp = res.res2d[res.res2d["FDR q-val"]<0.01].sort_values("NES")[::-1]
        tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{col}.csv')
    except:
        print('Insufficient # of Genes with agreement')
        tmp = pd.DataFrame(None, columns=['Name','Term','ES','NES','NOM p-val','FDR q-val','FWER p-val','Tag %','Gene %','Lead_genes'])
        tmp.to_csv(f'/home/jhpark/page_reproducibility/03_prior_perturbseq_compare/intermediate/{col}.csv')