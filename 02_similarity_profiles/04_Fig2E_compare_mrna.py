import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr, spearmanr, energy_distance
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import seaborn as sns
from scipy.stats import wilcoxon
import pdb

def compare(x, y, method):
    if method == 'pearsonr':
        return pearsonr(x, y)[0]
    elif method == 'spearmanr':
        return spearmanr(x, y)[0]
    elif method == 'cosine':
        return cosine_similarity(x[None,:], y[None,:])[0]


data_path_dict = {"replogle_k562_gwps":"/nas/project1/PAGE_PJH/02.processed/replogle_k562_gwps_min10cells_effectsize.csv.gz",
"replogle_k562_essential":"/nas/project1/PAGE_PJH/02.processed/replogle_k562_essential_min10cells_effectsize.csv.gz"}

data_dict = {}
for study in data_path_dict.keys():
    data_dict[study] = pd.read_csv(data_path_dict[study])

for method in ['pearsonr', 'spearmanr', 'cosine']:
    median_single_results_transposed = None
    median_pairwise_results_transposed = None
    single_results_transposed = []
    pairwise_results_transposed = []

    k562_gwps_norm = data_dict['replogle_k562_gwps'].set_index('guide')-data_dict['replogle_k562_gwps'].set_index('guide').mean(axis=0)
    k562_essential = data_dict['replogle_k562_essential'].set_index('guide')-data_dict['replogle_k562_essential'].set_index('guide').mean(axis=0)
    common_p = list(set(k562_gwps_norm.index).intersection(set(k562_essential.index)))
    common_g = list(set(k562_gwps_norm.columns).intersection(set(k562_essential.columns)))
    k562_gwps_norm = k562_gwps_norm.loc[common_p, common_g]
    k562_essential = k562_essential.loc[common_p, common_g]
    k562_gwps_norm_kernel = pd.DataFrame(cosine_similarity(np.array(k562_gwps_norm.T.loc[common_g, :]), np.array(k562_gwps_norm.T.loc[common_g, :])), index=common_g, columns=common_g)
    k562_essential_kernel = pd.DataFrame(cosine_similarity(np.array(k562_essential.T.loc[common_g, :]), np.array(k562_gwps_norm.T.loc[common_g, :])), index=common_g, columns=common_g)

    for iteration in tqdm(range(100)):
        cognate_data_transposed = []
        random_data_transposed = []

        for i in common_g:
            cognate_data_transposed.append(compare(np.array(k562_gwps_norm[i]), np.array(k562_essential[i]), method=method))
            random_data_transposed.append(compare(np.array(k562_gwps_norm[i]), np.array(k562_essential.T.iloc[random.randint(0, len(common_g)-1)].T), method=method))

        single_dist_transposed = energy_distance(np.array(random_data_transposed).flatten(), np.array(cognate_data_transposed).flatten())
        

        cognate_data_kernel_transposed = []
        random_data_kernel_transposed = []

        for i in common_g:
            cognate_data_kernel_transposed.append(compare(np.array(k562_gwps_norm_kernel.loc[i]), np.array(k562_essential_kernel.loc[i]), method=method))
            random_data_kernel_transposed.append(compare(np.array(k562_gwps_norm_kernel.loc[i]), np.array(k562_essential_kernel.iloc[random.randint(0, len(common_g)-1)]), method=method))
        pairwise_dist_transposed = energy_distance(np.array(random_data_kernel_transposed).flatten(), np.array(cognate_data_kernel_transposed).flatten())

        single_results_transposed.append(single_dist_transposed)
        pairwise_results_transposed.append(pairwise_dist_transposed)

        if single_dist_transposed == np.median(single_results_transposed):
            median_single_results_transposed = [cognate_data_transposed, random_data_transposed]
        if pairwise_dist_transposed == np.median(pairwise_results_transposed):
            median_pairwise_results_transposed = [cognate_data_kernel_transposed, random_data_kernel_transposed]

        
    plt.figure(figsize=(2, 5))
    sns.boxplot(x=0, y=single_results_transposed)
    sns.boxplot(x=1, y=pairwise_results_transposed)
    plt.xticks(ticks=[0,1], labels=['Raw\nProfile', 'Similarity\nProfile'])
    plt.ylabel('Energy Distance')
    plt.title('100 Repeats')
    plt.show()
    plt.savefig(f'./figure_out/{method}_Fig2E_3.pdf')


    plt.figure(figsize=(5, 5))
    
    cognate_data_transpoed, random_data_transpoed = median_single_results_transposed[0], median_single_results_transposed[1]
    cognate_data_kernel_transposed, random_data_kernel_transposed = median_pairwise_results_transposed[0], median_pairwise_results_transposed[1]
    sns.kdeplot(np.array(random_data_transposed).flatten(), c='black', label='Random Raw Profile')
    sns.kdeplot(np.array(cognate_data_transposed).flatten(), c='tab:blue', label='Cognate Raw Profile')
    sns.kdeplot(np.array(random_data_kernel_transposed).flatten(), color='black', label='Random Kernel', fill=True)
    sns.kdeplot(np.array(cognate_data_kernel_transposed).flatten(), color='tab:blue', label='Cognate Kernel', fill=True)
    plt.legend(prop={'size': 8}, loc='upper left')#, bbox_to_anchor=(1.04, 1))
    plt.title('mRNA Space')

    plt.show()
    plt.savefig(f'./figure_out/{method}_Fig2E_4.pdf')
    
    print('Method', method, 'Summary:')
    print(wilcoxon(single_results_transposed, pairwise_results_transposed))
    print(np.median(single_results_transposed), np.median(pairwise_results_transposed))
    #np.save(f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/{method}_single_mrna.npy',single_results_transposed)
    #np.save(f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/{method}_pairwise_mrna.npy',pairwise_results_transposed)

pdb.set_trace()