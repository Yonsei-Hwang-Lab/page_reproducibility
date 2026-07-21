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
    single_results = []
    pairwise_results = []

    k562_gwps_norm = data_dict['replogle_k562_gwps'].set_index('guide')-data_dict['replogle_k562_gwps'].set_index('guide').mean(axis=0)
    k562_essential = data_dict['replogle_k562_essential'].set_index('guide')-data_dict['replogle_k562_essential'].set_index('guide').mean(axis=0)
    common_p = list(set(k562_gwps_norm.index).intersection(set(k562_essential.index)))
    common_g = list(set(k562_gwps_norm.columns).intersection(set(k562_essential.columns)))
    k562_gwps_norm = k562_gwps_norm.loc[common_p, common_g]
    k562_essential = k562_essential.loc[common_p, common_g]
    k562_gwps_norm_kernel = pd.DataFrame(cosine_similarity(np.array(k562_gwps_norm.loc[common_p, :]), np.array(k562_gwps_norm.loc[common_p, :])), index=common_p, columns=common_p)
    k562_essential_kernel = pd.DataFrame(cosine_similarity(np.array(k562_essential.loc[common_p, :]), np.array(k562_gwps_norm.loc[common_p, :])), index=common_p, columns=common_p)

    median_single_results = None
    median_pairwise_results = None

    for iteration in tqdm(range(100)):
        cognate_data = []
        random_data = []

        for i in common_p:
            cognate_data.append(compare(np.array(k562_gwps_norm.loc[i]), np.array(k562_essential.loc[i]), method=method))
            random_data.append(compare(np.array(k562_gwps_norm.loc[i]), np.array(k562_essential.iloc[random.randint(0, len(common_p)-1)]), method=method))

        single_dist = energy_distance(np.array(random_data).flatten(), np.array(cognate_data).flatten())
        
        cognate_data_kernel = []
        random_data_kernel = []

        for i in common_p:
            cognate_data_kernel.append(compare(np.array(k562_gwps_norm_kernel.loc[i]), np.array(k562_essential_kernel.loc[i]), method=method))
            random_data_kernel.append(compare(np.array(k562_gwps_norm_kernel.loc[i]), np.array(k562_essential_kernel.iloc[random.randint(0, len(common_p)-1)]), method=method))

        pairwise_dist = energy_distance(np.array(random_data_kernel).flatten(), np.array(cognate_data_kernel).flatten())
        single_results.append(single_dist)
        pairwise_results.append(pairwise_dist)

        if single_dist == np.median(single_results):
            median_single_results = [cognate_data, random_data]
        if pairwise_dist == np.median(pairwise_results):
            median_pairwise_results = [cognate_data_kernel, random_data_kernel]

    
    plt.figure(figsize=(2, 5))
    sns.boxplot(x=0, y=single_results)
    sns.boxplot(x=1, y=pairwise_results)
    plt.xticks(ticks=[0,1], labels=['Raw\nProfile', 'Similarity\nProfile'])
    plt.ylabel('Energy Distance')
    plt.title('100 Repeats')

    plt.show()
    plt.savefig(f'./figure_out/{method}_Fig2E_1.pdf')


    plt.figure(figsize=(5, 5))
    cognate_data, random_data = median_single_results[0], median_single_results[1]
    cognate_data_kernel, random_data_kernel = median_pairwise_results[0], median_pairwise_results[1]
    sns.kdeplot(np.array(random_data).flatten(), c='black', label='Random Raw Profile')
    sns.kdeplot(np.array(cognate_data).flatten(), c='tab:blue', label='Cognate Raw Profile')
    sns.kdeplot(np.array(random_data_kernel).flatten(), color='black', label='Random Kernel', fill=True)
    sns.kdeplot(np.array(cognate_data_kernel).flatten(), color='tab:blue', label='Cognate Kernel', fill=True)
    plt.legend(prop={'size': 8}, loc='upper left')#, bbox_to_anchor=(1.04, 1))
    plt.title('Perturbation Space')

    plt.show()
    plt.savefig(f'./figure_out/{method}_Fig2E_2.pdf')
    
    print('Method', method, 'Summary:')
    print(wilcoxon(single_results, pairwise_results))
    print(np.median(single_results), np.median(pairwise_results))
    #np.save(f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/{method}_single_pert.npy',single_results_transposed)
    #np.save(f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/{method}_pairwise_pert.npy',pairwise_results_transposed)