# Baseline Cluster Comorbidity Analysis

We applied **hierarchical clustering** to group baseline diseases based on the **effect sizes** of metabolites on each disease.  
The number of clusters was determined by evaluating cluster solutions ranging from **10 to 40**, and selecting the optimal number based on the **average silhouette score**.

- The optimal number of clusters was determined to be **10**.
- A detailed list of diseases included in each cluster can be found in the supplementary cluster category table.

## Rationale for Clustering

The goal of clustering was to identify **disease groups** that share **similar metabolic profiles**.  
Diseases within the same cluster are likely to:

- Exhibit similar metabolic perturbations
- Potentially share **common metabolic pathways** or **pathophysiological mechanisms**

## Statistical Analysis per Cluster

For each of the **10 disease clusters**, we conducted two types of analyses:

1. **Binary Cluster Comorbidity Analysis**  
   - Defined individuals as **cases** if they had **two or more diseases** within the same cluster  
   - Defined as **controls** if they had **fewer than two diseases** within the cluster  
   - Applied **logistic regression** to test associations with metabolites

2. **Comorbidity Burden Analysis**  
   - Among individuals with at least one disease in the cluster  
   - Applied **ordinal logistic regression** to assess associations between metabolite levels and the **number of comorbid diseases** within the cluster

## Covariate Adjustments

All models were adjusted for the following baseline covariates:

- **Demographic factors**:  
  - Age  
  - Sex  
  - Race  
  - Education level  
- **Socioeconomic status**:  
  - Townsend Deprivation Index  
- **Lifestyle and anthropometric factors**:  
  - Body Mass Index (BMI)  
  - Smoking status  
- **Metabolite-related technical variables**:  
  - Fasting time  
  - Season of blood draw  
  - Sample storage duration  
  - Instrument ID  

## Statistical Framework

- Regression models were implemented using the `statsmodels` package in **Python**.
- **Multiple testing correction** was applied using the **Bonferroni method**, with a correction factor of:  
  `10 clusters × 251 metabolic traits`
- Significance was defined as **p < 0.05** after correction.

## Code for Hierarchical Clustering
```python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib

matplotlib.use('Agg')

possible_paths = [
    "C:/Users/Administrator/Desktop/metabolic_all",
    "/mnt/c/Users/Administrator/Desktop/metabolic_all",
    "/Users/mrli/Desktop/metabolic_all"
]

root_path = None
for path in possible_paths:
    if os.path.exists(path):
        root_path = path
        break

if root_path is None:
    raise FileNotFoundError("no root_path。")
else:
    print(f"root_path: {root_path}")
    
data_folder = "data"
results_folder = 'Results/association/disease/ba'
disease_code = pd.read_csv(os.path.join(root_path, data_folder, 'disease/disease_code_total_revise.csv'))
disease_code = disease_code[~disease_code['Chapter'].isin([16, 17])]
disease_code_need = disease_code[disease_code['Case_ba'].notna()]
disease_lst = disease_code_need.NAME.tolist()

all_files = os.listdir(os.path.join(root_path, results_folder))
COX_files = [f'LG_{d}.csv' for d in disease_lst]
COX_file_lst = [f for f in all_files if f in COX_files]

matrix_df = pd.DataFrame()
need_col = ['Pro_code', 'or','nb_case','nb_individuals']

less_100_lst = []

nb_df = pd.DataFrame({'NAME':disease_lst, 'nb_case':0, 'nb_control':0})

for file_name in tqdm(COX_file_lst):
    disease_name = file_name[3:-4]
    file_path = os.path.join(root_path, results_folder, file_name)
    df = pd.read_csv(file_path)
    df = df[need_col]
    df.rename(columns={'Pro_code': 'Pro_code', 'or': f'{disease_name}'}, inplace=True)
    nb_case = df['nb_case'][1]
    nb_df.loc[nb_df['NAME'] == disease_name, 'nb_case'] = nb_case
    nb_individuals = df['nb_individuals'][1]
    nb_df.loc[nb_df['NAME'] == disease_name, 'nb_control'] = nb_individuals - nb_case
    df.drop(columns='nb_case', inplace=True)
    df.drop(columns='nb_individuals', inplace=True)
    if nb_case >= 100:

        if matrix_df.empty:
            matrix_df = df
        else:
            matrix_df = pd.merge(matrix_df, df, on='Pro_code', how='outer')
    else:
        print(f"Case not enough: {disease_name}")
        less_100_lst.append(disease_name)
        continue
matrix_df.shape

disease_code = pd.merge(disease_code, nb_df, left_on='NAME', right_on='NAME', how='left')
disease_code.sort_values(by='Chapter', ascending=False, inplace=True)
disease_code.drop(columns=['Case_ba', 'Control_ba', 'Case_in', 'Control_in'], inplace=True)
disease_code.rename(columns={'nb_case': 'Case_ba', 'nb_control': 'Control_ba'}, inplace=True)
disease_code.to_csv(os.path.join(root_path, data_folder, 'disease/disease_code_ba.csv'), index=False)


missing_values = matrix_df.isnull().sum()

matrix_values = matrix_df.drop(columns='Pro_code').values
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix_values)

matrix_scaled_df = pd.DataFrame(matrix_scaled,
                                index=matrix_df['Pro_code'],
                                columns=matrix_df.columns[1:])

matrix_T = matrix_scaled_df.T

distance_matrix = pdist(matrix_T, metric='euclidean')
linkage_matrix = linkage(distance_matrix, method='ward')


Figure_dir = os.path.join(root_path, 'Results/s4_Cluster/Figure')
os.makedirs(Figure_dir, exist_ok=True)

plt.figure(figsize=(110, 10))
dendrogram(linkage_matrix, labels=matrix_T.index, leaf_rotation=90)
plt.title('Hierarchical Clustering of Diseases based on Metabolite Associations')
plt.xlabel('Diseases')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(Figure_dir, 'Disease_cluster_ba.pdf'), dpi=300)
plt.savefig(os.path.join(Figure_dir, 'Disease_cluster_ba.png'), dpi=300)
plt.close()

sil_scores = []
cluster_range = range(10, 40)

for k in tqdm(cluster_range):
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(matrix_T.values, labels, metric='euclidean')
    sil_scores.append(score)

plt.plot(cluster_range, sil_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Cluster Numbers')
plt.grid(True)
plt.savefig(os.path.join(Figure_dir, 'Silhouette_Scores_ba.png'), dpi=300)
plt.savefig(os.path.join(Figure_dir, 'Silhouette_Scores_ba.pdf'), dpi=300)
plt.close()

best_k = cluster_range[np.argmax(sil_scores)]
print(f'nb_Cluster：{best_k}')

fig_df = pd.DataFrame({'cluster_range': cluster_range, 'sil_scores': sil_scores})
fig_df.to_csv(os.path.join(Figure_dir, 'Silhouette_Scores_ba.csv'), index=False)

cluster_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust')

cluster_df = pd.DataFrame({
    'Disease': matrix_T.index,
    'Cluster': cluster_labels
})

cluster_counts = cluster_df['Cluster'].value_counts().reset_index()

disease_code_Cluster = pd.merge(disease_code, cluster_df, left_on='NAME', right_on='Disease', how='left')
disease_code_Cluster.rename(columns={'Cluster': 'Cluster_ba'}, inplace=True)
disease_code_Cluster.to_csv(os.path.join(Figure_dir, '../disease_code_cluster_ba.csv'), index=False)
```

## Code for Comorbidity Analysis
```python
import matplotlib
import gc
import time
import numpy as np
import pandas as pd
import pyreadr
import os
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from mne.stats import bonferroni_correction
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
import traceback
matplotlib.use('Agg')

possible_paths = [
    "C:/Users/Administrator/Desktop/metabolic_all",
    "/mnt/c/Users/Administrator/Desktop/metabolic_all",
    "/Users/mrli/Desktop/metabolic_all"
]

root_path = None
for path in possible_paths:
    if os.path.exists(path):
        root_path = path
        break

if root_path is None:
    raise FileNotFoundError("no root_path。")
else:
    print(f"root_path: {root_path}")
    
data_folder = "data"
cluster_file = 'Results/s4_Cluster'

dd1 = pyreadr.read_r(os.path.join(root_path, data_folder, 'metabolism/metabolism0.RData'))
dd2 = pyreadr.read_r(os.path.join(root_path, data_folder, 'cov/cov_metabolic.RData'))

metabolism = dd1['metabolism']
pro_lst = metabolism.columns[1:].tolist()

cov = dd2['cov_clean']
cols_label = ['sex', 'Ethnic_group', 'edu3', 'smoke', 'season', 'Spectrometer']
cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)
cov_lst = ['age', 'sex', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cov_needed = cov_lst

disease_df = pd.read_csv(os.path.join(root_path, cluster_file, 'data/disease_data_cluster_ba.csv'))
cluster_cols = [col for col in disease_df.columns if col.startswith('Cluster_')]

disease_lg = disease_df.copy()
disease_lg[cluster_cols] = disease_lg[cluster_cols].apply(lambda x: np.where(x > 1, 1, 0))

disease_ord = disease_df.copy()
disease_ord[cluster_cols] = disease_ord[cluster_cols].replace(0, np.nan)

def results_summary(tgt_out_df):
    oratio_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        oratio = f'{tgt_out_df.oratio.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.or_lbd.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.or_ubd.iloc[i]:.2f}'
        oratio_out_lst.append(oratio + ' [' + lbd + '-' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (oratio_out_lst, p_out_lst)

def process_lg(pro, target, cov_needed):
    try:
        tmp_df = df[['id'] + cov_needed + [target, pro]].copy()
        tmp_df.rename(columns={pro: 'x_pro', target: 'target_y'}, inplace=True)
        tmp_df.dropna(subset=['target_y'], inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        nb_all = len(tmp_df)
        nb_case = tmp_df['target_y'].sum()
        prop_case = np.round(nb_case / nb_all * 100, 3) if nb_all > 0 else 0

        Y = tmp_df['target_y']
        X = tmp_df[cov_needed + ['x_pro']]
        log_mod = sm.Logit(Y, sm.add_constant(X)).fit(disp=False)

        oratio = np.round(np.exp(log_mod.params['x_pro']), 5)
        pval = log_mod.pvalues['x_pro']
        ci_mod = log_mod.conf_int(alpha=0.05)
        lbd = np.round(np.exp(ci_mod.loc['x_pro'][0]), 5)
        ubd = np.round(np.exp(ci_mod.loc['x_pro'][1]), 5)

        return [pro, nb_all, nb_case, prop_case, oratio, lbd, ubd, pval]

    except Exception as e:
        print(f"Error processing {pro} (target: {target}): {e}")
        return [pro, nb_all if 'nb_all' in locals() else 0,
                nb_case if 'nb_case' in locals() else 0,
                prop_case if 'prop_case' in locals() else 0,
                np.nan, np.nan, np.nan, np.nan]

def results_summary_ord(tgt_out_df):
    coeff_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        coeff = f'{tgt_out_df.coeff.iloc[i]:.3f}'
        lbd = f'{tgt_out_df.lbd.iloc[i]:.3f}'
        ubd = f'{tgt_out_df.ubd.iloc[i]:.3f}'
        coeff_out_lst.append(coeff + ' [' + lbd + ',' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (coeff_out_lst, p_out_lst)

def process_ord(pro, target, cov_needed):

    try:
        tmp_df = df[['id'] + cov_needed + [target, pro]].copy()
        tmp_df.rename(columns={pro: 'x_pro', target: 'target_y'}, inplace=True)
        tmp_df.dropna(subset=['target_y', 'x_pro'], inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        nb_all = len(tmp_df)
        mean_target = np.round(tmp_df['target_y'].mean(), 3)
        std_target = np.round(tmp_df['target_y'].std(), 3)
        min = np.round(tmp_df['target_y'].min(), 3)
        max = np.round(tmp_df['target_y'].max(), 3)
        nb_catagory = len(tmp_df['target_y'].unique())

        Y = tmp_df['target_y']
        X = tmp_df[cov_needed + ['x_pro']]

        X = X.apply(pd.to_numeric, errors='coerce')
        Y = pd.to_numeric(Y, errors='coerce')

        model = OrderedModel(Y, X, distr='logit')
        res = model.fit(method='bfgs', disp=False)

        coeff = np.round(res.params['x_pro'], 5)
        pval = res.pvalues['x_pro']
        ci_mod = res.conf_int(alpha=0.05)
        lbd = np.round(ci_mod.loc['x_pro'][0], 5)
        ubd = np.round(ci_mod.loc['x_pro'][1], 5)

        return [pro, nb_all, mean_target, std_target, coeff, lbd, ubd, pval, min, max, nb_catagory]

    except Exception as e:
        print(f"Error processing {pro} (target: {target}): {e}")
        return [pro, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

nb_cpus = 10
save_dir = os.path.join(root_path, 'Results/s4_Cluster/Association/ba')
os.makedirs(save_dir, exist_ok=True)

df = pd.merge(metabolism, cov, on='id', how='left')
df = pd.merge(df, disease_lg, on='id', how='left')

bad_target = []

for target_name in tqdm(cluster_cols):

    save_path = os.path.join(save_dir, f'LG_{target_name}.csv')
    if os.path.exists(save_path):
        print(f"✔️Already processed {target_name}, skipping.")
        continue

    print(f"Processing target: {target_name}")
    time_start = time.time()
    try:
        dd = df[['id'] + [target_name]].copy()
        dd.rename(columns={target_name: 'target_y'}, inplace=True)
        dd.dropna(subset=['target_y'], inplace=True)
        dd.reset_index(drop=True, inplace=True)
        sample = len(dd)
        nb_case = dd['target_y'].value_counts().get(1, 0)
        nb_control = dd['target_y'].value_counts().get(0, 0)

        if sample < 1000:
            print(f"❌Sample size for {target_name} is less than 1000, skipping.")
            bad_target.append(target_name)
            with open(os.path.join(save_dir, f'nb_{target_name}.txt'), 'a') as f:
                f.write(f"{target_name}: {sample}\n")
            continue
        elif nb_case < 100 or nb_control < 100:
            print(f"❌Sample size for {target_name} is less than 100 for either case or control, skipping.")
            bad_target.append(target_name)
            with open(os.path.join(save_dir, f'nb_{target_name}.txt'), 'a') as f:
                f.write(f"{target_name}: Sample size {sample}, Cases {nb_case}, Controls {nb_control}\n")
            continue

        else:
            tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process_lg)(pro, target_name, cov_needed) for pro in tqdm(pro_lst, desc=target_name, leave=False))
            tgt_out_df = pd.DataFrame(tgt_out_df, columns=['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd', 'or_ubd', 'pval'])
            _, p_f_bfi = bonferroni_correction(tgt_out_df.pval.fillna(1), alpha=0.05)
            _, p_fdr, _, _ = multipletests(tgt_out_df.pval.fillna(1), method='fdr_bh')
            tgt_out_df['pval_bfi'] = p_f_bfi
            tgt_out_df['pval_fdr'] = p_fdr
            tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
            tgt_out_df['or_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)
            tgt_out_df = tgt_out_df[['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd',
                                     'or_ubd', 'pval', 'pval_bfi','pval_fdr', 'or_output', 'pval_significant']]
            tgt_out_df.rename(columns={'oratio': 'or'}, inplace=True)
            tgt_out_df.to_csv(save_path, index=False)
    except Exception as e:
        print(f"Error processing {target_name}: {e}")
        traceback.print_exc()
        bad_target += [target_name]
        continue

    gc.collect()
    time_end = time.time()
    print(f"✔️ Finished {target_name} in {time_end - time_start:.2f} seconds.")

bad_target_df = pd.DataFrame(bad_target, columns=['bad_target'])
bad_target_df.to_csv(os.path.join(save_dir, 'AAA_bad_target_lg.csv'), index=False)

df = pd.merge(metabolism, cov, on='id', how='left')
df = pd.merge(df, disease_ord, on='id', how='left')
bad_tgt_lst = []

for target_name in tqdm(cluster_cols):

    save_path = os.path.join(save_dir, f'ORD_{target_name}.csv')
    if os.path.exists(save_path):
        print(f"✔️Already processed {target_name}, skipping.")
        continue

    print(f"Processing target: {target_name}")
    time_start = time.time()
    try:
        dd = df[['id'] + cov_needed + [target_name]].copy()
        dd.rename(columns={target_name: 'target_y'}, inplace=True)
        dd.dropna(subset=['target_y'], inplace=True)
        dd.reset_index(drop=True, inplace=True)
        sample = len(dd)
        if sample < 1000:
            print(f"❌Sample size for {target_name} is less than 1000, skipping.")
            bad_tgt_lst.append(target_name)
            with open(os.path.join(save_dir, f'nb_O_{target_name}.txt'), 'a') as f:
                f.write(f"{target_name}: {sample}\n")
            continue

        else:
            tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process_ord)(pro, target_name, cov_needed) for pro in tqdm(pro_lst, desc=target_name, leave=False))

            tgt_out_df = pd.DataFrame(tgt_out_df,columns=['Pro_code', 'nb_individuals', 'mean_target', 'std_target', 'coeff',
                                                          'lbd', 'ubd', 'pval', 'min', 'max', 'nb_catagory'])

            tgt_out_df['pval'] = tgt_out_df['pval'].fillna(1)
            _, p_f_bfi = bonferroni_correction(tgt_out_df.pval, alpha=0.05)
            _, p_fdr, _, _ = multipletests(tgt_out_df.pval, method='fdr_bh')
            tgt_out_df['pval_bfi'] = p_f_bfi
            tgt_out_df['pval_fdr'] = p_fdr
            tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1

            tgt_out_df['coeff_output'], tgt_out_df['pval_significant'] = results_summary_ord(tgt_out_df)

            tgt_out_df = tgt_out_df[['Pro_code', 'nb_individuals', 'mean_target', 'std_target', 'min', 'max', 'nb_catagory',
                 'coeff', 'lbd', 'ubd', 'pval', 'pval_bfi', 'pval_fdr', 'coeff_output', 'pval_significant']]
            tgt_out_df.rename(columns={'coeff': 'Coefficient'}, inplace=True)
            tgt_out_df.to_csv(save_path, index=False)


    except Exception as e:
        print(f"❌Error processing {target_name}: {e}")
        traceback.print_exc()
        bad_tgt_lst.append(target_name)
        with open(os.path.join(save_dir, f'error_{target_name}.txt'), 'a') as f:
            f.write(f"{target_name}: {e}\n")
        continue
    gc.collect() 
    time_end = time.time()
    print(f"✔️ Finished {target_name} in {time_end - time_start:.2f} seconds.")

bad_tgt_df = pd.DataFrame({'Disease_code': bad_tgt_lst})
bad_tgt_df.to_csv(os.path.join(save_dir, 'AAA_bad_target_ord.csv'), index=False)
```