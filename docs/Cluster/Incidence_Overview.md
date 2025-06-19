# Incidence Cluster Comorbidity Analysis

We employed a **data-driven clustering approach** based on metabolite-disease associations, to gain deeper insights into the **metabolomic patterns underlying incident disease comorbidity**.

Rather than analyzing diseases individually or strictly within ICD chapters, we grouped diseases into **clusters** that exhibited similar **metabolite effect profiles**. This approach allowed us to identify latent comorbidity structures that may reflect **shared biological mechanisms**.

## Clustering Strategy

- We constructed a matrix of metabolite-disease associations across all incident diseases.
- **Hierarchical clustering** was applied to this matrix to uncover groups of diseases with similar metabolic signatures.
- The optimal number of clusters was determined through **silhouette analysis**, evaluating candidate solutions between 10 and 40 clusters.
- The final model identified **27 clusters** as the most stable and biologically interpretable solution.

Each cluster is hypothesized to represent a **metabolically coherent disease group**, potentially reflecting common etiological pathways or systemic dysregulation.

## Analytical Design

For each of the 27 clusters, two complementary analyses were conducted:

### 1. Comorbidity Status Analysis  
- Individuals were classified as **comorbid** if they developed **two or more incident diseases** within a given cluster.  
- Those with **zero or one disease** in the cluster served as the **reference group**.  
- We applied **logistic regression** to evaluate the association between baseline metabolites and the risk of cluster-level comorbidity.

### 2. Comorbidity Burden Analysis  
- This analysis was restricted to individuals with at least one disease within the cluster.  
- We used **ordinal logistic regression** to assess whether baseline metabolite levels were associated with the **severity of comorbidity**, defined by the number of incident diseases in the cluster.

## Covariate Adjustment

All regression models were adjusted for the following baseline covariates:

- **Demographic variables**:  
  - Age  
  - Sex  
  - Race  
  - Education level  
- **Socioeconomic factor**:  
  - Townsend Deprivation Index  
- **Lifestyle and body composition**:  
  - Body Mass Index (BMI)  
  - Smoking status  
- **Sample-related technical factors**:  
  - Fasting duration  
  - Season of sample collection  
  - Time in storage  
  - Instrument ID

## Statistical Considerations

- All models were fitted using the `statsmodels` package in **Python**.
- To correct for multiple comparisons, a **Bonferroni correction** was applied, using the factor:  
  `27 clusters × 251 metabolic traits`
- Statistical significance was defined as **p < 0.05** after correction.

## Code for Analysis
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

disease_df = pd.read_csv(os.path.join(root_path, cluster_file, 'data/disease_data_cluster_in.csv'))
cluster_cols = [col for col in disease_df.columns if col.startswith('Cluster_')]

disease_lg = disease_df.copy()
disease_lg[cluster_cols] = disease_lg[cluster_cols].apply(lambda x: np.where(x > 1, 1, 0)) # 共病

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
save_dir = os.path.join(root_path, 'Results/s4_Cluster/Association/in')
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