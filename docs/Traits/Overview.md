# Health-Related Traits Analysis

A total of **884 health-related traits** were included in this study. These traits were grouped into **11 chapters** based on the UK Biobank (UKB) data category pathways, in descending order of the number of traits per chapter:

- Diet and food preferences  
- Health and medical history  
- Mental health  
- Brain structure  
- Physical measures  
- Blood and urine assays  
- Lifestyle factors  
- Working and living environment  
- Eye measures  
- Physical activities  
- Cognitive function  

## Variable Types and Modeling Strategy

All traits were classified into three major types based on their data structure:

- **Continuous variables** → analyzed using **linear regression**  
- **Binary variables** → analyzed using **logistic regression**  
- **Ordered categorical variables** → analyzed using **ordinal logistic regression**  

For **unordered categorical variables**, we applied **one-hot encoding**, converting them into multiple binary variables for analysis.

## Covariate Adjustments

All models were adjusted for the following covariates at baseline:

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

For **sex-specific traits**, covariate adjustment was modified by:

- Removing the **sex** variable from the model  
- Restricting the analysis to the appropriate **sex-specific subgroup**

> **Note:** If **BMI** was the outcome trait (i.e., the dependent variable), it was **not included as a covariate** in that specific analysis.

## Statistical Analysis

- Logistic regression,  Linear regression and Ordinal logistic regression were fitted using the `statsmodels` package in **Python**.
- Multiple testing correction was applied using the **Bonferroni method**, with a correction factor of:
  `909 health-related traits × 251 metabolic traits`.
- The significance threshold was set at **p < 0.05** after correction.

## Partial Code for Health-Related Traits Analysis
```python
import time
import numpy as np
import pandas as pd
import pyreadr
import os
from joblib import Parallel, delayed
import statsmodels.api as sm
from mne.stats import bonferroni_correction
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
import traceback

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
dd1 = pyreadr.read_r(os.path.join(root_path, data_folder, 'metabolism/metabolism0.RData'))
dd2 = pyreadr.read_r(os.path.join(root_path, data_folder, 'cov/cov_metabolic.RData'))
dd3 = pyreadr.read_r(os.path.join(root_path, data_folder, 'health_traits/traits_770.RData'))
metabolism = dd1['metabolism']
pro_code = dd1['pro_code']
cov = dd2['cov_clean']
traits = dd3['traits_770']
traits_code = dd3['traits_code']
del dd1, dd2, dd3

pro_lst = metabolism.columns[1:].tolist()

Cont_traits = traits_code[traits_code['Variable.Type'] == 'Continuous']
Cont_traits = Cont_traits['Code'].tolist()

cov_lst = ['age', 'sex', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cov_lst_no_sex = ['age', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cols_label = ['sex', 'Ethnic_group', 'edu3', 'smoke', 'season', 'Spectrometer']
cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)

df = pd.merge(metabolism, cov, on='id', how='left')
df = pd.merge(df, traits, on='id', how='left')
del metabolism, cov, traits

df = df.apply(pd.to_numeric, errors='coerce')

def results_summary_cont(tgt_out_df):
    beta_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        beta = f'{tgt_out_df.beta.iloc[i]:.3f}'
        lbd = f'{tgt_out_df.beta_lbd.iloc[i]:.3f}'
        ubd = f'{tgt_out_df.beta_ubd.iloc[i]:.3f}'
        beta_out_lst.append(beta + ' [' + lbd + ',' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (beta_out_lst, p_out_lst)

def process_cont(pro, target, cov_needed):
    try:
        tmp_df = df[['id'] + cov_needed + [target, pro]].copy()
        tmp_df.rename(columns={pro: 'x_pro', target: 'target_y'}, inplace=True)
        tmp_df.dropna(subset=['target_y'], inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        nb_all = len(tmp_df)
        mean_target = np.round(tmp_df['target_y'].mean(), 3)
        std_target = np.round(tmp_df['target_y'].std(), 3)

        Y = tmp_df['target_y']
        X = tmp_df[cov_needed + ['x_pro']]
        lin_mod = sm.OLS(Y, sm.add_constant(X)).fit()

        beta = np.round(lin_mod.params['x_pro'], 5)
        pval = lin_mod.pvalues['x_pro']
        ci_mod = lin_mod.conf_int(alpha=0.05)
        lbd = np.round(ci_mod.loc['x_pro'][0], 5)
        ubd = np.round(ci_mod.loc['x_pro'][1], 5)

        return [pro, nb_all, mean_target, std_target, beta, lbd, ubd, pval]

    except Exception as e:
        print(f"Error processing {pro} (target: {target}): {e}")
        return [pro, nb_all if 'nb_all' in locals() else 0,
                mean_target if 'mean_target' in locals() else np.nan,
                std_target if 'std_target' in locals() else np.nan,
                np.nan, np.nan, np.nan, np.nan]

nb_cpus = 8
save_dir = os.path.join(root_path, 'Results/association/traits')
os.makedirs(save_dir, exist_ok=True)

for target_name in tqdm(Cont_traits):
    print(f"Processing target: {target_name}")
    time_start = time.time()
    try:
        match_row = traits_code.loc[traits_code.Code == target_name]
        if len(match_row) == 0:
            raise ValueError(f"No match for {target_name} in traits_code")

        sex_specific = match_row.Sex_specific.values[0]
        cov_needed = cov_lst if sex_specific == '' else cov_lst_no_sex

        tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process_cont)(pro, target_name, cov_needed) for pro in tqdm(pro_lst, desc=target_name, leave=False))
        tgt_out_df = pd.DataFrame(tgt_out_df, columns=['Pro_code', 'nb_individuals', 'mean_target', 'std_target', 'beta', 'beta_lbd', 'beta_ubd', 'pval'])

        tgt_out_df['pval'] = tgt_out_df['pval'].fillna(1)
        _, p_f_bfi = bonferroni_correction(tgt_out_df.pval, alpha=0.05)
        _, p_fdr, _, _ = multipletests(tgt_out_df.pval, method='fdr_bh')
        tgt_out_df['pval_bfi'] = p_f_bfi
        tgt_out_df['pval_fdr'] = p_fdr
        tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
        tgt_out_df['beta_output'], tgt_out_df['pval_significant'] = results_summary_cont(tgt_out_df)

        tgt_out_df = tgt_out_df[['Pro_code', 'nb_individuals', 'mean_target', 'std_target', 'beta', 'beta_lbd',
                                 'beta_ubd', 'pval', 'pval_bfi','pval_fdr', 'beta_output', 'pval_significant']]
        tgt_out_df.to_csv(os.path.join(save_dir, f'LN_{target_name}.csv'), index=False)

    except Exception as e:
        print(f"Error processing {target_name}: {e}")
        traceback.print_exc()
        continue
    time_end = time.time()
    print(f"Time taken for {target_name}: {time_end - time_start:.2f} seconds")
```