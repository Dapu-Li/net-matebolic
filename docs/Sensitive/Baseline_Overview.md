# Sensitivity Analysis: Baseline Disease

## Objective

To test the robustness of our main findings, we conducted a **sensitivity analysis** for the associations between baseline diseases and circulating metabolites.

---

## Additional Covariate Adjustments

In this sensitivity model, we included additional covariates beyond the primary model to control for potential confounding and interaction effects. Specifically, the following variables were added:

- **Age²**
- **Age × Sex** interaction  
- **Age² × Sex** interaction  
- **Top 10 genetic principal components (PCs)** to account for population stratification

These were included **in addition to** the original covariates:

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

> **Note**: For sex-specific diseases, the sex variable was removed and the analysis was restricted to the relevant subgroup.

---

## Statistical Methods

- Logistic regression models were re-fitted using the `statsmodels` package in **Python**.
- The same set of **827 baseline diseases** and **251 metabolites** (including ratios) were tested.
- Multiple testing correction was performed using the **Bonferroni method** with the same correction factor:  
  `827 diseases × 251 metabolites`

---

## Interpretation

Associations that remained statistically significant after this extended adjustment indicate **greater robustness** and **reduced susceptibility to residual confounding**.  
Discrepancies with the primary results may highlight potential confounding due to non-linear age effects, sex interactions, or population structure.

## Code for Sensitivity Analysis

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
dd3 = pyreadr.read_r(os.path.join(root_path, data_folder, 'disease/d_ba_total.RData'))
dd4 = pyreadr.read_r(os.path.join(root_path, data_folder, 'disease/out_df_ba.RData'))
disease_code = pd.read_csv(os.path.join(root_path, data_folder, 'disease/disease_code_total_revise.csv'))

metabolism = dd1['metabolism']
pro_code = dd1['pro_code']
cov = dd2['cov_clean']
disease_df = dd3['d_ba_total']
out_df_ba = dd4['out_df_ba']
del dd1, dd2, dd3, dd4

pro_lst = metabolism.columns[1:].tolist()
disease_code_need = disease_code[disease_code['Case_ba'].notna()]
disease_lst = disease_code_need.NAME.tolist()

cov_plus = pd.read_csv(os.path.join(root_path, data_folder, 'gwas/gwas_cov.csv'),usecols=['id','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
cov_plus.rename(columns={cov_plus.columns[0]: 'id'}, inplace=True)
cov = pd.merge(cov, cov_plus, on='id', how='left')

cov['sex'] = cov['sex'].replace({'Female': 1, 'Male': 2}).astype(int)
cov['age2'] = cov['age'] ** 2
cov['age_sex'] = cov['age'] * cov['sex']
cov['age2_sex'] = cov['age2'] * cov['sex']

cov = cov.dropna()

cov_lst = ['age', 'sex', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cov_lst_plus = ['age2','age_sex','age2_sex','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
cov_lst = cov_lst + cov_lst_plus

cov_lst_no_sex = ['age', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer',
                  'age2','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
cols_label = ['Ethnic_group', 'edu3', 'smoke', 'season', 'Spectrometer']

cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)

df = pd.merge(metabolism, cov, on='id', how='inner')
df = pd.merge(df, disease_df, on='id', how='left')

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


def process(pro, target, cov_needed):
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

nb_cpus = 20
save_dir = os.path.join(root_path, 'Results/s6_Sensitive/cov_plus/ba')
os.makedirs(save_dir, exist_ok=True)

bad_target = []
disease_lst1 = disease_lst
for target_name in tqdm(disease_lst1):
    print(f"Processing target: {target_name}")
    time_start = time.time()

    save_path = os.path.join(save_dir, f'LG_{target_name}.csv')
    if os.path.exists(save_path):
        print(f"✔️Already processed {target_name}, skipping.")
        continue

    try:
        match_row = disease_code.loc[disease_code.NAME == target_name]
        if len(match_row) == 0:
            with open(os.path.join(save_dir, f'no_match_{target_name}.txt'), 'a') as f:
                f.write(f"{target_name}: No match in disease_code\n")
            raise ValueError(f"No match for {target_name} in traits_code")

        sex_specific = match_row.SEX.values[0]
        cov_needed = cov_lst_no_sex if (sex_specific == 1 or sex_specific == 2) else cov_lst

        tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(pro, target_name, cov_needed) for pro in tqdm(pro_lst, desc=target_name, leave=False))
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
        print(f"❌Error processing {target_name}: {e}")
        traceback.print_exc()
        bad_target.append(target_name)
        with open(os.path.join(save_dir, f'error_{target_name}.txt'), 'a') as f:
            f.write(f"{target_name}: {e}\n")
        continue

    time_end = time.time()
    print(f"✔️ Finished {target_name} in {time_end - time_start:.2f} seconds.")

bad_target_df = pd.DataFrame(bad_target, columns=['bad_target'])
bad_target_df.to_csv(os.path.join(save_dir, 'bad_target.csv'), index=False)
```


