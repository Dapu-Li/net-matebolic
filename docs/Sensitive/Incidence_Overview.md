# Sensitivity Analysis: Incidence Disease

## Objective

To ensure the robustness of the associations between circulating metabolites and **incident diseases**, we conducted a sensitivity analysis by incorporating additional covariates into our regression models.

---

## Extended Covariate Adjustments

In addition to the baseline adjustment variables, we included higher-order and interaction terms, as well as genetic ancestry components to control for potential confounding. The full set of covariates included:

- **Demographic and socioeconomic factors**:  
  - Age  
  - Sex  
  - Race  
  - Education level  
  - Townsend Deprivation Index  
- **Lifestyle and anthropometric variables**:  
  - Body Mass Index (BMI)  
  - Smoking status  
- **Technical variables** related to sample collection:  
  - Fasting time  
  - Season of blood draw  
  - Sample storage duration  
  - Instrument ID  

**Additional sensitivity-specific covariates**:

- Age squared (**Age²**)  
- Age-by-sex interaction (**Age × Sex**)  
- Quadratic age-by-sex interaction (**Age² × Sex**)  
- Top 10 **genetic principal components (PCs)** for population stratification

> For sex-specific diseases, the **sex variable was excluded**, and analyses were restricted to the corresponding sex subgroup.

---

## Statistical Approach

- Cox regression models were used to assess binary disease incidence (case vs. control).  
- Each of the **incident diseases** (diagnosed during follow-up) was tested against **251 metabolites** (including ratios).
- The analysis was conducted using the `statsmodels` package in **Python**.
- A strict multiple testing correction was applied using the **Bonferroni method**, with a correction factor of:  
  `1212 diseases × 251 metabolic traits`.

---

## Interpretation

Significant associations observed in both the main and sensitivity models reinforce the **stability and validity** of our findings.  
Any differences between the models may suggest the influence of **age-related non-linearities**, **sex interactions**, or **underlying genetic structure**.

## Code for Sensitivity Analysis

```python
import time
import numpy as np
import pandas as pd
import pyreadr
import os
from joblib import Parallel, delayed
import statsmodels.api as sm
from lifelines import CoxPHFitter
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
dd3 = pyreadr.read_r(os.path.join(root_path, data_folder, 'disease/d_in_total.RData'))
dd4 = pyreadr.read_r(os.path.join(root_path, data_folder, 'disease/out_df_in.RData'))
disease_code = pd.read_csv(os.path.join(root_path, data_folder, 'disease/disease_code_total_revise.csv'))

metabolism = dd1['metabolism']
pro_code = dd1['pro_code']
cov = dd2['cov_clean']
disease_df = dd3['d_in_total']
out_df_ba = dd4['out_df_in']
del dd1, dd2, dd3, dd4

pro_lst = metabolism.columns[1:].tolist()
disease_code_need = disease_code[disease_code['Case_in'].notna()]
disease_lst = disease_code_need.NAME.tolist()

cov_plus = pd.read_csv(os.path.join(root_path, data_folder, 'gwas/gwas_cov.csv'),usecols=['id','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
cov_plus.rename(columns={cov_plus.columns[0]: 'id'}, inplace=True)
cov = pd.merge(cov, cov_plus, on='id', how='left')

cov['sex'] = cov['sex'].replace({'Female': 1, 'Male': 2}).astype(int)
cov['age2'] = cov['age'] ** 2
cov['age_sex'] = cov['age'] * cov['sex']
cov['age2_sex'] = cov['age2'] * cov['sex']

for col in ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']:
    cov[col] = cov[col].fillna(cov[col].median())

cov_lst = ['age', 'sex', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cov_lst_plus = ['age2','age_sex','age2_sex','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
cov_lst = cov_lst + cov_lst_plus

cov_lst_no_sex = ['age', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer',
                  'age2','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
cols_label = ['Ethnic_group', 'edu3', 'smoke', 'season', 'Spectrometer']

cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)


df = pd.merge(metabolism, cov, on='id', how='inner')
df = df.apply(pd.to_numeric, errors='coerce')
df = pd.merge(df, disease_df, on='id', how='left')
del metabolism, cov, disease_df

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
        duration_col = target + '_durations'
        tmp_df = df[['id'] + cov_needed + [target, pro, duration_col]].copy()
        tmp_df.rename(columns={pro: 'x_pro', target: 'event', duration_col: 'duration'}, inplace=True)
        tmp_df.dropna(subset=['event', 'duration'], inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        nb_all = len(tmp_df)
        nb_case = tmp_df['event'].sum()
        prop_case = np.round(nb_case / nb_all * 100, 3) if nb_all > 0 else 0
        X = tmp_df[cov_needed + ['x_pro']]
        df_model = pd.concat([tmp_df[['duration', 'event']], X], axis=1)
        df_model = df_model.apply(pd.to_numeric, errors='coerce')

        tmpout = []
        i = 1e-7
        max_penalizer = 1e3

        while len(tmpout) == 0 and i <= max_penalizer:
            try:
                cph = CoxPHFitter(penalizer=i)
                cph.fit(df_model, duration_col='duration', event_col='event', show_progress=False)

                hr = np.round(cph.hazard_ratios_['x_pro'], 5)
                ci = np.exp(cph.confidence_intervals_)
                lbd = np.round(ci.loc['x_pro'].iloc[0], 5)
                ubd = np.round(ci.loc['x_pro'].iloc[1], 5)
                pval = cph.summary.loc['x_pro', 'p']

                tmpout = [pro, nb_all, nb_case, prop_case, hr, lbd, ubd, pval]
            except:
                i *= 10
                continue

        if len(tmpout) == 0:
            tmpout = [pro, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
        return tmpout
    except Exception as e:
        print(f"Error in processing {pro}: {e}")
        return [pro, 0, 0, 0, np.nan, np.nan, np.nan, np.nan]

from joblib import parallel_backend

nb_cpus = 6
batch_size = 48
save_dir = os.path.join(root_path, 'Results/s6_Sensitive/cov_plus/in')
os.makedirs(save_dir, exist_ok=True)

disease_lst1 = disease_lst

bad_target = []

for target_name in tqdm(disease_lst1):
    save_path = os.path.join(save_dir, f'COX_{target_name}.csv')
    if os.path.exists(save_path):
        print(f"✔️Already processed {target_name}, skipping.")
        continue

    print(f"\nProcessing target: {target_name}")
    time_start = time.time()
    try:
        match_row = disease_code.loc[disease_code.NAME == target_name]
        if len(match_row) == 0:
            with open(os.path.join(save_dir, f'no_match_{target_name}.txt'), 'a') as f:
                f.write(f"{target_name}: No match in disease_code\n")
            raise ValueError(f"No match for {target_name} in traits_code")

        sex_specific = match_row.SEX.values[0]
        cov_needed = cov_lst_no_sex if (sex_specific == 1 or sex_specific == 2) else cov_lst

        results = []
        with parallel_backend("threading"):
            for i in tqdm(range(0, len(pro_lst), batch_size), desc=f"{target_name} batches", leave=False):
                batch_pros = pro_lst[i:i+batch_size]
                batch_results = Parallel(n_jobs=nb_cpus)(
                    delayed(process)(pro, target_name, cov_needed) for pro in batch_pros
                )
                results.extend(batch_results)

        tgt_out_df = pd.DataFrame(results, columns=['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd', 'or_ubd', 'pval'])
        _, p_f_bfi = bonferroni_correction(tgt_out_df.pval.fillna(1), alpha=0.05)
        _, p_fdr, _, _ = multipletests(tgt_out_df.pval.fillna(1), method='fdr_bh')

        tgt_out_df['pval_bfi'] = p_f_bfi
        tgt_out_df['pval_fdr'] = p_fdr
        tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
        tgt_out_df['or_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)

        tgt_out_df = tgt_out_df[['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd',
                                 'or_ubd', 'pval', 'pval_bfi', 'pval_fdr', 'or_output', 'pval_significant']]
        tgt_out_df.rename(columns={'oratio': 'HR'}, inplace=True)
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
    time.sleep(5)

bad_target_df = pd.DataFrame(bad_target, columns=['bad_target'])
bad_target_df.to_csv(os.path.join(save_dir, 'AAA_bad_target.csv'), index=False)
```
