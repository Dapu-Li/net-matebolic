# Incidence Disease Analysis

**Incident diseases** refer to conditions that were newly diagnosed during the follow-up period, excluding individuals with a diagnosis at baseline.  
To investigate the associations between circulating metabolites (including their ratios) and the **onset of 1210 diseases**, we conducted **Cox regression analyses**. These diseases were categorized into **15 Chapters** based on the **ICD-10** classification system, using the **first 3 digits** of the ICD codes. Given that the UK Biobank cohort comprises individuals aged 40 and above, the 15th chapter pertaining to pregnancy-related diseases includes only two conditions.

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

For **sex-specific diseases**, covariate adjustment was tailored by:

- Removing the **sex** variable from the model  
- Restricting the analysis to the appropriate **sex-specific subgroup**

## Statistical Analysis

- Cox regression models were fitted using the `lifelines` package in **Python**.
- Only participants free of the target disease at baseline were included in each analysis.
- Follow-up time was accounted for by **time since baseline**.
- Multiple testing correction was applied using the **Bonferroni method**, with a correction factor of:  
  `1210 diseases × 251 metabolic traits`.
- The significance threshold was set at **p < 0.05** after correction.

## Code for Incidence Disease Analysis
```python
import time
import numpy as np
import pandas as pd
import pyreadr
import os
from joblib import Parallel, delayed
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
disease_code_need = disease_code[disease_code['Case_in'].notna()]
disease_code_need = disease_code_need[~disease_code_need['Chapter'].isin([16, 17])]

disease_lst = disease_code_need.NAME.tolist()

metabolism = dd1['metabolism']
pro_code = dd1['pro_code']
cov = dd2['cov_clean']
disease_df = dd3['d_in_total']
out_df_ba = dd4['out_df_in']
del dd1, dd2, dd3, dd4

pro_lst = metabolism.columns[1:].tolist()

cov_lst = ['age', 'sex', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cov_lst_no_sex = ['age', 'Ethnic_group', 'edu3','Towns','BMI','smoke', 'season', 'fasting_time', 'sample_to_measurement', 'Spectrometer']
cols_label = ['sex', 'Ethnic_group', 'edu3', 'smoke', 'season', 'Spectrometer']
cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)

df = pd.merge(metabolism, cov, on='id', how='left')
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
save_dir = os.path.join(root_path, 'Results/association/disease/R_in')
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
        print(f"Error processing {target_name}: {e}")
        traceback.print_exc()
        bad_target.append(target_name)
        continue

    time_end = time.time()
    print(f"✔️ Finished {target_name} in {time_end - time_start:.2f} seconds.")

bad_target_df = pd.DataFrame(bad_target, columns=['bad_target'])
bad_target_df.to_csv(os.path.join(save_dir, 'AAA_bad_target.csv'), index=False)
```