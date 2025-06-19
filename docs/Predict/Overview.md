# Diagnosis and Prediction Analysis

To comprehensively assess the predictive power of metabolites for disease onset, we developed four types of models based on **time of diagnosis**:

1. **Baseline Diagnosis Model**  
   - Aimed at identifying individuals with diseases already present at baseline.

2. **Full Follow-up Prediction Model**  
   - Predicts whether an individual will develop the disease at any point during the follow-up period.

3. **5-Year Onset Prediction Model**  
   - Focuses on predicting diseases that occur **within the first 5 years** of follow-up.

4. **Post-5-Year Onset Prediction Model**  
   - Targets diseases that develop **after the first 5 years**.

## Modeling Framework

For each of the four prediction tasks, we implemented the following modeling strategy:

### 1. Feature Selection and Comparison

- Initially, models were trained using **all available metabolic features**.
- Based on **feature importance rankings**, the **top 30 metabolites** were selected for subsequent modeling.
- To evaluate the added value of metabolic markers, we compared three sets of predictors:
  - **Demographic variables only**
  - **Top 30 metabolites only**
  - **Combined model**: Demographics + Top 30 metabolites

### 2. Classification Algorithm

We used the **Light Gradient Boosting Machine (LightGBM)** classifier implemented via the `lightgbm` Python package.  
Two sets of predefined hyperparameters were used for the modeling tasks:

#### Full Metabolite Set
```python
my_params0 = {
    'n_estimators': 500,
    'max_depth': 15,
    'num_leaves': 10,
    'subsample': 0.8,
    'learning_rate': 0.01,
    'colsample_bytree': 0.7
}
```
#### Top 30 Metabolites
```python
my_params1 = {
    'n_estimators': 100,
    'max_depth': 3,
    'num_leaves': 7,
    'subsample': 1,
    'learning_rate': 0.01,
    'colsample_bytree': 1
}
```
#### Evaluation Metrics
To assess model performance, the following classification metrics were computed:
- Area Under the Receiver Operating Characteristic Curve (AUROC)
- Accuracy
- Precision
- Recall
- F1 Score

Model performance was evaluated using **area 10-fold cross-validation** to ensure robustness.

#### Statistical Comparison of ROC Curves
To compare the discriminative performance between models, we conducted **pairwise comparisons of ROC curves** using the **DeLong test**.
This test allows for the statistical comparison of two correlated ROC AUCs (e.g., Demographics vs. Metabolites, or Demographics vs. Combined model), providing **p-values** to assess whether performance differences are statistically significant.

## Code for Prediction Analysis
```python
import os
import time
import traceback
import pandas as pd
from pyreadr import pyreadr
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter
import warnings
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
import numpy as np
warnings.filterwarnings('error')

my_seed = 2024
np.random.seed(my_seed)
my_params0 = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.8,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7} 

my_params1 = {'n_estimators': 100,
             'max_depth': 3,
             'num_leaves': 7,
             'subsample': 1,
             'learning_rate': 0.01,
             'colsample_bytree': 1}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key ] /mysum
    return mydict

def get_cov_f_lst(disease_code, target_name):
    match_row = disease_code.loc[disease_code.NAME == target_name]
    sex_specific = match_row.SEX.values[0]
    if (sex_specific == 1) | (sex_specific == 2):
        cov_f_lst = ['age', 'Ethnic_group', 'edu3', 'Towns', 'BMI', 'smoke', 'units', 'SBP']
    else:
        cov_f_lst = ['age', 'Ethnic_group', 'edu3', 'Towns', 'BMI', 'smoke', 'units', 'SBP', 'sex']
    return cov_f_lst

def get_pro_f_lst(mydf, train_idx, f_lst, my_params):
    X_train, y_train = mydf.iloc[train_idx][f_lst], mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=-1, seed=2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    tg_imp_df = pd.DataFrame({'Pro_code': list(totalgain_imp.keys()), 'TotalGain': list(totalgain_imp.values())})
    tg_imp_df.sort_values(by = 'TotalGain', inplace = True, ascending = False)
    return tg_imp_df.Pro_code.tolist()[:30]

def model_training(mydf, train_idx, test_idx, f_lst, my_params):
    X_train, X_test = mydf.iloc[train_idx][f_lst], mydf.iloc[test_idx][f_lst]
    y_train = mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=-1, seed=2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred = my_lgb.predict_proba(X_test)[:, 1].tolist()
    return y_pred, my_lgb

def get_iter_predictions(mydf, full_pro_f_lst, cov_f_lst, fold_id, my_params0, my_params):
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    pro_f_lst = get_pro_f_lst(mydf, train_idx, full_pro_f_lst, my_params0)
    y_pred_pro, lgb_pro = model_training(mydf, train_idx, test_idx, pro_f_lst, my_params)
    y_pred_cov, lgb_cov = model_training(mydf, train_idx, test_idx, cov_f_lst, my_params)
    y_pred_pro_cov, lgb_pro_cov = model_training(mydf, train_idx, test_idx, cov_f_lst + pro_f_lst, my_params)
    y_test_lst = mydf.target_y.iloc[test_idx].tolist()
    id_lst = mydf.id.iloc[test_idx].tolist()
    y_pred_pro_lst = y_pred_pro
    y_pred_cov_lst = y_pred_cov
    y_pred_pro_cov_lst = y_pred_pro_cov
    totalgain_imp = lgb_pro.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(lgb_pro.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = lgb_pro.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(lgb_pro.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv = Counter(normal_imp(totalgain_imp))
    tc_imp_cv = Counter(normal_imp(totalcover_imp))
    return (tg_imp_cv, tc_imp_cv, id_lst, y_test_lst, y_pred_pro_lst, y_pred_cov_lst, y_pred_pro_cov_lst)

print("Loading data...")

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
disease_code = pd.read_csv(os.path.join(root_path, data_folder, 'disease/disease_code_ba.csv'))
region = pd.read_csv(os.path.join(root_path, data_folder, 'metabolism/region.csv'), usecols=['id', 'Region_code'])

print("Data loaded successfully.")
metabolism = dd1['metabolism']
pro_code = dd1['pro_code']
cov = dd2['cov_clean']
cov = cov[['id','age', 'Ethnic_group', 'edu3', 'Towns', 'BMI', 'smoke', 'units', 'SBP', 'sex']]
disease_df = dd3['d_ba_total']
out_df_ba = dd4['out_df_ba']
del dd1, dd2, dd3, dd4

cols_label = ['sex', 'Ethnic_group', 'edu3', 'smoke']
cov[cols_label] = cov[cols_label].apply(LabelEncoder().fit_transform)

metabolism.rename(columns=lambda x: x.replace('Clinical LDL-C', 'Clinical_LDL-C'), inplace=True)

pro_lst = metabolism.columns[1:].tolist()
disease_lst = disease_df.columns[1:].tolist()

print("Processing data...")
df = pd.merge(metabolism, cov, on='id', how='left')
df = df.apply(pd.to_numeric, errors='coerce')
df = pd.merge(df, disease_df, on='id', how='left')
df = pd.merge(df, region, on='id', how='left')
del metabolism, cov, disease_df,region

fold_id_lst = list(set(df.Region_code))
fold_id_lst.sort()
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "5"

nb_cpus = 5
save_dir = os.path.join(root_path, 'Results/predict/ba')
os.makedirs(save_dir, exist_ok=True)

bad_tgt_lst, nb_tgt_lst = [], []
disease_lst1 = disease_lst

from joblib import parallel_backend
import gc

print("Starting parallel processing...")
for tgt in tqdm(disease_lst1):
    print(f"Processing target: {tgt}")
    time_start = time.time()
    try:
        cov_f_lst = get_cov_f_lst(disease_code, tgt)
        tmp_df = df[['id'] + cov_f_lst + [tgt] + pro_lst + ['Region_code']].copy()
        tmp_df.rename(columns=lambda x: x.replace('Clinical LDL-C', 'Clinical_LDL-C'), inplace=True)
        tmp_df.rename(columns={tgt: 'target_y'}, inplace=True)
        tmp_df.dropna(subset=['target_y'], inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)
        id_lst, y_test_lst, y_pred_pro_lst, y_pred_cov_lst, y_pred_pro_cov_lst = [], [], [], [], []
        tg_imp_cv, tc_imp_cv = Counter(), Counter()

        try:
            with parallel_backend("loky", n_jobs=nb_cpus):
                fold_results_lst = Parallel(n_jobs=nb_cpus)(
                    delayed(get_iter_predictions)(tmp_df, pro_lst, cov_f_lst, fold_id, my_params0, my_params1)
                    for fold_id in fold_id_lst
                )
        except Exception as e:
            print(f"❌ Parallel processing failed for {tgt}: {e}")
            traceback.print_exc()
            bad_tgt_lst.append(tgt)
            continue 

        for fold_results in fold_results_lst:
            tg_imp_cv += fold_results[0]
            tc_imp_cv += fold_results[1]
            id_lst += fold_results[2]
            y_test_lst += fold_results[3]
            y_pred_pro_lst += fold_results[4]
            y_pred_cov_lst += fold_results[5]
            y_pred_pro_cov_lst += fold_results[6]

        tg_imp_cv = normal_imp(tg_imp_cv)
        tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()), 'TotalGain_cv': list(tg_imp_cv.values())})
        tc_imp_cv = normal_imp(tc_imp_cv)
        tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()), 'TotalCover_cv': list(tc_imp_cv.values())})
        imp_df = pd.merge(left=tc_imp_df, right=tg_imp_df, how='left', on=['Pro_code'])
        imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)

        pred_df = pd.DataFrame({
            'id': id_lst,
            'target_y': y_test_lst,
            'y_pred_pro': y_pred_pro_lst,
            'y_pred_cov': y_pred_cov_lst,
            'y_pred_pro_cov': y_pred_pro_cov_lst
        })

        imp_df.to_csv(os.path.join(save_dir, f'IMP_{tgt}.csv'), index=False)
        pred_df.to_csv(os.path.join(save_dir, f'PRED_{tgt}.csv'), index=False)

    except Exception as e:
        print(f"Error processing {tgt}: {e}")
        traceback.print_exc()
        bad_tgt_lst.append(tgt)
        continue
    gc.collect() 
    time_end = time.time()
    print(f"✔️ Finished {tgt} in {time_end - time_start:.2f} seconds.")

bad_tgt_df = pd.DataFrame({'Disease_code': bad_tgt_lst})
bad_tgt_df.to_csv(os.path.join(save_dir, 'AAA_bad_target.csv'), index=False)
```
