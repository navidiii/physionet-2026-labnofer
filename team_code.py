#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy import signal as scipy_signal
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sys
from tqdm import tqdm

from helper_code import *

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Feature subset selected via SHAP importance ranking (out of the 81 raw demographic +
# physiological + algorithmic features below), with `age`, `waso_min`, `resp_mav`, and
# `limb_auto` then explicitly dropped on top of that via a leave-one-out + combined-removal
# search using the Challenge's own metrics (Reward, Age-conditioned AUROC) instead of plain
# AUROC. `age` was the single highest-SHAP-importance feature but a harmful shortcut: since
# Age-conditioned AUROC only compares similarly-aged patients and Reward is very sensitive to
# calibration, leaning on raw age hurt both. The other three had a smaller but real version
# of the same problem. Removing all four raised mean CV Reward from 0.40 (age included) to
# 0.56 and mean CV Age-conditioned AUROC from 0.77 to 0.80 (5-seed average), at a cost of
# only ~0.01 on plain AUROC. `bmi` looks similar to age/waso_min at a glance but is NOT a
# shortcut — removing it collapses Reward to ~0.17, so it stays. Going further and also
# dropping `prob_arous`/`eeg_zcr` was tried but made both Reward and Age-AUROC worse, so the
# search stopped here. Indices refer to the position in np.hstack([demographic,
# physiological, algorithmic]): bmi, eeg_zcr, eeg_rel_theta, eog_zcr, eog_cmp, chin_zcr,
# chin_mob, ecg_cmp, spo2_rms, n3_pct, r_pct, prob_arous.
SELECTED_FEATURE_INDICES = [9, 12, 18, 24, 28, 31, 34, 49, 60, 70, 71, 75]

# XGBoost/LightGBM ensemble blend weight, re-tuned via multi-seed CV weight search after
# each SELECTED_FEATURE_INDICES change (the optimal blend shifts as the feature set changes):
# mean CV Reward+Age-AUROC is maximized around weight=0.10 on the current age-free feature set.
XGB_ENSEMBLE_WEIGHT = 0.10

# Binary decision threshold, re-tuned via multi-seed CV grid search on the Challenge Reward
# metric (which pays ~1/prevalence-1 for a caught true positive vs -1 for a miss or false
# alarm, so with ~7.6% prevalence it strongly favors a low threshold): smoothed mean CV
# Reward ~0.55 at 0.012 on the current feature set vs ~0.06 at the naive 0.5.
DECISION_THRESHOLD = 0.012

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # Iterate over the records to extract the features and labels.
    features = list()
    labels = list()
    
    pbar = tqdm(range(num_records), desc="Extracting Features", unit="record", disable=not verbose)
    for i in pbar:
        try:
            # Extract identifiers for this specific record
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({"patient": patient_id})

            # Load the patient data.
            patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)

            # Load signal data.

            # Load the physiological signal.
            physiological_data_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
            # --- Check if the file actually exists before proceeding ---
            if not os.path.exists(physiological_data_file):
                if verbose:
                    print(f"  ! Missing physiological data for {patient_id}. Skipping...")
                continue # skip record
            physiological_data, physiological_fs = load_signal_data(physiological_data_file)
            physiological_features = extract_physiological_features(physiological_data, physiological_fs, csv_path=csv_path) # This function can rename, re-reference, resample, etc. the signal data.

            # Load the algorithmic annotations.
            algorithmic_annotations_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            algorithmic_annotations, algorithmic_fs = load_signal_data(algorithmic_annotations_file)
            algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)

            # Load the human annotations; these data will not be available in the hidden validation and test sets.
            human_annotations_file = os.path.join(data_folder, HUMAN_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_expert_annotations.edf")
            human_annotations, human_fs = load_signal_data(human_annotations_file)
            human_features = extract_human_annotations_features(human_annotations)

            # Load the diagnoses; these data will not be available in the hidden validation and test sets.
            diagnosis_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            label = load_diagnoses(diagnosis_file, patient_id)

            # Store the features and labels, but the human annotations are not available on the hidden validation and test sets.
            if label == 0 or label == 1:
                features.append(np.hstack([demographic_features, physiological_features, algorithmic_features]))
                labels.append(label)

            if 'physiological_data' in locals():
                del physiological_data
            if 'algorithmic_annotations' in locals():
                del algorithmic_annotations

        except Exception as e:
            # If an error occurs (e.g., a record is corrupted), log it and move to the next
            tqdm.write(f"  !!! Error processing record {i+1} ({patient_id}): {e}")
            continue

    pbar.close()

    features = np.asarray(features, dtype=np.float32)
    ages = features[:, 0]  # age is demographic feature index 0; needed for Reward/Age-AUROC even though it's excluded from SELECTED_FEATURE_INDICES
    features = features[:, SELECTED_FEATURE_INDICES]
    labels = np.asarray(labels, dtype=bool)

    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

    labels_array = np.asarray(labels, dtype=np.float32)
    n_pos = int(np.sum(labels_array))
    n_neg = int(len(labels_array) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if verbose:
        print(f'  Classes: {n_neg} negative, {n_pos} positive (scale_pos_weight={scale_pos_weight:.2f})')

    xgb_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', XGBClassifier(
            n_estimators=357,
            max_depth=6,
            learning_rate=0.053335,
            scale_pos_weight=scale_pos_weight,
            subsample=0.804855,
            colsample_bytree=0.736161,
            min_child_weight=9,
            reg_alpha=0.104466,
            reg_lambda=0.045377,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1,
        ))
    ])

    lgb_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ))
    ])

    # 5-fold cross-validation to estimate real AUROC before final fit
    if verbose:
        print('  Running 5-fold CV to estimate generalization AUROC...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_probs_cv = cross_val_predict(xgb_model, features, labels, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
    lgb_probs_cv = cross_val_predict(lgb_model, features, labels, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
    blend_probs_cv = XGB_ENSEMBLE_WEIGHT * xgb_probs_cv + (1 - XGB_ENSEMBLE_WEIGHT) * lgb_probs_cv
    if verbose:
        print(f'  XGB CV AUROC: {roc_auc_score(labels, xgb_probs_cv):.4f}')
        print(f'  LGB CV AUROC: {roc_auc_score(labels, lgb_probs_cv):.4f}')
        print(f'  Ensemble CV AUROC (weight={XGB_ENSEMBLE_WEIGHT}): {roc_auc_score(labels, blend_probs_cv):.4f}')
        # Honest CV estimate using the Challenge's own scoring functions (Reward is what the
        # leaderboard ranks on, and Age-conditioned AUROC penalizes relying on raw age as a
        # shortcut) instead of just plain AUROC.
        from evaluate_model import compute_prevalence, compute_reward, compute_auroc_age
        labels_float = labels.astype(float)
        age_to_prevalence = compute_prevalence(ages, labels_float, ages, gap=2)
        blend_preds_cv = (blend_probs_cv >= DECISION_THRESHOLD).astype(float)
        print(f'  Ensemble CV Reward (threshold={DECISION_THRESHOLD}): '
              f'{compute_reward(labels_float, blend_preds_cv, ages, age_to_prevalence):.4f}')
        print(f'  Ensemble CV Age-conditioned AUROC: {compute_auroc_age(labels_float, blend_probs_cv, ages, gap=2):.4f}')

    # Fit both models on all data.
    xgb_model.fit(features, labels)
    lgb_model.fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the ensemble.
    save_model(model_folder, xgb_model, lgb_model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    # Support both ensemble {'xgb', 'lgb'} and legacy {'model'} formats
    if 'xgb' in model:
        xgb_model = model['xgb']
        lgb_model = model.get('lgb')
    else:
        xgb_model = model['model']
        lgb_model = None

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Load the patient data.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demographic_features = extract_demographic_features(patient_data)

    # Load the signal data.
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        # Ensure csv_path is accessible or defined
        physiological_features = extract_physiological_features(phys_data, phys_fs)
    else:
        physiological_features = np.full(54, float('nan')) # Fallback if signal data does not exist

    # Load the algorithmic annotations.
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
    else:
        algorithmic_features = np.full(17, float('nan')) # Fallback if algorithmic annotations do not exist

    features = np.hstack([demographic_features, physiological_features, algorithmic_features]).reshape(1, -1)
    features = features[:, SELECTED_FEATURE_INDICES]

    # Apply the ensemble to the features.
    xgb_prob = xgb_model.predict_proba(features)[0][1]
    if lgb_model is not None:
        lgb_prob = lgb_model.predict_proba(features)[0][1]
        probability_output = XGB_ENSEMBLE_WEIGHT * xgb_prob + (1 - XGB_ENSEMBLE_WEIGHT) * lgb_prob
    else:
        probability_output = xgb_prob
    binary_output = int(probability_output >= DECISION_THRESHOLD)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age
    age = load_age(data)
    age = np.array([age])

    # 2. Sex feature (one-hot encoding for Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (5 dimensions)
    # Standardizes the raw text into one of five categories using the helper function
    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    if race == 'Asian':
        race_vec[0] = 1
    elif race == 'Black':
        race_vec[1] = 1
    elif race == 'Others':
        race_vec[2] = 1
    elif race == 'Unavailable':
        race_vec[3] = 1
    elif race == 'White':
        race_vec[4] = 1
    else:
        race_vec[2] = 1 # Default to 'Others' for any unrecognized

    # 4. Body mass index (BMI)
    bmi = load_bmi(data)
    bmi = np.array([bmi])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    
    return np.concatenate([age, sex_vec, race_vec, bmi])


def compute_eeg_spectral(sig, fs):
    """
    Compute EEG frequency band relative powers.
    Returns 5 features: rel_delta, rel_theta, rel_alpha, rel_beta, theta/alpha ratio.
    Key Alzheimer's biomarkers: increased theta, decreased alpha.
    """
    try:
        max_samples = int(30 * 60 * fs)
        sig = sig[:max_samples] if len(sig) > max_samples else sig
        nperseg = min(int(4 * fs), len(sig))
        if nperseg < 4:
            return [float('nan')] * 5
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
        def bp(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else float('nan')
        delta = bp(0.5, 4); theta = bp(4, 8); alpha = bp(8, 13); beta = bp(13, 30)
        total = bp(0.5, 30)
        if not total or total <= 0:
            return [float('nan')] * 5
        rel_d = delta / total; rel_t = theta / total
        rel_a = alpha / total; rel_b = beta / total
        t_a = (theta / alpha) if alpha > 0 else float('nan')
        return [rel_d, rel_t, rel_a, rel_b, t_a]
    except Exception:
        return [float('nan')] * 5


def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            # Report error and stop if no FS is found for a kept channel
            raise KeyError(f"Sampling frequency (fs) not found for channel '{old_label}' ")
        
    if 'physiological_data' in locals(): del physiological_data

    # Step 3: Construct Bipolar Derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        # 1. Skip if target already exists or pos channel missing
        if target in processed_channels or pos not in processed_channels:
            continue
        
        # 2. Check all neg channels exist
        if not all(n in processed_channels for n in neg_list):
            continue

        # 3. Check sampling rate consistency
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        
        if len(set(fs_values)) > 1:
            raise ValueError(f"Sampling rate mismatch for {target}: {dict(zip(all_involved, fs_values))}")

        # 4. Derive bipolar signal
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'] # Added sao2 as fallback for spo2
    }
    
    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        fs = None
        
        # Identify the first available candidate
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate)
                break 

        if sig is not None and len(sig) > 1:
            # --- Time Domain Features (Very Fast) ---
            std_val = np.std(sig)
            mav_val = np.mean(np.abs(sig))

            # Zero Crossing Rate (Proxy for frequency/slowing)
            zcr = np.mean(np.diff(np.sign(sig)) != 0)

            # Root Mean Square
            rms = np.sqrt(np.mean(sig**2))

            # Signal Activity (Variance)
            activity = np.var(sig)

            # Mobility (Hjorth Parameter) - Proxy for mean frequency
            # sqrt(var(diff(sig)) / var(sig))
            diff_sig = np.diff(sig)
            mobility = np.sqrt(np.var(diff_sig) / activity) if activity > 0 else 0.0

            # Complexity (Hjorth Parameter) - Proxy for bandwidth
            diff2_sig = np.diff(diff_sig)
            var_d2 = np.var(diff2_sig)
            var_d1 = np.var(diff_sig)
            complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0.0

            final_features.extend([std_val, mav_val, zcr, rms, activity, mobility, complexity])

            # --- EEG Spectral Features (Alzheimer's biomarkers) ---
            if lead_type == 'eeg' and fs is not None:
                final_features.extend(compute_eeg_spectral(sig, fs))
            elif lead_type == 'eeg':
                final_features.extend([float('nan')] * 5)

        else:
            # Padding: 7 time-domain + 5 spectral (EEG only)
            final_features.extend([float('nan')] * 7)
            if lead_type == 'eeg':
                final_features.extend([float('nan')] * 5)

    if 'processed_channels' in locals(): del processed_channels

    return np.array(final_features)

def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 17
    """
    if not algo_data:
        return np.full(17, float('nan'))

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return float('nan')
        
        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)
        
        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)
        
        return num_events / total_hours
    
    ahi_auto = count_discrete_events('resp_caisr')      # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events('arousal_caisr') # Automated Arousal Index
    limb_auto = count_discrete_events('limb_caisr')    # Automated Limb Movement Index
    
    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 5=W, 4=R, 3=N1, 2=N2, 1=N3 (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        
        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [float('nan')]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [float('nan')]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [float('nan')]))
    
    # Standardize '9.0' or other filler values to NaN
    clean_prob = lambda x: x if x < 1.0 else float('nan')
    features.extend([clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    # --- 4. Sleep Fragmentation ---
    if len(valid_stages) > 1 and total_hours > 0:
        transitions_hr = float(np.count_nonzero(np.diff(valid_stages))) / total_hours
        sleep_onset = np.where((valid_stages >= 1) & (valid_stages <= 4))[0]
        if len(sleep_onset) > 0:
            post_onset = valid_stages[sleep_onset[0]:]
            waso_min = float(np.sum(post_onset == 5)) * 30.0 / 60.0
        else:
            waso_min = float('nan')
    else:
        transitions_hr = float('nan')
        waso_min = float('nan')
    features.extend([transitions_hr, waso_min])

    # --- 5. AHI Severity Flags ---
    ahi_mild   = float(ahi_auto >= 5)  if not np.isnan(ahi_auto) else float('nan')
    ahi_mod    = float(ahi_auto >= 15) if not np.isnan(ahi_auto) else float('nan')
    ahi_severe = float(ahi_auto >= 30) if not np.isnan(ahi_auto) else float('nan')
    features.extend([ahi_mild, ahi_mod, ahi_severe])

    return np.array(features)

def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.full(12, float('nan'))

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0
    
    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return float('nan')
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')      # Human AHI
    arousal_human = count_discrete_events('arousal_expert') # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')       # Human PLMI
    
    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))
    
    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]
    
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else float('nan')
    else:
        transitions = waso_minutes = rem_latency = float('nan')

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)


# Save your trained model.
def save_model(model_folder, xgb_model, lgb_model=None):
    d = {'xgb': xgb_model}
    if lgb_model is not None:
        d['lgb'] = lgb_model
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)