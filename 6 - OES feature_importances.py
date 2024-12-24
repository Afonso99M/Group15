import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn import base

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.impute import KNNImputer

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier


from sklearn.feature_selection import RFE
from sklearn.metrics import precision_recall_curve
from collections import defaultdict 

from sklearn.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

import scipy.stats as stats


from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from utils import *
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from collections import Counter
import inspect
from collections import defaultdict

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier



df = pd.read_csv("train_new_feats.csv")

df.columns


target = [[f"target_{i}" for i in range(1, 9)] + ["Claim Injury Type"] + ["WCB Decision"] + ["Agreement Reached"] + ["Claim Injury Type_encoded"]]
target = [item for sublist in target for item in sublist]
target

binary_target = [f"target_{i}" for i in range(1, 9)]

original_target  = [col for col in target if col not in binary_target]

ordinal_target = ["Claim Injury Type_encoded"]

features = [feat for feat in df.columns if feat not in target]

features = [feat for feat in features if df[feat].dtype != "datetime64[ns]"]

num_feats = [feat for feat in features if df[feat].dtype != "object"]

cat_feats = [feat for feat in features if df[feat].dtype == "object"]
cat_feats_index = [features.index(feat) for feat in cat_feats]



def num_imputing(X_train, X_val):
    feats_imput_max = ["C2_Accident_gap_weeks", "C3_Accident_gap_weeks", "Accident Date_assembly_gap_days", "Hearing_C3 gap_months", "Hearing_C2 gap_months", "Hearing_assembly_gap_months", "Days to First Hearing"]

    feat_imput_min = ["C3-C2_gap_days"]
    
    for feat in X_train.columns:
        if X_train[feat].isna().sum() > 0 or X_val[feat].isna().sum() > 0:
            if feat in feats_imput_max:
                X_train[feat] = X_train[feat].fillna(X_train[feat].max())
                X_val[feat] = X_val[feat].fillna(X_train[feat].max())
            elif feat in feat_imput_min:
                X_train[feat] = X_train[feat].fillna(X_train[feat].min())
                X_val[feat] = X_val[feat].fillna(X_train[feat].min())
            else:
                X_train[feat] = X_train[feat].fillna(X_train[feat].mean())
                X_val[feat] = X_val[feat].fillna(X_train[feat].mean())
    return X_train, X_val

def frequency_encoding(train_df, val_df, column):
    """
    Apply frequency encoding on the training set and use the same encoding to impute the validation set.
    
    Parameters:
    train_df (pd.DataFrame): Training dataset.
    val_df (pd.DataFrame): Validation dataset.
    column (str): Column to encode.
    
    Returns:
    train_encoded (pd.DataFrame): Encoded training set.
    val_encoded (pd.DataFrame): Encoded validation set.
    freq_map (dict): Mapping of frequency counts for the column.
    """
    # Compute frequency encoding for the training set
    freq_map = train_df[column].value_counts(normalize=True)  # Relative frequency
    train_df[f"{column}_freq"] = train_df[column].map(freq_map)

    # Impute frequency encoding on the validation set using the same mapping
    val_df[f"{column}_freq"] = val_df[column].map(freq_map)

    # Handle unseen categories in validation by imputing 0 frequency
    val_df[f"{column}_freq"] = val_df[f"{column}_freq"].fillna(0)
    
    train_df = train_df.drop(columns=[column])
    val_df = val_df.drop(columns=[column])

    # Return encoded datasets and frequency map
    return train_df, val_df, freq_map

def target_guided_ordinal_encoding(X_train, X_val, categorical_column, target_column, y_train, i):
    # Combine X_train with y_train temporarily to calculate means
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_train_encoded[target_column] = y_train

    means = X_train_encoded.groupby(categorical_column)[target_column].mean()
    #print(means)

    sorted_means = means.sort_values(by=target_column)
    #print(sorted_means)
    # if i == 1:
    #     print(f"Showing sorted means for {categorical_column}")
    #     lst_names = sorted_means.index.tolist()
    #     lst_values = sorted_means.values.tolist()
    #     dict_final = dict(zip(lst_names, lst_values))
    #     print(dict_final)
    
    ordinal_mapping = {category: rank for rank, category in enumerate(sorted_means.index, start=1)}
    # if i == 1:
    #     print(f"Showing ordinal mapping for {categorical_column}")
    #     print(ordinal_mapping)
    #     print("--------------------------------")
        
    X_train_encoded[f"{categorical_column}_encoded"] = X_train_encoded[categorical_column].map(ordinal_mapping)
    X_val_encoded[f"{categorical_column}_encoded"] = X_val_encoded[categorical_column].map(ordinal_mapping)

    #X_train_encoded = X_train_encoded.drop(columns=[categorical_column])
    X_train_encoded = X_train_encoded.drop(columns=[target_column[0]])
    #X_val_encoded = X_val_encoded.drop(columns=[categorical_column])
    X_train_encoded = X_train_encoded.fillna(1)
    X_val_encoded = X_val_encoded.fillna(1)

    return X_train_encoded, X_val_encoded, ordinal_mapping


selected_features=['Attorney/Representative',
 'IME-4 Count',
 'Accident Date_year',
 'Accident Date_assembly_gap_days',
 'C3-C2_gap_days',
 'C2_missing',
 'C3_missing',
 'C3_Accident_gap_weeks',
 'Hearing_C3 gap_months',
 'Hearing_C2 gap_months',
 'Days to Assembly',
 'Days to First Hearing',
 'Average Weekly Wage_log',
 'Carrier Name_encoded',
 'Carrier Type_encoded',
 'Industry Code Description_encoded',
 'WCIO Cause of Injury Description_encoded',
 'WCIO Nature of Injury Description_encoded',
 'WCIO Part Of Body Description_encoded',
 'Carrier Name_freq',
 'Carrier Type_freq',
 'Industry Code Description_freq',
 'WCIO Nature of Injury Description_freq',
 'WCIO Part Of Body Description_freq']

naive_features = [feat.replace("_encoded", "") for feat in selected_features]
naive_features = [feat.replace(f"_freq", "") for feat in naive_features]
naive_features = set(naive_features)
naive_features = list(naive_features)

cat_feats = [feat for feat in naive_features if feat in cat_feats]





X = df[naive_features]
y = df[ordinal_target]
# # ---------------  ------------------------------------

# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# X_train_encoded = X_train.copy()
# X_val_encoded = X_val.copy()

# # --------------- ------------------------------------
X_encoded = X.copy()
X_encoded_ = X.copy()

print(f"Ordinal encoding...")
X_train_encoded = X_encoded.copy()
X_val_encoded = X_encoded_.copy()
for cat in cat_feats:
    X_train_encoded, X_val_encoded, ordinal_mapping = target_guided_ordinal_encoding(X_train_encoded, X_val_encoded, cat, ordinal_target, y, 0)

print(f"Frequency encoding...")
for cat in cat_feats:
    X_train_encoded, X_val_encoded, freq_map = frequency_encoding(X_train_encoded, X_val_encoded, cat)


X_train_encoded  = X_train_encoded[selected_features]
X_val_encoded = X_val_encoded[selected_features]

X_train_imputed, X_val_imputed = num_imputing(X_train_encoded, X_val_encoded)


clf = CatBoostClassifier(random_state=42, verbose=10, iterations=1000, depth=6, boosting_type='Ordered', auto_class_weights='SqrtBalanced', loss_function="MultiClassOneVsAll")

clf.fit(X_train_imputed, y)




# -------------------- Global feature importance
print("Feature importances")

feature_importance = clf.get_feature_importance()
feature_names = clf.feature_names_
len(feature_importance), len(feature_names)
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# --------------------- Specific target feature importances
""
!pip install shap
import shap

# Train the model on the entire dataset

# Feature importances
print("Feature Importances:")
print(final_clf.get_feature_importance())

# SHAP Values for deeper insights
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

"""
OneVsRestClassifier() basically builds as much binary classifiers as there are classes. Each has its own set of importances (assuming the base classifier supports them), showing the importance of features to distinguish a certain class from all others when generalizing on the train set. Those can be accessed with .estimators_[i].feature_importances_.
"""

