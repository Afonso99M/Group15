import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, precision_recall_curve

# ------------------------------------------ Plot Numericals ------------------------------------------

def freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr == 0: 
        return int(np.ceil(np.log2(len(data)) + 1))
    bin_width = 2 * iqr * (len(data) ** (-1 / 3))
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(1, num_bins)  

def plot_numerical(df, col):
    sns.set(style="white")
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.7], hspace=0.03)
    
    col_median = np.median(df[col])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[col], bins=freedman_diaconis_bins(df[col]), alpha=0.9, color="skyblue", edgecolor="black")
    ax1.axvline(col_median, color='orange', linestyle='-', label=f"Median: {round(col_median)}", alpha=0.4)
    ax1.set_xticks([])
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    sns.set(style="whitegrid")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.boxplot(df[col], flierprops=dict(marker='o', alpha=0.5, markersize=3), 
                vert=False, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", color="black"))
    ax2.axvline(col_median, color='orange', linestyle='-', label=f"Median: {round(col_median)}", alpha=0.5)
    ax2.legend()
    ax2.set_xlabel(col)
    ax2.set_yticks([]) 
    
    plt.suptitle(f"'{col}' - Histogram and Boxplot", fontsize=14)
    plt.show()
    return None
    

# ------------------------------------------ Plot Outliers ------------------------------------------

def compare_figure_outliers(df_original, df, num_feats):
    sns.set_style('whitegrid')
    frows = math.ceil(len(num_feats) / 2)
    fcols = 2
    
    fig = plt.figure(figsize=(15, 5 * frows))
    
    subfigs = fig.subfigures(frows, fcols, wspace=0.03, hspace=0.03)
    
    for sfig, feat in zip(subfigs.flatten(), num_feats):
        axes = sfig.subplots(2, 1, sharex=True)
        
        sns.boxplot(x=df_original[feat], ax=axes[0])
        axes[0].set_ylabel("Original")
        axes[0].set_title(feat, fontsize="large")
        
        sns.boxplot(x=df[feat], ax=axes[1])
        axes[1].set_ylabel("Outliers\nRemoved")
        axes[1].set_xlabel("")
        
        sfig.set_facecolor("#F9F9F9")
        sfig.subplots_adjust(left=0.2, right=0.95, bottom=0.1)
        
    plt.show()
    sns.set()
    return None



# ------------------------------------------ Plot Numericals vs Target ------------------------------------------

def plot_numerical_vs_target(df, col, target):
    colors = ['#468B79', '#CF7842', '#827EB3', '#D9658D', 
              '#6BA656', '#D4AF4F', '#A9855A', '#808080']
    
    
    median_order = df.groupby(target)[col].median().sort_values().index
    
    
    ax = sns.boxplot(x=target, y=col, data=df, order=median_order, palette=colors[:len(median_order)])
    
    
    for i, line in enumerate(ax.lines[4::6]):  
        line.set_color("red")
        line.set_linewidth(2)
    
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(f"{target} by {col}")
    plt.show()

    return None
    
    # ------------------------------------------ Plot categoricals vs Target ------------------------------------------

def plot_side_by_side_crosstabs(dataframe, column1, column2, normalize="index", figsize=(16, 6)):
    """
    Plots two stacked bar charts side-by-side based on the cross-tabulations of two columns,
    ensuring both charts include legends and use a custom color palette.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        column1 (str): The first column for crosstab (rows).
        column2 (str): The second column for crosstab (columns).
        normalize (str): Normalize along 'index' or 'columns'. Default is 'index'.
        figsize (tuple): Size of the overall figure. Default is (16, 6).
    """
    # Define the custom color palette
    colors = ['#468B79', '#CF7842', '#827EB3', '#D9658D', 
              '#6BA656', '#D4AF4F', '#A9855A', '#808080']
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # First plot
    crosstab1 = pd.crosstab(dataframe[column1], dataframe[column2], normalize=normalize)
    crosstab1.plot(kind="bar", stacked=True, ax=axes[0], color=colors[:len(crosstab1.columns)])  # Use the custom colors
    axes[0].set_title(f"Distribution of {column2} by {column1}")
    axes[0].set_ylabel("Proportion")
    axes[0].set_xlabel(column1)
    axes[0].legend(title=column2, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second plot
    crosstab2 = pd.crosstab(dataframe[column2], dataframe[column1], normalize=normalize)
    crosstab2.plot(kind="bar", stacked=True, ax=axes[1], color=colors[:len(crosstab2.columns)])  # Use the custom colors
    axes[1].set_title(f"Distribution of {column1} by {column2}")
    axes[1].set_ylabel("Proportion")
    axes[1].set_xlabel(column2)
    axes[1].legend(title=column1, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
# ------------------------------------------ Plot Confusion Matrix ------------------------------------------



def plot_confusion_matrix(y_true, y_pred, clf_name, normalize=False, cmap='Blues'):
    """
    Plots a confusion matrix using matplotlib.

    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list, optional
        List of class names to display. Defaults to numeric labels if None.
    normalize : bool, optional
        Whether to normalize the confusion matrix values (proportion instead of count). Default is False.
    cmap : str, optional
        Matplotlib colormap to use for the heatmap. Default is 'Blues'.

    Returns:
    --------
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    class_names = unique_labels(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(f'{clf_name} Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
    return None


# ------------------------------------------ Data Preprocessing ------------------------------------------
# --------------------------------------------------------------------------------------------------------

# ------------------------------------------ Scaling features ------------------------------------------

def num_scaling(X_train, X_val):
    # benefits most models that are distance based (KNN) or gradient descent based (Neural Networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val


# ------------------------------------------ Imputing missing values ------------------------------------------

def num_imputing(X_train, X_val):
    feats_imput_max = ["C2_Accident_gap_weeks", "C3_Accident_gap_weeks", "Accident Date_assembly_gap_days", "Hearing_C3 gap_months", "Hearing_C2 gap_months", "Hearing_assembly_gap_months", "Days to First Hearing"]

    feat_imput_min = ["C3-C2_gap_days"]
    
    for feat in X_train.columns:
        if X_train[feat].isna().sum() > 0:
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


# ------------------------------------------ Encoding Categoricals ------------------------------------------

def calculate_woe(df, col, target_col):
    # based on https://ishanjainoffical.medium.com/understanding-weight-of-evidence-woe-with-python-code-cd0df0e4001e
    category_counts = df[col].value_counts()
    category_counts_pos = df[df[target_col] == 1][col].value_counts()
    category_counts_neg = df[df[target_col] == 0][col].value_counts()
    woe_values = {}
    epsilon = 1e-4  # Small constant to avoid division by zero
    for category in category_counts.index:
        pos_rate = (category_counts_pos.get(category, 0) + epsilon) / (category_counts[category] + epsilon)
        neg_rate = (category_counts_neg.get(category, 0) + epsilon) / (category_counts[category] + epsilon)
        woe = np.log(pos_rate / neg_rate)
        woe_values[category] = woe
    return woe_values

def weight_evidence_encoding(X_train, X_val, y_train, target_name, drop_original=False):
    cat_feats = [col for col in X_train.columns if X_train[col].dtype == "object"]
    
    X_train_y = pd.concat([X_train, y_train], axis=1)
    
    for cat in cat_feats:
        cat_encoded_name = f"{cat}_WOE_{target_name}"
        woe_mapping = calculate_woe(X_train_y, cat, target_name)
        X_train[cat_encoded_name] = X_train[cat].map(woe_mapping)
        X_val[cat_encoded_name] = X_val[cat].map(woe_mapping)
        if drop_original:
            X_train.drop(columns=cat, inplace=True)
            X_val.drop(columns=cat, inplace=True)
        #plot_numerical(X_train, cat_encoded_name)
        #plot_numerical(X_val, cat_encoded_name)
    return X_train, X_val
    
    

def target_guided_ordinal_encoding(X_train, X_val, categorical_column, target_column, y_train, i):
    # Combine X_train with y_train temporarily to calculate means
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_train_encoded[target_column] = y_train

    means = X_train_encoded.groupby(categorical_column)[target_column].mean()
    #print(means)

    sorted_means = means.sort_values(by=target_column)
    if i == 1:
        print(f"Showing sorted means for {categorical_column}")
        lst_names = sorted_means.index.tolist()
        lst_values = sorted_means.values.tolist()
        dict_final = dict(zip(lst_names, lst_values))
        print(dict_final)
    
    ordinal_mapping = {category: rank for rank, category in enumerate(sorted_means.index, start=1)}
    if i == 1:
        print(f"Showing ordinal mapping for {categorical_column}")
        print(ordinal_mapping)
        print("--------------------------------")
        
    X_train_encoded[f"{categorical_column}_encoded"] = X_train_encoded[categorical_column].map(ordinal_mapping)
    X_val_encoded[f"{categorical_column}_encoded"] = X_val_encoded[categorical_column].map(ordinal_mapping)

    X_train_encoded = X_train_encoded.drop(columns=[categorical_column])
    X_train_encoded = X_train_encoded.drop(columns=[target_column[0]])
    X_val_encoded = X_val_encoded.drop(columns=[categorical_column])

    return X_train_encoded, X_val_encoded, ordinal_mapping




# ------------------------------------------ Redundant Features selection ------------------------------------------

def cross_corr_target(df_input, df_target, corr_coeff=0.90):
    # adjusted from: https://github.com/adityav95/variable_reduction_correlation/blob/master/variable_reduction_by_correlation.ipynb
    # in the original they used pearson correlation here we use spearman 
	""" The function retuns a list of features to be dropped from the input features.
	
	INPUTS:
	1. df_input: n numeric input features (pandas dataframe)
	2. df_target: Target values (ensure same order as input features)
	3. corr_coeff: Coefficient threshold (absolute value, no negatives) for a pair of variables above which one of the two will be dropped

	
	PLEASE NOTE:
	- The dataframe df_input should contain only the n numeric input features i.e. no ID and targets) 
	- The pandas series df_target should only be 1 column (if multiclass it should include all classes) and should be in the same order as the input dataset df_input

	SUMMARY OF LOGIC:
	1. The n numeric input variables are taken and a n X n matrix of correlation is created (these are absolute values i.e. a correlation of -0.8 is treated as 0.8)
	2. Variable pairs with correlation higher than the corr_coeff threshold are picked and one of the two variables will be dropped
	3. Which of the two will be dropped is based on the one having lower correlation with target variable

	"""

	# Combining the input and target data
	df = pd.DataFrame(df_input)
	df["target"] = pd.Series(df_target)

	# Generating correlation matrix of input features
	corr_matrix = df.corr(method = 'spearman') 

	corr_target = (corr_matrix["target"])

	# Preparing data
	features_drop_list = [] # This will contain the list of features to be dropped
	features_index_drop_list = [] # This will contain the index of features to be dropped as per df_input
	corr_matrix = abs(corr_matrix)
	corr_target = abs(corr_target)

	# Selecting features to be dropped (Using two for loops that runs on one triangle of the corr_matrix to avoid checking the correlation of a variable with itself)
	for i in range(corr_matrix.shape[0]):
		for j in range(i+1,corr_matrix.shape[0]):

			# The following if statement checks if each correlation value is higher than threshold (or equal) and also ensures the two columns have NOT been dropped already.  
			if corr_matrix.iloc[i,j]>=corr_coeff and i not in features_index_drop_list and j not in features_index_drop_list:
			
				# The following if statement checks which of the 2 variables with high correlation has a lower correlation with target and then drops it. If equal we can drop any and it drops the first one (This is arbitrary)
				if corr_target[corr_matrix.columns[i]] >= corr_target[corr_matrix.columns[j]]:
					features_drop_list.append(corr_matrix.columns[j])	# Name of variable that needs to be dropped appended to list
					features_index_drop_list.append(j)	# Index of variable that needs to be dropped appended to list. This is used to not check for the same variables repeatedly
				else:
					features_drop_list.append(corr_matrix.columns[i])
					features_index_drop_list.append(i)

	return features_drop_list






def cross_corr_mean(df_input, corr_coeff=0.95):
    # adjusted from: https://github.com/adityav95/variable_reduction_correlation/blob/master/variable_reduction_by_correlation.ipynb
    # in the original they used pearson correlation here we use spearman
	""" The function retuns a list of features to be dropped from the input features.
	
	INPUTS:
	1. df_input: n input features (pandas dataframe)
	2. corr_coeff: Coefficient threshold (absolute value, no negatives) for a pair of variables above which one of the two will be dropped
	
	NOTICE:
	- The dataframe df_input (should contain only the n input features i.e. no ID and targets) 
	
	SUMMARY OF LOGIC:
	1. The n input variables are taken and a n X n matrix of correlation is created (these are absolute values i.e. a correlation of -0.8 is treated as 0.8)
	2. Variable pairs with correlation higher than the corr_coeff threshold are picked and one of the two variables will be dropped
	3. Which of the two will be dropped is based on the one having lower mean absolute correlation with all other variables 

	"""


	# Generating correlation matrix of input features
	corr_matrix = df_input.corr(method = 'spearman')

	# Generating correlation with the target
	corr_mean = abs(corr_matrix).mean()

	# Preparing data
	features_drop_list = [] # This will contain the list of features to be dropped
	features_index_drop_list = [] # This will contain the index of features to be dropped as per df_input
	corr_matrix = abs(corr_matrix)

	# Selecting features to be dropped (Using two for loops that runs on one triangle of the corr_matrix to avoid checking the correlation of a variable with itself)
	for i in range(corr_matrix.shape[0]):
		for j in range(i+1,corr_matrix.shape[0]):

			# The following if statement checks if each correlation value is higher than threshold (or equal) and also ensures the two columns have NOT been dropped already.  
			if corr_matrix.iloc[i,j]>=corr_coeff and i not in features_index_drop_list and j not in features_index_drop_list:
			
				# The following if statement checks which of the 2 variables with high correlation has a lower correlation with target and then drops it. If equal we can drop any and it drops the first one (This is arbitrary)
				if corr_mean[corr_matrix.columns[i]] >= corr_mean[corr_matrix.columns[j]]:
					features_drop_list.append(corr_matrix.columns[i])	# Name of variable that needs to be dropped appended to list
					features_index_drop_list.append(i)	# Index of variable that needs to be dropped appended to list. This is used to not check for the same variables repeatedly
				else:
					features_drop_list.append(corr_matrix.columns[j])
					features_index_drop_list.append(j)

	return features_drop_list


# ------------------------------------------ Model Evaluation Evaluation ------------------------------------------

def model_predictions(X_train, y_train, X_val, y_val, model, clf_name, specific_target, cv_i, params):

    train_probas = model.predict_proba(X_train)[:, 1]
    val_probas = model.predict_proba(X_val)[:, 1]
    
    
    precision, recall, thresholds = precision_recall_curve(y_val, val_probas)
    f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    best_threshold = thresholds[np.argmax(f1)]
    adjusted_train_predictions = np.where(train_probas > best_threshold, 1, 0)
    adjusted_val_predictions = np.where(val_probas > best_threshold, 1, 0)
    
    train_score = f1_score(y_train, adjusted_train_predictions)
    val_score = f1_score(y_val, adjusted_val_predictions)
    
    print(f"{clf_name}...")
    print(f"Params: {params}")
    print(f"Train F1-score: {round(train_score, 3)}")
    print(f"Validation F1-score: {round(val_score, 3)}")
    print(classification_report(y_val, adjusted_val_predictions))
    
    return params, best_threshold, train_score, val_score 



def model_predictions_global(X_train, y_train, X_val, y_val, model, clf_name, specific_target, cv_i, params):
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    f1_score_train = f1_score(y_train, train_preds, average="macro")
    f1_score_val = f1_score(y_val, val_preds, average="macro")
    
    print(f"{clf_name}...")
    print(f"Params: {params}")
    print(f"Train F1-score: {round(f1_score_train, 3)}")
    print(f"Validation F1-score: {round(f1_score_val, 3)}")
    print(classification_report(y_val, val_preds))
    
    return params, f1_score_train, f1_score_val 
