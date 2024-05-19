import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import  pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import scipy
import os
from sklearn.decomposition import PCA

from mappings import *
from ucimlrepo import fetch_ucirepo

import warnings

seed = 0
np.random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)

def load_data(from_begin=False, nationality=True):
    """
    This function loads and preprocesses data related to predicting student dropout and academic success from the UCI repository.

    Returns:
        features (DataFrame): DataFrame containing selected features for analysis.
        features_full (DataFrame): DataFrame containing all features, including the target variable 'Dropout'.
        labels (DataFrame): DataFrame containing binary labels for dropout prediction.
    """
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
    df = predict_students_dropout_and_academic_success.data.features
    df.columns = [i.lower().replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace("\t", "") for i in df.columns]
    df.rename(columns={'nacionality': 'nationality'}, inplace=True)

    if from_begin:
       df.drop([x for x in df.columns if 'curricular' in x], axis=1, inplace=True)

    # Make a binary column for dropout
    data_y = predict_students_dropout_and_academic_success.data.targets
    data_y = data_y.Target.apply(lambda x: 1 if x == 'Graduate' else 0)
    df['graduated'] = data_y

    for i,j in enumerate(columns_to_replace):
        df[columns_to_replace[i]] = df[columns_to_replace[i]].replace(dicts[i])

        
    if nationality:
        
        one_feats = ['application_mode', 'course', 
                     'nationality', 'previous_qualification', 'mothers_qualification',
                     'fathers_qualification', 'mothers_occupation',
                     'fathers_occupation', 'marital_status']
    else:
        df.drop('nationality', axis=1, inplace=True)
        one_feats = ['application_mode', 'course', 
                     'previous_qualification', 'mothers_qualification',
                     'fathers_qualification', 'mothers_occupation',
                     'fathers_occupation', 'marital_status']

    df["mothers_occupation"] = df["mothers_occupation"].replace(inverse)
    df["fathers_occupation"] = df["fathers_occupation"].replace(inverse)
    df["mothers_qualification"] = df["mothers_qualification"].replace(inverse_education)
    df["fathers_qualification"] = df["fathers_qualification"].replace(inverse_education)
    df["previous_qualification"] = df["previous_qualification"].replace(inverse_education)

    labels = df[['graduated']]
    #groups = df[protected_cols]

    features_full = df.copy()
    df.drop('graduated', axis=1, inplace=True)

    features = pd.get_dummies(df, columns= one_feats, dtype='int')

    ## create 'default' student and drop columns:
    # we selected manually rather than using the 'drop_first' for control
    features.drop(['application_mode_1st_phase_general_contingent', 'course_nursing', 
                    'previous_qualification_upper_secondary',
                    'mothers_qualification_upper_secondary', 'fathers_qualification_upper_secondary',
                    'mothers_occupation_unskilled','fathers_occupation_unskilled', 'marital_status_single'
                    ], axis=1, inplace=True)
    
    if 'nationality_portuguese' in features.columns:
       features.drop(['nationality_portuguese'], axis=1, inplace=True )
    
    
     

    return features, features_full, labels#, groups, protected_cols


def get_test_and_training_data(features, labels, group, protected_cols, test_size=0.2):
    """
    Splits the dataset into training and testing sets, and separates protected and non-protected features.

    Parameters:
        features (DataFrame): DataFrame containing features.
        labels (DataFrame): DataFrame containing target labels.
        group (DataFrame or pandas.core.series.Series): containing a group e.g. features['Gender']
        protected_cols (list): List of protected columns to seperate protected and non-protected features.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (DataFrame): Training labels.
        y_test (DataFrame): Testing labels.
        X_train_p (DataFrame): Training protected features.
        X_train_np (DataFrame): Training non-protected features.
        X_test_p (DataFrame): Testing protected features.
        X_test_np (DataFrame): Testing non-protected features.
        group_train (DataFrame or pandas.core.series.Series): Grouping variable for training set.
        group_test (DataFrame or pandas.core.series.Series): Grouping variable for testing set.
    """

    X, y = features, labels

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, group, test_size=test_size, random_state=seed)

    # Subset for protected and non-protected features
    X_train_p = X_train[protected_cols]
    X_train_np = X_train.drop(columns=protected_cols)
    X_test_p = X_test[protected_cols]
    X_test_np = X_test.drop(columns=protected_cols)

    return X_train, X_test, y_train, y_test, X_train_p, X_train_np, X_test_p, X_test_np, group_train, group_test

def standard_scale(X_train, X_test): ## We need to scale within the cross val step to avoid df leakage
    """
    Scale X_train and X_test using a StandardScalar fitted to X_train.

    Parameters:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.

    Returns:
        X_train_scaled (DataFrame): Scaled training features.
        X_test_scaled (DataFrame): Scaled testing features.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test

def debias_features(X, protected_idx, l=0):
    """
    Debias the features using de-correlation. The code is based on the exercises

    Parameters:
        X (DataFrame): Input features.
        protected_idx (list): Indices of protected features.
        l (float, optional): Lambda parameter for adjusting the level of debiasing. Defaults to 0 (max. debiased).

    Returns:
        DataFrame: Debias features.
    """

    #FunctionTransfomer makes it an array and I want it to be a dataFrame
    X = pd.DataFrame(X) #This makes no difference

    X_p = X[protected_idx]
    X_np = X.drop(columns=protected_idx)

    # Find the basis from the protected attributes
    orthbasis = scipy.linalg.orth(X_p)

    # Debias nonprotected features by projecting them onto the basis
    X_np_debiased = X_np - orthbasis @ orthbasis.T @ X_np

    # Return debiased nonprotected features, tempered by lambda: r′_j(λ) = r_j + λ⋅ (x_j− r_j)
    return pd.DataFrame(X_np_debiased + l * (X_np - X_np_debiased))

def get_debiased_data(X_train, X_test, protected_cols, lambd=0):
  """
  Get debiased df by applying standard scaling and debias_features().

  Parameters:
      X_train (DataFrame): Training features.
      X_test (DataFrame): Testing features.
      protected_cols (list): List of protected columns.
      lambd (float, optional): Lambda parameter for adjusting debias level. Defaults to 0 (max. debiased).

  Returns:
      DataFrame: Debiased training features (de-correlated and protected columns removed).
      DataFrame: Debiased testing features (protected columns removed).
  """
  nonprotected_cols = [col for col in X_train.columns if col not in protected_cols]
  protected_idx = [X_train.columns.get_loc(col) for col in protected_cols]

  X_train, X_test = standard_scale(X_train, X_test)

  X_train_debiased = debias_features(X_train, protected_idx, l=lambd) # using decorrelation
  X_train_debiased.columns = nonprotected_cols # Assume the order is the same

  X_test = X_test.drop(columns=protected_idx)
  X_test.columns = nonprotected_cols

  return X_train_debiased, X_test

# Define a class for Fair PCA. The code is taken from our assignment 2.
class FairPCA:
    """
    Fair Principal Component Analysis (PCA) class for debiasing df.

    Attributes:
        U (array): Projection matrix defined in fit().
    """

    def __init__(self, Xs, p_idxs, n_components):
        """
        Initializes FairPCA instance and fits the model.

        Parameters:
            Xs (DataFrame): Input features.
            p_idxs (list): Indices of protected features.
            n_components (int): Number of principal components.
        """

        self.fit(Xs, p_idxs, n_components)

    def fit(self, Xs, p_idxs, n_components):
        """
        Fits the FairPCA model.

        Parameters:
            Xs (DataFrame): Input features.
            p_idxs (list): Indices of protected features.
            n_components (int): Number of principal components.
        """
        # Extract protected features
        Xs_p = Xs.iloc[:, p_idxs]

        # Compute projection matrix (U)
        # Set z
        Z = Xs_p

        # Compute orthonormal null-space spanned by Z.T @ Xs
        Z = Z - Z.mean(0) #center
        R = scipy.linalg.null_space(Z.T @ Xs)

        # Compute orthonormal eigenvectors (L)
        eig_vals, L = scipy.linalg.eig(R.T @ Xs.T @ Xs @ R)

        # U = R * Eigenvectors
        self.U = R @ L[:, :n_components]

    def project(self, Xs):
        """
        Projects df into fair space using projection matrix U.

        Parameters:
            Xs (DataFrame): Input features.

        Returns:
            DataFrame: Projected features.
        """
        # Project df into fair space using projection matrix U
        return Xs @ self.U


# Define a class for Normal PCA. The code is taken from our assignment 2.
class NormalPCA:
    """
    Normal Principal Component Analysis (PCA) class.

    Attributes:
        pca (PCA): PCA model.

    """

    def __init__(self, n_components):
        """
        Initializes NormalPCA instance with the number of components.

        Parameters:
            n_components (int): Number of principal components.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=seed)

    def fit(self, X):
        """
        Fits the NormalPCA model.

        Parameters:
            X (DataFrame): Input features.
        """
        self.pca.fit(X)

    def project(self, X):
        """
        Projects df into PCA space.

        Parameters:
            X (DataFrame): Input features.

        Returns:
            DataFrame: Projected features.
        """
        return self.pca.transform(X)


def debias_features_adjust_fairness_level(X_fair, X_pca, l=0):
    """
    Adjusts fairness level of debiased features.

    Parameters:
        X_fair (DataFrame): Fair features.
        X_pca (DataFrame): PCA features.
        l (float, optional): Lambda parameter for tempering. Defaults to 0.

    Returns:
        DataFrame: Adjusted debiased features.
    """

    # r′j(λ) = rj + λ⋅ (xj− rj)
    return X_fair + l * (X_pca - X_fair)

def get_pca_data(X_train, X_test, protected_cols, lambd=0):
  """
  Get debiased df using FairPCA.

  Parameters:
      X_train (DataFrame): Training features.
      X_test (DataFrame): Testing features.
      protected_cols (list): List of protected columns.
      lambd (float, optional): Lambda parameter for tempering. Defaults to 0 (max. fair).

  Returns:
      DataFrame: Debiased and PCA-transformed training features.
      DataFrame: Debiased and PCA-transformed testing features.
  """
  nonprotected_cols = [col for col in X_train.columns if col not in protected_cols]
  protected_idx = [X_train.columns.get_loc(col) for col in protected_cols]

  n_components = len(X_train.columns) - len(protected_idx)

  fair = FairPCA(X_train, protected_idx, n_components)
  norm = NormalPCA(n_components)
  norm.fit(X_train)

  X_train_debiased = fair.project(X_train)
  X_test_debiased = fair.project(X_test)
  X_train_pca = norm.project(X_train)
  X_test_pca = norm.project(X_test)

  X_train_final = debias_features_adjust_fairness_level(X_train_debiased, X_train_pca, l=lambd)
  X_test_final = debias_features_adjust_fairness_level(X_test_debiased, X_test_pca, l=lambd)

  return X_train_final, X_test_final, fair

def cross_validator(model, X, y, protected_cols=[], n=5, debias=None, lambd=0):
    """
    Perform cross-validation with optional debiasing.

    Parameters:
        model: The model to be trained and tested.
        X (DataFrame): The input features.
        y (DataFrame or Series): The target variable.
        protected_cols (list, optional): List of protected columns. Defaults to []. Only needed when debiasing df.
        n (int, optional): Number of folds for cross-validation. Defaults to 5.
        debias (str, optional): Method for debiasing. Options: 'fairPCA', 'geometric', or None. Defaults to None.
        lambd (float, optional): Lambda parameter for adjusting debias. Defaults to 0 (max. debiasing).

    Returns:
        ndarray: Predicted labels.
    """

    kfold = StratifiedKFold(n, shuffle=True, random_state=seed)

    y_hat = np.zeros(len(y))

    for train_idx, test_idx in kfold.split(X, y): ## Split based on both X and y
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        if debias == 'fairPCA':
          X_train, X_test,_ = get_pca_data(X_train, X_test, protected_cols=protected_cols, lambd=lambd)
        elif debias == 'geometric':
          X_train, X_test = get_debiased_data(X_train, X_test, protected_cols, lambd=lambd) #### This is if you want to use the geometric version

        else:
          X_train, X_test = standard_scale(X_train, X_test)


        model = model.fit(X_train, y_train)
        y_hat[test_idx] = model.predict(X_test)

    return y_hat

def tune_lambda(model, X, y, protected_cols, groups):
  """
  Tune the lambda parameter for debiasing using fairPCA.

  Parameters:
      model: The model to be trained and tested.
      X (DataFrame): The input features.
      y (Series): The target variable.
      protected_cols (list): List of protected columns.

  Returns:
      tuple: Lambda values, accuracies, F1 scores, positive rates.
  """

   # Define a function to calculate true positives
  def calculate_true_positives(y_preds):
    tn, fp, fn, tp = confusion_matrix(y, y_preds).ravel()
    true_positives = tp / (tp + fn)
    return true_positives


  lambda_values = np.linspace(0, 1, 11)  # 11 values from 0 to 1

  # Create column names for results DataFrames
  col_names = [f"{g} {v}" for g in protected_cols for v in groups[g].unique()]

  # Initialize DataFrames to store results
  accuracies = pd.DataFrame(columns=col_names)
  f1_scores = pd.DataFrame(columns=col_names)
  positive_rates = pd.DataFrame(columns=col_names)

  y_preds = []

  # Loop over lambda values
  for l in lambda_values:

      y_preds = cross_validator(model, X, y, protected_cols=protected_cols, debias='fairPCA', lambd=l)

      # Initialize lists to store metrics
      acs = []
      fs = []
      pos = []
      all_true_positive = calculate_true_positives(y_preds)

      # Loop over protected columns and calculate the accuracies, f1-scores and positive rates.
      for g in protected_cols:
        for j in range(groups[g].nunique()): # I don't think the order of j is correct.. but I don't know if it matters
          acs.append(balanced_accuracy_score(y[groups[g]==j], y_preds[groups[g]==j]))
          fs.append(f1_score(y[groups[g]==j], y_preds[groups[g]==j]))
          pos.append(np.mean(y_preds[groups[g]==j]))

      # Concatenate metrics to result DataFrames
      accuracies = pd.concat([accuracies, pd.DataFrame([acs], columns=col_names)])
      f1_scores = pd.concat([f1_scores, pd.DataFrame([fs], columns=col_names)])
      positive_rates = pd.concat([positive_rates, pd.DataFrame([pos], columns=col_names)])

      print(f'Lambda {l:.1f} completed')

  # Set lambda values as index for result DataFrames
  accuracies.index = lambda_values
  f1_scores.index = lambda_values
  positive_rates.index = lambda_values
  return lambda_values, accuracies, f1_scores, positive_rates

## Code taken from assignment 1

def make_df(y_test, y_pred, group_test):
    """
    Generate a DataFrame for computing fairness metrics.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        group_test (array-like): Grouping variable.

    Returns:
        DataFrame: DataFrame with predicted labels ('S'), grouping variable ('G'), and true labels ('A').
    """
    df = pd.DataFrame(y_pred, columns=['S'])
    df['G'] = group_test
    df['A'] = y_test
    return df


def equal_odds(df, target=1, groups=None):
    """
    Calculates equal odds for a specified target outcome and multiple groups.

    Parameters:
        df (DataFrame): DataFrame containing predicted labels ('S'), true labels ('A'), and grouping variable ('G').
        target (int, optional): Target outcome. Defaults to 1.
        groups (array-like, optional): List of groups. Defaults to None.

    Returns:
        list: List of equal odds scores for each group.
    """
    return [sum(df[(df['G'] == g) & (df['A'] == target)].S) / df[(df['G'] == g) & (df['A'] == target)].shape[0] for g in groups]


def accuracy(df, groups=None):
    """
    Calculates balanced accuracy for multiple groups.

    Parameters:
        df (DataFrame): DataFrame containing predicted labels ('S'), true labels ('A'), and grouping variable ('G').
        groups (array-like, optional): List of groups. Defaults to None.

    Returns:
        list: List of balanced accuracy scores for each group.
    """
    return [balanced_accuracy_score(df[(df['G'] == g) & df['A'].notna()].A, df[(df['G'] == g) & df['A'].notna()].S) for g in groups]

# Names of the fairness metrics
metric_names = ["odds_t0", "odds_t1", "accuracy"]

# Computes fairness metric scores
def metric_scores(y_test, y_pred, group_test, df=None):
    """
    Computes fairness metric scores.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        group_test (array-like): Grouping variable.
        df (DataFrame, optional): DataFrame with predicted labels, true labels, and grouping variable. Defaults to None.

    Returns:
        list: List of fairness metric scores.
    """
    if df is None:
        df = make_df(y_test, y_pred, group_test)

    groups = df['G'].unique()
    return [
            equal_odds(df, target=0, groups=groups),
            equal_odds(df, target=1, groups=groups),
            accuracy(df, groups=groups)]


def metric_df(scores, group_names, names=metric_names):
    """
    Generates a DataFrame of fairness metric scores.

    Parameters:
        scores (list): List of fairness metric scores.
        group_names (list): List of group names.
        names (list, optional): List of fairness metric names. Defaults to ["odds_t0", "odds_t1", "accuracy"] as defined above.

    Returns:
        DataFrame: DataFrame with fairness metric scores, metric names, and group names.
    """

    df_fair_metrics = pd.DataFrame({"score":[], "metric" : [], "group":[]})
    for i in range(len(names)):
        new_metrics = {"score": scores[i], "metric" : [names[i] for _ in range(len(group_names))], "group":group_names}
        df_fair_metrics = pd.concat([df_fair_metrics, pd.DataFrame(new_metrics)], ignore_index=True)

    return df_fair_metrics

def corr_mat(X, method='pearson'):
    """
    Compute correlations and their statistical significance between all features in a DataFrame.
    The code is adapted from solution to df debias exercises

    Parameters:
        X (DataFrame): Input DataFrame.
        method (str, optional): Method for correlation calculation. Options: 'pearson' or 'spearman'. Defaults to 'pearson'.

    Returns:
        tuple of arrays: Correlation matrix and p-values matrix.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Compute correlation matrix
    n_features = X.shape[1]

    corr_ = np.zeros((n_features, n_features))
    p_ = np.zeros((n_features, n_features))

    if method == 'pearson':
        corr_func = pearsonr
    elif method == 'spearman':
        corr_func = spearmanr
    else:
        raise ValueError("Unsupported correlation method. Supported methods are 'pearson' and 'spearman'.")

    for i in range(n_features):
        for j in range(n_features):
            corr_[i,j], p_[i,j] = corr_func(X[:,i], X[:,j])
            corr_ = np.nan_to_num(corr_, 0)
            # Handle NaN values in correlation coefficient by setting p-value to 1
            if np.isnan(corr_[i,j]):
                p_[i,j] = 1

    return corr_, p_

def plot_corr(df, method='pearson', feature_cols=[], show_specific_features=False, num_corr=None, figsize=(8,6)):
    """
    Plot correlation matrix heatmap.

    Parameters:
        df (DataFrame): Input DataFrame.
        method (str, optional): Method for correlation calculation. Options: 'pearson' or 'spearman'. Defaults to 'pearson'.
        feature_cols (list, optional): List of feature columns. Defaults to [].
        show_specific_features (bool, optional): Whether to show correlations with specific features. Defaults to False.
        num_corr (int, optional): Number of highest correlations to show with specific features. Defaults to None.
          If the length of feature_cols != 1 or show_specific_features is False, then it will show all features

    Example use:
        plot_corr(features_full, feature_cols=['Dropout', 'Gender'], show_specific_features=True)
        # Will give a vertical plot showing gender's and dropout's correlation to all other features.

        plot_corr(features_full, feature_cols=['Gender'], show_specific_features=True, num_corr=10)
        # Will give a horizontal bar plot showing the 10 most correlated features to Gender.

        plot_corr(features_full)
        # Will show all the correlations.

    """
    corr_name = ""
    if method == 'pearson':
        corr_name = "Pearson's "
    elif method == 'spearman':
        corr_name = "Spearman's "
    else:
        raise ValueError("Unsupported correlation method. Supported methods are 'pearson' and 'spearman'.")

    corr, p = corr_mat(df, method)
    feature_idx = [df.columns.get_loc(col) for col in feature_cols]
    

    alpha = 0.05 # Significance level
    corrected_alpha = alpha / ((df.shape[1]**2)/2) #bonferronni correction

    _, ax = plt.subplots(1,1, figsize=figsize)

    if show_specific_features and len(feature_cols)==1 and num_corr:
      # make bonferroni correction!
      for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
          if p[i,j] > corrected_alpha:
            corr[i,j] = 0
         
      corr_spec_feature = corr[feature_idx[0]]

      features_sorted_spec_feature = [ y for x,y in sorted(zip(corr_spec_feature, df.columns), key = lambda x: abs(x[0]), reverse = True)]
      corr_sorted_spec_feature = [ x for x,y in sorted(zip(corr_spec_feature, df.columns), key = lambda x: abs(x[0]), reverse = True)]


      features_num_corr_spec_feature = features_sorted_spec_feature[1:(num_corr+1)]
      corr_num_corr_spec_feature = corr_sorted_spec_feature[1:(num_corr+1)]

      # Plot 10 highest correlations with specific feature
      x_pos = np.arange(len(features_num_corr_spec_feature))

      colors = []
      for i in corr_num_corr_spec_feature:
          if i > 0:
              colors.append("#ffcea5")
          else:
              colors.append("#c9ffc7")

      for i in range(len(x_pos)):
          ax.bar(x_pos[i], corr_num_corr_spec_feature[i], color = colors[i], align='center')

      ax.set_xticks(x_pos, [x.replace("_", " ").title() for x in features_num_corr_spec_feature], rotation = 60, ha = 'right')
      ax.set_xlabel("Feature", fontsize = 12)
      ax.set_ylabel('Correlation', fontsize = 12)
      ax.set_title(f'All p-values are significant (lower than {alpha}%)', fontsize = 12)
      plt.suptitle(f'{feature_cols[0].replace("_", " ").title()}: {num_corr} highest {corr_name}correlations', fontsize = 14)

    else:
      if show_specific_features:

        sns.heatmap(corr[:, feature_idx], cmap="coolwarm", 
                    xticklabels=[x.replace("_", " ").title() for x in df.columns[feature_idx]], 
                    yticklabels=[x.replace("_", " ").title() for x in df.columns],
                    mask = p[:, feature_idx] > corrected_alpha, 
                    vmin=-1, vmax=1, square=True, ax=ax)
      else:
        sns.heatmap(corr, cmap="coolwarm",
                    xticklabels=[x.replace("_", " ").title() for x in  df.columns], yticklabels=[x.replace("_", " ").title() for x in  df.columns],
                    square=True, vmin=-1, vmax=1, mask= p > corrected_alpha,ax=ax)

      ax.set_title(f"{corr_name}Correlation Coeff between all features (filtered by p > {alpha})")
    
    plt.tight_layout()
    plt.show()

def plot_feature(feature_name, features_full, 
                 column_feature='graduated', order=[], 
                 colors=['#c9ffc7', '#ffcea5'], show_ratio=True, 
                 labels=['No', 'Yes'], figsize=(7,5),
                 save=False, file_name='img/default'):
  """
  Plot distribution of a feature and its relationship with another binary column feature.

  Parameters:
      feature_name (str): Name of the feature to plot.
      feature_mapping (dict): Mapping dictionary to encode feature values.
      features_full (DataFrame): Full DataFrame containing the features.
      column_feature (str, optional): Binary column feature. Defaults to 'Dropout'.
      colors (list, optional): Colors for plotting. Defaults to ['#c9ffc7', '#ffcea5'].
      show_ratio (bool, optional): Whether to show the ratio of 'Yes' to 'No' in column_feature. Defaults to True.
      labels (list, optional): Labels for column_feature values. Defaults to ['No', 'Yes'].

  Example:
      plot_feature(feature_name='Course', feature_mapping=course_names, features_full=features_full, column_feature='Gender', colors=['pink', 'skyblue'], labels=['Female', 'Male'], show_ratio=False)
      # Makes a bar plot showing distribution of Courses by Gender

      plot_feature("Father's qualification", qualification_groups, features_full)
      # Makes a barplot showing distribution of Father's qualification and Dropout. The ratio is also shown.
  """
  # Data
  tmp_full_features = features_full.copy()
  counts = tmp_full_features.groupby([feature_name, column_feature]).size().unstack(fill_value=0)

  # Calculate column_feature ratio
  if show_ratio:
    counts[f'{column_feature} Ratio'] = counts[1] / (counts[0] + counts[1])

  # Plotting
  _, ax1 = plt.subplots(figsize=figsize)

  # Plotting column_feature counts
  bar_width = 0.35
  positions = np.arange(len(counts))
  counts.reset_index(inplace=True)
  mapping = {cor: i for i, cor in enumerate(order)}
  key = counts[feature_name].map(mapping)
  counts = counts.iloc[key.argsort()]

  ax1.bar(positions - bar_width/2, counts[1], width=bar_width, color=colors[1], label=labels[1])
  ax1.bar(positions + bar_width/2, counts[0], width=bar_width, color=colors[0], label=labels[0])
  ax1.set_ylabel('Number of Students', color='black')
  ax1.tick_params(axis='y', labelcolor='black')
  ax1.set_xticks(positions)
  ax1.set_xticklabels([x.replace('_', ' ').title() for x in counts[feature_name]], rotation=60, ha='right')
  ax1.legend(title=column_feature, loc='upper left')

  # Creating a second y-axis for column_feature ratio
  if show_ratio:
    ax2 = ax1.twinx()
    ax2.plot(positions, counts[f'{column_feature} Ratio'], color='red', marker='o', linestyle='-', label='_')
    ax2.set_ylabel(f"Proportion {column_feature.replace('_', ' ').title()}", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f"Distribution of {feature_name.replace('_', ' ').title()} and {column_feature.replace('_', ' ').capitalize()} Ratio")
    plt.ylim(0,1)
  else:
    plt.title(f"Distribution of {feature_name.replace('_', ' ').title()} by {column_feature.replace('_', ' ').capitalize()}")
  
  plt.tight_layout()
  ax1.legend(title=f"{column_feature.replace('_', ' ').title()}")
  if save:
     plt.savefig(f'img/{file_name}.png', transparent=True)

def plot_scores_and_group(df, title=None):
  """
  Plot scores by metric and group from a DataFrame.

  Parameters:
      df (DataFrame): Input DataFrame containing scores, metrics, and groups. Can be obtained by using metric_scores() and metric_df()
      title (str, optional): Title for the plot. Defaults to None, setting the title to 'Scores by Metric and Group'.

  Example:
      plot_scores_and_group(df)
  """
  # Plotting
  fig, ax = plt.subplots(figsize=(10, 6))
  #groups = df['group'].unique()
  metrics = df['metric'].unique()
  bar_width = 0.35
  index = range(len(metrics))

  color_palette = sns.color_palette("pastel", n_colors=len(df['group'].unique()))


  for i, group in enumerate(df['group'].unique()):
      values = df[df['group'] == group]['score']
      ax.bar([x + i * bar_width for x in index], values, bar_width, label=group, color=color_palette[i])

  ax.set_xlabel('Metric')
  ax.set_ylabel('Score')
  if title:
    ax.set_title(title)
  else:
    ax.set_title('Scores by Metric and Group')
  ax.set_xticks([x + bar_width for x in index])
  ax.set_xticklabels(metrics)
  ax.legend()

  plt.tight_layout()
  plt.show()

def plot_scores_and_group_compare(dfs=[], titles=None, suptitle='', color_palette=None, bar_width=0.35, figsize=(15, 6)):
    """
    Plot and compare scores by metric and group from two DataFrames.

    Parameters:
        df1 (DataFrame): First input DataFrame containing scores, metrics, and groups. Can be obtained by using metric_scores() and metric_df()
        df2 (DataFrame): Second input DataFrame containing scores, metrics, and groups. Can be obtained by using metric_scores() and metric_df()
        titles (list, optional): Titles for the plots. Defaults to None, setting it to 'Scores by Metric and Group'-
        color_palette (list, optional): Color palette for the plots. Defaults to None, setting it to 'pastel'.
        bar_width (float, optional): Width of the bars in the plot. Defaults to 0.35.

    Example:
        plot_scores_and_group_compare(gender_log_metric_df, gender_RF_metric_df, titles=['Logistic Regression Gender','Random Forest Classification Gender'], color_palette=['pink', 'skyblue'])
    """

    # Plotting
    fig, axes = plt.subplots(1, len(dfs), figsize=figsize, sharey=True)  # Create subplots side by side ####################

    # Iterate over DataFrames
    for i, (ax, df) in enumerate(zip(axes, dfs)):
        metrics = df['metric'].unique()
        index = range(len(metrics))

        if not color_palette:
          color_palette = sns.color_palette("pastel", n_colors=len(df['group'].unique()))


        for j, group in enumerate(df['group'].unique()):
            values = df[df['group'] == group]['score']
            ax.bar([x + j * bar_width for x in index], values, bar_width, label=group, color=color_palette[j])

        ax.set_xlabel('Metric')
        if titles:
            ax.set_title(titles[i])
        else:
            ax.set_title('Scores by Metric and Group')

        ax.set_xticks([x + bar_width for x in index])
        ax.set_xticklabels(metrics)

    # Set common y-label and legend
    axes[0].set_ylabel('Score')
    axes[-1].legend(title='Group', loc='upper right', bbox_to_anchor=(1.47,1))
    plt.suptitle(suptitle)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

def tune_hyper_params(features, model, fixed_model_params, param_to_tune, param_values, protected_cols=[], labels=[]):
  """
  Tune hyperparameters of a model and visualize their effect on balanced accuracy.
  3 plots are printed: The influence on the model trained on 'original df', 'nonprotected features' and 'debiased df' respectively.

  Parameters:
      features (DataFrame): Input features DataFrame.
      model (class): Model class to be used for training.
      fixed_model_params (dict): Fixed parameters for the model.
      param_to_tune (str): Name of the hyperparameter to tune.
      param_values (list): List of values for the hyperparameter.

  Example:
      tune_hyper_params(features=features, model = LogisticRegression, param_to_tune='C', fixed_model_params = {'random_state':seed, 'max_iter':1000}, param_values = np.linspace(start=0.01,stop=5,num=25))
  """
  def get_model(value):
    fixed_model_params[param_to_tune] = value
    return model(**fixed_model_params)

  debiased_data_np, data_np,_ = get_pca_data(features, features, protected_cols) ## Use FairPCA

  descriptions = ['original df', 'nonprotected features', 'debiased df']
  datasets = [features, data_np, debiased_data_np]
  for data_X, desc in zip(datasets, descriptions):
    accuracies = []
    for v in param_values:
      model_with_params = get_model(v)
      preds = cross_validator(model_with_params, data_X, labels, protected_cols=protected_cols)
      acc = balanced_accuracy_score(preds, labels)
      accuracies.append(acc)
    
    plt.ylim(0,1)
    plt.plot(param_values, accuracies)
    plt.title(f'Accuracy at different values of {param_to_tune} for {desc}')
    plt.xlabel(f'{param_to_tune}')
    plt.ylabel('Accuracy')
    plt.show()

def plot_tune_lambda(lambda_values, accuracies, f1_scores, positive_rates):
  """
  Plot the effect of lambda values on accuracy, F1 scores, and positive rates.

  Parameters:
      lambda_values (array): Array of lambda values.
      accuracies (DataFrame): DataFrame containing accuracies for different lambda values.
      f1_scores (DataFrame): DataFrame containing F1 scores for different lambda values.
      positive_rates (DataFrame): DataFrame containing positive rates for different lambda values.

  Example:
      plot_tune_lambda(lambda_values, accuracies, f1_scores, positive_rates)
  """
  _, ax = plt.subplots(3,1, figsize=(5, 10), sharex=True)
  titles = ['Accuracies', 'F1 scores', 'Positive rates']

  for i, df in enumerate([accuracies, f1_scores, positive_rates]):
      for j in range(len(accuracies.columns)):
          ax[i].plot(df.iloc[:,j], label=df.columns[j])
          ax[i].set_title(f'{titles[i]}')

  ax[1].legend(bbox_to_anchor=(1,1))

  plt.xticks(lambda_values, labels=[round(x,1) for x in lambda_values])
  plt.xlabel('lambda value')
  plt.show()

## Taken from assignment 1
def feature_weights(features, model, top_n=10, protected_cols=[], labels=[]):
  """
  Get the feature weights of a model and their odds ratios.

  Parameters:
      features (DataFrame): Input features DataFrame.
      model (object): Trained model object.
      top_n (int, optional): Number of top features to return. Defaults to 10.

  Returns:
      list: List of DataFrames containing feature weights and odds ratios for different datasets.
            The first df of the list is the weights of the model trained on the original df,
            the second df contains the weights of the model trained using nonprotected features, and
            the third df contains the weights of the model trained using debiased df

  Example:
      weights_orgin_data, weights_np, weights_debiased = feature_weights(features=features, model=LogisticRegression(max_iter=1000, random_state=seed, C=0.05), top_n=10)
  """

  debiased_data_np, data_np, pca = get_pca_data(features, features, protected_cols) ## Use FairPCA

  descriptions = ['original df', 'nonprotected features', 'debiased df']
  datasets = [features, data_np, debiased_data_np]

  weights_dfs = []

  for dataset in datasets:
    model.fit(dataset,labels)
    odds = pd.DataFrame(model.coef_[0], dataset.columns, columns=['coef'])
    odds['odds'] = np.exp(odds.coef)
    odds['abs_odds'] = abs(odds.odds)
    odds = odds.sort_values(by='abs_odds', ascending=False).drop('abs_odds', axis=1)
    odds = odds.round(decimals=3)

    weights_dfs.append(odds.head(top_n))

  return weights_dfs, pca