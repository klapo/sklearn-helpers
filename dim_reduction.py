import copy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.pipeline import Pipeline

from sklearn_helpers import column_transformer_builder


def combine_pca_original_data(
    sample_pca,
    pca,
    pca_index,
    df_original,
    vars_column_type,
    component_name=None,
):
    """Combines the dimensionality results (from sklearn) with the original data.

    Written for pca, but should be applicable to any sklearn dimensionality reduction
    method that has the `n_components` attribute.

    Parameters
    ----------
    sample_pca : numpy array
        Original data projected into the reduced dimensional space, e.g., result from
        pca.transform(X).
    pca : sklearn dim red object
        The result from the `fit` or `fit_transform` operation.
    df_original : pandas DataFrame
        Original data used to fit the dimensionality reduction.
    vars_column_type : dict

    component_name : str, optional
        Name to prepend to the dimension name (i.e., "{component name} dim {n}")

    Returns
    -------
        df_combined : pandas DataFrame
    """
    # Build a Dataframe of the pca results with each column labeled by the
    # dimension number.
    df_sample_pca = pd.DataFrame(
        sample_pca,
        columns=pd.Series(
            ['{} dim {}'.format(component_name, n) for n in np.arange(0, pca.n_components)],
            index=np.arange(0, pca.n_components),
        )
    )
    # Restore the index to be the same as the original data.
    df_sample_pca.index = pca_index

    df_combined = pd.concat([df_sample_pca, df_original], axis=1)

    # For the categorical variables to be strings so that seaborn plays nicely.
    for k, v in vars_column_type.items():
        if v == "categorical":
            df_combined[k] = df_combined[k].astype(str)
    return df_combined


def prepare_mixed_data_for_dim_reduction(ds, dict_datavars_type):
    datavars_list = list(dict_datavars_type.keys())
    ds = ds[datavars_list]
    ds = ds.dropna(dim='job', how='any')
    df = ds.to_dataframe()

    numeric_cols = [v for v in datavars_list if
                    dict_datavars_type[v] == 'numerical']
    cat_cols = [v for v in datavars_list if
                dict_datavars_type[v] == 'categorical']

    # numeric process
    normalized_df = normalize_data(df, numeric_cols)
    normalized_df = normalized_df[numeric_cols]

    # categorical process
    cat_one_hot_df, one_hot_cols = one_hot_encode(df, cat_cols)
    cat_one_hot_norm_df = normalize_column_modality(
        cat_one_hot_df, one_hot_cols
    )
    cat_one_hot_norm_center_df = center_columns(
        cat_one_hot_norm_df, one_hot_cols
    )

    # Merge DataFrames
    processed_df = pd.concat(
        [normalized_df, cat_one_hot_norm_center_df], axis=1
    )

    return processed_df


def methods_generator(df_original, var_types, dim_red_methods=None):
    """Generate a pipeline of preprocessor and dimensionality reduction.

    Preprocessor pipeline is the ColumnTransformer and normalization that
    depends on the data type.

    Parameter
    ---------
    df_original : pandas DataFrame
        Data used for performing the dimensionality reduction. Columns
        should match `var_types`.
    var_types : dict
        Keys are variables to be included in the pipeline. Items are
        either "categorical" or "numerical" indicating which pipeline
        is applied to the variable.
    dim_red_methods : list of tuples, optional
        Default=[
            ('pca', sklearn.decomposition.PCA),
            ('kernel_pca', sklearn.decomposition.KernelPCA),
            'factor_analysis', sklearn.decomposition.FactorAnalysis]
        First item is the pipeline name for the method and the 2nd item
        is an sklearn.decomposition-like method with a `.fit_transform()`
        functionality.

    Returns
    -------
    df_pipeline : pandas DataFrame
        Original data with the dimensionality reduction results appended.
    feature_names : list of strings
        Feature names from the transformations.
    collected_dim_red : dict
        Keys are dimensionality reduction names, Items are the actual
        sklearn.decomposition methods.
        Useful for plotting/evaluation, e.g., in `pca_contribution_plot`
    """
    preprocessor = column_transformer_builder(var_types)

    # Instantiate dim reduction methods
    if dim_red_methods is None:
        pca = PCA(n_components=9)
        kernel_pca = KernelPCA(
            kernel="rbf", n_components=9, gamma=0.01
        )
        fa = FactorAnalysis(n_components=9, rotation='varimax')
        dim_red_methods = [
            ('pca', pca),
            ('kernel_pca', kernel_pca),
            ('factor_analysis', fa),
        ]

    collected_pipelines = []
    collected_dim_red = {}
    for method_name, method in dim_red_methods:
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                (method_name, method)
            ]
        )
        collected_pipelines.append([method_name, method, pipeline])
        collected_dim_red[method_name] = method

    df_pipeline = copy.deepcopy(df_original)
    for method_name, method, dim_red in collected_pipelines:
        reduced_data = dim_red.fit_transform(df_pipeline)

        df_pipeline = combine_pca_original_data(
            reduced_data,
            method,
            df_pipeline.index,
            df_pipeline,
            var_types,
            component_name=method_name,
        )

    # Feature names should be identical between methods
    feature_names = collected_pipelines[0][-1][0].get_feature_names_out()
    for n, f in enumerate(feature_names):
        feature_names[n] = f.split('__')[-1]

    return df_pipeline, feature_names, collected_dim_red


def factor_analysis_explained_variance(fa):
    '''
    The variance explained by each latent variable and the estimated noise of
    this latent feature.

    I went through the math and believe it should be correct, but use at your own risk.
    It is possible that the underlying assumptions do not support this approach.

    Literature:
    The wikipedia article is surprisingly direct to follow:
    https://en.wikipedia.org/wiki/Factor_analysis#Confirmatory_factor_analysis

    And the stack overflow that sent me down the rabbit hole:
    https://stackoverflow.com/questions/41388997/factor-analysis-in-sklearn-explained-variance
    '''
    m = fa.components_
    n = fa.noise_variance_

    m1 = m**2

    m2 = np.sum(m1, axis=1)

    # Fraction of variance explained by each component
    explained_variance = m2
    explained_variance_ratio = m2 / np.sum(m2)

    # Fraction of variance explained by each component, including the
    # estimate of "noise"
    explained_variance_ratio_with_noise = m2 / (np.sum(m2) + np.sum(n))

    return explained_variance, explained_variance_ratio, explained_variance_ratio_with_noise
