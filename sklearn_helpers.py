import numpy as np
import xarray as xr
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    _OneToOneFeatureMixin,
    _ClassNamePrefixFeaturesOutMixin
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class normalize_column_modality(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """ Transformer that scales categorical data for Factor Analysis of Mixed Data (FAMD).
    Parameters
    ----------
    center : bool, default=True
        Indicates if the normalized column modalities should be centered to zero mean.
    accept_sparse : bool, default=False
        The class cannot currently accept sparse matrices.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`. Unused.
    weight_ : float
        Weight of each column applied in :meth:`fit`.
    X_means_ : float
        Mean of each column after weighting. Used to center a column in :meth:`fit`.

    """
    def __init__(self, center=True, accept_sparse=False):
        self.center = center
        self.accept_sparse = accept_sparse
        
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in fit
        if hasattr(self, "weight_"):
            del self.weight_
            del self.n_features_in_
            del self.X_means_

    def fit(self, X, y=None):
        """Compute the column weights and centering by mean removal.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=self.accept_sparse)
        self._reset()
        self.n_features_in_ = X.shape[1]

        length = X.shape[0]
        self.weight_ = (np.sqrt(X.sum(axis=0) / length))        
        self.X_means_ = (np.divide(X, self.weight_)).mean(axis=0)

        # Return the transformer
        return self

    def transform(self, X):
        """ Weight the columns and center them.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_in_')

        # Input validation
        X = check_array(X, accept_sparse=self.accept_sparse)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        X = np.divide(X, self.weight_)
        
        if self.center:
            X = X - self.X_means_
        
        return X


def gridsearch_results_to_xarray(df, dim_reduction_list=None, scorers=None):
    # Apparent gaps in the rank of a score are a result of multiple combinations generating
    # identical scores -- the rank of a metric are not unique values and cannot be reliably
    # used as a dimension for labeling the data.
    
    # if type(scorers) is not list:
    #     scorers = [scorers]

    if dim_reduction_list is None:
        dim_reduction_list = []
    dim_red_fill = [None]
    
    if scorers is None:
        score_dim = 'rank_test_score'
        scorers = ['test']
    else:
        score_dim = ['rank_test_' + s for s in scorers]

    ds = xr.Dataset.from_dataframe(df)
    dv_to_dim = [dv for dv in ds.data_vars if 'param_' in dv]
    dv_to_dim.extend(score_dim)
    ds = ds.set_coords(dv_to_dim).drop(['params'])
    
    # param_rename = {p: p.replace('param_', '') for p in ds.coords if 'param_' in p}
    # ds = ds.rename(param_rename)

    for score in scorers:
        list_splits = [datavar for datavar in ds.data_vars if 'split' in datavar and score in datavar]
        var_splits = [split for split in ds.reset_coords()[list_splits].values()]

        ds['split_test_{}'.format(score)] = xr.DataArray(
            var_splits,
            dims=['split', 'index'],
            coords={
                'split': np.arange(0, len(list_splits)),
                'index': ds['index'],
            }
        )
        ds = ds.drop(list_splits)
    
    # Parameters that identify steps that are turned on or off can be identified by
    # not having a double-underscore. The data model from the results dictionary has
    # confusing logic that it is nan when it used and "passthrough" when it not used.
    # This creates two issues: (1) it is a mixed data type that doesn't play well with
    # conversions to better data models like pandas and xarray. (2) a value of nan is
    # not inuitive for a True value.
    
    # Both of these issues are fixed below, but note it uses some janky logic that
    # may break.
    identifier_vars = [c for c in ds.coords if '__' not in c and 'param_' in c]
    for idv in identifier_vars:
        ds[idv] = xr.where(ds[idv]=='passthrough', False, True)
    
    # Variable indicating which dim reduction was used
    ds['dim_reduction'] = (('index'), dim_red_fill * len(ds['index']))
    
    for dr in dim_reduction_list:
        if 'No dim reduction' not in dr:
            dim_red_param_name = 'param_' + dr
            ds['dim_reduction'] = xr.where(ds[dim_red_param_name] == True, dr, ds['dim_reduction'])
    
    # Handle the case with no dimensionality reduction
    ds['dim_reduction'] = xr.where(
        ds['dim_reduction'] == dim_red_fill,
        'No dim reduction',
        ds['dim_reduction']
    )

    return ds


def dataset_filter(ds, filters=None, TAT_threshold=100):
    '''
    Filters the mongodb xarray Dataset according to commonly used
    criteria for the performance model.
    '''
    default_filters = [
        'real_workers',
        'symmetric_bic',
        'few_dropped_workers',
        'reasonable_TAT',
        'v14',
        'v16.8',
        'rgmc_rotation_check',
        'mbw_version_2x',
        'only_newest_rasterizer',
        'wcu_off',
    ]
    
    if filters is None:
        filters = default_filters
    
    for f in filters:
        if f not in default_filters:
            raise ValueError('{} is not an option.'.format(f))
    
    if 'real_workers' in filters: 
        ds = ds.where(ds.worker_type == 'real', drop=True)
    if 'symmetric_bic' in filters:
        ds = ds.where(ds.bic_x == ds.bic_y, drop=True)
    if 'few_dropped_workers' in filters:
        ds = ds.where(ds.max_number_of_rasterizer_workers > 300, drop=True)
    if 'reasonable_TAT' in filters:
        ds = ds.where(ds.TAT_monte_carlo_23104_scaled < TAT_threshold, drop=True)
    if 'v14' in filters:
        ds = ds.where(ds.sw_version == 'v14', drop=True)
        if 'only_newest_rasterizer' in filters:
            ds = ds.where(ds.rasterizer_version == '14.0.0.7ecf84be', drop=True)
    if 'v16.8' in filters:
        ds = ds.where(ds.sw_version == 'v16.8', drop=True)
        if 'only_newest_rasterizer' in filters:
            ds = ds.where(
                (ds.rasterizer_version.isin(['16.8.0.4c46bd92', '16.8.0.7e382c9e']))
                & (ds.emet_profile.isin(['emet-07.00.03', 'emet-07.00.02-hotfix-220218', 'emet-07.00.02-hotfix-22021'])),
                drop=True)
    if 'rgmc_rotation_check' in filters:
        ds = ds.where(~np.isnan(ds.rgmc_rotation_mrad), drop=True)
    if 'mbw_version_2x' in filters:
        ds = ds.where(ds.mbw_version.isin(['2.1', '2.2']), drop=True)
    if 'wcu_off' in filters:
        ds = ds.where(ds.wcu == 0, drop=True)
    
    return ds


def column_transformer_builder(var_types, one_hot_drop='first',
                               add_imputer=False):
    categorical_data = [
        col for col, col_type in var_types.items() if 'categorical' in col_type
    ]
    numerical_data = [
        col for col, col_type in var_types.items() if 'numerical' in col_type
    ]
    
    categorical_transformer = Pipeline(
        steps=[
            ("one_hot", OneHotEncoder(
                sparse=False, drop=one_hot_drop, handle_unknown='ignore')),
            ("normalize_column_modality", normalize_column_modality(center=True)),
        ]
    )

    if add_imputer:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
    else:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_data),
            ("categorical", categorical_transformer, categorical_data),
        ],
        sparse_threshold=0
    )
    
    return preprocessor

