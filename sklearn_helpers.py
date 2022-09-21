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


class NormalizeColumnModality(
    _OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """ Transformer that scales categorical data as in Factor Analysis of
    Mixed Data (FAMD).
    Parameters
    ----------
    center : bool, default=True
        Indicates if the normalized column modalities should be centered to
        have zero mean.
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


def column_transformer_builder(var_types, one_hot_drop='first',
                               add_imputer=False):
    """Builds the column transformer based on user defined values.

    The column transformer creates separate pipelines for categorical data
    and numerical data. The critical difference between this function and
    other similar ones is that the data type is specified by the user,
    instead of being inferred. This feature allows specifying data that are
    stored as numeric types to be processed as categorical data.

    Parameters
    ----------
    var_types : Dictionary of length n_features
        Indicates if a column should be treated as a categorical or
        numerical feature.
    one_hot_drop : string
        Keyword passed to OneHotEncoder
    add_imputer : bool
        Indicates if an imputer should be added to the column transformer.
    Returns
    -------
    preprocessor : sklearn pipeline

    """

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
            ("normalize_column_modality", NormalizeColumnModality(center=True)),
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

