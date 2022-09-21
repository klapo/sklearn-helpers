import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def tat_evaluation_dataset(results, models_to_plot, rank_var=None):
    '''For the input models, select the best ranked configuration for each
    and concatenate together.
    '''
    if rank_var is None:
        rank_var = 'rank_test_R2'
    
    ds_combined = xr.Dataset()
    model_config = []
    for run_name, model_name in models_to_plot.items():
        # Select the best ranked model iteration.
        model_config.append(model_name)
        ds_plot = results[run_name]
        
        # Model configurations can be degenerate. Force return a single
        # model configuration.
        ds_plot = ds_plot.where(
            ds_plot[rank_var]==1, drop=True).reset_coords().isel(index=0)
        # ds_plot = ds_plot.squeeze('index')
        
        # Select the variables to pass on.
        ds_plot = ds_plot[
            [
                'observed_TAT', 'predicted_TAT', 'train_test',
                'predicted_TAT_percent_error', 'MAPE_test',
                'mean_test_MAPE', 'std_test_MAPE', 'R2_test',
                'mean_test_R2', 'std_test_R2', 'split_test_R2',
                'split_test_MAPE', 'R2_train', 'MAPE_train',
            ]
        ]
        
        # This variable indicates the model configuration and train/test split
        # for each job. Useful for the concatenated result.
        ds_plot['model, split'] = (
            'job',
             model_name + ', ' + ds_plot.train_test.values
        )
        ds_combined = xr.concat([ds_combined, ds_plot], dim='model_config')

    ds_combined.coords['model_config'] = pd.Index(model_config, name='model_config')
    return(ds_combined)


def tat_eval_scatterplot(results, models_to_plot, suptitle=None, 
                         rank_var=None, fig_axes=None, sns_kwargs=None):
    if suptitle is None:
        suptitle = 'Predicted TATs from best estimator from GridSearchCV'
    
    ds_combined = tat_evaluation_dataset(
        results, models_to_plot, rank_var=rank_var
    )
    if fig_axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig = fig_axes[0]
        axes = fig_axes[1]
    
    if sns_kwargs is None:
        sns_kwargs = {}
    sns_kwargs['s'] = sns_kwargs.get('s', 20)
    sns_kwargs['palette'] = sns_kwargs.get('palette', 'Paired')
    
    ax = axes[0]
    sns.scatterplot(
        data=ds_combined.to_dataframe(),
        ax=ax,
        x="observed_TAT",
        y="predicted_TAT",
        hue="model, split",
        # hue_order=sorted(ds_combined['model, split'].values),
        style="train_test",
        # s=20,
        # palette='Paired',
        **sns_kwargs,
    )
    ax.set_ylabel('Predicted TAT (hours)')
    ax.set_xlabel('Observed TAT (hours)')
    ax.plot([0, 100], [0, 100], 'k--')

    ax = axes[1]
    sns.scatterplot(
        data=ds_combined.to_dataframe(),
        ax=ax,
        x="observed_TAT",
        y="predicted_TAT_percent_error",
        hue="model, split",
        style="train_test",
        s=20,
        palette='Paired',
        legend=False,
    )
    ax.axhline(0, color='0.5', ls='--')
    ax.set_ylabel('Predicted bias (%)')
    ax.set_xlabel('Observed TAT (hours)')
    
    # lots and lots of assumptions in these quantities
    font_size = sns.plotting_context()['font.size']
    if font_size > 14:
        text_box_height = 0.15
    elif font_size <= 14:
        text_box_height = 0.1
    for nmodel, model in enumerate(models_to_plot.values()):
        ds_print = ds_combined.sel(model_config=model)
        error_str = (
            '{}, MAPE:\n'
            'Test={:.1f}%\n'
            'CV={:.1f}%+/-{:.1f}%\n'.format(
                model,
                ds_print.MAPE_test.values * 100,
                ds_print.mean_test_MAPE.values.item() * 100,
                ds_print.std_test_MAPE.values.item() * 100,
            )
        )
        color = sns.color_palette('Paired')[nmodel * 2]

        ax.text(0.05, 0.9 - text_box_height * nmodel, error_str,
                transform=ax.transAxes,
                bbox=dict(facecolor=color, alpha=0.5))

    fig.suptitle(suptitle)
    fig.tight_layout()
    
    return ds_combined


def melt_model_results(results, models_to_plot, eval_vars=None):
    ds_combined = tat_evaluation_dataset(results, models_to_plot)
    df = ds_combined.drop_dims('job')

    if eval_vars is None:
        eval_vars = ['R2', 'MAPE']
    
    df_combined = pd.DataFrame()
    for nvar, eval_var in enumerate(eval_vars):
        df_var = df[
            [
                'split_test_{}'.format(eval_var),
                '{}_test'.format(eval_var),
                '{}_train'.format(eval_var)
            ]
        ]
        df_var = df_var.rename(
            {
                'split_test_{}'.format(eval_var): 'CV',
                '{}_test'.format(eval_var): 'test split',
                '{}_train'.format(eval_var): 'train split',
            }
        )
        # Dimension 0 coords get squeezed, but that is undesirable. For when a single
        # run is pased in, we add the dimension back.
        if 'mode_config' not in df_var.coords:
            df_var.coords['model_config'] = ds_combined.model_config
        
        df_var = df_var.to_dataframe()
        df_var = df_var.reset_index().melt(
            id_vars=['model_config'],
            value_vars=['CV', 'test split', 'train split'],
            var_name='Evaluation type',
            value_name=eval_var
        )
        if nvar > 0:
            df_combined = pd.concat(
                [
                    df_combined,
                    df_var.drop(['Evaluation type', 'model_config'], axis=1)
                ],
                axis=1
            )
        else:
            df_combined = df_var

    return df_combined


def barcharts(results, models_to_plot, suptitle=None, fig_axes=None):
    if suptitle is None:
        suptitle = 'Comparison of cross validation '
    df_combined = melt_model_results(results, models_to_plot, eval_vars=None)
    
    if fig_axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    else:
        fig = fig_axes[0]
        axes = fig_axes[1]
    
    sns.barplot(
        ax=axes[0],
        data=df_combined,
        x='model_config',
        y='MAPE',
        hue='Evaluation type',
    )

    sns.barplot(
        ax=axes[1],
        data=df_combined,
        x='model_config',
        y='R2',
        hue='Evaluation type',
    )
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.suptitle(suptitle, y=1.01)
    
    return df_combined


def cv_splits(results, models_to_plot, rank_var=None, eval_vars=None):
    if rank_var is None:
        rank_var = 'rank_test_R2'
    if eval_vars is None:
        eval_vars = ['R2', 'MAPE']
    num_subplots = len(eval_vars)
    
    fig, axes = plt.subplots(
        num_subplots, 1,
        figsize=(5 * num_subplots, 10), sharex=True)

    for model in models_to_plot.keys():
        ds = results[model]
        ds_best = ds.where(ds[rank_var]==1, drop=True).isel(index=0)
        for nvar, var in enumerate(eval_vars):
            ax = axes[nvar]
            ax.plot(
                ds_best.split,
                ds_best['split_test_{}'.format(var)],
                label=models_to_plot[model],
            )
            ax.set_ylabel('{} for each CV split'.format(var))
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('CV split')
                ax.legend()

    fig.tight_layout()
    

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ax=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate test and training learning curves. Directly lifted from sklearn's examples.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    ax : array-like of shape (1,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    score_dict = {
        'neg_mean_absolute_percentage_error': 'mean abs error (%)',
        'r2': 'correlation (-)'
    }
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    if isinstance(scoring, str):
        score_text = score_dict[scoring]
        ax.set_ylabel(score_text)
    else:
        ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    if isinstance(scoring, str) and scoring == 'neg_mean_absolute_percentage_error':
        train_scores = train_scores * -1
        test_scores = test_scores * -1

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    
    return plt, [train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std]