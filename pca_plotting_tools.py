import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dim_reduction import factor_analysis_explained_variance


def pca_contribution_plot(pca, df_columns, dim1=0, dim2=1, color_dim1='xkcd:blue',
                          color_dim2='xkcd:goldenrod', sup_title=None, highlight_var=None,
                          fig_kwargs=None):
    
    if isinstance(df_columns, pd.DataFrame):
        df_columns = df_columns.columns

    if highlight_var is not None:
        if highlight_var not in df_columns:
            raise KeyError('{} not found in provided columns.'.format(highlight_var))
    
    # Assume we were passed a pca object
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings_corr = pca.components_.T * np.sqrt(explained_variance)
    explained_variance_ratio_dim1 = pca.explained_variance_ratio_[dim1]
    explained_variance_ratio_dim2 = pca.explained_variance_ratio_[dim2]
    
    # Sort the loadings for each dimension so that they appear in order of
    # importance to the factor/dimension
    sort_index_dim1 = np.argsort(loadings_corr[:, dim1]**2)[::-1].astype(int)
    sort_index_dim2 = np.argsort(loadings_corr[:, dim2]**2)[::-1].astype(int)

    if fig_kwargs is None:
        fig_kwargs = dict()
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), **fig_kwargs)
    axes = axes.flatten()

    ax = axes[0]
    ax.set_xlabel('dimension {}: ({:2.1f} % explained variance)'.format(
        dim1, explained_variance_ratio_dim1 * 100))
    ax.set_ylabel('dimension {}: ({:2.1f} % explained variance)'.format(
        dim2, explained_variance_ratio_dim2* 100))

    for nvar, vts in enumerate(df_columns):
        if vts == highlight_var:
            color='red'
            zorder=len(df_columns) + 1
        else:
            color='black'
            zorder=None
        arrow_props_dict = dict(
            facecolor=color, shrink=0.02,
        )
        
        ax.annotate(
            '',
            xytext=(0, 0),
            xy=(
                loadings_corr[nvar, dim1],
                loadings_corr[nvar, dim2]
            ),
            arrowprops=arrow_props_dict,
            zorder=zorder,
        )

        ax.annotate(
            vts,
            xy=(0, 0),
            xytext=(
                loadings_corr[nvar, dim1],
                loadings_corr[nvar, dim2]
            ),
            zorder=zorder,
            color=color
        )
    ax.set_title('Loadings along each dimension', loc='left')
    ax.add_patch(plt.Circle((0, 0), 1, color='0.5', fill=False))
    ax.axis('equal')
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlim(-1.25, 1.25)

    ax = axes[1]
    ax.set_ylabel('Loading (-)')
    ax.set_ylabel('Loading on dimension {} (-)'.format(dim1 + 1))
    barlist = ax.bar(
        df_columns[sort_index_dim1],
        # loadings_corr[sort_index_dim1, dim1]**2,
        loadings_corr[sort_index_dim1, dim1],
        color=color_dim1
    )
    ax.tick_params(labelrotation=90)
    if highlight_var is not None:
        mask = np.flatnonzero(df_columns[sort_index_dim1] == highlight_var).astype(int)
        barlist[mask[0]].set_color('r')

    ax = axes[2]
    ax.set_ylabel('Loading on dimension {} (-)'.format(dim2 + 1))
    barlist = ax.bar(
        df_columns[sort_index_dim2],
        loadings_corr[sort_index_dim2, dim2],
        # loadings_corr[sort_index_dim2, dim2]**2,
        color=color_dim2
    )
    ax.tick_params(labelrotation=90)
    if highlight_var is not None:
        mask = np.flatnonzero(df_columns[sort_index_dim2] == highlight_var).astype(int)
        barlist[mask[0]].set_color('r')

    ax = axes[3]
    ax.set_title('Skree plot', loc='left')
    ax.plot(
        np.arange(0, len(explained_variance_ratio)) + 1,
        explained_variance_ratio,
        marker='o',
        color='0.5'
    )
    ax.set_xlabel('PCA Dimension (-)')
    ax.set_ylabel('Explained Variance (-)')
    ax.plot(
        dim1 + 1,
        explained_variance_ratio_dim1,
        color=color_dim1,
        ls='None',
        marker='o',
        markersize=10
    )
    ax.plot(
        dim2 + 1,
        explained_variance_ratio_dim2,
        color=color_dim2,
        ls='None',
        marker='o',
        markersize=10
    )

    if sup_title:
        fig.suptitle(sup_title)
    fig.tight_layout()


def pca_data_plotter(df, hue_vars, hue_norm, dim1, dim2, sup_title=None,
                     num_cols=None, num_rows=None, fig_axes=None, jitter=False, **sns_kwargs):
    n_hue_vars = len(hue_vars)

    if num_cols and num_rows:
        if num_cols * num_rows < n_hue_vars:
            raise ValueError(
                'Num rows x num cols is too small.')
    elif (num_cols and not num_rows) or (num_rows and not num_cols):
        raise ValueError('num_cols and num_rows must be provided together.')
    else:
        num_cols = 4
        num_rows = np.ceil(n_hue_vars / 4).astype(int)

    if fig_axes is None:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(5 * num_cols, 5 * num_rows),
            sharex=True,
            sharey=True
        )
        axes = axes.flatten()
    else:
        fig = fig_axes[0]
        axes = fig_axes[1]
        axes = np.atleast_1d(axes)
    
    g_list = []
    
    if jitter:
        df[dim1] = rand_jitter(df[dim1])
        df[dim2] = rand_jitter(df[dim2])
    
    for nh, hvar in enumerate(hue_vars):
        if hvar not in hue_norm:
            norm = None
        else:
            norm = hue_norm[hvar]
        
        ax = axes[nh]
        g = sns.scatterplot(
            data=df,
            x=dim1,
            y=dim2,
            hue=hvar,
            hue_norm=norm,
            legend=True,
            ax=ax,
            **sns_kwargs,
        )
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('{}'.format(dim2))
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('{}'.format(dim1))
        ax.set_title(hvar)
        g_list.append(g)

    if num_cols * num_rows > n_hue_vars:
        for n in np.arange(n_hue_vars, num_cols * num_rows):
            ax = axes[n]
            ax.axis('off')

    if sup_title:
        fig.suptitle(sup_title)
    fig.tight_layout()
    
    return g_list


def rand_jitter(arr):
    stdev = .025 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev
