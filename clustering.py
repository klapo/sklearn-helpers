import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics


def dbscan_hyperparameters_sweep(X, eps_array, min_samples_array):
    num_clusters = np.ones((len(eps_array), len(min_samples_array))) * np.nan
    num_noise = np.ones((len(eps_array), len(min_samples_array))) * np.nan
    ch_score = np.ones((len(eps_array), len(min_samples_array))) * np.nan
    silhouettes = np.ones((len(eps_array), len(min_samples_array))) * np.nan

    for neps, eps in enumerate(eps_array):
        for nmin, min_samples in enumerate(min_samples_array):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            try:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                sil = metrics.silhouette_score(X, labels)
                ch = metrics.calinski_harabasz_score(X, labels)
            # Catch when there is only one cluster
            except ValueError:
                n_clusters = np.nan
                n_noise = np.nan
                sil = np.nan
                ch = np.nan

            if n_clusters > 1:
                num_clusters[neps, nmin] = n_clusters
                num_noise[neps, nmin] = n_noise
                ch_score[neps, nmin] = ch
                silhouettes[neps, nmin] = sil
                
    return ch_score, silhouettes, num_noise, num_clusters


def dbscan_hyperparameters_plot(
    ch_score, silhouettes, num_clusters, num_noise, eps_array,
    min_samples_array):

    ind_sil_eps, ind_sil_ms = find_suggested_params(
        silhouettes, eps_array, min_samples_array)
    print('Silhouttes suggested eps={}, min_samples={}'.format(
        eps_array[ind_sil_eps], min_samples_array[ind_sil_ms]))

    ind_ch_eps, ind_ch_ms = find_suggested_params(
        ch_score, eps_array, min_samples_array)
    print('Calinski-Harabasz suggested eps={}, min_samples={}'.format(
        eps_array[ind_ch_eps], min_samples_array[ind_ch_ms]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    axes = axes.flatten()
    eps_legend_label = ['min_samples=5', 'min_samples=10', 'min_samples=15']

    ax = axes[0]
    ax.plot(eps_array, ch_score, label=eps_legend_label, marker='o')
    ax.scatter(
        eps_array[ind_sil_eps],
        ch_score[ind_sil_eps, ind_sil_ms],
        100,
        label='Best Silhoutte Score',
        marker='s',
        color='k',
        zorder=10,
    )
    ax.scatter(
        eps_array[ind_ch_eps],
        ch_score[ind_ch_eps, ind_ch_ms],
        100,
        label='Best Calinski-Harabasz Score',
        marker='^',
        color='0.5',
        zorder=10,
    )
    ax.legend()
    ax.set_ylabel('Calinski-Harabasz Score (big is better)')
    ax.set_xlabel('EPS value')

    ax = axes[1]
    ax.plot(eps_array, silhouettes, label=eps_legend_label, marker='o')
    ax.scatter(
        eps_array[ind_sil_eps],
        silhouettes[ind_sil_eps, ind_sil_ms],
        100,
        marker='s',
        color='k',
        zorder=10,
    )
    ax.scatter(
        eps_array[ind_ch_eps],
        silhouettes[ind_ch_eps, ind_ch_ms],
        100,
        marker='^',
        color='0.5',
        zorder=10,
    )
    ax.set_ylabel('Silhouette Coefficient (closer to 1 is better)')
    ax.set_xlabel('EPS value')

    ax = axes[2]
    ax.plot(eps_array, num_clusters, label=eps_legend_label, marker='o')
    ax.scatter(
        eps_array[ind_sil_eps],
        num_clusters[ind_sil_eps, ind_sil_ms],
        100,
        marker='s',
        color='k',
        zorder=10,
    )
    ax.scatter(
        eps_array[ind_ch_eps],
        num_clusters[ind_ch_eps, ind_ch_ms],
        100,
        marker='^',
        color='0.5',
        zorder=10,
    )
    ax.set_ylabel('Number of clusters')
    ax.set_xlabel('EPS value')

    ax = axes[3]
    ax.plot(eps_array, num_noise, label=eps_legend_label, marker='o')
    ax.scatter(
        eps_array[ind_sil_eps],
        num_noise[ind_sil_eps, ind_sil_ms],
        100,
        marker='s',
        color='k',
        zorder=10,
    )
    ax.scatter(
        eps_array[ind_ch_eps],
        num_noise[ind_ch_eps, ind_ch_ms],
        100,
        marker='^',
        color='0.5',
        zorder=10,
    )
    ax.set_ylabel('Points labeled as noise')
    ax.set_xlabel('EPS value')
    fig.tight_layout()
    
    return ind_sil_eps, ind_sil_ms, ind_ch_eps, ind_ch_ms


def find_suggested_params(param, eps_array, min_samples_array, maximize=True):
    if maximize:
        ind_eps, ind_ms = np.nonzero(param == np.nanmax(param))
    else:
        ind_eps, ind_ms = np.nonzero(param == np.nanmin(param))        
    return ind_eps, ind_ms
