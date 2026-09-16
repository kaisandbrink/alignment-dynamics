from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

################################################
## Define colour palettes for plotting
################################################

# curricula_names = {'static':'Interleaved', 'clean_early':'Curriculum'}
curricula_names = {'clean_early':'Imbalanced', 'static':'Balanced'}

cPal = {
    "Interleaved": "#7E6EB3",   
    "Curriculum": "#55A868",   
    "static": "#7E6EB3",
    "clean_early": "#55A868",
    "Balanced": "#7E6EB3",   
    "Imbalanced": "#55A868",
}

dataPal = {
    "training": "black",
    "misaligned": "#DD8452",   
    "aligned": "#56B4E9", 
}

interpPal = {
    "baseline": "#9CA3AF",   # neutral grey
    "ablated": "#D81B60",    # magenta
    "patched": "#DEA11C",    # amber/gold
    "attention": "#2E4E79",   # slate grey
}

# from theory_utils import compute_ntw_node_trajectories  # noqa: F401 — re-exported for callers


def format_axis(ax, line_width_multiplier=2, font_size_multiplier=1, dark_mode=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=8*font_size_multiplier)
    ax.yaxis.set_tick_params(size=8*font_size_multiplier)

    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2*line_width_multiplier)

    # increase tick width
    ax.tick_params(width=2*line_width_multiplier)

    ax.xaxis.label.set_fontsize(16*font_size_multiplier)
    ax.yaxis.label.set_fontsize(16*font_size_multiplier)

    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(14*font_size_multiplier)

    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(13*font_size_multiplier)
        
        for legobj in ax.get_legend().legend_handles:
            legobj.set_linewidth(2.0*line_width_multiplier)

        if ax.get_legend().get_title() is not None:
            ax.get_legend().get_title().set_fontsize(13*font_size_multiplier)

    for line in ax.lines:
        line.set_linewidth(2.5*font_size_multiplier)

    # Add space between title and plot by adjusting y position
    if ax.get_title():
        ax.set_title(ax.get_title(), pad=20)

    ax.title.set_fontsize(24*font_size_multiplier)

    # Dark mode styling
    if dark_mode:
        # Set text colors to white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        # Set tick colors to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Set spine colors to white
        for spine in ax.spines.values():
            spine.set_color('white')

        # Set tick label colors to white
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_color('white')

        # Set legend colors to white if legend exists
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            for item in legend.get_texts():
                item.set_color('white')
            if legend.get_title() is not None:
                legend.get_title().set_color('white')
            # Make legend background transparent
            legend.get_frame().set_facecolor('none')
            legend.get_frame().set_edgecolor('white')

        # Make figure background transparent
        ax.figure.patch.set_facecolor('none')
        ax.set_facecolor('none')


def plot_polar_trajectories(
    ax,
    trajectories: list,
    labels: list[str] | None = None,
    colors: list | None = None,
    r_max: float | None = None,
    title: str = "",
    phase_ends: list[int] | None = None,
    epoch_indices: np.ndarray | None = None,
):
    """
    Plot 2-D weight trajectories on a polar axis.

    Each trajectory is an (n_epochs, 2) array whose angle and radius are
    extracted via arctan2 / norm.  Alpha fades from 0.1 (start) to 1.0 (end).
    Phase boundaries are marked with a scatter of the trajectory point at that
    epoch.

    Parameters
    ----------
    ax            : polar Axes
    trajectories  : list of (n_epochs, 2) arrays
    labels        : one label per trajectory
    colors        : one colour per trajectory
    r_max         : shared radial limit; computed from data if None
    title         : subplot title
    phase_ends    : cumulative epoch values where phases end (for markers)
    epoch_indices : (n_epochs,) array mapping position → epoch number
                    (used to locate phase_end markers)
    """
    if labels is None:
        labels = [f"ctx {i}" for i in range(len(trajectories))]
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(trajectories))]

    # Convert each trajectory to polar coords
    polar = []
    for traj in trajectories:
        phis = np.arctan2(traj[:, 1], traj[:, 0])
        rs   = np.linalg.norm(traj, axis=-1)
        polar.append((phis, rs))

    if r_max is None:
        r_max = max(rs.max() for _, rs in polar) * 1.1

    n_steps = len(trajectories[0])
    for (phis, rs), label, color in zip(polar, labels, colors):
        for t in range(n_steps - 1):
            alpha = 0.15 + 0.85 * (t / max(n_steps - 1, 1))
            ax.plot(phis[t:t+2], rs[t:t+2], color=color, alpha=alpha, linewidth=1.5)
        ax.scatter(phis[0],  rs[0],  color=color, marker="o", s=60,  zorder=5, alpha=0.4)
        ax.scatter(phis[-1], rs[-1], color=color, marker="*", s=150, zorder=5, label=label)

        # phase boundary markers
        if phase_ends is not None and epoch_indices is not None:
            for end_epoch in phase_ends[:-1]:
                idx = int(np.searchsorted(epoch_indices, end_epoch, side="right")) - 1
                idx = max(0, min(idx, n_steps - 1))
                ax.scatter(phis[idx], rs[idx], color=color, marker="X",
                           s=100, zorder=6, alpha=1.0, linewidths=1)

    ax.set_rlim(0, r_max)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    if title:
        ax.set_title(title, pad=15)
    if any(l for l in labels):
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
def compute_moving_avg(timeseries,window_size=8):
    windows = pd.Series(timeseries).rolling(window_size)
    return windows.mean().tolist()
    
def plot_ribbon(arr, y=False, ax=None, label = '', color = 'blue', 
                smoothed_window = False, alpha=1, linewidth = 1, 
                linestyle='solid', shift=False, error_style='ribbon',
                capsize = 4, markersize = 3):
    
    """ plot mean timeseries with std ribbon.
    Args: arr is a numpy array (number of repeats) x (number of timepoints)"""
    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
        
    mean_vals = np.nanmean(arr, axis=0)
    std_vals = np.nanstd(arr, axis=0)/np.sqrt(arr.shape[0])
    
    if not y:
        x_vals = np.arange(arr.shape[1])
    else:
        x_vals = y
    
    if shift:
        x_vals = x_vals + shift
        
    if smoothed_window:
         mean_vals = compute_moving_avg(mean_vals, smoothed_window)
 
    if error_style == 'ribbon':
        (ax or plt).plot(x_vals, mean_vals, color = color, label = label, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        (ax or plt).fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, color = color, alpha = 0.4*alpha)
    elif error_style == 'errorbar':
        (ax or plt).errorbar(x_vals, mean_vals, yerr=std_vals, fmt='-o', capsize=capsize, color = color, label=label, markersize=markersize,
                              alpha=alpha, linewidth=linewidth)
    
    
def plot_ribbon_df(df, metrics, x='checkpoint', hue=None, palette=None, ax=None, title=None):
    """Plot mean ± std ribbon for one or more metrics from a DataFrame.

    Args:
        df: DataFrame with columns for x, metrics, and optionally hue.
        metrics: list of (col, label, color) tuples to plot.
        x: Column to use as x-axis (default 'checkpoint').
        hue: Optional column to split lines by (e.g. 'rule_idx'). If None,
             aggregates all rows.
        palette: Optional dict {hue_val: color} or seaborn palette name for
                 per-hue coloring. If None, all hue groups use the metric color.
        ax: Matplotlib axis to plot on. If None, creates a new figure.
        title: Optional axis title.

    Returns:
        The matplotlib axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    group_cols = [x] + ([hue] if hue else [])

    for col, label, default_color in metrics:
        stats = df.groupby(group_cols)[col].agg(['mean', 'std']).reset_index()

        if hue:
            hue_vals = stats[hue].unique()
            if palette is None:
                colors = {v: default_color for v in hue_vals}
            elif isinstance(palette, dict):
                colors = palette
            else:
                colors = dict(zip(hue_vals, sns.color_palette(palette, len(hue_vals))))

            for hue_val, grp in stats.groupby(hue):
                color = colors[hue_val]
                ax.plot(grp[x], grp['mean'], color=color, linewidth=2, label=str(hue_val))
                ax.fill_between(grp[x], grp['mean'] - grp['std'],
                                grp['mean'] + grp['std'], alpha=0.3, color=color)
        else:
            ax.plot(stats[x], stats['mean'], color=default_color, linewidth=2, label=label)
            ax.fill_between(stats[x], stats['mean'] - stats['std'],
                            stats['mean'] + stats['std'], alpha=0.3, color=default_color)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.legend()
    sns.despine(ax=ax)
    return ax

def plot_rule_alignment(probe_df):
    
    cols = {'p_aligned':'P(aligned)', 
            'p_misaligned':'P(misaligned)', 
            'p_ungrammatical':'P(ungrammatical)'}
    
    fig, axs = plt.subplots(1, 3, figsize = (3*5, 4), sharey=True)

    for ci, col in enumerate(cols):
        ax=axs[ci]
        
        plot_ribbon_df(
            probe_df,
            metrics=[(col, col, None)],
            hue='rule_id',
            palette='Set2',
            title=col,
            ax=ax
        )
    
    return fig, axs

def plot_rule_alignment_avg(probe_df, title='', ax=None):
    cols = {
        'p_aligned':      ('P(aligned)',      'steelblue'),
        'p_misaligned':   ('P(misaligned)',   'tomato'),
        'p_ungrammatical':('P(ungrammatical)','green'),
    }

    for col, (label, color) in cols.items():
        plot_ribbon_df(
            probe_df,
            metrics=[(col, label, color)],
            ax=ax,
        )
    ax.set_title(title)
    ax.legend()

### DEFINE COLOR PALETTES

# def plot_rule_alignment(rule_probe_alignment, title='', yscale=False):

#     rcols = sns.color_palette('Set2')[:len(rule_probe_alignment)]

#     fig, axs = plt.subplots(1, 3, figsize = (15, 4))

#     cols = ['p_ungrammatical', 'p_aligned' , 'p_misaligned']
#     for axi, col in enumerate(cols):
#         ax=axs[axi]
#         for ri, rule in enumerate(rule_probe_alignment):
#             sns.lineplot(rule_probe_alignment[rule], x = 'checkpoint', 
#                             y = col, color = rcols[ri],
#                             ax=ax,
#                             linewidth=3, label = rule)
#         ax.set_ylabel('')
#         ax.set_ylabel('P(aligned)')
#         ax.set_title(col)

#         if yscale:
#             ax.set_ylim(yscale[0], yscale[1])

#     sns.despine()
#     fig.tight_layout()
#     plt.legend()
#     plt.suptitle(title, y = 1.1, fontsize = 18)

# def plot_rule_alignment(rule_probe_alignment, title='', yscale=None):
#     df = pd.concat(
#         [d.assign(rule_id=rule_id) for rule_id, d in rule_probe_alignment.items()],
#         ignore_index=True
#     )
#     palette = dict(zip(rule_probe_alignment.keys(),
#                        sns.color_palette('Set2', len(rule_probe_alignment))))

#     cols = ['p_ungrammatical', 'p_aligned', 'p_misaligned']
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#     for ax, col in zip(axes, cols):
#         plot_ribbon_df(df, [(col, col, None)], hue='rule_id', palette=palette, ax=ax, title=col)
#         ax.set_ylabel('Probability')
#         if yscale:
#             ax.set_ylim(*yscale)

#     fig.suptitle(title, y=1.05, fontsize=18)
#     fig.tight_layout()
#     return fig



