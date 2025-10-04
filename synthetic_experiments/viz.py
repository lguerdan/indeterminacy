import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

sys.path.append('..')

from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, NullLocator, FormatStrFormatter
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from scipy import stats
import core.metrics as mets

def highlight_min_max(row):
    # Create an empty style series
    styles = pd.Series('', index=row.index)
    
    # Set style for minimum value
    styles[row == row.min()] = 'background-color: #FF7F7F; color: white'
    # Set style for maximum value
    styles[row == row.max()] = 'background-color: #50C878; color: white'
    
    return styles


def plot_target_exp_results(results, tau):

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for i, eval_measure in enumerate(['consistency', 'bias_mae']):
        plotdf = results[(results['eval_measure'] == eval_measure) & 
              (results['tau'].astype(float) >= tau-.01) &
              (results['tau'].astype(float) <= tau+.01)]

        g = sns.barplot(data=plotdf, x='target_measure', y='value', 
                   hue='target_measure_category', errorbar='sd', ax=axes[i])

        if i == 0:
            g.get_legend().remove()

        axes[i].tick_params(axis='x', rotation=30)
        axes[i].set_ylabel(eval_measure)
    axes[0].set_ylim([.4,1])
    axes[1].set_ylim([0,.4])

    axes[0].set_xlabel('')
    plt.tight_layout()
    plt.show()

def plot_fs_results(results_df, measures):

    results_df.loc[:, 'target_measure'] = results_df['target_measure'].map(metrics.METRIC_LOOKUP)

    m = [metrics.METRIC_LOOKUP[measure] for measure in measures]

    cdf = results_df[(results_df['eval_measure'] == 'consistency') &
                     results_df['target_measure'].isin(m)]

    maedf = results_df[(results_df['eval_measure'] == 'bias_mae') &
                     results_df['target_measure'].isin(m)]


    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First plot
    sns.lineplot(
        data=cdf,
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax1,  # specify which subplot to use,
        legend=False
    )
    ax1.set_xscale('log')
    ax1.set_xlabel('Ratings-per-Item', fontsize=14)
    ax1.set_ylabel('Consistency', fontsize=14)
    ax1.set_ylim([.7,1])

    # Second plot
    sns.lineplot(
        data=maedf,  # you can use different data here
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax2  # specify which subplot to use
    )
    ax2.set_xscale('log')
    ax2.set_xlabel('Ratings-per-Item', fontsize=14)
    ax2.set_ylabel('Bias (Mean Absolute Error)', fontsize=14)
    ax2.legend(title='Target Measure')

    def plot_metric_consistency_comparison(sdf, udf, save_path=None, figsize=(20, 16)):
        """
        Create a 2x2 grid comparing Cohen's kappa and MSE metrics against consistency
        for fully specified and underspecified tasks.
        
        Parameters:
        -----------
        sdf : pandas.DataFrame
            DataFrame containing data for the fully specified task
        udf : pandas.DataFrame
            DataFrame containing data for the underspecified task
        save_path : str, optional
            Path to save the figure (e.g., '../figures/rank_consistency_visualization.pdf')
        figsize : tuple, optional
            Figure size in inches (width, height), default is (20, 16)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot
        axes : numpy.ndarray
            2x2 array of the subplot axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import matplotlib.lines as mlines
        import seaborn as sns
        from scipy import stats
        
        # Create a 2x2 grid of subplots with larger overall figure size
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
        
        # Colors for each column
        colors = {
            'fully_specified': {
                'darker': '#1A5276',  # Dark navy blue for scatter points and confidence interval
                'lighter': '#2980B9'  # Medium blue for regression line
            },
            'underspecified': {
                'darker': '#C44536',  # Darker red for scatter points and confidence interval
                'lighter': '#E06666'  # Light red for regression line
            }
        }
        
        # List of metrics to plot (in row order)
        metrics = ['Cohen_kappa_h_h', 'MSE_srs_srs']
        metric_labels = ["Cohen's κ (h/h)↑", "MSE (srs/srs) ↓"]
        metric_titles = ["Cohen's κ (h/h) vs. Consistency", "MSE vs. Consistency"]
        
        # Define optimal positions for stars for each dataset type
        optimal_positions = {
            'fully_specified': {  # Fully specified dataset
                'MSE_srs_srs': [(1, 0.0001)],  # Single optimal point
                'Cohen_kappa_h_h': [(1, 1)]  # Multiple optimal points
            },
            'underspecified': {  # Underspecified dataset
                'MSE_srs_srs': [(0.95, 0.00015)],  # Different optimal point for underspecified
                'Cohen_kappa_h_h': [(.75, .34), (0.8, 0.34), (0.85, 0.34), (0.9, 0.34)]  # Different optimal points
            }
        }
        
        # Define dataset configurations (in column order)
        dataset_configs = [
            {'data': sdf, 'name': 'fully_specified', 'title': 'Fully Specified Task'},
            {'data': udf, 'name': 'underspecified', 'title': 'Underspecified Task'}
        ]
        
        # Plot the grid with datasets in columns and metrics in rows
        for row, (metric, label, title) in enumerate(zip(metrics, metric_labels, metric_titles)):
            for col, dataset_config in enumerate(dataset_configs):
                dataset = dataset_config['data']
                dataset_name = dataset_config['name']
                title_suffix = dataset_config['title']
                
                ax = axes[row, col]
                
                # Get colors for this dataset
                darker_color = colors[dataset_name]['darker']
                lighter_color = colors[dataset_name]['lighter']
        
                # First plot the confidence interval with regplot but without scatterplot
                sns.regplot(x='consistency', y=metric, data=dataset, ax=ax,
                        scatter=False,  # Don't plot scatter points
                        line_kws={'color': lighter_color, 'linewidth': 3.5},
                        ci=95,
                        color=darker_color)
                
                # Then manually add scatter points as X markers
                ax.scatter(dataset['consistency'], dataset[metric], 
                        marker='x', s=190, color=darker_color, alpha=0.7,
                        linewidth=3)
                
                # Set axis labels - only add x-label for bottom row, only add y-label for left column
                if row == 1:  # Bottom row
                    ax.set_xlabel('Downstream Metric\n Consistency ↑', fontsize=36)
                else:
                    ax.set_xlabel('')
                    
                if col == 0:  # Left column
                    ax.set_ylabel(f"Agreement Metric\n {label}", fontsize=36)
                else:
                    ax.set_ylabel('')
                
                # Set title
                if row == 0:
                    ax.set_title(f"{title_suffix}", fontsize=40, pad=25)
                
                # Add star markers for optimal values (specific to this dataset and metric)
                if metric in optimal_positions[dataset_name]:
                    for star_x, star_y in optimal_positions[dataset_name][metric]:
                        ax.scatter(star_x, star_y, s=2000, marker='*', color='gold',
                                edgecolor='black', linewidth=2.5, zorder=10)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Increase linewidth for the visible spines
                ax.spines['bottom'].set_linewidth(2.0)
                ax.spines['left'].set_linewidth(2.0)
                
                # Set tick parameters - hide x-axis labels on top row with LARGER tick labels
                if row == 0:  # Top row
                    ax.tick_params(axis='x', which='major', labelsize=32, labelbottom=False, width=2.0, length=8)
                    ax.tick_params(axis='y', which='major', labelsize=32, width=2.0, length=8)
                else:  # Bottom row
                    ax.tick_params(axis='both', which='major', labelsize=32, width=2.0, length=8)
                
                # Set x-axis tick formatter to display 2 decimal places
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                
                # Set y-axis tick formatter to display 2 decimal places for left column
                if col == 0:
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                
                # Ensure 1.00 is included in the x-axis for right column
                if col == 1:  # Right column
                    current_xlim = ax.get_xlim()
                    ax.set_xlim(current_xlim[0], 1.00)
                    
                # Calculate Spearman correlation
                spearman_corr, p_value = stats.spearmanr(dataset['consistency'], dataset[metric])
                
                # Add correlation annotation with LARGER font
                ax.annotate(f"Spearman's ρ = {spearman_corr:.2f}", 
                        xy=(0.05, 0.05), 
                        xycoords='axes fraction', 
                        fontsize=26,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=1, linewidth=2.0))
        
        # Add star legend to top-left plot
        star_legend = mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                                markersize=24, markeredgecolor='black', label='Optimal model for agreement metric')
        axes[0, 0].legend(handles=[star_legend], loc='upper left', fontsize=20, 
                        frameon=True, framealpha=0.9, edgecolor='gray')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15, wspace=0.25)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        
        return fig, axes

##################### Code for correlation bar plot  #####################

def plot_metric_consistency_comparison(sdf, udf, save_path=None, figsize=(20, 16)):
    """
    Create a 2x2 grid comparing Cohen's kappa and MSE metrics against consistency
    for fully specified and underspecified tasks.
    
    Parameters:
    -----------
    sdf : pandas.DataFrame
        DataFrame containing data for the fully specified task
    udf : pandas.DataFrame
        DataFrame containing data for the underspecified task
    save_path : str, optional
        Path to save the figure (e.g., '../figures/rank_consistency_visualization.pdf')
    figsize : tuple, optional
        Figure size in inches (width, height), default is (20, 16)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    axes : numpy.ndarray
        2x2 array of the subplot axes
    """

    
    # Create a 2x2 grid of subplots with larger overall figure size
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    
    # Colors for each column
    colors = {
        'fully_specified': {
            'darker': '#1A5276',  # Dark navy blue for scatter points and confidence interval
            'lighter': '#2980B9'  # Medium blue for regression line
        },
        'underspecified': {
            'darker': '#C44536',  # Darker red for scatter points and confidence interval
            'lighter': '#E06666'  # Light red for regression line
        }
    }
    
    # List of metrics to plot (in row order)
    metrics = ['Cohen_kappa_h_h', 'MSE_srs_srs']
    metric_labels = ["Cohen's κ (h/h)↑", "MSE (srs/srs) ↓"]
    metric_titles = ["Cohen's κ (h/h) vs. Consistency", "MSE vs. Consistency"]
    
    # Define optimal positions for stars for each dataset type
    optimal_positions = {
        'fully_specified': {  # Fully specified dataset
            'MSE_srs_srs': [(1, 0.0001)],  # Single optimal point
            'Cohen_kappa_h_h': [(1, 1)]  # Multiple optimal points
        },
        'underspecified': {  # Underspecified dataset
            'MSE_srs_srs': [(0.95, 0.00015)],  # Different optimal point for underspecified
            'Cohen_kappa_h_h': [(.75, .34), (0.8, 0.34), (0.85, 0.34), (0.9, 0.34)]  # Different optimal points
        }
    }
    
    # Define dataset configurations (in column order)
    dataset_configs = [
        {'data': sdf, 'name': 'fully_specified', 'title': 'Fully Specified'},
        {'data': udf, 'name': 'underspecified', 'title': 'Underspecified (Asymmetric)'}
    ]
    
    # Plot the grid with datasets in columns and metrics in rows
    for row, (metric, label, title) in enumerate(zip(metrics, metric_labels, metric_titles)):
        for col, dataset_config in enumerate(dataset_configs):
            dataset = dataset_config['data']
            dataset_name = dataset_config['name']
            title_suffix = dataset_config['title']
            
            ax = axes[row, col]
            
            # Get colors for this dataset
            darker_color = colors[dataset_name]['darker']
            lighter_color = colors[dataset_name]['lighter']
    
            # First plot the confidence interval with regplot but without scatterplot
            sns.regplot(x='consistency', y=metric, data=dataset, ax=ax,
                      scatter=False,  # Don't plot scatter points
                      line_kws={'color': lighter_color, 'linewidth': 3.5},
                      ci=95,
                      color=darker_color)
            
            # Then manually add scatter points as X markers
            ax.scatter(dataset['consistency'], dataset[metric], 
                      marker='x', s=190, color=darker_color, alpha=0.7,
                      linewidth=3)
            
            # Set axis labels - only add x-label for bottom row, only add y-label for left column
            if row == 1:  # Bottom row
                ax.set_xlabel('Downstream Metric\n Consistency ↑', fontsize=36)
            else:
                ax.set_xlabel('')
                
            if col == 0:  # Left column
                ax.set_ylabel(f"Agreement Metric\n {label}", fontsize=36)
            else:
                ax.set_ylabel('')
            
            # Set title
            if row == 0:
                ax.set_title(f"{title_suffix}", fontsize=40, pad=25)
            
            # Add star markers for optimal values (specific to this dataset and metric)
            if metric in optimal_positions[dataset_name]:
                for star_x, star_y in optimal_positions[dataset_name][metric]:
                    ax.scatter(star_x, star_y, s=2000, marker='*', color='gold',
                              edgecolor='black', linewidth=2.5, zorder=10)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Increase linewidth for the visible spines
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_linewidth(2.0)
            
            # Set tick parameters - hide x-axis labels on top row with LARGER tick labels
            if row == 0:  # Top row
                ax.tick_params(axis='x', which='major', labelsize=32, labelbottom=False, width=2.0, length=8)
                ax.tick_params(axis='y', which='major', labelsize=32, width=2.0, length=8)
            else:  # Bottom row
                ax.tick_params(axis='both', which='major', labelsize=32, width=2.0, length=8)
            
            # Set x-axis tick formatter to display 2 decimal places
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            
            # Set y-axis tick formatter to display 2 decimal places for left column
            if col == 0:
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                
            # Calculate Spearman correlation
            spearman_corr, p_value = stats.spearmanr(dataset['consistency'], dataset[metric])
            
            # Add correlation annotation with LARGER font
            ax.annotate(f"Spearman's ρ = {spearman_corr:.2f}", 
                       xy=(0.05, 0.05), 
                       xycoords='axes fraction', 
                       fontsize=26,
                       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=1, linewidth=2.0))
    
    # Add star legend to top-left plot
    star_legend = mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                               markersize=24, markeredgecolor='black', label='Optimal model for agreement metric')
    axes[0, 0].legend(handles=[star_legend], loc='upper left', fontsize=20, 
                     frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.25)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    
    return fig, axes


def plot_corr_df(df, settings, cols, figsize=(12, 10)):
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Professional color palette - muted tones that work well together
    palette = ['#4472C4', '#A5A5A5', '#E06666' ]  # Blue, Gray, Red - common in professional publications
    
    # Define hatching patterns for each category
    hatches = ['/', '\\', 'x']  # Classic professional patterns
    
    for ax, target_metric in zip([ax1, ax2], ['consistency', 'bias_mae']):
        df_filtered = df[df['target_metric'] == target_metric]
        
        df_melted = df_filtered[cols[:-3] + ['category']].melt(id_vars=['category'], 
                                                        var_name='metric', 
                                                        value_name='value')
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
        
        # Create the base plot
        plot = sns.barplot(
            data=df_melted,
            x='metric',
            y='value',
            hue='category',
            width=0.75,
            edgecolor='black',
            capsize=.2,
            zorder=2, 
            hue_order=settings,
            palette=palette,
            ax=ax)
        
        # Add hatching to the bars
        # We need to manually apply hatches since seaborn doesn't support this directly
        bars = plot.patches
        num_categories = len(settings)
        num_metrics = len(cols[:-3])
        
        # Apply hatches to each bar based on its category
        for i, bar in enumerate(bars):
            # Calculate which category this bar belongs to
            category_idx = i // num_metrics
            # Apply the corresponding hatch pattern
            if category_idx < len(hatches):
                bar.set_hatch(hatches[category_idx])
                # Make hatch lines more visible
                bar.set_edgecolor('black')
                bar.set_linewidth(1.0)
        
        # Set ylabel based on metric with LARGER font
        if target_metric == 'consistency':
            ax.set_ylabel(r'Consistency ($\rho$)', fontsize=26)
        else:
            ax.set_ylabel('Estimation Bias\n' +  r'MAE ($\rho$)', fontsize=26)
        
        ax.set_xlabel('', fontsize=18)
        
        # Get current tick positions and set new labels
        locs = ax.get_xticks()
        new_labels = [metrics.METRIC_LOOKUP[col] for col in cols[:-3]]
        ax.set_xticks(locs)
        
        # Set x-tick labels with 45 degree rotation and LARGER font
        ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=24)
        
        # Set yticks with LARGER font
        ax.tick_params(axis='y', labelsize=26)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        
        # Update the legend to include hatch patterns
        if target_metric == 'consistency':
            # Get the current handles and labels
            handles, labels = ax.get_legend_handles_labels()
            
            # Create new patch handles with hatches for the legend
            new_handles = []
            for i, handle in enumerate(handles):
                # Create a new patch with the same color but with hatches
                new_patch = mpatches.Patch(
                    facecolor=palette[i % len(palette)],
                    edgecolor='black',
                    hatch=hatches[i % len(hatches)],
                    label=labels[i]
                )
                new_handles.append(new_patch)
            
            # Create custom legend with the new handles
            legend = ax.legend(
                handles=new_handles,
                title="",
                frameon=False,
                framealpha=1,
                edgecolor='black',
                loc='center',
                ncol=1,
                title_fontsize=28,
                bbox_to_anchor=(0.25, 1.40),
                fontsize=22)
        else:
            ax.get_legend().remove()  # Remove legend from bottom plot
    
    # Adjust layout to accommodate larger text, rotated labels, and add space between plots
    plt.tight_layout()
    
    # Increase hspace to add more vertical space between plots
    # Also adjust bottom margin for rotated labels
    plt.subplots_adjust(bottom=0.15, hspace=0.55)
    
    return fig

def compute_corr_df(cdf, cols, target_metrics):
    
    metadata = ['n_items', 'n_options', 'n_response_sets', 'human_order', 'judge_order', 'human_beta', 'judge_beta','tau', 'trial', 'category']
    
    columns_to_invert = ['HR_h_h',  'Fleiss_kappa_h_h', 'Cohen_kappa_h_h', 'COV_h_hrs', 'consistency']
    
    cdf = cdf.copy()
    cdf.loc[:, columns_to_invert] *= -1


    trial_results = []

    for trial in cdf['trial'].unique():

        trialdf = cdf[cdf['trial'] == trial]
        evalfields = trialdf[metadata].iloc[0]
        
        for tm in target_metrics:

            corrdf = trialdf[cols].corr(method='spearman')[tm]
            corrdf['target_metric'] = tm
            trial_results.append(pd.concat((corrdf, evalfields)))

    return pd.DataFrame(trial_results)


# Create a function to filter the dataframe
def filter_df(df, conditions, cols, setting_name):
    filtered = df.copy()
    for col, value in conditions.items():
        filtered = filtered[filtered[col] == value]
    filtered = filtered[cols]
    filtered['setting'] = setting_name
    return filtered



def plot_human_rating_estimation_exp(df, cfg):

    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 4))  # Increased width from 8 to 10
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig, wspace=0.4)  # Added wspace parameter

    # Create the subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    colors = sns.color_palette('RdPu')[2:6]

    # First plot (upper panel)
    sns.lineplot(data=df, x='rpi', y='consistency', marker='o', hue='tau', ax=ax1, linewidth=2)
    ax1.set_xscale('log')
    ax1.set_ylabel('Decision Consistency \n $\mathbb{E}[s(\hat{Y}_{rs}) = s(Y_{rs})]$', fontsize=12)
    ax1.set_xticks(cfg.rpis)
    ax1.set_xticklabels(cfg.rpis)
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)
    ax1.legend(title=r'Cutoff ($\tau$)', loc='lower right', fontsize=10, title_fontsize=10)
    ax1.set_xlabel('Ratings per Item', fontsize=12)
    ax1.set_xticks(cfg.rpis)
    ax1.set_xticklabels(cfg.rpis)

    # Second plot (lower panel)
    sns.lineplot(data=df, x='rpi', y='bias', marker='o', hue='tau', ax=ax2, legend=False, linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Ratings per Item', fontsize=12)
    ax2.set_ylabel('Estimation Bias \n $\mathbb{E}[s(\hat{Y}_{rs})] - \mathbb{E}[s(Y_{rs})]$', fontsize=12)
    ax2.set_xticks(cfg.rpis)
    ax2.set_xticklabels(cfg.rpis)
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)

    # Save the figure
    plt.savefig('figures/{path}.pdf', bbox_inches='tight', dpi=400)

def plot_target_system_prevalence_results(df, tag):

    plt.figure(figsize=(6, 6))

    categories = [
        {'condition': (df['n_options'] == df['n_response_sets']), 'label': 'Fully Specified Rating Task', 'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': 4, 'markersize': 0},
        {'condition': ((df['n_options'] < df['n_response_sets']) & (df['setting'] == 'Positive')), 'label': 'Underspecified: Positive', 'color': '#2a9d8f', 'linestyle': '-', 'marker': 'o', 'linewidth': 4, 'markersize': 5},
        {'condition': ((df['n_options'] < df['n_response_sets']) & (df['setting'] == 'Randomized Selection')), 'label': 'Underspecified: Randomized', 'color': '#264653', 'linestyle': '-', 'marker': 'o', 'linewidth': 4, 'markersize':5},
        {'condition': ((df['n_options'] < df['n_response_sets']) & (df['setting'] == 'Negative')), 'label': 'Underspecified: Negative', 'color': '#e76f51', 'linestyle': '-', 'marker': 'o', 'linewidth': 4, 'markersize':5},
    ]

    # Create the plot
    for category in categories:
        sns.lineplot(
            data=df[category['condition']],
            x='tau',
            y='bias',
            marker=category['marker'],
            markersize=category['markersize'],
            errorbar='se',
            label=category['label'],
            linestyle=category['linestyle'],
            linewidth=category['linewidth'],
            color=category['color'],
        )

    plt.xlabel(r'Cutoff ($\tau$)', fontsize=22)
    plt.ylabel('Estimation Bias', fontsize=22)
    plt.title('Estimation Bias Induced \n by Task Underspecification', pad=14, fontsize=24)
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.legend(fontsize=17, loc='lower right', framealpha=1, bbox_to_anchor=(1.44, 0))


    plt.xticks(fontsize=18)  # Adjust x-axis tick label size
    plt.yticks(fontsize=18)  # Adjust y-axis tick label size

    plt.savefig(f'figures/{tag}.pdf', bbox_inches='tight', dpi=400)



#################################################################################


##################### Code for judge selection experiment  #####################

def plot_selection_results_main(df, mask, labels, cfg):
    # Create figure with fixed aspect ratio
    fig = plt.figure(figsize=(22, 8))  # Increased width to accommodate 5 columns
    
    # Define column widths and spacing
    width_per_column = 0.2
    spacing = 0.0
    
    
    # Define colors just like in your second code
    aggregate_colors = {
        'Categorical (Forced Choice)': '#990000',  # Dark red
        'Distributional (Forced Choice)': '#990000',  # Dark red
        'Discrete (Multi-Label)': '#000099',  # Dark blue
        'Continuous (Multi-Label)': '#000099'   # Dark blue
    }
    
    detailed_colors = {
        'HR_h_h': '#1f77b4',           # Blue
        'Krippendorff_h_h': '#ff7f0e', # Orange
        'Fleiss_kappa_h_h': '#2ca02c', # Green
        'Cohen_kappa_h_h': '#d62728',  # Red
        'KL_lh_s_s': '#6A0DAD',        # Purple
        'KL_hl_s_s': '#8c564b',        # Brown
        'CE_lh_s_s': '#4C9D9D',        # Pink
        'CE_hl_s_s': '#7f7f7f',        # Gray
        'JSD_s_s': '#bcbd22',          # Yellow-green
        'COV_h_hrs': '#17becf',        # Cyan
        'MSE_srs_srs': '#000099',      # Dark blue
        'MSE_h_h': '#ff9896',          # Light red
        'MSE_s_s': '#000099'           # Dark Blue
    }
    
    # Define markers and line styles similar to your second code
    aggregate_markers = {
        'Categorical (Forced Choice)': 's',  # Square marker
        'Distributional (Forced Choice)': '^',  # Triangle marker
        'Discrete (Multi-Label)': 'o',  # Circle marker
        'Continuous (Multi-Label)': '*'  # Star marker
    }
    
    detailed_markers = {
        'HR_h_h': 'o',                # Circle
        'Krippendorff_h_h': 's',      # Square
        'Fleiss_kappa_h_h': '^',      # Triangle up
        'Cohen_kappa_h_h': 'D',       # Diamond
        'KL_lh_s_s': 'p',             # Pentagon
        'KL_hl_s_s': '<',             # Triangle left
        'CE_lh_s_s': 'X',             # X filled
        'CE_hl_s_s': 'h',             # Hexagon 1
        'JSD_s_s': 'H',               # Hexagon 2
        'COV_h_hrs': 'v',             # Triangle down
        'MSE_srs_srs': '*',           # Star
        'MSE_h_h': '>',               # Triangle right
        'MSE_s_s': 'd'                # Thin diamond
    }
    
    aggregate_linestyles = {
        'Categorical (Forced Choice)': '-',   # Solid line
        'Distributional (Forced Choice)': '--',  # Dashed line
        'Discrete (Multi-Label)': '-',  # Solid line
        'Continuous (Multi-Label)': '--'  # Dashed line
    }
    
    # Create a 2x5 grid of subplots
    gs = gridspec.GridSpec(2, 5, figure=fig)
    gs.update(hspace=0.05, wspace=0.25)
    
    # Create the axes
    axs = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), 
        fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), 
        fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4])]
    ])
    
    # Make subplots in the same row share y-axis with the first subplot
    for j in range(1, 5):
        axs[0, j].sharey(axs[0, 0])
        axs[1, j].sharey(axs[1, 0])
        
        # Hide y-axis labels and ticks for all but the leftmost plots
        axs[0, j].tick_params(axis='y', which='both', labelleft=False)
        axs[1, j].tick_params(axis='y', which='both', labelleft=False)
    
    # Explicitly ensure y-axis ticks are shown for the leftmost plots in each row
    axs[0, 0].tick_params(axis='y', which='both', labelleft=True)
    axs[1, 0].tick_params(axis='y', which='both', labelleft=True)
    
    # Column titles
    column_titles = [
        "Fully Specified\nGrouped Metrics", 
        "Underspecified\nSymmetric\nGrouped Metrics", 
        "Underspecified\nAsymmetric\nGrouped Metrics",
        "Underspecified\nAsymmetric\nCategorical Metrics",
        "Underspecified\nAsymmetric\nDistributional Metrics"
    ]
    
    for j, title in enumerate(column_titles):
        axs[0, j].set_title(title, fontsize=22, pad=15)
    
    # Define conditions for the first three columns
    condition_functions = [
        # Fully Specified, No Error
        lambda df_row: (df_row['n_options'] == df_row['n_response_sets']) & (df_row['error_rate'] == 0),
        
        # Underspecified Symmetric, No Error
        lambda df_row: (df_row['n_options'] < df_row['n_response_sets']) & (df_row['human_beta'] == df_row['judge_beta']) & (df_row['error_rate'] == 0),
        
        # Underspecified Asymmetric, No Error
        lambda df_row: (df_row['n_options'] < df_row['n_response_sets']) & (df_row['human_beta'] != df_row['judge_beta']) & (df_row['error_rate'] == 0)
    ]
    
    # Create working copy of the DataFrame
    working_df = df.copy()
    
    # Add metric_type column to the dataframe based on METRIC_GROUPS
    working_df['metric_type'] = 'Other'  # Default value
    
    # Assign metric types based on METRIC_GROUPS
    all_grouped_metrics = []
    for metric_type, metric_list in mets.METRIC_GROUPS.items():
        # Mark metrics with their type
        working_df.loc[working_df['target_measure'].isin(metric_list), 'metric_type'] = metric_type
        all_grouped_metrics.extend(metric_list)
    
    # Filter to only include metrics that are in the METRIC_GROUPS
    working_df = working_df[working_df['target_measure'].isin(all_grouped_metrics)]
    
    # Convert mask to an indexer if it's a Series
    # This fixes the "Boolean Series key will be reindexed" warning
    if isinstance(mask, pd.Series):
        # Get indices where mask is True
        mask_indices = mask[mask].index
        # Filter working_df to only include these indices
        masked_df = working_df.loc[working_df.index.isin(mask_indices)].copy()
    else:
        # If mask is already an array or list, use it directly
        masked_df = working_df[mask].copy()
    
    # Metrics for detailed breakdowns
    categorical_metrics = mets.METRIC_GROUPS['Categorical (Forced Choice)']
    distributional_metrics = mets.METRIC_GROUPS['Distributional (Forced Choice)']
    
    # Process the data for each condition (first 3 columns)
    for col_idx, condition_func in enumerate(condition_functions):
        # Apply the specific condition
        condition_df = masked_df[condition_func(masked_df)].copy()
        
        # For consistency (first row)
        consistency_df = condition_df[condition_df['eval_measure'] == 'consistency'].copy()
        
        # Group by metric_type and calculate mean and standard error
        grouped_consistency = consistency_df.groupby(['rpi', 'metric_type'])['value'].agg(
            ['mean', 'sem']).reset_index()
        
        # Plot each metric type for consistency
        for metric_type in grouped_consistency['metric_type'].unique():
            type_data = grouped_consistency[grouped_consistency['metric_type'] == metric_type]
            
            # Get color, marker, and linestyle for this metric type
            color = aggregate_colors[metric_type]
            marker = aggregate_markers[metric_type]
            linestyle = aggregate_linestyles[metric_type]
            
            # Sort data by rpi for proper fill_between
            type_data = type_data.sort_values('rpi')
            
            # Plot the line
            axs[0, col_idx].plot(type_data['rpi'], type_data['mean'], 
                    color=color, linestyle=linestyle, alpha=0.8, linewidth=2)
            
            # Add shaded error region instead of error bars
            upper = type_data['mean'] + 1.4*type_data['sem']
            lower = type_data['mean'] - 1.4*type_data['sem']
            axs[0, col_idx].fill_between(type_data['rpi'], lower, upper, 
                                    color=color, alpha=0.25)
            
            # Plot scatter points on top
            axs[0, col_idx].scatter(type_data['rpi'], type_data['mean'], 
                    color=color, marker=marker, s=80, 
                    label=metric_type, zorder=10)
        
        # Format the top row plots
        apply_formatting(axs[0, col_idx], labels, cfg, 
                            ylabel='Consistency' if col_idx == 0 else '', 
                            xlabel='', 
                            subplot_label=f'({chr(65+col_idx)})', 
                            position='top',
                            show_ylabels=(col_idx == 0))
        
        # For bias MAE (second row)
        bias_df = condition_df[condition_df['eval_measure'] == 'bias_mae'].copy()
        
        # Group by metric_type and calculate mean and standard error
        grouped_bias = bias_df.groupby(['rpi', 'metric_type'])['value'].agg(
            ['mean', 'sem']).reset_index()
        
        # Plot each metric type for bias
        for metric_type in grouped_bias['metric_type'].unique():
            type_data = grouped_bias[grouped_bias['metric_type'] == metric_type]
            
            # Get color, marker, and linestyle for this metric type
            color = aggregate_colors[metric_type]
            marker = aggregate_markers[metric_type]
            linestyle = aggregate_linestyles[metric_type]
            
            # Sort data by rpi for proper fill_between
            type_data = type_data.sort_values('rpi')
            
            # Plot the line
            axs[1, col_idx].plot(type_data['rpi'], type_data['mean'], 
                    color=color, linestyle=linestyle, alpha=0.8, linewidth=2)
            
            # Add shaded error region instead of error bars
            upper = type_data['mean'] + type_data['sem']
            lower = type_data['mean'] - type_data['sem']
            axs[1, col_idx].fill_between(type_data['rpi'], lower, upper, 
                                    color=color, alpha=0.25)
            
            # Plot scatter points on top
            axs[1, col_idx].scatter(type_data['rpi'], type_data['mean'], 
                    color=color, marker=marker, s=80, 
                    label=metric_type, zorder=10)
        
        # Format the bottom row plots
        apply_formatting(axs[1, col_idx], labels, cfg, 
                            ylabel='Estimation Bias\n (Mean Absolute Error)' if col_idx == 0 else '', 
                            xlabel='Ratings per Item', 
                            subplot_label=f'({chr(65+col_idx+3)})', 
                            position='bottom',
                            show_ylabels=(col_idx == 0))
    
    # Add legend for the first three columns to the third column
    legend_elements = []
    forced_choice_header = mpatches.Patch(color='white', alpha=0.0, label='Forced Choice')
    legend_elements.append(forced_choice_header)
    legend_elements.append(Line2D([0], [0], color=aggregate_colors['Categorical (Forced Choice)'], 
                                marker=aggregate_markers['Categorical (Forced Choice)'], markersize=8,
                                linestyle=aggregate_linestyles['Categorical (Forced Choice)'], 
                                label='  Categorical'))
    legend_elements.append(Line2D([0], [0], color=aggregate_colors['Distributional (Forced Choice)'], 
                                marker=aggregate_markers['Distributional (Forced Choice)'], markersize=8,
                                linestyle=aggregate_linestyles['Distributional (Forced Choice)'], 
                                label='  Distributional'))
    multi_label_header = mpatches.Patch(color='white', alpha=0.0, label='Multi-Label')
    legend_elements.append(multi_label_header)
    legend_elements.append(Line2D([0], [0], color=aggregate_colors['Discrete (Multi-Label)'], 
                                marker=aggregate_markers['Discrete (Multi-Label)'], markersize=8,
                                linestyle=aggregate_linestyles['Discrete (Multi-Label)'], 
                                label='  Discrete'))
    legend_elements.append(Line2D([0], [0], color=aggregate_colors['Continuous (Multi-Label)'], 
                                marker=aggregate_markers['Continuous (Multi-Label)'], markersize=8,
                                linestyle=aggregate_linestyles['Continuous (Multi-Label)'], 
                                label='  Continuous'))
    
    axs[1, 0].legend(handles=legend_elements, loc='upper right', fontsize=14, title='',
                    title_fontsize=14, frameon=True, framealpha=1, ncol=1)
    
    # Now handle the categorical and distributional metric breakdowns (columns 4 and 5)
    for row_idx, eval_measure in enumerate(['consistency', 'bias_mae']):
        # Filter for the current evaluation measure
        measure_df = masked_df[masked_df['eval_measure'] == eval_measure].copy()
        
        # CATEGORICAL METRICS (column 4)
        cat_df = measure_df[measure_df['target_measure'].isin(categorical_metrics)].copy()
        
        # Group by target_measure (metric) and calculate stats
        grouped_cat = cat_df.groupby(['rpi', 'target_measure'])['value'].agg(
            ['mean', 'sem']).reset_index()
        
        # Plot each categorical metric
        for metric in categorical_metrics:
            metric_data = grouped_cat[grouped_cat['target_measure'] == metric]
            
            if len(metric_data) > 0:
                # Get color and marker
                color = detailed_colors.get(metric, '#333333')
                marker = detailed_markers.get(metric, 'o')
                
                # Sort data by rpi for proper fill_between
                metric_data = metric_data.sort_values('rpi')
                
                # Plot the line
                axs[row_idx, 3].plot(metric_data['rpi'], metric_data['mean'], 
                        color=color, alpha=0.8, linewidth=1.5)
                
                # Add shaded error region
                upper = metric_data['mean'] + 2.3*metric_data['sem']
                lower = metric_data['mean'] - 2.3*metric_data['sem']
                axs[row_idx, 3].fill_between(metric_data['rpi'], lower, upper, 
                                        color=color, alpha=0.25)
                
                # Plot scatter points on top
                axs[row_idx, 3].scatter(metric_data['rpi'], metric_data['mean'], 
                        color=color, marker=marker, s=60, 
                        label=mets.METRIC_LOOKUP.get(metric, metric), zorder=10)
        
        # Format the categorical breakdown plots
        apply_formatting(axs[row_idx, 3], labels, cfg, 
                        ylabel='', 
                        xlabel='' if row_idx == 0 else 'Ratings per Item', 
                        subplot_label=f'({chr(65+3+row_idx)})', 
                        position='top' if row_idx == 0 else 'bottom',
                        show_ylabels=False)
        
        # DISTRIBUTIONAL METRICS (column 5)
        dist_df = measure_df[measure_df['target_measure'].isin(distributional_metrics)].copy()
        
        # Group by target_measure (metric) and calculate stats
        grouped_dist = dist_df.groupby(['rpi', 'target_measure'])['value'].agg(
            ['mean', 'sem']).reset_index()
        
        # Plot each distributional metric
        for metric in distributional_metrics:
            metric_data = grouped_dist[grouped_dist['target_measure'] == metric]
            
            if len(metric_data) > 0:
                # Get color and marker
                color = detailed_colors.get(metric, '#333333')
                marker = detailed_markers.get(metric, 'o')
                
                # Sort data by rpi for proper fill_between
                metric_data = metric_data.sort_values('rpi')
                
                # Plot the line
                axs[row_idx, 4].plot(metric_data['rpi'], metric_data['mean'], 
                        color=color, alpha=0.8, linewidth=1.5)
                
                # Add shaded error region
                upper = metric_data['mean'] + 1.8*metric_data['sem']
                lower = metric_data['mean'] - 1.8*metric_data['sem']
                axs[row_idx, 4].fill_between(metric_data['rpi'], lower, upper, 
                                        color=color, alpha=0.25)
                
                # Plot scatter points on top
                axs[row_idx, 4].scatter(metric_data['rpi'], metric_data['mean'], 
                        color=color, marker=marker, s=60, 
                        label=mets.METRIC_LOOKUP.get(metric, metric), zorder=10)
        
        # Format the distributional breakdown plots
        apply_formatting(axs[row_idx, 4], labels, cfg, 
                        ylabel='', 
                        xlabel='' if row_idx == 0 else 'Ratings per Item', 
                        subplot_label=f'({chr(65+4+row_idx)})', 
                        position='top' if row_idx == 0 else 'bottom',
                        show_ylabels=False)
    
    # Add legends for the categorical and distributional breakdowns
    axs[1, 3].legend(loc='upper right', fontsize=14, 
                    title='Categorical Metrics', title_fontsize=14, frameon=True)
    
    axs[1, 4].legend(loc='upper right', fontsize=14,
                    title='Distributional Metrics', title_fontsize=14, frameon=True)
    
    # Hide x-axis labels for top row
    for j in range(5):
        axs[0, j].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Ensure y-axis labels are properly sized and visible
    for i in range(2):
        axs[i, 0].yaxis.label.set_size(22)
        axs[i, 0].tick_params(axis='y', labelsize=20)
    
    # Use subplots_adjust instead of tight_layout to avoid the warning
    plt.subplots_adjust(wspace=0.25, hspace=0.1, left=0.07, right=0.95, top=0.85, bottom=0.1)
    
    return fig


def apply_formatting(ax, labels, cfg, ylabel, subplot_label, xlabel='', fontsize=22, show_ylabels=True, position='top'):
    """Apply common axis formatting"""
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(cfg.n_ratings_per_item)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', labelsize=20)
    # Remove minor ticks/labels
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    if position == 'top':
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim([0.6, 1.0])
    else:
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        ax.set_ylim([0, 0.21])
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)
    ax.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)



def plot_selection_results(df, mask, labels, cfg):

    # Create figure with fixed aspect ratio
    fig = plt.figure(figsize=(20, 8))
    
    width_per_column = 0.15  # Set a standard column width
    spacing = 0.05          # Set consistent spacing between panels

    gs1 = gridspec.GridSpec(2, 1, left=0.08, right=0.08+width_per_column, figure=fig, bottom=0.2, top=0.95)
    gs2 = gridspec.GridSpec(2, 2, left=0.08+width_per_column+spacing, right=0.08+3*width_per_column+spacing, figure=fig, bottom=0.2, top=0.95)
    gs3 = gridspec.GridSpec(2, 2, left=0.08+3*width_per_column+2*spacing, right=0.08+5*width_per_column+2*spacing, figure=fig, bottom=0.2, top=0.95)
        
    # Set common parameters for all GridSpecs
    for gs in [gs1, gs2, gs3]:
        gs.update(hspace=0.1, wspace=0.2)
    
    # Create subplots
    ax1 = fig.add_subplot(gs1[0, 0])  # Panel 1 top
    ax2 = fig.add_subplot(gs1[1, 0])  # Panel 1 bottom
    
    ax3 = fig.add_subplot(gs2[0, 0])  # Panel 2 top-left
    ax4 = fig.add_subplot(gs2[1, 0])  # Panel 2 bottom-left
    ax5 = fig.add_subplot(gs2[0, 1])  # Panel 2 top-right
    ax6 = fig.add_subplot(gs2[1, 1])  # Panel 2 bottom-right
    
    ax7 = fig.add_subplot(gs3[0, 0])  # Panel 3 top-left
    ax8 = fig.add_subplot(gs3[1, 0])  # Panel 3 bottom-left
    ax9 = fig.add_subplot(gs3[0, 1])  # Panel 3 top-right
    ax10 = fig.add_subplot(gs3[1, 1]) # Panel 3 bottom-right
    

    # ##################################################
    #  First Column, Top Row -- Fully Specified Rating Task, no error
    condition_mask = mask & (df['eval_measure'] == 'consistency') &\
                             (df['n_options'] == df['n_response_sets']) & (df['error_rate'] == 0)
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax1,
        linewidth=2,
        legend=False
    )
    apply_formatting(ax1, labels, cfg, ylabel='Decision Consistency', xlabel='', subplot_label='(A)', position='top')
    ax1.set_title('Fully Specified \n No Error', pad=15, fontsize=16)



    # First Column, Bottom Row 
    condition_mask = mask & (df['eval_measure'] == 'bias_mae') &\
                     (df['n_options'] == df['n_response_sets']) & (df['error_rate'] == 0)
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax2,
        linewidth=2,
        legend=False
    )
    apply_formatting(ax2, labels, cfg, ylabel='Estimation Bias (MAE)', xlabel='Ratings per Item', subplot_label='(D)', position='bottom')

    # ##################################################

    # ##################################################
    # Second Column, Top Row - Underspecified, Symmetric Selection Effects, no error
    condition_mask = mask & (df['eval_measure'] == 'consistency') & (df['n_options'] < df['n_response_sets']) &\
                    (df['human_beta'] ==  df['judge_beta']) & (df['error_rate'] == 0)
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax3, 
        legend=False,
        linewidth=2
    )
    apply_formatting(ax3, labels, cfg, ylabel='', xlabel='', subplot_label='(B)', show_ylabels=False, position='top')
    ax3.set_title("Underspecified \n Symmetric  \n No Error", pad=15, fontsize=16)

    # Second Column, Bottom Row - Underspecified, symmetric forced choice bias, no error
    condition_mask = mask & (df['eval_measure'] == 'bias_mae') & (df['n_options'] < df['n_response_sets']) &\
                         (df['human_beta'] == df['judge_beta']) & (df['error_rate'] == 0)
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax4,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax4, labels, cfg, ylabel='', xlabel='Ratings per Item', subplot_label='(C)', show_ylabels=False, position='bottom')
    
    
    ##################################################
    # Third Column, Top Row - Underspecified, asymmetric , no error
    
    ax5.set_title("Underspecified  \n Asymmetric  \n No Error", pad=15, fontsize=16)
    condition_mask = mask & (df['eval_measure'] == 'consistency') & (df['n_options'] < df['n_response_sets']) &\
                    (df['human_beta'] !=  df['judge_beta']) & (df['error_rate'] == 0) 
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax5,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax5, labels, cfg, ylabel='', xlabel='', subplot_label='(E)', show_ylabels=False, position='top')


    # Third Column, Bottom Row - Underspecified, asymmetric , no error
    condition_mask = mask & (df['eval_measure'] == 'bias_mae') & (df['n_options'] < df['n_response_sets']) &\
                    (df['human_beta'] !=  df['judge_beta']) & (df['error_rate'] == 0) 
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax6,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax6, labels, cfg, ylabel='', xlabel='Ratings per Item', subplot_label='(F)', show_ylabels=False, position='bottom')

    ##################################################

    ##################################################
    # Forth Column, Top Row - Underspecified, asymmetric , error, no skew
    
    condition_mask = mask & (df['eval_measure'] == 'consistency') & (df['n_options'] < df['n_response_sets']) &\
                    (df['human_beta'] == 2) &\
                    (df['error_rate'] == 0.3)  & (df['skew'] == 0) 
    
    ax7.set_title("Underspecified \n Asymmetric  \n Random Error", pad=15, fontsize=16)
    
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax7,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax7, labels, cfg, ylabel='', xlabel='', subplot_label='(E)', show_ylabels=False, position='top')


    # Forth Column, Bottom Row - Underspecified, asymmetric , error, no skew
    condition_mask = mask & (df['eval_measure'] == 'bias_mae') & (df['n_options'] < df['n_response_sets']) &\
                            (df['human_beta'] == 2) &\
                              (df['error_rate'] == 0.3) & (df['skew'] == 0) 

    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax8,
        legend=True,
        linewidth=2
    )
    apply_formatting(ax8, labels, cfg, ylabel='', xlabel='Ratings per Item', subplot_label='(F)', show_ylabels=False, position='bottom')

    ##################################################

    ##################################################
    # Fifth Column, Top Row - Underspecified, asymmetric , error, skew
    
    condition_mask = mask & (df['eval_measure'] == 'consistency') & (df['n_options'] < df['n_response_sets']) &\
                    (df['human_beta'] == 2) &\
                    (df['error_rate']  == 0.3)  & (df['skew'] == 1) 

    ax9.set_title("Underspecified \n Asymmetric  \n Additive Error", pad=15, fontsize=16)
    
    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax9,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax9, labels, cfg, ylabel='', xlabel='', subplot_label='(E)', show_ylabels=False, position='top')


    # Fifth Column, Bottom Row - Underspecified, asymmetric , error, skew
    condition_mask = mask & (df['eval_measure'] == 'bias_mae') & (df['n_options'] < df['n_response_sets']) &\
                             (df['human_beta'] == 2) &\
                              (df['error_rate'] == 0.3) & (df['skew'] == 1) 

    sns.lineplot(
        data=df[condition_mask],
        x='rpi',
        y='value',
        hue='target_measure',
        marker='o',
        errorbar='se',
        ax=ax10,
        legend=False,
        linewidth=2
    )
    apply_formatting(ax10, labels, cfg, ylabel='', xlabel='Ratings per Item', subplot_label='(F)', show_ylabels=False, position='bottom')

    ##################################################

    # # Hide y-axis labels for all subplots except leftmost ones (ax1 and ax2)
    # for ax in [ax9]:  # top row except first
    #     ax.set_yticklabels([])
    # # for ax in [ax6, ax8, ax10]:  # bottom row except first
    # #     ax.set_yticklabels([])

    for ax in [ax1, ax3, ax5, ax7, ax9]:  # bottom row except first
        ax.set_xticklabels([])
        
    # Update legend to be on the first plot now (originally was on third)
    legend = ax8.legend(title='Agreement Metric', fontsize=12, title_fontsize=12)
    title = legend.get_title()

    for t in ax8.get_legend().texts:
        # Replace each legend text with its lookup value
        t.set_text(metrics.METRIC_LOOKUP[t.get_text()])

    return fig


def plot_error_ranking_corr_results(results):

    results.drop(columns=['bias', 'consistency', 'bias_mae', 'bias_mse'], inplace=True)

    # First row: only equal sets
    equal_sets = results[results['n_options'] == results['n_response_sets']]
    equal_combinations = equal_sets.groupby(['human_beta', 'judge_beta']).groups

    # Other rows: original grouping but only unequal sets
    unequal_sets = results[results['n_options'] != results['n_response_sets']]
    unequal_combinations = unequal_sets.groupby(['human_beta', 'judge_beta']).groups

    cols = ['MSE_srs_srs', 'JSD_s_s', 'HR_h_h']
    line_styles = ['--', '-', ':']
    metric_colors = {'MSE_srs_srs': 'blue', 'HR_h_h': 'red', 'JSD_s_s': 'green'}

    # Create figure with 4 subfigures (rows)
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    subfigs = fig.subfigures(nrows=4, ncols=1)

    # First row: equal sets
    subfigs[0].suptitle("            Fully Specified Rating Task", size=14, horizontalalignment='center')
    axes = subfigs[0].subplots(nrows=1, ncols=3, sharey=True)

    for idx, col in enumerate(cols):
        for i, skew_val in enumerate(sorted(equal_sets['skew'].unique())):
            skew_data = equal_sets[equal_sets['skew'] == skew_val]
            sns.lineplot(data=skew_data, 
                        x='error_rate', 
                        y=col,
                        label=f'skew={skew_val}',
                        color=metric_colors[col],
                        linestyle=line_styles[i],
                        ax=axes[idx])

        axes[idx].set_title(metrics.METRIC_LOOKUP[col], size=14)
        axes[idx].set_xlabel('Error Rate ($\epsilon$)', size=14)
        if idx == 0:
            axes[idx].set_ylabel('Correlation in \nJudge System Rankings ($\\rho$)', size=14)
        axes[idx].legend(fontsize=12, title_fontsize=12)
        axes[idx].grid(True, axis='both', linestyle='--', alpha=0.4, zorder=0)
        axes[idx].set_ylim([0.3, 1.03])
        axes[idx].set_xlim([0, 0.5])

    # Remaining rows: unequal sets with original grouping
    titles = ['$\Gamma^h=1, \Gamma^j=1$', '$\Gamma^h=.5, \Gamma^j=2$', '$\Gamma^h=2, \Gamma^j=.5$']
    for row_idx, (h_beta, j_beta) in enumerate(unequal_combinations, start=1):

        mask = (unequal_sets['human_beta'] == h_beta) & \
               (unequal_sets['judge_beta'] == j_beta)
        subset = unequal_sets[mask]

        subfigs[row_idx].suptitle(f"            Underspecified Rating Task ({titles[row_idx-1]})", size=14, horizontalalignment='center')
        axes = subfigs[row_idx].subplots(nrows=1, ncols=3, sharey=True)

        for idx, col in enumerate(cols):
            for i, skew_val in enumerate(sorted(subset['skew'].unique())):
                skew_data = subset[subset['skew'] == skew_val]
                sns.lineplot(data=skew_data, 
                            x='error_rate', 
                            y=col,
                            label=f'skew={skew_val}',
                            color=metric_colors[col],
                            linestyle=line_styles[i],
                            ax=axes[idx])

            axes[idx].set_title(metrics.METRIC_LOOKUP[col], size=14)
            axes[idx].set_xlabel('Error Rate ($\epsilon$)', size=14)
            if idx == 0:
                axes[idx].set_ylabel('Correlation in \nJudge System Rankings ($\\rho$)', size=14)
            axes[idx].legend(fontsize=12, title_fontsize=12)
            axes[idx].grid(True, axis='both', linestyle='--', alpha=0.4, zorder=0)
            axes[idx].set_ylim([0.3, 1.03])
            axes[idx].set_xlim([0, 0.5])

    return fig



def plot_error_hr_ablation_exp(df):
    # Create figure and subplots - now in 2x4 layout with shared y-axis per row
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey='row', sharex='col')

    eval_measures = ['consistency', 'bias_mae']

    for row, eval_measure in enumerate(eval_measures):
        # Define masks
        masks = [
            # Mask 1
            (df['eval_measure'] == eval_measure) & 
            (df['human_beta'] == 0) & 
            (df['target_measure'] == 'HR_h_h') & 
            (df['J'] < df['M']),

            # Mask 2
            (df['eval_measure'] == eval_measure) & 
            (df['human_beta'] == 2) & 
            (df['target_measure'] == 'HR_h_h') & 
            (df['J'] < df['M']),

            # Mask 3 
            (df['eval_measure'] == eval_measure) & 
            (df['human_beta'] != 0) & 
            (df['target_measure'] == 'HR_h_h') & 
            (df['J'] < df['M']),

            # Mask 4
            (df['eval_measure'] == eval_measure) & 
            (df['target_measure'] == 'HR_h_h') & 
            (df['J'] == df['M'])
        ]


        titles = [
            'Underspecified Rating Task \n $\Gamma^h=1$, $\Gamma^j=1$',
            'Underspecified Rating Task \n $\Gamma^h=2$, $\Gamma^j=.5$',
            'Underspecified Rating Task \n $\Gamma^h=.5$, $\Gamma^j=2$',
            'Fully Specified Rating Task'
        ]

        # Create each subplot in the current row
        for idx, (ax, mask, title) in enumerate(zip(axes[row], masks, titles)):
            sns.lineplot(
                data=df[mask],
                x='error_rate',
                y='value',
                hue='skew',
                marker='o',
                linewidth=2,
                ax=ax,
                legend=None if not (idx == 3 and row == 1 ) else True,
                palette="tab10"
            )
            ax.set_title(title, fontsize=17)
            ax.set_xlabel(r'Rater Error Rate ($\epsilon$)', fontsize=17)
            # Only set ylabel for the leftmost plots in each row
            if idx == 0:
                ax.set_ylabel(('Consistency' if eval_measure == 'consistency' else 'Bias MAE'), fontsize=17)
            else:
                ax.set_ylabel('')

            if idx == 3 and row == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(fontsize=15, title='Error Skew ($\eta$)', title_fontsize=15)

            ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()


def elicitation_error_prevalence_exp(df, cfg):

    # Create the main figure
    fig = plt.figure(figsize=(12, 12))
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.2)  # Increased hspace for more space between rows

    # Define subplot colors and line styles
    subplot_colors = ['#264653', '#e76f51', '#2a9d8f']
    line_styles = ['--', '-', ':']

    # Custom y-positions for each row title
    y_positions = [1.12, 1.12, 1.12]  # Adjust these values as needed

    # For each tau value (row)
    for row, tau in enumerate(cfg.taus):
        # Filter data for this tau
        tau_data = df[df['tau'] == tau]

        # Create a subfigure title with custom positioning
        subfigs[row].suptitle(f'Cutoff ($\\tau = {tau}$)', 
                              fontsize=15, 
                              y=y_positions[row],
                              x=0.5)  # Centered horizontally

        # Create 3 axes for this subfigure (for the 3 beta values)
        axes = subfigs[row].subplots(1, 3, sharey=True, sharex=True)

        # For each beta value (column)
        for col, beta in enumerate([0, 1, 2]):
            # Filter data for this beta
            beta_data = tau_data[tau_data['beta'] == beta]

            # Get unique skew values
            skew_values = sorted(beta_data['skew'].unique())

            # Plot each skew line
            for i, skew in enumerate(skew_values):
                sns.lineplot(
                    data=beta_data[beta_data['skew'] == skew],
                    x='error_rate', y='bias',
                    errorbar='sd', ax=axes[col],
                    color=subplot_colors[col],
                    linestyle=line_styles[i],
                    label=f"Skew = {skew}"
                )

            # Set titles for columns
            if beta == 0:
                axes[col].set_title(r'Rand. Selection ($\Gamma^h = 1$)', fontsize=14)
            elif beta == 1:
                axes[col].set_title(r'Neg. Selection Effects ($\Gamma^h = .5$)', fontsize=14)
            else:
                axes[col].set_title(r'Pos. Selection Effects ($\Gamma^h = 2$)', fontsize=14)

            # Add grid to all plots
            axes[col].grid(True, axis='both', linestyle='--', alpha=0.6, zorder=0)

            # X-axis label
            axes[col].set_xlabel('Error Magnitude ($\epsilon$)', fontsize=14)

            # Y-axis label (only leftmost column)
            if col == 0:
                axes[col].set_ylabel('Prevalence Estimation Bias', fontsize=14)


    # Set uniform y-limits for all plots
    for row in range(3):
        for col in range(3):
            subfigs[row].axes[col].set_ylim([-1, 0.5])

    # Adjust subplot spacing within each subfigure
    for row in range(3):
        subfigs[row].subplots_adjust(wspace=0.05)  # Reduce horizontal space between plots

    # Use tight_layout for the overall figure with top margin adjustment
    plt.subplots_adjust(top=0.95)
    
    return fig
