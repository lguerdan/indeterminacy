import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
import copy
import warnings
from pathlib import Path

# Custom imports
from core.rating_model import RatingModel
from core import metrics as mets
from config import tasks

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class Visualizer: 

    def __init__(self):

        self.rating_model = RatingModel()

    ############################################
    ###### Functions to score judge systems given raw results
    ############################################

    def score_judges_subsample(self, run_results, task_name, betas, taus,
            resample=True, n_samples=50, items_per_sample=100,
            estimate_F=False, n_f_samples=1, approach='grand_mean', random_seed=42):
        """
        Evaluate judge performance with optional bootstrap resampling.
        
        Parameters:
        -----------
        run_results : dict
            Dictionary containing the results to be evaluated
        task_name : str
            The specific task to evaluate
        betas : list
            List of beta values to evaluate
        taus : list
            List of tau values to evaluate
        resample : bool, default=True
            Whether to perform bootstrap resampling
        n_samples : int, default=50
            Number of bootstrap samples to generate
        items_per_sample : int, default=100
            Number of items to include in each bootstrap sample
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing performance metrics for each combination of judge model, beta, and tau
        """
        if not resample: 
            return self.score_judges(run_results, task_name, betas, taus,  estimate_F, n_f_samples, approach)

        results = []

        for sample in range(n_samples):
            run_results_subsample = self.subsample_results(task_name, run_results, items_per_sample, random_seed=random_seed+sample)
            rank_df = self.score_judges(run_results_subsample, task_name, betas, taus, estimate_F, n_f_samples, approach)
            rank_df['sample'] = sample
            results.append(rank_df)

        return pd.concat(results) 


    def subsample_results(self, task_name, run_results, items_per_sample, random_seed=42):
        """
        Create a subsampled version of run_results by randomly sampling rows from arrays and DataFrames.
        
        Parameters:
        -----------
        run_results : dict
            Original results dictionary containing nested arrays and DataFrames
        items_per_sample : int
            Number of items to sample per bootstrap
            
        Returns:
        --------
        dict
            Subsampled copy of the original results
        """

        np.random.seed(random_seed)

        ra_subset = copy.deepcopy(run_results)

        ratings_subset = ra_subset[task_name]['ratings'].copy()    
        n_items = ratings_subset.shape[0]
        subsample_inds = np.random.choice(n_items, size=min(items_per_sample, n_items), replace=False)

        ra_subset[task_name]['ratings'] = ratings_subset.iloc[subsample_inds]

        for model in ra_subset[task_name]['judge_results'].keys():
            resp_table = ra_subset[task_name]['judge_results'][model]['resp_table'].copy()
            ra_subset[task_name]['judge_results'][model]['resp_table'] = resp_table[subsample_inds]

            decomp_keys = ra_subset[task_name]['judge_results'][model]['p_judge_hat'].keys()

            for key in decomp_keys:
                if key != 'rs_option_lookup':
                    # Filter down into the subset of rows.
                    arr = ra_subset[task_name]['judge_results'][model]['p_judge_hat'][key].copy()
                    ra_subset[task_name]['judge_results'][model]['p_judge_hat'][key] = arr[subsample_inds]

        return ra_subset
    
    def score_judges(self, run_results, task_name, betas, taus, estimate_F=False, n_f_samples=1, approach='grand_mean'):
        """
        Calculate performance metrics for judge models across different parameter settings.
        
        Parameters:
        -----------
        run_results : dict
            Dictionary containing the results to be evaluated
        task_name : str
            The specific task to evaluate
        betas : list
            List of beta values to evaluate
        taus : list
            List of tau values to evaluate
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing performance metrics for each combination of judge model, beta, and tau
        """
        corpus = run_results[task_name]['ratings']
        judge_results = run_results[task_name]['judge_results']
        task_config = run_results[task_name]['task_config']
        positive_options = task_config['positive_categorization_options']

        results = []

        for model, jperf in judge_results.items():
            for beta in betas:
                p_human = self.rating_model.construct_human_rating_distribution(
                    corpus, task_name, task_config, beta
                )

                # If we are estimating F, sample from the rating distribution and reconstruct F
                if estimate_F == True:
                    p_human_hat = self.rating_model.sample_and_estimate_dgp(
                        p_human, n_samples=n_f_samples, estimation_approach=approach)
                else: 
                    p_human_hat = None

                for tau in taus:
                    agreement_metrics = mets.compute_performance_metrics(
                        p_human,
                        jperf['p_judge_hat'],
                        classification_j=positive_options,
                        classification_tau=tau,
                        h_rs_F_hat=p_human_hat
                    )


                    agreement_metrics['tau'] = tau
                    agreement_metrics['beta'] = beta
                    agreement_metrics['model'] = model
                    results.append(agreement_metrics)
                    
        return pd.DataFrame(results)

    ####################################################################################
    ######## Functions to generate beta values
    ####################################################################################

    def get_beta_value(self, v, mv):
    
        resp_table = mv['resp_table']
        
        pos_options = v['task_config']['positive_categorization_options']
        rs_lookup = mv['p_judge_hat']['rs_option_lookup']
        neg_options = [i for i in range(rs_lookup.shape[0]-1) if i not in pos_options]
        
        item_mask = mv['p_judge_hat']['O'][:, neg_options] > 0

        # (n_items, n_samples) where each value is one if the trial returned a negative FC option
        neg_option_trials = resp_table[:,neg_options,:,:].sum(axis=1).sum(axis=1)


        rs_lookup = mv['p_judge_hat']['rs_option_lookup']
        pos_rs = rs_lookup[pos_options,:].sum(axis=0) >= 1


        pos_rep_set_trials = resp_table[:,:,pos_rs,:].sum(axis=1).sum(axis=1)

        # Initialize an array to store fractions for each row
        fractions = np.zeros(pos_rep_set_trials.shape[0])

        # Calculate fraction for each row (first dimension)
        for i in range(pos_rep_set_trials.shape[0]):
            # Create mask where neg_option_trials equals 1 for this row
            mask = (neg_option_trials[i] == 1)

            # Skip rows where mask has no True values to avoid division by zero
            if np.sum(mask) > 0:
                # Count elements in pos_rep_set_trials that are 1 where the mask is True
                numerator = np.sum((pos_rep_set_trials[i] == 1) & mask)

                # Count total number of elements where the mask is True
                denominator = np.sum(mask)

                # Calculate the fraction for this row
                fractions[i] = numerator / denominator

        # Calculate the average fraction across all rows
        average_fraction = np.mean(fractions)
        
        return average_fraction


    def get_beta_distribution(self, run_results, n_samples=30, items_per_sample=100, random_seed=42):

        results = {}
        results['task'] = []
        results['model'] = []
        results['beta'] = []
        results['sample'] = []


        # Enumerate over tasks
        for k, v in run_results.items():

            # Repeatedly sub-sample
            for sample in range(n_samples):

                run_results_subsample = self.subsample_results(k, run_results, items_per_sample=items_per_sample, random_seed=random_seed+sample)
                judge_results = run_results_subsample[k]['judge_results']

                for mk, mv in judge_results.items():

                    beta_m = self.get_beta_value(v, mv)

                    results['task'].append(k)
                    results['model'].append(mk)
                    results['beta'].append(beta_m)
                    results['sample'].append(sample)
        
        return pd.DataFrame(results)

    def plot_beta_distributions(self, df, palette, display_names, plot_full=False):

        if plot_full == True:
            models_to_compare = palette.keys()
            figsize=(20, 18)
            n_legend_cols=2
        else:
            models_to_compare = ["mistral-mistral-small-latest", "openai-gpt-3.5-turbo", "anthropic-claude-3-haiku-20240307"]
            figsize=(8, 16)
            n_legend_cols=1


        # Filter for the models included in the comparison
        filtered_df = df[df["model"].isin(models_to_compare)].copy()

        # Set the style
        sns.set(style="whitegrid")
        plt.figure(figsize=figsize)


        ax = sns.boxplot(y="task", x="beta", hue="model", data=filtered_df, 
                        palette=palette, width=0.6, orient='h',saturation=1.0)


        # Improve readability
        plt.yticks(fontsize=39)
        plt.xticks(fontsize=25)
        plt.ylabel('')
        plt.xlabel(r'Sensitivity Parameter Estimate ($\hat{\beta}^J_t$)', fontsize=27)

        # Add grid lines for better readability of values
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Fix legend with display names
        handles, labels = ax.get_legend_handles_labels()

        # Create new display labels by mapping the original model names to display names
        display_labels = []
        for label in labels[:len(models_to_compare)]:
            if label in display_names:
                display_labels.append(display_names[label])
            else:
                display_labels.append(label)  # Fall back to original name if not in dictionary

        # Create the legend with display names
        plt.legend(handles[:len(models_to_compare)], display_labels, 
                title="Model", fontsize=22, title_fontsize=22,
                ncols=n_legend_cols, loc='upper right')

        # Set x-axis limits with some padding
        max_beta = filtered_df['beta'].max()
        min_beta = filtered_df['beta'].min()
        plt.xlim(min_beta - 0.02, max_beta + 0.02)

        # Sort tasks by the median beta value across models
        task_order = list(tasks.TASK_CONFIGS.keys())
        
        # Set tick positions AND labels
        ax = plt.gca()
        ax.set_yticks(range(len(task_order)))
        ax.set_yticklabels([tasks.TASK_CONFIGS[t]['display'] for t in task_order], fontsize=28)
        
    ####################################################################################
    
    
    def create_task_summary(self, run_results):
        """
        Create a summary DataFrame with basic information about each task.
        
        Parameters:
        -----------
        run_results : dict
            Dictionary containing results for each task
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing task information including name, property, and number of items
        """
        tasks = []
        
        for task_name, value in run_results.items():
            df = value['ratings']
            
            task_info = {}
            task_info['name'] = task_name
            task_info['property'] = value['task_config']['property']
            task_info['n_items'] = df.shape[0]
            task_info['n_options'] = value['task_config']['n_options']
            task_info['n_response_sets'] = value['task_config']['n_response_sets']
            task_info['n_items'] = df.shape[0]
            task_info['ratings_per_item'] =  value['task_config']['ratings_per_item']
            
            tasks.append(task_info)
        
        # Create DataFrame from the collected task information
        task_info_df = pd.DataFrame(tasks)

        task_info_df = task_info_df.rename(columns={
            'name': 'Task',
            'property':'Propertiy',
            'n_items': 'Items'
        })
        
        return task_info_df


    ######## Rank analysis post-processing

    def get_diff_df(self, betas, taus, rank_df, downstream_metric):


        diff_dict = {}
        diff_dict['beta'] = []
        diff_dict['tau'] = []
        diff_dict['sample'] = []
        diff_dict['metric'] = []
        diff_dict['abs_difference'] = []
        diff_dict['pct_difference'] = []

        samples = rank_df['sample'].unique()

        for beta in betas:
            for tau in taus:
                for sample in samples:

                    # Compute HJ agreement assuming FC distribution = RS distribution (beta=0)
                    hj_df = rank_df[(rank_df['beta'] == 0) & (rank_df['tau'] == tau) & (rank_df['sample'] == sample)]


                    # Compute DS metric using true RS distribution
                    downstream_df = rank_df[(rank_df['beta'] == beta) & (rank_df['tau'] == tau) & (rank_df['sample'] == sample)]

                    opt_ind_downstream = mets.METRIC_TABLE[downstream_metric]['opt_ind']
                    ascending_downstream = (opt_ind_downstream == 0)
                    best_model_downstream = downstream_df.sort_values(by=downstream_metric, ascending=ascending_downstream).iloc[0]

                    # Lower=Better => ascending=True

                    for metric in mets.TARGET_METRICS:

                        # Sort by metric (check ordering based on metric type) and pick best model
                        opt_ind_hj_agreement = mets.METRIC_TABLE[metric]['opt_ind']
                        ascending_hj_agreement = (opt_ind_hj_agreement == 0)

                        # Best model w.r.t. H-J agreement 
                        if metric in mets.FC_RS_GROUPING['FC']:
                            # Compute at the value beta=0
                            best_model_hj_agreement = hj_df.sort_values(by=metric, ascending=ascending_hj_agreement).iloc[0]['model']

                        else:
                            # Compute at the "true" value of beta
                            best_model_hj_agreement = downstream_df.sort_values(by=metric, ascending=ascending_hj_agreement).iloc[0]['model']


                        # Value of consistency at the "true" value of beta, assuming we use the hj-agreement selected model
                        hj_agreement_model = downstream_df[downstream_df['model'] == best_model_hj_agreement].iloc[0]

                        
                        best_value_hj = hj_agreement_model[downstream_metric]
                        best_value_downstream = best_model_downstream[downstream_metric]

                        abs_difference = abs(best_value_downstream - best_value_hj)

                        # Lower values of the downstream metric are better.
                        if ascending_downstream == True:
                            # Calculate percentage difference where lower values are better
                            if best_value_downstream != 0:
                                pct_difference = ((best_value_hj - best_value_downstream) / abs(best_value_downstream)) * 100

                            else:
                                pct_difference = 0.0  # No difference
                            
      
                        # Higher values of the downstream metric are better.
                        else:
                            # Calculate percentage difference where higher values are better
                            if best_value_downstream != 0:
                                # Normal case - can do regular percent calculation
                                pct_difference = ((best_value_downstream - best_value_hj) / abs(best_value_downstream)) * 100
                            else:

                                pct_difference = 0.0  # Both values are 0
                            

                        diff_dict['beta'].append(beta)
                        diff_dict['tau'].append(tau)
                        diff_dict['sample'].append(sample)
                        diff_dict['metric'].append(metric)
                        diff_dict['abs_difference'].append(abs_difference)
                        diff_dict['pct_difference'].append(pct_difference)


        return pd.DataFrame(diff_dict)



    def all_task_raw_metric_df(self, run_results,task_names, betas, taus, 
                n_samples, items_per_sample, estimate_F=False, n_f_samples=1, approach='grand_mean'):


        all_task_dfs = []

        for task_name in task_names: 

            for downstream_metric in ['bias_mae', 'consistency']:


                rank_df = self.score_judges_subsample(
                        run_results, task_name, betas, taus,
                        resample=True, n_samples=n_samples, items_per_sample=items_per_sample,
                        estimate_F=estimate_F, n_f_samples=n_f_samples, approach=approach
                    )


                rank_df['downstream_metric'] = downstream_metric
                rank_df['task_name'] = task_name

                all_task_dfs.append(rank_df)

        return pd.concat(all_task_dfs)

    def all_task_diff_df(self, run_results, task_names, betas, taus,
            downstream_metrics, n_samples, items_per_sample,
            estimate_F=False, n_f_samples=1, approach='grand_mean'):
        
        all_dfs = []

        for downstream_metric in downstream_metrics:
        
            for task_name in task_names:

                rank_df = self.score_judges_subsample(
                    run_results, task_name, betas, taus,
                    resample=True, n_samples=n_samples, items_per_sample=items_per_sample,
                    estimate_F=estimate_F, n_f_samples=n_f_samples, approach=approach
                )
                
                diff_df = self.get_diff_df(betas, taus, rank_df, downstream_metric=downstream_metric)
                diff_df['task'] = task_name
                diff_df['downstream_metric'] = downstream_metric
                
                all_dfs.append(diff_df)
            
        return pd.concat(all_dfs)


    def create_ranking_facit_plots(self, all_metric_df, task_names_facit,
                      metrics_facit, beta_1_values, beta_2_values,
                      display_names, tau_value=0.5):
        """
        Create a figure with facit_value_plot_error_bars that shows the relationship between
        metrics and downstream metrics, with shared styling and legend.

        Parameters:
        -----------
        task_names_facit : list
            List of task names for the facit plot
        all_metric_df : pandas.DataFrame
            DataFrame containing performance metrics
        metrics_facit : list
            List of metrics for the facit plot (last one should be downstream_metric)
        beta_1_values : list
            List of beta parameter values for the first metric in facit plot
        beta_2_values : list
            List of beta parameter values for the second metric in facit plot
        display_names : dict
            Mapping from model names to display names
        tau_value : float, default=0.5
            Tau threshold value to filter results

        Returns:
        --------
        fig : matplotlib figure
            The facit plot figure
        axes : dict
            Dictionary containing all axes from the plot
        """

        # Define common styling elements
        color_palette = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Yellow-green
            '#17becf',  # Cyan
            '#aa40fc',  # Bright purple
            '#aaffc3',  # Mint
            '#000099',  # Dark blue
            '#ff9896',  # Light red
            '#fab0e4'   # Light pink
        ]

        # Predefined marker styles
        marker_styles = [
            'o',  # Circle
            's',  # Square
            '^',  # Triangle up
            'D',  # Diamond
            'p',  # Pentagon
            '*',  # Star
            'X',  # X filled
            'h',  # Hexagon 1
            'v',  # Triangle down
            '>',  # Triangle right
            '<',  # Triangle left
            'd',  # Thin diamond
            'H',  # Hexagon 2
            'P',  # Plus filled
            '8'   # Octagon
        ]

        # Function to calculate standard error of the mean
        def sem(values):
            return stats.sem(values) if len(values) > 1 else 0

        # =================================================================================
        # Facit Plot Setup - First extract downstream_metric and prepare metrics
        # =================================================================================

        # Extract downstream metric (should be the last one in metrics_facit)
        downstream_metric = metrics_facit[-1]

        # Filter out the downstream metric if it's in metrics list
        plot_metrics = [m for m in metrics_facit if m != downstream_metric]
        n_plots_facit = len(plot_metrics)

        # Ensure beta values have the right length for facit plot
        if len(beta_1_values) < n_plots_facit:
            beta_1_values = beta_1_values * n_plots_facit
        if len(beta_2_values) < n_plots_facit:
            beta_2_values = beta_2_values * n_plots_facit

        # Ensure task_names_facit has the right length
        if len(task_names_facit) < n_plots_facit:
            print(f"Warning: Not enough task names ({len(task_names_facit)}) for the number of plots ({n_plots_facit})")
            # Duplicate the last task name if needed
            task_names_facit = task_names_facit + [task_names_facit[-1]] * (n_plots_facit - len(task_names_facit))

        # =================================================================================
        # Find all unique models for consistent color and marker assignment
        # =================================================================================

        # Get the set of all unique models across all filtered conditions
        all_models = set()

        # For facit plot
        for i in range(n_plots_facit):
            task = task_names_facit[i]
            beta1 = beta_1_values[i]
            beta2 = beta_2_values[i]

            # Make sure to filter by the correct task_name for each plot
            df1_all = all_metric_df[(all_metric_df['beta'] == beta1) & 
                                (all_metric_df['tau'] == tau_value) & 
                                (all_metric_df['task_name'] == task)]
            df2_all = all_metric_df[(all_metric_df['beta'] == beta2) & 
                                (all_metric_df['tau'] == tau_value) & 
                                (all_metric_df['task_name'] == task)]

            all_models.update(df1_all['model'].unique())
            all_models.update(df2_all['model'].unique())

        # Create model colors and markers dictionaries
        model_colors = {}
        model_markers = {}

        # Get a sorted list of all unique models for consistent assignment
        sorted_models = sorted(all_models)

        for i, model in enumerate(sorted_models):
            # Assign color from palette (cycling if needed)
            color_index = i % len(color_palette)
            model_colors[model] = color_palette[color_index]

            # Assign marker from styles (cycling if needed)
            marker_index = i % len(marker_styles)
            model_markers[model] = marker_styles[marker_index]

        # =================================================================================
        # Create the figure layout
        # =================================================================================

        # Set up the figure with sizing appropriate for facit plots only
        fig_width = 6 * n_plots_facit  # Adjust width based on number of plots
        fig = plt.figure(figsize=(fig_width, 10))

        # Use GridSpec for layout control
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig, wspace=0.35, hspace=0.4)

        # Create a subgridspec for the facit plots
        gs_facit = gridspec.GridSpecFromSubplotSpec(1, n_plots_facit, subplot_spec=gs[0, 0], wspace=0.3)

        # Create the axes for facit plots
        facit_axes = []
        facit_twin_axes = []
        for i in range(n_plots_facit):
            ax = fig.add_subplot(gs_facit[0, i])
            ax_twin = ax.twinx()
            facit_axes.append(ax)
            facit_twin_axes.append(ax_twin)

        # =================================================================================
        # Facit Plot Implementation
        # =================================================================================

        # First, determine the global min/max values for the downstream metric
        # to ensure consistent y-axis scaling on the right side
        all_downstream_values = []
        all_downstream_sems = []

        # Collect all downstream metric values and their standard errors
        for i in range(n_plots_facit):
            beta2 = beta_2_values[i]

            # Make sure to filter by the correct task_name for each plot
            df_all = all_metric_df[(all_metric_df['beta'] == beta2) & 
                                (all_metric_df['tau'] == tau_value) & 
                                (all_metric_df['task_name'] == task_names_facit[i])]

            # Skip if we don't have the metric or no data
            if downstream_metric not in df_all.columns or df_all.empty:
                continue

            # Compute statistics
            df_stats = df_all.groupby('model')[downstream_metric].agg(['mean', sem]).reset_index()

            all_downstream_values.extend(df_stats['mean'].tolist())
            all_downstream_sems.extend(df_stats['sem'].tolist())

        # Calculate the global min/max with padding for the downstream metric
        if all_downstream_values:
            downstream_min = min(all_downstream_values) - 2 * max(all_downstream_sems) if all_downstream_sems and max(all_downstream_sems) > 0 else min(all_downstream_values)
            downstream_max = max(all_downstream_values) + 2 * max(all_downstream_sems) if all_downstream_sems and max(all_downstream_sems) > 0 else max(all_downstream_values)
            downstream_padding = 0.1 * (downstream_max - downstream_min)  # 10% padding
        else:
            # Default values if no data is found
            downstream_min, downstream_max, downstream_padding = 0, 1, 0.1

        # Create the facit plots
        for i in range(n_plots_facit):
            # Get the current metric and axis
            metric1 = plot_metrics[i]  # Agreement metric
            metric2 = downstream_metric  # Downstream metric
            ax1 = facit_axes[i]
            ax2 = facit_twin_axes[i]

            beta1 = beta_1_values[i]
            beta2 = beta_2_values[i]

            # Filter the data for the specific parameters including task_name
            df1_all = all_metric_df[(all_metric_df['beta'] == beta1) & 
                                (all_metric_df['tau'] == tau_value) & 
                                (all_metric_df['task_name'] == task_names_facit[i])]
            df2_all = all_metric_df[(all_metric_df['beta'] == beta2) & 
                                (all_metric_df['tau'] == tau_value) & 
                                (all_metric_df['task_name'] == task_names_facit[i])]

            # Check if dataframes are empty or missing metrics
            if df1_all.empty or df2_all.empty or metric1 not in df1_all.columns or metric2 not in df2_all.columns:
                # Set default limits and skip this iteration
                ax1.set_ylim(0, 1)
                ax2.set_ylim(0, 1)
                ax1.set_title(f"No data for {task_names_facit[i]}")
                continue

            # Calculate means and standard errors for each model
            df1_stats = df1_all.groupby('model')[metric1].agg(['mean', sem]).reset_index()
            df1_stats.rename(columns={'mean': metric1, 'sem': f'{metric1}_sem'}, inplace=True)

            df2_stats = df2_all.groupby('model')[metric2].agg(['mean', sem]).reset_index()
            df2_stats.rename(columns={'mean': metric2, 'sem': f'{metric2}_sem'}, inplace=True)

            # Merge the data
            merged_df = pd.merge(
                df1_stats,
                df2_stats,
                on='model'
            )

            # Add display names
            merged_df['display_name'] = merged_df['model'].map(display_names)

            # Determine sort direction based on metric optimization goal
            # For metrics where lower is better (opt_ind == 0), we want to sort ascending
            # For metrics where higher is better (opt_ind == 1), we want to sort descending
            try:
                metric1_ascending = (mets.METRIC_TABLE[metric1]['opt_ind'] == 0)
            except (NameError, KeyError):
                # If mets is not defined, use a default (assuming higher is better)
                metric1_ascending = False

            # Sort appropriately so that best-performing models come first
            merged_df = merged_df.sort_values(by=metric1, ascending=metric1_ascending)

            # First, set up the x-values for our data points
            x1, x2 = 0, 1

            # Get metric names for labels
            try:
                metric1_name = mets.METRIC_LOOKUP_SHORT[metric1] if hasattr(mets, 'METRIC_LOOKUP_SHORT') else metric1
                metric2_name = mets.METRIC_LOOKUP_SHORT[metric2] if hasattr(mets, 'METRIC_LOOKUP_SHORT') else metric2
            except (NameError, KeyError):
                # If mets is not defined, use the original metric names
                metric1_name = metric1
                metric2_name = metric2

            # Safely calculate the min/max values with padding for the left axis (agreement metric)
            if not merged_df.empty and f'{metric1}_sem' in merged_df.columns:
                sem_max = merged_df[f'{metric1}_sem'].max()
                if np.isfinite(sem_max) and sem_max > 0:
                    metric1_min = merged_df[metric1].min() - 2 * sem_max
                    metric1_max = merged_df[metric1].max() + 2 * sem_max
                else:
                    metric1_min = merged_df[metric1].min()
                    metric1_max = merged_df[metric1].max()

                # Ensure we don't get flat limits
                if metric1_min == metric1_max:
                    metric1_min -= 0.1
                    metric1_max += 0.1

                metric1_padding = 0.1 * (metric1_max - metric1_min)  # 10% padding
            else:
                # Default values if no data
                metric1_min, metric1_max, metric1_padding = 0, 1, 0.1

            # If lower values are better, invert the y axis
            try:
                if mets.METRIC_TABLE[metric1]['opt_ind'] == 0:
                    ax1.set_ylim(metric1_max + metric1_padding, metric1_min - metric1_padding)
                else:
                    # Regular low-to-high ordering
                    ax1.set_ylim(metric1_min - metric1_padding, metric1_max + metric1_padding)
            except (NameError, KeyError):
                # Default if mets is not defined
                ax1.set_ylim(metric1_min - metric1_padding, metric1_max + metric1_padding)

            # Use the global min/max for the right axis (downstream metric)
            try:
                if mets.METRIC_TABLE[metric2]['opt_ind'] == 0:
                    ax2.set_ylim(downstream_max + downstream_padding, downstream_min - downstream_padding)
                else:
                    ax2.set_ylim(downstream_min - downstream_padding, downstream_max + downstream_padding)
            except (NameError, KeyError):
                # Default if mets is not defined
                ax2.set_ylim(downstream_min - downstream_padding, downstream_max + downstream_padding)

            # Always show y-axis labels on the left axis for all plots
            ax1.tick_params(axis='y', labelleft=True, labelsize=24)

            # Only show y-axis labels on the right axis for the last plot
            if i == n_plots_facit - 1:  # Last plot
                ax2.tick_params(axis='y', labelright=True, labelsize=24)
            else:
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(axis='y', labelright=False)

                # Remove tick marks and labels from secondary axis
                ax2.tick_params(axis='y', which='both', 
                            length=0,           # Set tick length to 0
                            width=0,            # Set tick width to 0
                            labelright=False,   # Hide tick labels
                            right=False)        # Hide tick marks

            # Set x-tick labels
            ax1.set_xticks([x1, x2])

            # Set x-axis limits with extra padding on the right for model names
            ax1.set_xlim(-0.2, 1.2)

            # Only proceed with plotting if we have data
            if not merged_df.empty:
                # Draw connecting lines using a properly scaled approach
                for j, (_, row) in enumerate(merged_df.iterrows()):
                    model = row['model']
                    value1 = row[metric1]
                    value2 = row[metric2]
                    # Use model-specific color from the dictionary
                    color = model_colors[model]

                    # Convert to display coordinates
                    point1 = ax1.transData.transform((x1, value1))
                    point2 = ax2.transData.transform((x2, value2))

                    # Convert back to figure coordinates
                    point1_fig = fig.transFigure.inverted().transform(point1)
                    point2_fig = fig.transFigure.inverted().transform(point2)

                    # Create a line in figure coordinates
                    line = plt.Line2D(
                        [point1_fig[0], point2_fig[0]],
                        [point1_fig[1], point2_fig[1]],
                        transform=fig.transFigure,
                        color=color,
                        alpha=0.7,
                        linewidth=6,
                        zorder=1
                    )

                    # Add the line to the figure
                    fig.add_artist(line)

                # Plot all the points first
                for j, (_, row) in enumerate(merged_df.iterrows()):
                    model = row['model']
                    display_name = row['display_name']
                    value1 = row[metric1]
                    value2 = row[metric2]
                    sem1 = row[f'{metric1}_sem']
                    sem2 = row[f'{metric2}_sem']
                    # Use model-specific color
                    color = model_colors[model]

                    # Left axis points
                    ax1.scatter(x1, value1, 
                    s=200,  # Size 
                    color=color, 
                    marker=model_markers[model],  # Use assigned marker
                    zorder=10)

                    # Add error bars to left axis points
                    ax1.errorbar(
                        x1, value1, 
                        yerr=sem1, 
                        fmt='none', 
                        ecolor=color,
                        elinewidth=5, 
                        capsize=6, 
                        capthick=5, 
                        alpha=0.9,
                        zorder=5
                    )

                    # Right axis points
                    ax2.scatter(x2, value2, 
                    s=200, 
                    color=color, 
                    marker=model_markers[model],  # Use assigned marker
                    zorder=10)

                    # Add error bars to right axis points
                    ax2.errorbar(
                        x2, value2, 
                        yerr=sem2, 
                        fmt='none', 
                        ecolor=color,
                        elinewidth=5, 
                        capsize=6, 
                        capthick=5, 
                        alpha=0.9,
                        zorder=5
                    )

            # Use x-ticks for the metric identification instead
            try:
                direction1 = "↓" if mets.METRIC_TABLE[metric1]['opt_ind'] == 0 else "↑"
                direction2 = "↓" if mets.METRIC_TABLE[metric2]['opt_ind'] == 0 else "↑"
            except (NameError, KeyError):
                # Default if mets is not defined
                direction1 = "↑"
                direction2 = "↑"

            ax1.set_xticklabels([f"{metric1_name} {direction1}", f"{metric2_name} {direction2}"], fontsize=24, rotation=25)

            # Set the title for each subplot with the task name
            try:
                task_display = tasks.TASK_CONFIGS[task_names_facit[i]]['display']
            except (NameError, KeyError):
                task_display = task_names_facit[i]

            ax1.set_title(f"{task_display}\n$\\beta^H_t={beta2}$", fontsize=24)

            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # Remove unneeded spines
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            
            ax1.grid(False)
            ax2.grid(False)

        # Remove spines from all facit plots except the rightmost one
        for i, ax in enumerate(facit_axes):
            # For all plots except the rightmost one
            if i < n_plots_facit - 1:
                # Get the corresponding secondary y-axis
                right_axis = facit_twin_axes[i]

                # Remove right spine from primary (left) axis
                ax.spines['right'].set_visible(False)

                # Remove all spines from the secondary (right) axis
                right_axis.spines['top'].set_visible(False)
                right_axis.spines['bottom'].set_visible(False)
                right_axis.spines['left'].set_visible(False)
                right_axis.spines['right'].set_visible(False)

                # Also make sure no tick marks show on the right axis
                right_axis.tick_params(axis='y', which='both', length=0)

        # For the rightmost facit plot, just ensure the top spines are hidden
        if n_plots_facit > 0:
            # Primary (left) axis
            facit_axes[-1].spines['top'].set_visible(False)

            # Secondary (right) axis
            right_axis = facit_twin_axes[-1]
            right_axis.spines['top'].set_visible(False)
            right_axis.spines['bottom'].set_visible(False)
            right_axis.spines['left'].set_visible(False)

        # Set the title for the entire figure

        # =================================================================================
        # Create a legend
        # =================================================================================

        # Create legend elements
        legend_elements = []
        for model in sorted(all_models):
            if model in display_names:
                legend_elements.append(
                    Line2D([0], [0], 
                        marker=model_markers[model],  # Use assigned marker
                        color=model_colors[model],    # Use assigned color
                        label=display_names[model], 
                        markersize=12,  # Increased marker size
                        markerfacecolor=model_colors[model],
                        linestyle='-', 
                        linewidth=4)
                )

        # Add the legend underneath the facit plots with better positioning
        legend = fig.legend(
            handles=legend_elements,
            loc='upper center',
            frameon=True,
            framealpha=1,
            ncol=3,                  # Adjust number of columns as needed
            fontsize=30,
            bbox_to_anchor=(0.5, 0.2),  # Position underneath facit plots
        )

        # Adjust layout
        # plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust the rect to make room for the title

        # Return the figure and axes
        return fig

    
    def plot_headline_figure(self, task_name, rank_df, tau_value=0.5, betas=None, 
                                        downstream_metric=None, display_names=None, metric_lower_is_better=False):
        """
        Create a line plot showing how model rankings vary with beta values.

        Parameters:
        -----------
        task_name : str
            Name of the task being evaluated
        rank_df : pandas.DataFrame
            DataFrame containing performance metrics across different samples
        tau_value : float, default=0.5
            Tau threshold value to filter results
        betas : list or array
            List of beta values to plot
        downstream_metric : str
            The metric to rank models by
        display_names : dict
            Mapping from model names to display names
        metric_lower_is_better : bool, default=False
            Whether lower values of the metric indicate better performance

        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """

        # Make sure we have valid display names
        if display_names is None:
            display_names = {}

        # Filter the DataFrame for the specific task and tau value
        filtered_df = rank_df[(rank_df['tau'] == tau_value)]

        # Get all unique models in the filtered data
        models = filtered_df['model'].unique()

        # If betas not provided, use all unique beta values in the data
        if betas is None:
            betas = sorted(filtered_df['beta'].unique())

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set up a colormap for different models
        # Get original tab10 colors
        original_colors = list(get_cmap('tab10').colors)

        # Define a specific shuffled order
        # Define a new shuffled order with different indices
        shuffled_colors = [
            original_colors[5],  # Green
            original_colors[7],  # Gray
            original_colors[4],  # Purple-ish
            original_colors[0],  # Blue
            original_colors[9],  # Light green
            original_colors[1],  # Red
            original_colors[6],  # Brown
            original_colors[3],  # Orange
            original_colors[8],  # Pink
            original_colors[2],  # Purple
        ]


        # Create a new colormap
        cmap = mcolors.ListedColormap(shuffled_colors, name='tab10_shuffled')

        
        # Create a marker cycle
        markers = ['o', 's', '^', 'D', 'p','h', '*', 'X',  'v', '>', '<', 'd', 'H', 'P', '8']

        # Store rankings for each beta value
        all_rankings = {}

        # First, calculate rankings at each beta value
        for beta in betas:
            beta_data = filtered_df[filtered_df['beta'] == beta]

            # Calculate mean performance for each model at this beta
            model_means = {}
            for model in models:
                model_data = beta_data[beta_data['model'] == model]
                if not model_data.empty:
                    model_means[model] = model_data[downstream_metric].mean()

            # Sort models by performance
            # If lower is better, use ascending=True; if higher is better, use ascending=False
            reverse = not metric_lower_is_better  # reverse=True means descending order (higher is better)
            sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=reverse)

            # Assign rankings (1 = best)
            rankings = {}
            for rank, (model, _) in enumerate(sorted_models, 1):
                rankings[model] = rank

            # Fill in NaN for models with no data at this beta
            for model in models:
                if model not in rankings:
                    rankings[model] = np.nan

            all_rankings[beta] = rankings

        # Identify which models are ever ranked #1
        top_models = set()
        for beta in betas:
            beta_rankings = all_rankings[beta]
            for model, rank in beta_rankings.items():
                if rank == 1:
                    top_models.add(model)

        # Plot rankings for each model
        for i, model in enumerate(sorted(models)):
            display_name = display_names.get(model, model)
            color = cmap(i % 10)
            marker = markers[i % len(markers)]

            # Get rankings for this model across all betas
            model_rankings = []
            for beta in betas:
                model_rankings.append(all_rankings[beta].get(model, np.nan))

            # Set visual properties based on whether this model is ever ranked #1
            if model in top_models:
                alpha = 1.0  # Full opacity for top-ranked models
                linewidth = 5  # Original thickness
                markersize = 18  # Original size
                zorder = 2  # Bring to front
            else:
                alpha = 0.3  # Reduced opacity for others
                linewidth = 3  # Thinner line
                markersize = 16  # Smaller markers
                zorder = 1  # Send to back

            # Plot the line for this model
            ax.plot(
                betas,
                model_rankings,
                label=display_name,
                color=color,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder
            )

        # Set labels and title
        ax.set_ylabel('Rank', fontsize=30)

        # Invert y-axis so that rank 1 is at the top
        ax.invert_yaxis()

        # Set tick label sizes
        ax.tick_params(axis='both', labelsize=24)

        # Add a legend
        ax.legend(
            fontsize=16,
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=0.9,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.48),  # Position above the plot
            ncol=3,
            columnspacing=1.5,
            handletextpad=0.5,
            borderpad=0.5
        )

        # Set axis limits
        ax.set_xlim(min(betas) - 0.05, max(betas) + 0.05)
        ax.set_xticks(betas)
        
        ax.set_xticks([0.0, 0.3])
        ax.set_xticklabels(['Ranking Under\nForced Choice Elicitation', 
                            'Ranking Under\nResponse Set Elicitation'], 
                        fontsize=16, ha='center')
        
        # Increase x-axis label size
        ax.tick_params(axis='x', labelsize=20)

        # Set y-axis to show integer ranks
        max_rank = len(models)
        ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_yticks(range(1, max_rank + 1))

        # Tight layout for better spacing
        plt.tight_layout()

        return fig, ax

    def plot_aggregate_loss_scores(self, run_results=None, df=None, 
                                    categorical_metrics=None, distributional_metrics=None,
                                    tau_filter=None, difference_metric='abs_difference', 
                                    figsize=(24, 14), cache_path=None, 
                                    task_names=None, betas=None, taus=None,
                                    downstream_metrics=None, n_samples=10, 
                                    estimate_F=False, n_f_samples=1,
                                    approach='grand_mean',
                                    items_per_sample=100, force_recompute=False):
        """
        Generate a comprehensive transposed plot with:
        - Row 1: Consistency (Aggregate, Categorical breakdown, Distributional breakdown)
        - Row 2: Bias MAE (Aggregate, Categorical breakdown, Distributional breakdown)

        Parameters:
        - df: pandas DataFrame with your data
        - categorical_metrics: list of categorical metrics to include in the categorical breakdown
        - distributional_metrics: list of distributional metrics to include in the distributional breakdown
        - mets: metrics module containing METRIC_LOOKUP_SHORT for metric name mapping
        - tau_filter: value to filter tau by (if None, will use all tau values)
        - difference_metric: which difference metric to plot ('abs_difference' or 'pct_difference')
        - figsize: size of the figure (width, height)

        Returns:
        - fig, axs: matplotlib figure and axes objects
        """
        
        # Handle data loading/computation
        if df is None:
            if cache_path is not None:
                cache_path = Path(cache_path)
                
                # Check if cache exists and we're not forcing recomputation
                if cache_path.exists() and not force_recompute:
                    print("Loading cached results...")
                    df = pd.read_csv(cache_path)
                else:
                    print("Computing scores...")
                    if run_results is None:
                        raise ValueError("run_results must be provided when computing from scratch")
                    
                    # Set defaults for computation parameters
                    if task_names is None:
                        task_names = list(run_results.keys())
                    if betas is None:
                        betas = np.arange(0, 0.7, 0.1).round(2)
                    if taus is None:
                        taus = np.arange(0, 1.1, 0.1)
                    if downstream_metrics is None:
                        downstream_metrics = ['consistency', 'bias_mae']
                    
                    df = self.all_task_diff_df(
                        run_results,
                        task_names=task_names,
                        betas=betas,
                        taus=taus,
                        downstream_metrics=downstream_metrics,
                        estimate_F=estimate_F,
                        n_f_samples=n_f_samples,
                        approach=approach,
                        n_samples=n_samples,
                        items_per_sample=items_per_sample
                    )
                    
                    # Save to cache
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                    print(f"Results cached to {cache_path}")
            else:
                raise ValueError("Either df or cache_path must be provided")

        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create a filtered copy of the dataframe
        plot_df = df.copy()

        # Apply tau filter if specified
        if tau_filter is not None:
            plot_df = plot_df[plot_df['tau'] == tau_filter]

        # Define downstream metrics
        downstream_metrics = ['consistency', 'bias_mae']

        fig = plt.figure(figsize=figsize, dpi=150)
        axs = np.array([
            [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3)],
            [plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]
        ])

        # Then manually set sharing
        for j in range(1, 3):
            # Make subplots in the same row share y-axis with the first subplot
            axs[0, j].sharey(axs[0, 0])
            axs[1, j].sharey(axs[1, 0])

            # Hide y-axis labels and ticks for all but the leftmost plots
            axs[0, j].tick_params(axis='y', which='both', labelleft=False)
            axs[1, j].tick_params(axis='y', which='both', labelleft=False)

        # For even more clarity, you can also remove the y-axis spines for non-leftmost plots
        for i in range(2):
            for j in range(1, 3):
                # Remove the left spine (the vertical line) on all but leftmost plots
    #             axs[i, j].spines['left'].set_visible(False)
                pass



        # Create separate copies of dataframe to avoid interference
        aggregate_df = plot_df.copy()
        categorical_df = plot_df.copy()
        distributional_df = plot_df.copy()

        # Color schemes for each column
        detailed_colors = {
            # Categorical metrics - browns and greens (no orange)
            'HR_h_h': '#654321',           # Dark brown
            'Krippendorff_h_h': '#8b4513', # Saddle brown
            'Fleiss_kappa_h_h': '#556b2f', # Dark olive green
            'Cohen_kappa_h_h': '#228b22',  # Forest green

            # Distributional metrics - keep the current purple palette
            'KL_lh_s_s': '#4b0082',        # Indigo (darker purple)
            'KL_hl_s_s': '#9370db',        # Medium purple
            'CE_lh_s_s': '#da70d6',        # Orchid (purple-pink)
            'CE_hl_s_s': '#ff69b4',        # Hot pink
            'JSD_s_s': '#4169e1',          # Royal blue

            # Discrete Multi-Label - keep the teal
            'COV_h_hrs': '#4682b4',        # Turquoise/teal

            # Continuous Multi-Label (MSE) - highlight proposed method
            'MSE_srs_srs': '#ff0000',      # Bright red (PROPOSED METHOD)
            'MSE_hat_srs_srs': '#ff0000',  # Medium gray (baseline)
            'MSE_h_h': '#404040',          # Dark gray
            'MSE_s_s': '#606060'           # Light gray
        }

        # Update aggregate colors to match
        aggregate_colors = {
            # Average of new categorical colors (browns/greens)
            'Categorical (Forced Choice)': '#6b5d39',     # Olive brown blend

            # Keep the purple average for distributional
            'Distributional (Forced Choice)': '#9c5eb8',  # Mauve (blended purple-pink)

            # Keep the teal for discrete
            'Discrete (Multi-Label)': '#4682b4',          # Turquoise/teal

            # Update to reflect red for proposed method
            'Continuous (Multi-Label)': '#ff0000'         # Darker red blend
        }
        # Marker styles
        aggregate_markers = {
            'Categorical (Forced Choice)': 's',  # Square marker
            'Distributional (Forced Choice)': '^',  # Star marker
            'Discrete (Multi-Label)': 'o',  # Circle marker
            'Continuous (Multi-Label)': '*'  # Triangle marker
        }

        detailed_markers = {
            'HR_h_h': 'o',                # Circle
            'Krippendorff_h_h': 's',      # Square
            'Fleiss_kappa_h_h': '^',      # Triangle up
            'Cohen_kappa_h_h': 'D',       # Diamond
            'KL_lh_s_s': 'p',             # Pentagon
            'KL_hl_s_s': '<',             # Star
            'CE_lh_s_s': 'X',             # X filled
            'CE_hl_s_s': 'h',             # Hexagon 1
            'JSD_s_s': 'H',               # Hexagon 2
            'COV_h_hrs': 'v',             # Triangle down
            'MSE_srs_srs': '*',           # Triangle left
            'MSE_hat_srs_srs': 'o',           # Triangle left
            'MSE_h_h': '>',               # Triangle right
            'MSE_s_s': 'd'                # Thin diamond
        }

        # Line styles
        aggregate_linestyles = {
            'Categorical (Forced Choice)': '-',   # Solid line
            'Distributional (Forced Choice)': '--',  # Dashed line
            'Discrete (Multi-Label)': '-',  # Solid line
            'Continuous (Multi-Label)': '--'  # Dashed line
        }

        # Create metric labels dictionary for detailed plots
        try:
            metric_labels = {metric: mets.METRIC_LOOKUP[metric] 
                            for metric in categorical_metrics + distributional_metrics}
        except:
            print('error')
            # Fallback if mets.METRIC_LOOKUP_SHORT doesn't exist
            metric_labels = {metric: metric for metric in categorical_metrics + distributional_metrics}


        # Process each downstream metric (row)
        for i, downstream_metric in enumerate(downstream_metrics):

            #------------------------------------------------------------
            # COLUMN 1: Aggregate Plot (Plot by Metric Type)
            #------------------------------------------------------------

            # Add metric_type column to the aggregate dataframe
            aggregate_df['metric_type'] = 'Other'  # Default value

            # Assign metric types based on mets.METRIC_GROUPS
            all_grouped_metrics = []
            for metric_type, metric_list in mets.METRIC_GROUPS.items():
                # Mark metrics with their type
                aggregate_df.loc[aggregate_df['metric'].isin(metric_list), 'metric_type'] = metric_type
                all_grouped_metrics.extend(metric_list)

            # Filter to only include metrics that are in the METRIC_GROUPS
            agg_filtered_df = aggregate_df[aggregate_df['metric'].isin(all_grouped_metrics)]

            # Filter for the current downstream metric
            metric_df = agg_filtered_df[agg_filtered_df['downstream_metric'] == downstream_metric]

            # Group by beta, metric_type, and sample
            type_grouped_df = metric_df.groupby(['beta', 'metric_type', 'sample'])[difference_metric].mean().reset_index()

            # Calculate statistics for each group
            stats_df = type_grouped_df.groupby(['beta', 'metric_type'])[difference_metric].agg(
                mean='mean',
                sem=lambda x: x.sem()  # Use standard error of the mean directly
            ).reset_index()

            # Invert the values for better visualization
            stats_df['mean'] = -stats_df['mean']

            metric_linestyles = {
                'MSE_srs_srs': '--',  # Make MSE_srs_srs dashed in all plots
                'MSE_hat_srs_srs': ':'  # Make MSE_srs_srs dashed in all plots
            }

            # Default linestyle for metrics without special handling
            default_linestyle = '-'

            # Define marker sizes with larger stars
            marker_sizes = {
                '*': 120,   # Larger size for star markers (default is 60)
                '^': 80,    # Triangle markers slightly larger
                'o': 60     # Keep circle markers the same
            }
            default_marker_size = 60  # Default size for other markers


            # Plot each metric type
            for metric_type in stats_df['metric_type'].unique():
                # Get data for this metric type
                type_data = stats_df[stats_df['metric_type'] == metric_type]

                # Get color, marker, and linestyle for this metric type
                color = aggregate_colors[metric_type]
                marker = aggregate_markers[metric_type]
                linestyle = aggregate_linestyles[metric_type]

                # Plot the connecting line
                axs[i, 0].plot(type_data['beta'], type_data['mean'], 
                        color=color, linestyle=linestyle, alpha=0.8, linewidth=2)

                # Plot scatter points
                s = marker_sizes.get(marker, default_marker_size) 
                axs[i, 0].scatter(type_data['beta'], type_data['mean'], 
                                color=color, marker=marker, s=s, 
                                label=metric_type, zorder=10)

                # Add error bars
                axs[i, 0].errorbar(type_data['beta'], type_data['mean'], 
                            yerr=type_data['sem'], color=color, 
                            fmt='none', alpha=0.8, capsize=5, capthick=1)

            # Set subplot title
            metric_title = 'Consistency' if downstream_metric == 'consistency' else 'Bias (Mean Absolute Error)'

            if i == 0:
                axs[i, 0].set_title(f"Results Grouped by\n Metric Type", fontsize=19)

            axs[0, 0].set_ylim(-0.25, -0.04)

            axs[1, 0].set_ylim(-.28, -.06)


            # Add y-axis label
            if i == 0:
                axs[i, 0].set_ylabel('Consistency of Selected Judge \nRelative to Optimal', fontsize=19)
            if i == 1:
                axs[i, 0].set_ylabel('Bias (MAE) of Selected Judge \nRelative to Optimal', fontsize=19)

            # Add grid and set tick size
            axs[i, 0].grid(True, linestyle='--', alpha=0.7)
            axs[i, 0].tick_params(axis='both', labelsize=14)

            # Add legend for first column
            legend_elements = []
            multi_label_header = mpatches.Patch(color='white', alpha=0.0, label='Multi-Label')
            legend_elements.append(multi_label_header)
            legend_elements.append(Line2D([0], [0], color=aggregate_colors['Discrete (Multi-Label)'], 
                                        marker=aggregate_markers['Discrete (Multi-Label)'], markersize=5,
                                        linestyle=aggregate_linestyles['Discrete (Multi-Label)'], 
                                        label='  Discrete'))
            legend_elements.append(Line2D([0], [0], color=aggregate_colors['Continuous (Multi-Label)'], 
                                        marker=aggregate_markers['Continuous (Multi-Label)'], markersize=6,
                                        linestyle=aggregate_linestyles['Continuous (Multi-Label)'], 
                                        label='  Continuous'))
            legend_elements.append(mpatches.Patch(color='white', alpha=0.0, label=''))
            forced_choice_header = mpatches.Patch(color='white', alpha=0.0, label='Forced Choice')
            legend_elements.append(forced_choice_header)
            legend_elements.append(Line2D([0], [0], color=aggregate_colors['Categorical (Forced Choice)'], 
                                        marker=aggregate_markers['Categorical (Forced Choice)'], markersize=6,
                                        linestyle=aggregate_linestyles['Categorical (Forced Choice)'], 
                                        label='  Categorical'))
            legend_elements.append(Line2D([0], [0], color=aggregate_colors['Distributional (Forced Choice)'], 
                                        marker=aggregate_markers['Distributional (Forced Choice)'], markersize=6,
                                        linestyle=aggregate_linestyles['Distributional (Forced Choice)'], 
                                        label='  Distributional'))

            # Add legend to the lower left panel (bottom row)
            if i == 1:  # Second row (bottom row)
                axs[i, 0].legend(handles=legend_elements, loc='upper left', fontsize=13,
                            frameon=True, framealpha=0.95, ncol=1,
                            bbox_to_anchor=(0.2, 1.65))

            #------------------------------------------------------------
            # COLUMN 2: Categorical Metrics Breakdown
            #------------------------------------------------------------

            # Filter for the current downstream metric
            metric_df = categorical_df[categorical_df['downstream_metric'] == downstream_metric]

            # Filter for only categorical metrics
            metric_df = metric_df[metric_df['metric'].isin(categorical_metrics)]

            # Group by beta, metric, and sample
            grouped_df = metric_df.groupby(['beta', 'metric', 'sample'])[difference_metric].mean().reset_index()

            # Calculate statistics
            stats_df = grouped_df.groupby(['beta', 'metric'])[difference_metric].agg(
                mean='mean',
                sem=lambda x: x.sem()
            ).reset_index()

            # Invert values
            stats_df['mean'] = -stats_df['mean']

            # Plot each categorical metric
            for metric in categorical_metrics:
                # Get data for this metric
                metric_data = stats_df[stats_df['metric'] == metric]

                if len(metric_data) > 0:
                    # Get color and marker
                    color = detailed_colors.get(metric, '#333333')
                    marker = detailed_markers.get(metric, 'o')

                    # Get human-readable label
                    label = metric_labels.get(metric, metric)

                    # Plot line
                    linestyle = metric_linestyles.get(metric, default_linestyle)
                    axs[i, 1].plot(metric_data['beta'], metric_data['mean'], 
                            color=color, alpha=0.8, linewidth=1.5, linestyle=linestyle)

                    # Plot scatter points
                    s = marker_sizes.get(marker, default_marker_size)
                    axs[i, 1].scatter(metric_data['beta'], metric_data['mean'], 
                            color=color, marker=marker, s=s, 
                            label=label, zorder=10)

                    # Add error bars
                    axs[i, 1].errorbar(metric_data['beta'], metric_data['mean'], 
                            yerr=metric_data['sem'], color=color, 
                            fmt='none', alpha=0.6, capsize=4, capthick=1)

            # Set subplot title
            if i == 0:
                axs[i, 1].set_title(f"Categorical Forced Choice vs.\n Continuous Multi-Label", fontsize=19)


            # Add grid and set tick size
            axs[i, 1].grid(True, linestyle='--', alpha=0.7)
            axs[i, 1].tick_params(axis='both', labelsize=14)

            # Add legend for categorical metrics with groupings
            if i == 1 and not metric_df.empty:
                # Create custom legend elements with groupings
                cat_legend_elements = []

                # Add header for MSE metrics first
                mse_header = mpatches.Patch(color='white', alpha=0.0, label='Multi-Label Continuous')
                cat_legend_elements.append(mse_header)

                # Add MSE metrics
                mse_metrics = [m for m in categorical_metrics if m.startswith('MSE')]
                for metric in mse_metrics:
                    if metric in detailed_colors and metric in detailed_markers:
                        cat_legend_elements.append(Line2D([0], [0], 
                            color=detailed_colors[metric], 
                            marker=detailed_markers[metric], 
                            markersize=6,
                            linestyle=metric_linestyles.get(metric, '-'),
                            label=f'  {metric_labels.get(metric, metric)}'))

                cat_legend_elements.append(mpatches.Patch(color='white', alpha=0.0, label=''))

                # Add header for Categorical metrics
                categorical_header = mpatches.Patch(color='white', alpha=0.0, label='Categorical')
                cat_legend_elements.append(categorical_header)

                # Add categorical metrics (excluding MSE metrics)
                categorical_only = [m for m in categorical_metrics if not m.startswith('MSE')]
                for metric in categorical_only:
                    if metric in detailed_colors and metric in detailed_markers:
                        cat_legend_elements.append(Line2D([0], [0], 
                            color=detailed_colors[metric], 
                            marker=detailed_markers[metric], 
                            markersize=6,
                            linestyle=metric_linestyles.get(metric, '-'),
                            label=f'  {metric_labels.get(metric, metric)}'))


                axs[i, 1].legend(handles=cat_legend_elements, loc='upper center', 
                                frameon=True, framealpha=1, fontsize=13, 
                                bbox_to_anchor=(0.5, 1.7), ncol=1)

            #------------------------------------------------------------
            # COLUMN 3: Distributional Metrics Breakdown
            #------------------------------------------------------------

            # Filter for the current downstream metric
            metric_df = distributional_df[distributional_df['downstream_metric'] == downstream_metric]

            # Filter for only distributional metrics
            metric_df = metric_df[metric_df['metric'].isin(distributional_metrics)]

            # Group by beta, metric, and sample
            grouped_df = metric_df.groupby(['beta', 'metric', 'sample'])[difference_metric].mean().reset_index()

            # Calculate statistics
            stats_df = grouped_df.groupby(['beta', 'metric'])[difference_metric].agg(
                mean='mean',
                sem=lambda x: x.sem()
            ).reset_index()

            # Invert values
            stats_df['mean'] = -stats_df['mean']

            # Plot each distributional metric
            for metric in distributional_metrics:
                # Get data for this metric
                metric_data = stats_df[stats_df['metric'] == metric]

                if len(metric_data) > 0:
                    # Get color and marker
                    color = detailed_colors.get(metric, '#333333')
                    marker = detailed_markers.get(metric, 'o')

                    # Get human-readable label
                    label = metric_labels.get(metric, metric)

                    # Plot line
                    linestyle = metric_linestyles.get(metric, default_linestyle)
                    axs[i, 2].plot(metric_data['beta'], metric_data['mean'], 
                            color=color, alpha=0.8, linewidth=1.5, linestyle=linestyle)

                    # Plot scatter points
                    s = marker_sizes.get(marker, default_marker_size)
                    axs[i, 2].scatter(metric_data['beta'], metric_data['mean'], 
                            color=color, marker=marker, s=s, 
                            label=label, zorder=10)

                    # Add error bars
                    axs[i, 2].errorbar(metric_data['beta'], metric_data['mean'], 
                            yerr=metric_data['sem'], color=color, 
                            fmt='none', alpha=0.6, capsize=4, capthick=1)

            # Set subplot title
            if i == 0:
                axs[i, 2].set_title(f"Distributional Forced Choice vs.\n Continuous Multi-Label", fontsize=19)


            # Add grid and set tick size
            axs[i, 2].grid(True, linestyle='--', alpha=0.7)
            axs[i, 2].tick_params(axis='both', labelsize=13)

            # Add legend for distributional metrics with groupings
            if i == 1 and not metric_df.empty:
                # Create custom legend elements with groupings
                dist_legend_elements = []

                # Add header for MSE metrics first
                mse_header = mpatches.Patch(color='white', alpha=0.0, label='Multi-Label Continuous')
                dist_legend_elements.append(mse_header)

                # Add MSE metrics
                mse_metrics = [m for m in distributional_metrics if m.startswith('MSE')]
                for metric in mse_metrics:
                    if metric in detailed_colors and metric in detailed_markers:
                        dist_legend_elements.append(Line2D([0], [0], 
                            color=detailed_colors[metric], 
                            marker=detailed_markers[metric], 
                            markersize=6,
                            linestyle=metric_linestyles.get(metric, '-'),
                            label=f'  {metric_labels.get(metric, metric)}'))

                # Add spacing
                dist_legend_elements.append(mpatches.Patch(color='white', alpha=0.0, label=''))

                # Add header for Distributional metrics
                distributional_header = mpatches.Patch(color='white', alpha=0.0, label='Distributional')
                dist_legend_elements.append(distributional_header)

                # Add KL and JSD metrics
                kl_jsd_metrics = [m for m in distributional_metrics if m.startswith(('KL', 'JSD', 'CE'))]
                for metric in kl_jsd_metrics:
                    if metric in detailed_colors and metric in detailed_markers:
                        dist_legend_elements.append(Line2D([0], [0], 
                            color=detailed_colors[metric], 
                            marker=detailed_markers[metric], 
                            markersize=6,
                            linestyle=metric_linestyles.get(metric, '-'),
                            label=f'  {metric_labels.get(metric, metric)}'))

                axs[i, 2].legend(handles=dist_legend_elements, loc='upper center', 
                                frameon=True, framealpha=1, fontsize=13, 
                                bbox_to_anchor=(0.5, 1.69), ncol=1)


        # Add x-axis labels and set consistent ticks
        for j in range(3):
            # Set consistent x-ticks for all plots
            x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,]

            # Apply to both rows
            for i in range(2):
                axs[i, j].set_xticks(x_ticks)
                axs[i, j].set_xticklabels([f'{x:.1f}' for x in x_ticks])
                axs[i, j].tick_params(axis='x', labelsize=14)

            # Hide x-axis labels for top row
            axs[0, j].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            # Add x-axis label only to bottom row
            axs[1, j].set_xlabel(r'Sensitivity Parameter ($\beta^H$)', fontsize=19)


        # Adjust spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(left=0.07, wspace=0.1, hspace=0.65, bottom=0.08)

        return fig, axs
