import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch


from core.rating_model import RatingModel
from core import metrics as mets
from config import models

class Validator:
    """
    Evaluates judge model performance against human ratings.
    """
    
    def __init__(self):
        self.rating_model = RatingModel()
    
    def score_judges(self, run_results, task_name, beta=0.0, tau=0.5):
        """
        Calculate performance metrics for judge models.
        
        Parameters:
        -----------
        run_results : dict
            Dictionary containing the results to be evaluated.
            Expected structure:
            {
                task_name: {
                    'ratings': DataFrame with human ratings,
                    'judge_results': dict of model results,
                    'task_config': task configuration dict
                }
            }
        task_name : str
            The specific task to evaluate
        beta : float, default=0.0
            Sensitivity parameter for constructing human rating distribution
        tau : float, default=0.5
            Classification threshold value
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns: model, beta, tau, and all agreement metrics
        """
        # Extract components from run_results
        corpus = run_results[task_name]['ratings']
        judge_results = run_results[task_name]['judge_results']
        task_config = run_results[task_name]['task_config']
        positive_options = task_config['positive_categorization_options']
        
        # Construct human rating distribution for this beta
        p_human = self.rating_model.construct_human_rating_distribution(
            corpus, task_name, task_config, beta
        )
        
        # Collect results for all models
        results = []
        
        for model_id, judge_perf in judge_results.items():
            # Compute agreement metrics between human and judge
            agreement_metrics = mets.compute_performance_metrics(
                h_rs=p_human,
                llm_rs=judge_perf['p_judge_hat'],
                classification_j=positive_options,
                classification_tau=tau
            )
            
            # Add metadata
            agreement_metrics['model'] = model_id
            agreement_metrics['beta'] = beta
            agreement_metrics['tau'] = tau
            
            results.append(agreement_metrics)
        
        return pd.DataFrame(results)


    def plot_downstream_metrics(self, metrics_df, figsize=(12, 5)):
        """
        Plot downstream metrics (consistency and bias_mae) by model.
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            DataFrame from evaluator.score_judges() containing metrics
        figsize : tuple, default=(12, 5)
            Figure size
        """
        
        # Extract relevant columns
        plot_data = metrics_df[['model', 'consistency', 'bias_mae', 'beta', 'tau']].copy()

        # Shorten model names for display
        plot_data['model_short'] = plot_data['model'].apply(
            lambda x: models.DISPLAY_NAMES[x]
        )
        
        # Map models to colors
        colors = [models.MODEL_PALETTE.get(model, '#808080') for model in plot_data['model']]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Consistency
        axes[0].bar(plot_data['model_short'], plot_data['consistency'], color=colors)
        axes[0].set_ylabel('Consistency', fontsize=12)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_title('Decision Consistency', fontsize=13)
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Bias MAE
        axes[1].bar(plot_data['model_short'], plot_data['bias_mae'], color=colors)
        axes[1].set_ylabel('Estimation Bias (MAE)', fontsize=12)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_title('Prevalence Estimation Bias (Mean Absolute Error)', fontsize=13)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(
            f"Downstream Metrics (β={plot_data['beta'].iloc[0]}, τ={plot_data['tau'].iloc[0]})",
            fontsize=14,
            y=1.02
        )
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"{'Model':<40} {'Consistency':>12} {'Bias (MAE)':>12}")
        print("-" * 75)
        for _, row in plot_data.iterrows():
            print(f"{row['model']:<40} {row['consistency']:>12.3f} {row['bias_mae']:>12.3f}")

    def plot_facit(self, metrics_df, metric_left, metric_right, figsize=(8, 6)):
        merged_df = metrics_df.copy()
        merged_df['display_name'] = merged_df['model'].map(models.DISPLAY_NAMES)
        
        # Sort by left metric to get ranking order
        metric1_ascending = (mets.METRIC_TABLE[metric_left]['opt_ind'] == 0)
        merged_df = merged_df.sort_values(by=metric_left, ascending=metric1_ascending).reset_index(drop=True)
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        x1, x2 = 0.0, 1.0
        
        # Get y-limits with safe padding
        m1min, m1max = merged_df[metric_left].min(), merged_df[metric_left].max()
        m2min, m2max = merged_df[metric_right].min(), merged_df[metric_right].max()
        pad1 = 0.1 * (m1max - m1min) if m1max > m1min else 0.05 * (abs(m1max) + 1e-6)
        pad2 = 0.1 * (m2max - m2min) if m2max > m2min else 0.05 * (abs(m2max) + 1e-6)
        
        # Set y-limits based on metric optimization direction
        if mets.METRIC_TABLE[metric_left]['opt_ind'] == 0:
            ax1.set_ylim(m1max + pad1, m1min - pad1)
        else:
            ax1.set_ylim(m1min - pad1, m1max + pad1)
        
        if mets.METRIC_TABLE[metric_right]['opt_ind'] == 0:
            ax2.set_ylim(m2max + pad2, m2min - pad2)
        else:
            ax2.set_ylim(m2min - pad2, m2max + pad2)
        
        ax1.set_xlim(-0.25, 1.25)
        ax1.set_xticks([x1, x2])
        
        # Store exact positions to ensure perfect alignment
        positions = []
        for _, row in merged_df.iterrows():
            left_val = row[metric_left]
            right_val = row[metric_right]
            positions.append((left_val, right_val, row['model']))
        
        # First draw ALL connections
        for _, row in merged_df.iterrows():
            color = models.MODEL_PALETTE.get(row['model'], '#808080')
            con = ConnectionPatch(
                xyA=(x1, row[metric_left]), coordsA=ax1.transData,
                xyB=(x2, row[metric_right]), coordsB=ax2.transData,
                linewidth=3, alpha=0.7, color=color, zorder=1
            )
            fig.add_artist(con)

        # Then draw ALL points on top
        for _, row in merged_df.iterrows():
            color = models.MODEL_PALETTE.get(row['model'], '#808080')
            ax1.scatter(x1, row[metric_left], s=40, color=color, zorder=10,
                    edgecolors='white', linewidth=2)
            ax2.scatter(x2, row[metric_right], s=40, color=color, zorder=10,
                    edgecolors='white', linewidth=2)
                
        # Format axes
        metric1_name = mets.METRIC_LOOKUP_SHORT.get(metric_left, metric_left)
        metric2_name = mets.METRIC_LOOKUP_SHORT.get(metric_right, metric_right)
        d1 = "↓" if mets.METRIC_TABLE[metric_left]['opt_ind'] == 0 else "↑"
        d2 = "↓" if mets.METRIC_TABLE[metric_right]['opt_ind'] == 0 else "↑"
        ax1.set_xticklabels([f"{metric1_name} {d1}", f"{metric2_name} {d2}"],
                            fontsize=11, rotation=25)
        
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.grid(False)
        ax2.grid(False)
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color=models.MODEL_PALETTE.get(m, '#808080'),
                label=models.DISPLAY_NAMES.get(m, m), markersize=8, linestyle='-', linewidth=2)
            for m in merged_df['model'].unique()
        ]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1),
                frameon=True, fontsize=9)
        
        # Add title
        beta = metrics_df['beta'].iloc[0]
        tau = metrics_df['tau'].iloc[0]
        plt.title(f'Facit Plot (β={beta}, τ={tau})', fontsize=13, fontweight='bold', pad=20)
        
        # Adjust layout
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        plt.show()