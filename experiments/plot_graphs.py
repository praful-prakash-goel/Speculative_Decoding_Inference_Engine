import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
import sys
from .benchmark_tps import RESULTS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks.csv")
CUSTOM_STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test.csv")
GPT2_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks_gpt2.csv")
OPT_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks_opt.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Standardized colours
main_color = "#779BE8"       
draft_small_color = "#97BC60"
draft_med_color = "#DA712B"
edge_color= "#949494"
distilgpt2_color = "#548235"
opt_125m_color = "#9333EA"

def plot_graphs():
    # Load dataframe containing benchmark results
    if os.path.exists(CUSTOM_BENCHMARKS_PATH):
        df = pd.read_csv(CUSTOM_BENCHMARKS_PATH)
        df = df.drop(columns=['Unnamed: 0'])
    else:
        print(f">> No csv file found at {CUSTOM_BENCHMARKS_PATH}")
        sys.exit(1)
    
    # Load dataframe containing stress results
    if os.path.exists(CUSTOM_STRESS_PATH):
        stress_df = pd.read_csv(CUSTOM_STRESS_PATH)
        stress_df = stress_df.drop(columns=['Unnamed: 0'])
        stress_df = stress_df.pivot(index='configuration', columns='context_length', values='tps')
    else:
        print(f">> No csv file found at {CUSTOM_STRESS_PATH}")
        sys.exit(1)
    
    if os.path.exists(GPT2_BENCHMARKS_PATH):
        df_gpt2 = pd.read_csv(GPT2_BENCHMARKS_PATH)
        df_gpt2 = df_gpt2.drop(columns=['Unnamed: 0'])
    else:
        print(f">> No csv file found at {GPT2_BENCHMARKS_PATH}")
        sys.exit(1)
    
    if os.path.exists(OPT_BENCHMARKS_PATH):
        df_opt = pd.read_csv(OPT_BENCHMARKS_PATH)
        df_opt = df_opt.drop(columns=['Unnamed: 0'])
    else:
        print(f">> No csv file found at {OPT_BENCHMARKS_PATH}")
        sys.exit(1)
    
    # Get unique gamma values used in experiments
    gammas = df['gamma'].dropna().unique()
    
    # Partition df based on draft model size and cache usage
    df_speculative_small_cache = df[(df['method'] == 'speculative') & (df['cache'] == True) & (df['draft'] == 'small')]
    df_speculative_small_no_cache = df[(df['method'] == 'speculative') & (df['cache'] == False) & (df['draft'] == 'small')]
    df_speculative_medium_cache = df[(df['method'] == 'speculative') & (df['cache'] == True) & (df['draft'] == 'medium')]
    df_speculative_medium_no_cache = df[(df['method'] == 'speculative') & (df['cache'] == False) & (df['draft'] == 'medium')]
    # df containing baseline models benchmark results
    df_baseline = df[df['method'].isin(['Main', 'Draft small', 'Draft medium'])] \
                .pivot(index='cache', columns='method', values='tps') \
                .reindex(columns=['Main', 'Draft small', 'Draft medium'])
                
    # Partition gpt2 df based on cache usage
    df_gpt2_cache = df_gpt2[(df_gpt2['method'] == 'speculative') & (df_gpt2['cache'] == True)]
    df_gpt2_no_cache = df_gpt2[(df_gpt2['method'] == 'speculative') & (df_gpt2['cache'] == False)]
    
    # Partition opt df based on cache usage
    df_opt_cache = df_opt[(df_opt['method'] == 'speculative') & (df_opt['cache'] == True)]
    df_opt_no_cache = df_opt[(df_opt['method'] == 'speculative') & (df_opt['cache'] == False)]
    
    # Graph 1: Baseline Tokens Per Second (TPS)      
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    df_baseline.plot(
        kind='bar', 
        rot=0, 
        ax=ax,
        color=[main_color, draft_small_color, draft_med_color],
        edgecolor=edge_color,
        linewidth=1.0 
    )
    ax.set_ylabel('Tokens Per Second', fontsize='13')
    ax.set_xlabel('cache', fontsize='15')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "baseline_tps.pdf"), bbox_inches='tight')
    
    # Graph 2: Subplot for comparison of speculative engine with and without cache
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)

    # Left subplot (without cache)
    sb.lineplot(ax=axes[0], x=gammas, y=df_speculative_small_no_cache['tps'], color=draft_small_color, marker='o', label='Speculative (30M draft)')
    sb.lineplot(ax=axes[0], x=gammas, y=df_speculative_medium_no_cache['tps'], color=draft_med_color, marker='s', label='Speculative (70M draft)')
    axes[0].axhline(y=df_baseline['Main'].iloc[0], color=main_color, linestyle='dashed', label='Baseline Main')

    axes[0].set_xlabel('\u03B3', fontsize='14')
    axes[0].set_ylabel('Tokens Per Second', fontsize='15')
    axes[0].get_legend().remove()

    # Right subplot (with cache)
    sb.lineplot(ax=axes[1], x=gammas, y=df_speculative_small_cache['tps'], color=draft_small_color, marker='o', label='speculative (30M draft)')
    sb.lineplot(ax=axes[1], x=gammas, y=df_speculative_medium_cache['tps'], color=draft_med_color, marker='s', label='speculative (70M draft)')
    axes[1].axhline(y=df_baseline['Main'].iloc[1], color=main_color, linestyle='dashed', label='Baseline Main')

    axes[1].set_xlabel('\u03B3', fontsize='15')
    axes[1].set_ylabel("Tokens Per Second", fontsize='15')
    axes[1].get_legend().remove()

    # Create shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "speculative_cache_comparison.pdf"), bbox_inches='tight')
    
    # Graph 3: Gamma sweep to compare speculative speedup vs different gamma values
    fig = plt.figure(figsize=(5, 3.5))

    sb.lineplot(x=gammas, y=df_speculative_small_no_cache['speedup'], color=draft_small_color, marker='o', label='30M - No KV Cache')
    sb.lineplot(x=gammas, y=df_speculative_medium_no_cache['speedup'], color=draft_med_color, marker='s', label='70M - No KV Cache')
    sb.lineplot(x=gammas, y=df_speculative_small_cache['speedup'], color=draft_small_color, marker='o', label='30M - KV Cache', linestyle='dashed')
    sb.lineplot(x=gammas, y=df_speculative_medium_cache['speedup'], color=draft_med_color, marker='s', label='70M - KV Cache', linestyle='dashed')
    sb.lineplot(x=gammas, y=1.0, color=main_color, label='Autoregressive baseline', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel("\u03B3", fontsize='15')
    plt.ylabel("Speedup", fontsize='15')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "gamma_sweep_speedup.pdf"), bbox_inches='tight')
    
    # Graph 4: Subplot for comparison of acceptance and mean accepted vs different gamma values
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))

    # Left subplot (acceptance vs gamma)
    sb.lineplot(ax=axes[0], x=gammas, y=df_speculative_small_cache['acceptance'], color=draft_small_color, marker='o', label='Speculative (30M draft)')
    sb.lineplot(ax=axes[0], x=gammas, y=df_speculative_medium_cache['acceptance'], color=draft_med_color, marker='s', label='Speculative (70M draft)')

    axes[0].set_xlabel("\u03B3", fontsize='15')
    axes[0].set_ylabel("Acceptance Rate", fontsize='14')
    axes[0].get_legend().remove()

    # Right subplot (mean accepted vs gamma)
    sb.lineplot(ax=axes[1], x=gammas, y=df_speculative_small_cache['mean_accepted'], color=draft_small_color, marker='o', label='Speculative (30M draft)')
    sb.lineplot(ax=axes[1], x=gammas, y=df_speculative_medium_cache['mean_accepted'], color=draft_med_color, marker='s', label='Speculative (70M draft)')

    axes[1].set_xlabel("\u03B3", fontsize='15')
    axes[1].set_ylabel("Mean Accepted", fontsize='14')
    axes[1].get_legend().remove()

    # Create shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "gamma_sweep_acceptance.pdf"), bbox_inches='tight')
    
    # Graph 5: Stress test across different context lengths
    fig = plt.figure(figsize=(5, 3.5))
    sb.lineplot(x=stress_df.columns, y=stress_df.iloc[0], color=main_color, linestyle='dashed', label='Main - KV Cache')
    sb.lineplot(x=stress_df.columns, y=stress_df.iloc[1], color=draft_med_color, linestyle='dashed', label='Speculative - KV Cache')
    sb.lineplot(x=stress_df.columns, y=stress_df.iloc[2], color=draft_med_color, label='Speculative - No KV Cache')

    plt.ylabel("Tokens Per Second", fontsize='12')
    plt.xlabel("Max Tokens Generated", fontsize='12')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "stress_test.pdf"), bbox_inches='tight')
    
    # Graph 6: Speedup comparison across gamma for standardized gpt2 and opt models
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)
    
    # Left Subplot: GPT2
    sb.lineplot(ax=axes[0], x=gammas, y=df_gpt2_cache['speedup'], color=distilgpt2_color, marker='o', label='GPT2 - KV Cache', linestyle='dashed')
    sb.lineplot(ax=axes[0], x=gammas, y=df_gpt2_no_cache['speedup'], color=distilgpt2_color, marker='s', label='GPT2 - No KV Cache')
    axes[0].axhline(y=1.0, color=main_color, linestyle='--', label='Autoregressive baseline', linewidth=1.5, alpha=0.7)
    
    axes[0].set_xlabel('\u03B3', fontsize='14')
    axes[0].set_ylabel('Speedup', fontsize='15')
    axes[0].get_legend().remove()
    
    # Right Subplot: OPT
    sb.lineplot(ax=axes[1], x=gammas, y=df_opt_cache['speedup'], color=opt_125m_color, marker='o', label='OPT - KV Cache', linestyle='dashed')
    sb.lineplot(ax=axes[1], x=gammas, y=df_opt_no_cache['speedup'], color=opt_125m_color, marker='s', label='OPT - No KV Cache')
    axes[1].axhline(y=1.0, color=main_color, linestyle='--', label='Autoregressive baseline', linewidth=1.5, alpha=0.7)
    
    axes[1].set_xlabel('\u03B3', fontsize='14')
    axes[1].set_ylabel('Speedup', fontsize='15')
    axes[1].get_legend().remove()
    
    # Create shared legend at the top
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles, labels = h1 + h2, l1 + l2[:-1]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "standardized_speedup.pdf"), bbox_inches='tight')
    
    print(f">> Plots saved at {PLOT_DIR}")
    
if __name__ == '__main__':
    plot_graphs()