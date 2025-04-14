from ..modeling import DataFrame
from .plot_configs import *
import matplotlib.pyplot as plt
import seaborn as sns

#plots 

def plot_null_values_heatmap(dataframe: DataFrame, size: tuple = (10, 7)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_facecolor(background_color)
    
    null_mask = dataframe.isna()
    print(null_mask)
    
    sns.heatmap(null_mask, cmap=['black', neon_pink], cbar_kws={'label': 'Is Null'}, yticklabels=False)
    
    plt.title('Null Values Distribution', fontsize=18, color=neon_yellow, pad=20, weight='bold')
    plt.xlabel('Features', fontsize=16, color=neon_green, labelpad=15, weight='bold')
    plt.ylabel('Records', fontsize=16, color=neon_green, labelpad=15, weight='bold')
    plt.xticks(rotation=45, ha='right', color=neon_blue)
    plt.tight_layout()
    plt.show()

def plot_null_values_bar(dataframe: DataFrame, size: tuple = (10, 7)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_facecolor(background_color)  
    plt.bar(range(len(dataframe.columns)), dataframe.isna().sum(), color=neon_blue)
    plt.xticks(range(len(dataframe.columns)), dataframe.columns, rotation=45, ha='right')
    plt.ylabel('Number of Null Values')
    for spine in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[spine].set_color(neon_purple)
        plt.gca().spines[spine].set_linewidth(2.5)
    plt.show()

def plot_distribution(dataframe: DataFrame, column: str, size: tuple = (10, 7)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    sns.histplot(dataframe[column], kde=True, color=neon_blue, edgecolor=neon_purple, linewidth=2.3, alpha=0.8)
    for spine in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[spine].set_color(neon_purple)
        plt.gca().spines[spine].set_linewidth(2.5)
    plt.show()

def plot_count_values(dataframe: DataFrame, column: str, size: tuple = (10, 7)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.grid(True, linestyle='-', alpha=0.3, color=neon_cyan, linewidth=0.8)
    
    sns.countplot(x=column, data=dataframe, color=neon_blue, edgecolor=neon_purple, linewidth=2.3, alpha=0.8)
    
    ax.set_title(f"Distribution of {column}", fontsize=18, color=neon_yellow, pad=20, weight='bold')
    ax.set_xlabel(column, fontsize=16, color=neon_green, labelpad=15, weight='bold')
    ax.set_ylabel("Count", fontsize=16, color=neon_green, labelpad=15, weight='bold')
    
    for spine in ax.spines.values():
        spine.set_color(neon_purple)
        spine.set_linewidth(2.5)
    
    ax.tick_params(colors=neon_blue, labelsize=14)
    for line in ax.lines:
        line.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=3, foreground=background_color)])
    
    plt.tight_layout()
    plt.show()