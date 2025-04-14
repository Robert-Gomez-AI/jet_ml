import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Packages loaded!!!")


# Set the style

# Set cyberpunk style colors and theme
plt.style.use('dark_background')
neon_pink = '#FF006E'  
neon_blue = '#00FFF5'
neon_cyan = '#00FF8C'
neon_purple = '#8A2BE2'
neon_yellow = '#FFD700'
neon_green = '#39FF14'
background_color = '#0D0221'





class DataFrame(pd.DataFrame):
    """
    A class that extends pandas.DataFrame with additional methods for data preprocessing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = self.shape[0]
        self.p = self.shape[1]

    def null_values_treatment(self, method: str, column: list[str] = None):
        """
        Treats null values in the DataFrame based on the specified method.

        Args:
            method (str): The method to use for treating null values.
            - "drop": Drops rows with any null values.
            - "mean": Replaces null values with the mean of the column.
            - "median": Replaces null values with the median of the column.
            - "mode": Replaces null values with the mode of the column.

        Returns:
            DataFrame: Returns self after treating null values.
        """
        if column is None:
            try:
                if method == "drop":
                    self._update_inplace(self.drop(self[self.isna().any(axis=1)].index))
                elif method == "mean":
                    self._update_inplace(self.fillna(self.mean()))
                elif method == "median":
                    self._update_inplace(self.fillna(self.median()))
                elif method == "mode":
                    self._update_inplace(self.fillna(self.mode().iloc[0]))
                    print(self)
                    print(self.isna().sum())
                return self
            except Exception as e:
                print(f"Error: {e}")
                return self
        else:
            for col in column:
                if method == "drop":
                    self._update_inplace(self.drop(self[self[col].isna()].index))
                elif method == "mean":
                    self._update_inplace(self.fillna(self[col].mean()))
                elif method == "median":
                    self._update_inplace(self.fillna(self[col].median()))
                elif method == "mode":
                    self._update_inplace(self.fillna(self[col].mode().iloc[0]))
                return self
                
    def plot_count_values(self, column: str, size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.grid(True, linestyle='-', alpha=0.3, color=neon_cyan, linewidth=0.8)
        
        sns.countplot(x=column, data=self, color=neon_blue, edgecolor=neon_purple, linewidth=2.3, alpha=0.8)
        
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

    def plot_distribution(self, column: str, size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        sns.histplot(self[column], kde=True, color=neon_blue, edgecolor=neon_purple, linewidth=2.3, alpha=0.8)
        for spine in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[spine].set_color(neon_purple)
            plt.gca().spines[spine].set_linewidth(2.5)
        plt.show()

    def plot_null_values(self, heatmap: bool = True, size: tuple = (10, 7)):
        """Plots a heatmap showing the distribution of null values in the DataFrame."""
        if heatmap:
            fig, ax = plt.subplots(figsize=size)
            ax.set_facecolor(background_color)
            
            null_mask = self.isna()
            print(null_mask)
            
            sns.heatmap(null_mask, cmap=['black', neon_pink], cbar_kws={'label': 'Is Null'}, yticklabels=False)
            
            plt.title('Null Values Distribution', fontsize=18, color=neon_yellow, pad=20, weight='bold')
            plt.xlabel('Features', fontsize=16, color=neon_green, labelpad=15, weight='bold')
            plt.ylabel('Records', fontsize=16, color=neon_green, labelpad=15, weight='bold')
            plt.xticks(rotation=45, ha='right', color=neon_blue)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=size)
            ax.set_facecolor(background_color)  
            plt.bar(range(len(self.columns)), self.isna().sum(), color=neon_blue)
            plt.xticks(range(len(self.columns)), self.columns, rotation=45, ha='right')
            plt.ylabel('Number of Null Values')
            for spine in ['top', 'bottom', 'left', 'right']:
                plt.gca().spines[spine].set_color(neon_purple)
                plt.gca().spines[spine].set_linewidth(2.5)
            plt.show()

def read_csv(path: str):
    return DataFrame(pd.read_csv(path))

def read_excel(path: str):
    return DataFrame(pd.read_excel(path))

def read_json(path: str):
    return DataFrame(pd.read_json(path))

def read_parquet(path: str):
    return DataFrame(pd.read_parquet(path))

