"""
Visualization Module
This module contains functions for data visualization
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class DataVisualizer:
    """Class for creating data visualizations"""
    
    def __init__(self, save_dir='reports/figures'):
        """
        Initialize DataVisualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_distribution(self, df, column, save_name=None):
        """
        Plot distribution of a column
        
        Args:
            df: pandas.DataFrame
            column: Column name
            save_name: Name to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        if df[column].dtype in ['int64', 'float64']:
            # Numerical column - histogram with KDE
            sns.histplot(data=df, x=column, kde=True, bins=30)
            plt.title(f'Distribution of {column}')
        else:
            # Categorical column - count plot
            sns.countplot(data=df, x=column)
            plt.title(f'Count Plot of {column}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_matrix(self, df, save_name='correlation_matrix.png'):
        """
        Plot correlation matrix
        
        Args:
            df: pandas.DataFrame
            save_name: Name to save the plot
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            logger.warning("No numerical columns found for correlation matrix")
            return
        
        # Calculate correlation
        corr = df[numerical_cols].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_target_distribution(self, df, target_column, save_name='target_distribution.png'):
        """
        Plot target variable distribution
        
        Args:
            df: pandas.DataFrame
            target_column: Name of target column
            save_name: Name to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Count plot
        ax = sns.countplot(data=df, x=target_column)
        plt.title(f'Distribution of {target_column}')
        
        # Add percentages
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_feature_vs_target(self, df, feature, target, save_name=None):
        """
        Plot feature vs target relationship
        
        Args:
            df: pandas.DataFrame
            feature: Feature column name
            target: Target column name
            save_name: Name to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        if df[feature].dtype in ['int64', 'float64']:
            # Numerical feature - box plot
            sns.boxplot(data=df, x=target, y=feature)
            plt.title(f'{feature} by {target}')
        else:
            # Categorical feature - count plot with hue
            sns.countplot(data=df, x=feature, hue=target)
            plt.title(f'{feature} by {target}')
            plt.xticks(rotation=45)
            plt.legend(title=target)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_missing_values(self, df, save_name='missing_values.png'):
        """
        Plot missing values heatmap
        
        Args:
            df: pandas.DataFrame
            save_name: Name to save the plot
        """
        # Calculate missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            logger.info("No missing values found")
            return
        
        # Plot
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.xlabel('Column')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def create_interactive_scatter(self, df, x_col, y_col, color_col=None, 
                                   title='Interactive Scatter Plot'):
        """
        Create interactive scatter plot using Plotly
        
        Args:
            df: pandas.DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Column for color coding
            title: Plot title
        """
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=title, hover_data=df.columns)
        fig.show()
    
    def create_dashboard(self, df, target_column):
        """
        Create a comprehensive dashboard with multiple plots
        
        Args:
            df: pandas.DataFrame
            target_column: Name of target column
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Target distribution
        df[target_column].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title(f'Distribution of {target_column}')
        axes[0, 0].set_xlabel(target_column)
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values')
            axes[0, 1].set_xlabel('Column')
            axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Numerical features distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numerical_cols):
            if i < 2:
                df[col].hist(bins=30, ax=axes[1, i], edgecolor='black')
                axes[1, i].set_title(f'Distribution of {col}')
                axes[1, i].set_xlabel(col)
                axes[1, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save dashboard
        filepath = os.path.join(self.save_dir, 'dashboard.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_model_performance(self, metrics_dict, save_name='model_performance.png'):
        """
        Plot model performance comparison
        
        Args:
            metrics_dict: Dictionary of model names and their metrics
            save_name: Name to save the plot
        """
        # Convert to DataFrame
        df_metrics = pd.DataFrame(metrics_dict).T
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_metrics.plot(kind='bar', ax=ax)
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()

def visualize_data(df, target_column='Loan_Status'):
    """
    Convenience function to create multiple visualizations
    
    Args:
        df: pandas.DataFrame
        target_column: Name of target column
    """
    visualizer = DataVisualizer()
    
    logger.info("Creating visualizations...")
    
    # Target distribution
    visualizer.plot_target_distribution(df, target_column)
    
    # Correlation matrix
    visualizer.plot_correlation_matrix(df)
    
    # Missing values
    visualizer.plot_missing_values(df)
    
    # Dashboard
    visualizer.create_dashboard(df, target_column)
    
    logger.info("Visualizations created successfully")

if __name__ == '__main__':
    # Test visualization
    # Create sample data
    sample_data = {
        'ApplicantIncome': np.random.normal(5000, 2000, 100),
        'LoanAmount': np.random.normal(150, 50, 100),
        'Credit_History': np.random.choice([0, 1], 100),
        'Loan_Status': np.random.choice(['Y', 'N'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    visualizer = DataVisualizer()
    visualizer.plot_target_distribution(df, 'Loan_Status')
    visualizer.plot_correlation_matrix(df)
