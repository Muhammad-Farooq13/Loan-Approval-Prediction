"""
Data Loading Module
This module handles loading data from various sources
"""

import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading data from various sources"""
    
    def __init__(self, data_dir='data'):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.processed_data_dir = os.path.join(data_dir, 'processed')
    
    def load_raw_data(self, filename='loan_data.csv'):
        """
        Load raw data from CSV file
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            filepath = os.path.join(self.raw_data_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")
            
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_processed_data(self, filename='processed_loan_data.csv'):
        """
        Load processed data from CSV file
        
        Args:
            filename: Name of the processed CSV file
            
        Returns:
            pandas.DataFrame: Loaded processed data
        """
        try:
            filepath = os.path.join(self.processed_data_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Processed file not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def save_processed_data(self, df, filename='processed_loan_data.csv'):
        """
        Save processed data to CSV file
        
        Args:
            df: pandas.DataFrame to save
            filename: Name of the output file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.processed_data_dir, exist_ok=True)
            
            filepath = os.path.join(self.processed_data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully saved {len(df)} rows to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def get_data_info(self, df):
        """
        Get information about the dataset
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            dict: Dictionary containing dataset information
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info

def load_loan_data(data_path='data/raw/loan_data.csv'):
    """
    Convenience function to load loan data
    
    Args:
        data_path: Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded loan data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading loan data: {str(e)}")
        raise

if __name__ == '__main__':
    # Test the data loader
    loader = DataLoader()
    df = loader.load_raw_data()
    info = loader.get_data_info(df)
    
    print("Dataset Information:")
    print(f"Rows: {info['rows']}")
    print(f"Columns: {info['columns']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
