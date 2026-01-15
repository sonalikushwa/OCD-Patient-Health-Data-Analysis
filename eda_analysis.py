import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

def load_data(filepath):
    """
    Load the dataset.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis.
    """
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # Missing Values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Feature Engineering for EDA
    if 'Y-BOCS Score (Obsessions)' in df.columns and 'Y-BOCS Score (Compulsions)' in df.columns:
        df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
        
    # Visualizations
    
    # 1. Distribution of Total Y-BOCS Score
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Y-BOCS Score'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Total Y-BOCS Score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('eda_ybocs_dist.png')
    print("Saved 'eda_ybocs_dist.png'")
    # plt.show()
    
    # 2. Gender vs Y-BOCS Score
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Gender', y='Total Y-BOCS Score', data=df, palette="Set2")
    plt.title('Total Y-BOCS Score by Gender')
    plt.savefig('eda_gender_boxplot.png')
    print("Saved 'eda_gender_boxplot.png'")
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('eda_correlation_heatmap.png')
    print("Saved 'eda_correlation_heatmap.png'")
    
    # 4. Obsession Type Counts
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Obsession Type', data=df, order=df['Obsession Type'].value_counts().index, palette="viridis")
    plt.title('Frequency of Obsession Types')
    plt.tight_layout()
    plt.savefig('eda_obsession_types.png')
    print("Saved 'eda_obsession_types.png'")
    
    print("\nEDA Completed. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    file_path = 'ocd_patient_dataset.csv'
    df = load_data(file_path)
    if df is not None:
        perform_eda(df)
