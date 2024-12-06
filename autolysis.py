pip install pandas matplotlib seaborn

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_csv(file_path):
    # Load CSV
    data = pd.read_csv(file_path)
    report_lines = []

    # General information
    report_lines.append("# Automated Data Analysis Report")
    report_lines.append(f"### Dataset: {file_path}")
    report_lines.append(f"Number of rows: {data.shape[0]}")
    report_lines.append(f"Number of columns: {data.shape[1]}\n")

    # Column descriptions
    report_lines.append("## Dataset Overview")
    report_lines.append(data.describe(include='all').transpose().to_markdown())

    # Visualization 1: Correlation Heatmap
    if data.select_dtypes(include=['number']).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        heatmap_path = "correlation_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        plt.close()
        report_lines.append(f"![Correlation Heatmap]({heatmap_path})")

    # Visualization 2: Distribution of Numerical Columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    for col in numeric_columns[:3]:  # Limit to 3 distributions
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True, bins=30)
        dist_path = f"{col}_distribution.png"
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.savefig(dist_path)
        plt.close()
        report_lines.append(f"![{col} Distribution]({dist_path})")

    # Save README.md
    with open("README.md", "w") as readme_file:
        readme_file.write("\n".join(report_lines))
    
    print("Analysis complete. Check README.md and generated charts.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not Path(csv_file).is_file():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    analyze_csv(csv_file)
