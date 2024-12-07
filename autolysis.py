import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from pathlib import Path

def explain_headers(headers):
    """
    Use an AI model to contextualize dataset headers.
    Requires an `AIPROXY_TOKEN` environment variable for authentication.
    """
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")

    api_url = "https://api.openai.com/v1/completions"  # Replace with your AI proxy endpoint
    prompt = f"Provide context and potential meanings for the following dataset headers: {', '.join(headers)}"
    
    headers_dict = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.7,
    }

    response = requests.post(api_url, headers=headers_dict, json=payload)
    response.raise_for_status()  # Raise error if API call fails

    explanation = response.json().get("choices", [{}])[0].get("text", "").strip()
    return explanation
    
def analyze_csv(file_path):
    # Load CSV
    data = pd.read_csv(file_path)
    report_lines = []

    # Explain dataset headers
    try:
        header_context = explain_headers(data.columns)
        report_lines.append("# Automated Data Analysis Report")
        report_lines.append(f"### Dataset: {file_path}")
        report_lines.append("## Contextualized Dataset Headers")
        report_lines.append(header_context)
    except Exception as e:
        report_lines.append("## Contextualized Dataset Headers")
        report_lines.append(f"Could not retrieve header context: {e}")
    
    # General information
    report_lines.append(f"## Dataset Overview")
    report_lines.append(f"Number of rows: {data.shape[0]}")
    report_lines.append(f"Number of columns: {data.shape[1]}\n")

    # Column descriptions
    report_lines.append("### Descriptive Statistics")
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
    print(data.dtypes)
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
