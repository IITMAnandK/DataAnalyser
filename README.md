# Automated Data Analysis Tool

This tool, powered by `autolysis.py`, provides automated data analysis for CSV files. It leverages data science techniques and machine learning to generate insights and visualizations, helping users quickly understand their data.

## Features

- **Data Profiling:** Generates descriptive statistics, identifies missing values, and determines data types for each column.
- **Outlier Detection:** Identifies and visualizes potential outliers using the Interquartile Range (IQR) method.
- **Correlation Analysis:** Creates a heatmap to visualize correlations between numerical columns.
- **Clustering:** Performs density-based (DBSCAN) or hierarchical clustering to identify patterns and groupings in the data.
- **Categorical Data Visualization:** Generates plots (e.g., bar charts, count plots) to visualize categorical data distributions.
- **Time Series Analysis:** Creates time series plots to visualize trends and patterns over time.
- **Report Generation:** Compiles analysis results into a comprehensive report with embedded visualizations and narrative summaries.

## Usage

1. **Install Dependencies:**
2. **Set API Key (Optional):**
   - If you want to use the LLM capabilities (e.g., for narrative generation or column selection), set the `AIPROXY_TOKEN` environment variable with your API key.
3. **Run the Analysis:**
    Format: uv run autolysis.py dataset.csv
   
*Replace `dataset.csv` with the path to your CSV file.

## Output

The analysis results will be saved in a new directory named after your CSV file (without the extension). Inside this directory, you'll find:

- **README.md:** A comprehensive report containing analysis summaries, visualizations, and narrative insights.
- **Images:** PNG files of the generated visualizations (e.g., correlation heatmap, cluster plots, time series plots).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
