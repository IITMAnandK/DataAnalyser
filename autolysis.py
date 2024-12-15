# Required files are given in meta data as requires and dependencies. So no need to install each time in pip command
# /// script
# requires-python = ">=3.11"
# requires-openai=">=0.27.0"
# dependencies = [
#   "httpx",
#   "pandas",
#   "tabulate",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "chardet",
#   "openai",
#   "statsmodels",
#   "networkx",
#   "ipykernel",
#   "requests",
#   "geopandas",
#   "scipy"
# ]
# ///
#!pip install python-dotenv==1.0.0
import os
import sys
import pandas as pd
import matplotlib.pyplot as mathplt
import seaborn as sns
import requests
import chardet
#import dotenv
import numpy as np
import seaborn as sns
import httpx
import chardet
import time
import base64
import tempfile
import json
import shutil
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image
from pathlib import Path
from tabulate import tabulate
#from google.colab import userdata

img_cnt = 0
max_img = 10

# Set your AIPROXY TOKEN
API_KEY = os.getenv("AIPROXY_TOKEN")
#api_key = userdata.get('AIPROXY_TOKEN')
#API_KEY = api_key
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Main function to analyze and generate output of the csv dataset
def analyze_and_generate_output(csv_file):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    This function takes a CSV file as an input and performs a comprehensive data analysis, 
    generating various insights and visualizations. It essentially automates the 
    process of exploring and understanding a dataset.

    Data Loading: It reads the CSV file using the get_data function.
    
    Standard Analysis: Performs descriptive statistics analysis using
    standard_analysis_csv, including summary statistics and missing values.
    
    Outlier Detection: Identifies outliers using anomoly_detection 
    and visualizes them with plot_outliers.
    
    Correlation Heatmap: Generates a correlation heatmap using 
    gen_correlation_heatmap to visualize relationships between numerical columns.
    
    Clustering: Performs clustering analysis using either 
    DBSCAN (dclustering) or hierarchical clustering (hclustering) based on suitable columns.
    
    Categorical Plotting: Creates plots for categorical data using plot_categorical_data.
    
    Time Series Plotting: Generates time series plots using 
    plot_time_series to analyze trends over time.
    
    Report Generation: Saves the overall analysis summary to a 
    
    README file in the output directory using save_readme.

    In essence, analyze_and_generate_output automates the exploration, 
    analysis, and visualization of a dataset, providing a 
    comprehensive overview of its characteristics and patterns.

    Args:
        csv_file: Input CSV file for analysis

    Returns:
        None

    '''
    overall_analysis_summary = []

    curfilename = os.path.basename(csv_file)
    new_dir_name, ext = os.path.splitext(curfilename)

    # Get the directory of the current file
    current_dir = os.getcwd()
    #UNCOMMENT current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a new directory (e.g., 'CSV File Name') in the current directory
    new_dir_path = os.path.join(current_dir, new_dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(new_dir_path, exist_ok=True)

    # Execute a function named get_data to load the csv file into dataframe
    loadedcsv = get_data(csv_file)

    columnslist = ', '.join(loadedcsv.columns)

    LLM_Input = {}

    # Perform basic analysis
    standard_analysis = standard_analysis_csv(loadedcsv, columnslist)
    overall_analysis_summary.append(standard_analysis)
    LLM_Input["summary"] = standard_analysis

    prompt_outliers = f"""
    I have a dataset with the following columns: {columnslist}.
    I want to perform outlier detection to identify unusual or extreme values.
    Which columns would be most suitable for this analysis, considering their
    data types and potential for containing outliers? Please provide a list
    of column names.
    """

    # Get recommended columns for outliers from LLM
    recommended_columns_outliers = []
    #UNCOMMENT recommended_columns_outliers  = column_selector(loadedcsv, prompt_outliers, num_columns=5, column_type="numeric")

    # Check if recommended_columns_outliers  contains valid columns before proceeding, if not select relevant numberic columns using custom built function
    if not (recommended_columns_outliers  and all(col in loadedcsv.columns for col in recommended_columns_outliers )):
        recommended_columns_outliers  = select_relevant_numeric_column(loadedcsv)

    # Perform Anomoly Detection and outlier plot
    anomoly_detect = anomoly_detection(loadedcsv, recommended_columns_outliers)
    overall_analysis_summary.append(anomoly_detect)
    LLM_Input["anomoly"] = anomoly_detect

    gen_plot_outliers = plot_outliers(loadedcsv, recommended_columns_outliers, new_dir_path)
    overall_analysis_summary.append(gen_plot_outliers)
    
    if img_cnt <= max_img:
        prompt_heatmap = f"""
        I have a dataset with the following columns: {columnslist}.
        I want to create a correlation heatmap to visualize the relationships
        between numerical columns. Which columns would be most suitable for
        this analysis, considering their data types and potential for revealing
        meaningful correlations? Please provide a list of column names.
        """

        # Get recommended columns for heatmap from LLM
        relevant_columns_heatmap = []
        #UNCOMMENT relevant_columns_heatmap = column_selector(loadedcsv, prompt_heatmap, num_columns=5, column_type="numeric")

        # Check if relevant_columns_heatmap contains valid columns before proceeding, if not select relevant numberic columns using custom built function
        if not (relevant_columns_heatmap and all(col in loadedcsv.columns for col in relevant_columns_heatmap)):
            relevant_columns_heatmap = select_relevant_numeric_column(loadedcsv)

        # Perform Correlation Heatmap along with png image
        correlation_heatmap = gen_correlation_heatmap(loadedcsv, new_dir_path,relevant_columns_heatmap)
        overall_analysis_summary.append(correlation_heatmap)

    if img_cnt <= max_img:
        # Perform Clustering Analysis
        prompt_clustering = f"""
        I have a dataset with the following columns: {columnslist}.
        I want to create clustering charts (e.g., DBSCAN, hierarchical clustering).
        Which columns would be most suitable for clustering analysis, considering
        their data types and potential for revealing meaningful patterns?
        Please provide a list of column names.
        """

        recommended_columns = []
        #UNCOMMENT recommended_columns=column_selector(loadedcsv, prompt_clustering, num_columns=3, column_type="numeric")
        if len(recommended_columns) >= 1:
            best_column1 = recommended_columns[0]
        if len(recommended_columns) >= 2:
            best_column2 = recommended_columns[1]
        if len(recommended_columns) >= 3:
            best_column3 = recommended_columns[3]

        # Check if recommended_columns contains valid columns before proceeding, if not select best numberic columns using custom built function
        if not recommended_columns or len(recommended_columns) == 0:
            if loadedcsv.select_dtypes(include=np.number).dropna().shape[1] > 1:
                # Select the best columns for clustering using select_best_numeric_column
                best_column1 = select_best_numeric_column(loadedcsv)
                # Get remaining numeric columns and select the next best
                remaining_cols = [col for col in loadedcsv.select_dtypes(include=np.number).columns if col != best_column1]
                if remaining_cols:  # Check if there are any remaining columns
                    best_column2 = select_best_numeric_column(loadedcsv[remaining_cols])

                    remaining_cols = [col for col in loadedcsv.select_dtypes(include=np.number).columns if col != best_column1]
                    if remaining_cols:  # Check if there are any remaining columns
                        best_column3 = select_best_numeric_column(loadedcsv[remaining_cols])
                    else:
                        best_column3 = None  # If no remaining columns, set to None
                else:
                    best_column2 = None  # If no remaining columns, set to None

    if img_cnt <= max_img:
        prompt_clustering_alg = f"""
        I have selected the following columns for clustering analysis: {recommended_columns}.
        Based on these columns, can you recommend the most suitable clustering algorithms and explain why they are appropriate?
        """
        selected_columns = recommended_columns
        recommended_algorithm = []
        #UNCOMMENT recommended_algorithm=clustering_algorithm_selector(loadedcsv, prompt_clustering_alg, selected_columns)
        if not recommended_algorithm or len(recommended_algorithm) == 0:
            recommended_algorithm = "DBSCAN"

        if recommended_algorithm == "DBSCAN" and best_column1 and best_column2:
            # Perform Clustering - Density-based
            dbclustering = dclustering(loadedcsv, new_dir_path, best_column1, best_column2)
            overall_analysis_summary.append(dbclustering)
        else:
            # Perform Clustering - Hierarchical
            hiclustering = hclustering(loadedcsv, new_dir_path, best_column1, best_column2, best_column3)
            overall_analysis_summary.append(hiclustering)

    if img_cnt <= max_img:
        # Perform Categorical Plot
        prompt_categorical = f"From the following list of columns: {columnslist}, please identify and list only the categorical columns that are suitable for creating categorical plots"

        relevant_categorical_columns = []
        #UNCOMMENT relevant_categorical_columns = column_selector(loadedcsv, prompt_categorical, num_columns=1, column_type="categorical")
        if not relevant_categorical_columns or len(relevant_categorical_columns) == 0:
            relevant_categorical_columns = select_best_categorical_column(loadedcsv)
        
        # Call plot_categorical_data with relevant columns (if any are found), 
        if relevant_categorical_columns:
            run_plot_categorical_data = plot_categorical_data(loadedcsv, new_dir_path, relevant_categorical_columns)
            overall_analysis_summary.append(run_plot_categorical_data)

    if img_cnt <= max_img:
        # Perform Time Series Plot
        prompt_timeseries = f"From the following list of columns: {columnslist}, please identify and list only the numerical columns that are suitable for creating a time series plot"

        relevant_columns = []
        #UNCOMMENT relevant_columns = column_selector(loadedcsv, prompt_timeseries, num_columns=1, column_type="numeric")

        # Check if relevant_columns contains valid columns before proceeding, if not select best numberic columns using custom built function
        if not relevant_columns or len(relevant_columns) == 0:
            relevant_columns = select_best_numeric_column(loadedcsv)

        # Identify date or year column
        date_column = identify_date_or_year_column_timeseries(loadedcsv)

        # Execute Time Series Plot    
        plot_time_series_result = plot_time_series(loadedcsv, date_column, relevant_columns, new_dir_path)
        overall_analysis_summary.append(plot_time_series_result)

    if img_cnt <= max_img:
        # Perform Histogram Plot
        relevant_columns = []
        relevant_columns = select_relevant_numeric_column(loadedcsv)
        # Execute Time Series Plot    
        histplot_result = histplot(loadedcsv, new_dir_path, relevant_columns)
        overall_analysis_summary.append(histplot_result)

    # Low resolution images for LLM
    current_image_folder = new_dir_path
    output_folder = new_dir_path + "/low_detail_images"  # Create a new folder
    os.makedirs(output_folder, exist_ok=True)

    image_base64_strings = []
    image_count = 0  # Initialize a counter
    for filename in os.listdir(current_image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            if image_count < 3:  # Process only the first 3 images
                image_path = os.path.join(current_image_folder, filename)
                dest_path = os.path.join(output_folder, filename)
                reduce_image_detail(image_path, dest_path)
                encoded_string = encode_image_base64(image_path)
                image_base64_strings.append(encoded_string)
                image_count += 1  # Increment the counter
            else:
                break  # Exit the loop after processing 3 images

    # Send data to LLM for analysis and suggestions
    data_info = {
        "summary_missing_values": LLM_Input["summary"],
        "outliers": LLM_Input["anomoly"],
        "images_base64": image_base64_strings,  # Add the images
    }

    narrative = ""
    narrative_summary = []
    narrative = explain_headers(data_info)
    if narrative is not None:
        narrative_summary.append("# Automated Data Analysis Report")
        narrative_summary.append(f"\n{narrative}")

    final_summary = narrative_summary + overall_analysis_summary

    # Save the narrative to a README file
    save_readme(final_summary,new_dir_path)

    # Save the narrative to a README file
    folder_to_delete = output_folder
    delete_folder(folder_to_delete)


# Get the 'csv' dataset

def get_data(csv_dataset):
    global img_cnt, max_img, API_KEY, API_URL

    '''
    Loads a CSV dataset into a pandas DataFrame.

    This function reads a CSV file, automatically detects its encoding, 
    and returns a pandas DataFrame containing the data.

    Args:
        csv_dataset (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded CSV data as a pandas DataFrame.
                        Returns None if there's an error during loading.
    '''

    try:
        # Detect the encoding
        with open(csv_dataset, 'rb') as f:  # Open in binary mode for detection
          resencod = chardet.detect(f.read())

        # Get the detected encoding
        encoding = resencod['encoding']

        # Load CSV
        loadedcsv = pd.read_csv(csv_dataset, encoding=encoding)
        return loadedcsv

    except Exception as exp:
        print(f"Error in extracting the file {csv_dataset}: {exp}")
        sys.exit(1)

# To perform standard descriptive analysis like summary stats, missing values, etc.

def standard_analysis_csv(loadedcsv, columnslist):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Performs standard descriptive analysis on a CSV dataset.

    This function takes a loaded pandas DataFrame and a list of columns, 
    calculates descriptive statistics (summary statistics and missing values), 
    and returns a list of strings representing the analysis summary.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        columnslist (str): A string containing the list of column names, 
                           typically generated using ', '.join(loadedcsv.columns).

    Returns:
        list: A list of strings containing the formatted analysis summary.
    '''
    # Quick summary statistics of the loaded csv dataset
    summary = loadedcsv.describe(include='all').to_dict()

    # Missing values from the loaded csv dataset
    missing_values = loadedcsv.isnull().sum().to_dict()

    # Determining each column types from loaded csv dataset
    column_info = loadedcsv.dtypes.to_dict()

    standard_analysis = []

    # General information
    standard_analysis.append(f"## A Glimpse into the Data")
    standard_analysis.append(f"Total number of rows present in the dataset: {loadedcsv.shape[0]}\n")
    standard_analysis.append(f"Total number of columns present in the dataset: {loadedcsv.shape[1]}\n")

    # Adding column descriptions to the summary
    standard_analysis.append("### Descriptive Statistics")

    prompt = f"""
    You are a data science assistant.
    Given the following columns from a dataset: {columnslist}
    Please identify and list only the numerical columns that would be suitable for generating descriptive statistics.
    Exclude columns that are likely to be identifiers, categorical, or contain irrelevant numerical data for descriptive analysis (e.g., IDs, codes, timestamps).
    Provide your answer as a comma-separated list of column names.
    """

    relevant_columns = []
    #UNCOMMENT relevant_columns = column_selector(loadedcsv, prompt, num_columns=None, column_type="numeric")
    if not relevant_columns or len(relevant_columns) == 0:
        relevant_columns = select_relevant_numeric_column(loadedcsv)

   # Use the custom formatting function
    styled_table = format_describe_table(loadedcsv, relevant_columns)
    standard_analysis.append(styled_table)
    standard_analysis.append("\n")

    # Adding missing values to the summary
    standard_analysis.append("### Missing Values")
    standard_analysis.append("This section identifies the number of missing values in each column of the dataset. Missing values are represented by NaN (Not a Number).\n")  # Description

    missing_dict = loadedcsv.isnull().sum().to_dict()
    for column, missing_count in missing_dict.items():  # Iterate through missing_dict
        if missing_count != 0:  # Exclude columns with 0 missing values
            standard_analysis.append(f"- **{column}:** {missing_count} missing values")
    standard_analysis.append("\n")

    return standard_analysis

# Detect anomolies/outlier using Interquartile Range (IQR)

def anomoly_detection(loadedcsv, recommended_columns):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Detects anomalies (outliers) in the specified columns of a dataset.

    This function uses the Interquartile Range (IQR) method to identify 
    outliers in the given columns of a pandas DataFrame. It returns a 
    formatted text report summarizing the detected anomalies.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        recommended_columns (list): A list of column names where anomaly 
                                     detection should be performed.

    Returns:
        list: A list of strings containing the formatted outlier report. 
                        Returns None if there's an error during loading.
    '''    
    #Calculation of inter quartile range for the selected numeric columns

    numeric_data = loadedcsv[recommended_columns].dropna()
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    interquartilerange = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * interquartilerange)) | (numeric_data > (Q3 + 1.5 * interquartilerange))).sum().to_dict()
    outlier_info = ""
    no_outliers_cols = []  # List to store columns with no outliers

    for column, count in outliers.items():
        if count > 0:
            outlier_info += f"- **{column}:** {count} outliers\n"
        else:
            no_outliers_cols.append(column)  # Add column to no_outliers_cols list

    # Add the aggregated "No outliers detected" line
    if no_outliers_cols:
        outlier_info += f"- **No outliers detected in:** {', '.join(no_outliers_cols)}\n"

    # Add the summary of the outlier analysis
    outlier_report = []
    outlier_report.append("### Outlier Expedition: Unmasking the Unusual\n")
    outlier_report.append("I embarked on an expedition to uncover outliers—those data points that strayed from the familiar path. Like detectives searching for clues, we used statistical techniques to identify values that stood out from the crowd.\n")
    outlier_report.append("Imagine our dataset as a map, with most data points clustered in well-trodden areas. Outliers, however, ventured into uncharted territories, marking anomalies or potential areas of interest.\n")
    outlier_report.append("I used the Interquartile Range (IQR) method as our compass, flagging data points that fell outside the expected range. These outliers could represent:\n")
    outlier_report.append("* **Data Entry Errors:** Typos or mistakes during data collection.\n")
    outlier_report.append("* **Unusual Events:** Rare occurrences that deviate from the norm.\n")
    outlier_report.append("* **Interesting Insights:** Potential areas for further investigation or deeper understanding.\n\n")
    outlier_report.append("#### Outlier Summary\n")
    outlier_report.append(outlier_info.strip())
    outlier_report.append("\n")

    return outlier_report

# Function to visualize outliers

def plot_outliers(loadedcsv, recommended_columns, output_path):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Creates and saves a box plot visualizing outliers in the specified columns.

    This function generates a box plot using Seaborn to visualize the 
    distribution of data and highlight outliers in the given columns 
    of a pandas DataFrame. The plot is saved as a PNG image in the 
    specified output directory.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        recommended_columns (list): A list of column names to include in the box plot.
        output_path (str): The directory where the box plot image will be saved.

    Returns:
        list: A list of strings containing markdown elements, 
              including the embedded image tag for the box plot and a narrative summary.
                        Returns None if there's an error during loading.
    '''

    if len(recommended_columns) > 1:  # Check if there are at least 2 relevant columns
        # Create the box plot for the outliers
        numeric_data = loadedcsv[recommended_columns].dropna()
        if numeric_data.empty:
            return
        mathplt.figure(figsize=(512/100, 512/100), dpi=100)
        sns.boxplot(data=numeric_data)
        mathplt.title("Anomoly Detection - Box-Plot")

        # Rotate the x-axis labels to be vertical
        mathplt.xticks(rotation=90)
        anomoly_path = os.path.join(output_path, "Anomoly_Detection_BoxPlot.png")
        if img_cnt <= max_img:
            mathplt.savefig(anomoly_path)
            img_cnt += 1
        mathplt.close()
        outlier_report = []
        # Embed image tag
        anomoly_tag = embed_image_in_markdown(anomoly_path, alt_text="Anomoly Detection BoxPlot")

        # Add the summary of the outlier analysis - box plot
        outlier_report.append(anomoly_tag)
        outlier_report.append("\n")
        outlier_report.append("The visualization above showcases these outliers, highlighting them against the backdrop of the normal data distribution. By understanding these unusual data points, we could either correct errors, investigate anomalies, or uncover hidden patterns that might have otherwise gone unnoticed. Our outlier expedition helped us refine our data and gain a more comprehensive understanding of its nuances.\n")

        return outlier_report
    else:
        print("Not enough relevant numerical columns for correlation heatmap.")
        return []  # Return an empty list if not enough columns

# Correlation between all the numaric fields
def gen_correlation_heatmap(loadedcsv, output_path, relevant_cols):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Generates and saves a correlation heatmap for the selected relevant columns.

    This function calculates the correlation between the specified numerical 
    columns of a pandas DataFrame and creates a heatmap visualization using 
    Seaborn. The heatmap is saved as a PNG image in the specified output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        output_path (str): The directory where the heatmap image will be saved.
        relevant_cols (list): A list of numerical column names to include in the correlation calculation.

    Returns:
        list: A list of strings containing markdown elements, 
              including the embedded image tag for the heatmap and a narrative summary.
                        Returns None if there's an error during loading.
    '''   

    if len(relevant_cols) > 1:  # Check if there are at least 2 relevant columns
        # Subset the DataFrame with relevant columns
        relevant_data = loadedcsv[relevant_cols]

        # Create correlation heatmap with relevant columns
        mathplt.figure(figsize=(512/100, 512/100), dpi=100)
        sns.heatmap(relevant_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        heatmap_path = os.path.join(output_path, "Correlation_heatmap.png")
        mathplt.title("Correlation Heatmap")
        if img_cnt <= max_img:
            mathplt.savefig(heatmap_path, dpi=100)
            img_cnt += 1
        mathplt.close()

        # Embed image tag
        heatmap_tag = embed_image_in_markdown(heatmap_path, alt_text="Correlation Heatmap")

        # Add the summary of the correlation heatmap

        corr_heatmap = []
        corr_heatmap.append("### Unveiling Relationships: A Heatmap's Narrative\n")
        corr_heatmap.append("I delved into the relationships between numerical features using a correlation heatmap. This visual tool painted a picture of how different variables in our dataset danced together—sometimes in harmony, sometimes in opposition. Imagine it as a ballroom where each feature is a dancer, and the heatmap reveals their choreography.\n")
        corr_heatmap.append("Warm colors, like reds and oranges, highlighted positive correlations, where features tended to waltz together, rising and falling in unison. Cool colors, like blues and greens, indicated negative correlations, where features moved in opposite directions, creating a dynamic interplay.\n")
        corr_heatmap.append("The intensity of the color reflected the strength of the relationship—the brighter the color, the stronger the bond or the more pronounced the opposition.\n")
        corr_heatmap.append(heatmap_tag)
        corr_heatmap.append("\n\n")
        corr_heatmap.append("This heatmap narrative unveiled the hidden connections within our data, allowing us to:\n")
        corr_heatmap.append("* **Identify Strong Partnerships:** We spotted features that shared a strong positive correlation, suggesting a close relationship or shared influence.\n")
        corr_heatmap.append("* **Uncover Opposing Forces:** We identified features with negative correlations, indicating a potential trade-off or inverse relationship.\n")
        corr_heatmap.append("* **Spot Potential Conflicts:** We observed areas with weak or no correlation, suggesting that some features might be dancing independently of others.\n")
        corr_heatmap.append("By understanding these relationships, we gained a deeper understanding of the data's dynamics, helping us to build better models, make more accurate predictions, and uncover hidden patterns.\n")
        return corr_heatmap

    else:
        print("Not enough relevant numerical columns for correlation heatmap.")
        return []  # Return an empty list if not enough columns

# Clustering (Density-Based Clustering)
def dclustering(loadedcsv, output_path, best_column1, best_column2):
    global img_cnt, max_img, API_KEY, API_URL

    '''
    Performs density-based clustering (DBSCAN) on the specified columns.

    This function applies the DBSCAN clustering algorithm to the data in the 
    specified columns of a pandas DataFrame. It generates a scatter plot 
    visualizing the clusters and saves it as a PNG image in the output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        output_path (str): The directory where the cluster plot image will be saved.
        best_column1 (str): The name of the first column to use for clustering.
        best_column2 (str): The name of the second column to use for clustering.

    Returns:
        list: A list of strings containing markdown elements, 
              including the embedded image tag for the cluster plot and a narrative summary.
                        Returns None if there's an error during loading.
    '''

    if best_column1 and best_column2: # Check if there are relevant columns - best column 1 and 2

        # Perform standard scalar and fit dbscan to get the clustering

        numeric_data = loadedcsv[[best_column1, best_column2]].dropna()
        sscaler = StandardScaler()
        sscaled_data = sscaler.fit_transform(numeric_data)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(sscaled_data)
        numeric_data['cluster'] = clusters
        mathplt.figure(figsize=(512/100, 512/100), dpi=100)
        sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="cividis")
        mathplt.title("Density-Based Clustering")
        dbcluster_path = os.path.join(output_path, "Density_based_clusters.png")
        if img_cnt <= max_img:
            mathplt.savefig(dbcluster_path)
            img_cnt += 1
        mathplt.close()

        dcluster_tag = embed_image_in_markdown(dbcluster_path, alt_text="Clustering - Density-based")

        # Add the summary of the dbscan clustering
        dcluster = []
        dcluster.append("### Unraveling Hidden Galaxies: A Density-Based Journey\n")
        dcluster.append("I embarked on a journey to uncover hidden galaxies within our data using density-based clustering. This method, like a cosmic cartographer, charted the data universe, identifying clusters based on the density of data points—like stars forming constellations in the night sky.\n")
        dcluster.append("Imagine our dataset as a vast expanse of space, with data points representing celestial objects. Density-based clustering, employing the DBSCAN algorithm, sought to identify regions where these objects were densely packed, forming clusters—akin to galaxies teeming with stars.\n")
        dcluster.append("Unlike traditional clustering methods that search for spherical shapes, DBSCAN excels at identifying clusters of arbitrary shapes, like cosmic nebulae or irregular galaxies. It also filters out the 'cosmic noise'—those solitary data points that don't belong to any particular cluster.\n")
        dcluster.append(dcluster_tag)
        dcluster.append("\n\n")
        dcluster.append("The visualization above showcases the discovered galaxies within our data, with each color representing a distinct cluster and the outliers (noise) marked as solitary points. This density-based journey allowed us to:\n")
        dcluster.append("* **Identify Densely Populated Regions:** I observed areas where data points were tightly clustered, revealing distinct communities or patterns.\n")
        dcluster.append("* **Uncover Irregular Shapes:** I discovered clusters with non-spherical or arbitrary shapes, reflecting the complex relationships within the data.\n")
        dcluster.append("* **Filter Cosmic Noise:** I identified and separated noise points that did not belong to any cluster, providing a clearer view of the underlying structure.\n")
        dcluster.append("By understanding the density distribution and patterns within our data, I gained insights into the inherent groupings, guiding further exploration and analysis. The journey through this data universe helped uncover valuable knowledge hidden within the cosmic tapestry of our dataset.\n")
        return dcluster
    else:
        print("Not enough suitable columns found for DBSCAN clustering.")
        return []  # Return an empty list if clustering is not possible

# Clustering (Hierarchical)
def hclustering(loadedcsv, output_path, best_column1, best_column2, method='ward', metric='euclidean'):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Performs hierarchical clustering on the specified columns and generates a dendrogram.

    This function applies hierarchical clustering to the data in the specified 
    columns of a pandas DataFrame using the specified method and metric. 
    It creates a dendrogram visualization of the clustering hierarchy and 
    saves it as a PNG image in the output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        output_path (str): The directory where the dendrogram image will be saved.
        best_column1 (str): The name of the first column to use for clustering.
        best_column2 (str): The name of the second column to use for clustering.
        method (str, optional): The linkage method to use for hierarchical clustering. Defaults to 'ward'.
        metric (str, optional): The distance metric to use for clustering. Defaults to 'euclidean'.

    Returns:
        list: A list of strings containing markdown elements, 
              including the embedded image tag for the dendrogram and a narrative summary.
                        Returns None if there's an error during loading.
    '''

    if best_column1 and best_column2: # Check if there are relevant columns - best column 1 and 2
    
        data = loadedcsv[best_column1,best_column2]

        # Scale the data (important for distance-based clustering)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Perform hierarchical clustering
        linked = linkage(scaled_data, method=method, metric=metric)

        # Plot the dendrogram
        mathplt.figure(figsize=(512/100, 512/100), dpi=100)
        dendrogram(linked,
                  labels=data.index,  # Use data index as labels
                  orientation='top',
                  distance_sort='descending',
                  show_leaf_counts=True)
        mathplt.xlabel('Data Points')
        mathplt.ylabel('Distance')
        mathplt.title("Hierarchical Clustering - Dendrogram")

        # Rotate x-axis labels for better readability (optional)
        mathplt.xticks(rotation=90)

        hc_path = os.path.join(output_path, "Hierarchical_clustering_dendrogram.png")
        if img_cnt <= max_img:
            mathplt.savefig(hc_path)
            img_cnt += 1
        mathplt.close()

        # Embed image tag
        hc_tag = embed_image_in_markdown(hc_path, alt_text="Clustering - Hierarchical - Dendrogram")

        # Add the summary of the hierarchical clustering
        hcluster = []
        hcluster.append("### Data Family Tree: A Hierarchical History\n")
        hcluster.append("I traced the lineage of our data using hierarchical clustering, creating a family tree that revealed how data points were related. This method grouped similar data points into branches, forming a hierarchy that showed their relationships from the closest kin to distant relatives.\n")
        hcluster.append("Imagine our dataset as a collection of individuals with diverse characteristics. Hierarchical clustering, like a genealogist, sought to build a family tree that connected these individuals based on their similarities.\n")
        hcluster.append("Starting with individual data points, the method gradually merged similar points into clusters, forming branches that grew into larger clusters, eventually culminating in a single root representing the entire dataset.\n")
        hcluster.append(hc_tag)
        hcluster.append("\n\n")
        hcluster.append("The dendrogram above visualizes this data family tree, with each branch representing a cluster and the height of the branch indicating the distance or dissimilarity between clusters. This hierarchical history allowed us to:\n")
        hcluster.append("* **Understand Data Relationships:** I observed how data points were related to each other, from close neighbors to distant relatives.\n")
        hcluster.append("* **Identify Subgroups:** I discovered subgroups within the data, revealing a hierarchical structure of communities and their characteristics.\n")
        hcluster.append("* **Explore Data Organization:** I gained insights into the natural grouping tendencies of the data, aiding in pattern recognition and data interpretation.\n")
        hcluster.append("By understanding the data's family tree, I gained a deeper understanding of its structure and organization, informing our analysis and guiding further exploration.\n")
        return hcluster
    else:
        print("Not enough suitable columns found for hierarchical clustering.")
        return []  # Return an empty list if clustering is not possible

# Time series analysis

def plot_time_series(loadedcsv, date_column, relevant_columns, output_path):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Generates and saves a time series plot for the specified data.

    This function creates a time series plot using Seaborn, visualizing 
    the trend of the specified relevant columns over time, using the 
    date_column as the x-axis. The plot is saved as a PNG image in 
    the output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        date_column (str): The name of the column containing the date or time data.
        relevant_columns (list or str):  The column(s) to plot on the y-axis. 
                                        Can be a single column name or a list.
        output_path (str): The directory where the time series plot image will be saved.

    Returns:
        list: A list of strings containing markdown elements, 
              including the embedded image tag for the time series plot 
              and a narrative summary. Returns an empty list if no plot was created.
    '''
    try:
        # Make a copy to avoid modifying the original data
        df = loadedcsv.copy()

        # Check and standardize the date column
        if df[date_column].dtype == 'int64':  # If it's a year
            df[date_column] = pd.to_datetime(df[date_column], format='%Y')
        elif df[date_column].dtype == 'object':  # If it's a string
            try:
                df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d')  # Or your format
            except ValueError:
                raise ValueError(f"Cannot parse dates in column '{date_column}'. Ensure it's a valid date or year format.")
        elif not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            raise TypeError(f"Column '{date_column}' must be a datetime, string, or integer year format.")

        # Sort by the date column for proper plotting
        df = df.sort_values(by=date_column)

        best_column = relevant_columns
        non_numeric_df = loadedcsv.select_dtypes(include=['number'])
        if best_column in non_numeric_df.columns:
            mathplt.figure(figsize=(512/100, 512/100), dpi=100)
            sns.lineplot(data=loadedcsv, x=date_column, y=best_column)
            mathplt.title(f"Time Series Analysis for {best_column}")
            ts_path = os.path.join(output_path, "Time_Series_Analysis.png")
            if img_cnt <= max_img:
                mathplt.savefig(ts_path)
                img_cnt += 1
            mathplt.close()

            # Embed image tag
            ts_tag = embed_image_in_markdown(ts_path, alt_text="Time Series Analysis")
            
            # Add the summary of the time series plot
            tsanalysis = []
            tsanalysis.append("### Riding the Waves of Time: A Time Series Tale\n")
            tsanalysis.append("I embarked on a journey through time, charting the course of our data using time series analysis. This method, like a seasoned mariner, mapped the ebb and flow of variables over time, revealing trends, seasonality, and potential turning points.\n")
            tsanalysis.append("Imagine our data as a vast ocean, with each variable represented by a wave, cresting and troughing over time. Time series analysis, like a skilled navigator, helped us understand the rhythm of these waves, revealing hidden patterns and guiding us through the turbulent waters of change.\n")
            tsanalysis.append("By plotting the values of variables against time, we could visualize the long-term trajectory of the data, identifying periods of growth, decline, or stability. We also explored the presence of seasonality—recurring patterns that emerged at regular intervals, like the tides or the changing seasons.\n")
            tsanalysis.append(ts_tag)
            tsanalysis.append("\n\n")
            tsanalysis.append("The time series visualization above reveals the dynamic nature of our data, allowing us to:\n")
            tsanalysis.append("* **Identify Trends:** We observed the overall direction of movement in variables over time, revealing long-term patterns like growth or decline.\n")
            tsanalysis.append("* **Spot Seasonality:** We discovered recurring patterns that occurred at specific time intervals, suggesting regular fluctuations or cyclical behaviors.\n")
            tsanalysis.append("* **Detect Anomalies:** We identified unusual spikes or dips in the data, indicating potential outliers or significant events that warrant further investigation.\n")
            tsanalysis.append("By understanding the temporal dynamics of our data, we gained insights into how variables evolve, helping us to forecast future values, understand past behaviors, and identify key periods of change or stability.\n")
            return tsanalysis
    except Exception as e:
        tsanalysis = []  # Return an empty list if no time series plot was created
        return tsanalysis

# Function for categorical data analysis
def plot_categorical_data(loadedcsv, output_path, relevant_categorical_columns):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Generates and saves plots for categorical data.

    This function creates visualizations for categorical data in the specified 
    columns of a pandas DataFrame, typically using bar charts or count plots 
    to show the frequency of each category. The plots are saved as PNG images 
    in the output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        output_path (str): The directory where the categorical plot images will be saved.
        relevant_categorical_columns (list or str): The column(s) containing 
                                                categorical data to be visualized. 
                                                Can be a single column name or a list.

    Returns:
        list: A list of strings containing markdown elements, 
              including embedded image tags for the categorical plots 
              and a narrative summary. Returns an empty list if no plot was created.
    '''

    def adjust_labels(ax, labels, max_chars_per_line=10, rotate=False):
        '''
        Helper function to adjust labels by splitting long ones into multiple lines
        and optionally rotating them.
        '''
        new_labels = []
        for label in labels:
            split_label = "\n".join(
                [label[i:i + max_chars_per_line] for i in range(0, len(label), max_chars_per_line)]
            )
            new_labels.append(split_label)
        ax.set_xticks(range(len(new_labels)))
        ax.set_xticklabels(new_labels, rotation=90 if rotate else 0, ha='center' if rotate else 'right')
    
    best_column = relevant_categorical_columns
    non_numeric_df = loadedcsv.select_dtypes(exclude=['number'])
    if best_column in non_numeric_df.columns:
        value_counts = loadedcsv[best_column].value_counts()
        num_unique_values = len(value_counts)
        # Create a consolidated figure
        fig, axs = mathplt.subplots(2 if num_unique_values > 15 else 1, 1, figsize=(16, 12), squeeze=False)
        axs = axs.flatten()

        if num_unique_values <= 15:
            # Single plot for up to 15 unique values
            ax = axs[0]
            sns.countplot(x=relevant_categorical_columns, data=loadedcsv, order=value_counts.index, ax=ax)
            ax.set_title(f"Distribution of {relevant_categorical_columns}")
            adjust_labels(ax, value_counts.index, max_chars_per_line=10, rotate=False)

        elif 15 < num_unique_values <= 30:
            # Single plot with rotated labels for 16-30 unique values
            ax = axs[0]
            sns.countplot(x=relevant_categorical_columns, data=loadedcsv, order=value_counts.index, ax=ax)
            ax.set_title(f"Distribution of {relevant_categorical_columns}")
            adjust_labels(ax, value_counts.index, max_chars_per_line=15, rotate=True)

        else:
            # Two plots for more than 30 unique values: Top 15 and Bottom 15
            top_15 = value_counts.head(15)
            bottom_15 = value_counts.tail(15)

            # Top 15 graph
            sns.barplot(x=top_15.index, y=top_15.values, ax=axs[0])
            axs[0].set_title(f"Top 15 Distribution of {relevant_categorical_columns}")
            adjust_labels(axs[0], top_15.index, max_chars_per_line=15, rotate=True)

            # Bottom 15 graph
            sns.barplot(x=bottom_15.index, y=bottom_15.values, ax=axs[1])
            axs[1].set_title(f"Bottom 15 Distribution of {relevant_categorical_columns}")
            adjust_labels(axs[1], bottom_15.index, max_chars_per_line=15, rotate=True)

        mathplt.tight_layout()
        consolidated_path = os.path.join(output_path, f"Categorical_Distribution_{relevant_categorical_columns}.png")
        if img_cnt <= max_img:
            mathplt.savefig(consolidated_path)
            img_cnt += 1
        mathplt.close()

        cat_tag = embed_image_in_markdown(consolidated_path, alt_text="Categorical Data Distribution")

        catanalysis = []
        catanalysis.append("### Unmasking the Categories: A Categorical Chronicle\n")
        catanalysis.append("I delved into the realm of categories, unraveling the distribution and relationships within our data using categorical analysis. This method, like a skilled detective, examined the patterns and frequencies of different groups, revealing hidden insights and potential connections.\n")
        catanalysis.append("Imagine our data as a diverse population, with each data point belonging to a specific category or group. Categorical analysis, like a census taker, helped us understand the composition of this population, identifying the dominant groups, the outliers, and the relationships between different categories.\n")
        catanalysis.append("By visualizing the distribution of categories using bar charts or count plots, we could see which groups were most prevalent, which were rare, and how they compared to each other. We also explored the relationships between categories, looking for potential associations or dependencies.\n")
        catanalysis.append(cat_tag)
        catanalysis.append("\n\n")
        catanalysis.append("The categorical visualizations above unveiled the categorical landscape of our data, allowing us to:\n")
        catanalysis.append("* **Identify Dominant Categories:** We observed which categories were most frequent, providing insights into the prevalent characteristics or groups within the data.\n")
        catanalysis.append("* **Spot Outliers:** We identified categories with unusually low or high frequencies, suggesting potential anomalies or unique data points.\n")
        catanalysis.append("* **Explore Relationships:** We investigated the associations between different categories, looking for patterns or dependencies that might reveal underlying connections.\n")
        catanalysis.append("By understanding the categorical composition and relationships within our data, we gained valuable insights into the diversity and structure of our dataset, enriching our analysis and guiding further exploration.\n")

        return catanalysis
    else:
        catanalysis = []  # Return an empty list if no categorical analysis was created
        return tsanalysis

#Distribution of Numerical Columns
def histplot(loadedcsv, output_path, relevant_numeric_column):
    """Generates and saves a histogram plot for numerical data.

    This function creates a histogram visualization for the specified 
    numerical column of a pandas DataFrame. It uses Seaborn to generate 
    the histogram, including a kernel density estimate (KDE), and saves 
    the plot as a PNG image in the output path.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        output_path (str): The directory where the histogram image will be saved.
        relevant_numeric_column (str): The name of the numerical column 
                                        to create the histogram for.

    Returns:
        list: A list of strings containing markdown elements, including the 
              embedded image tag for the histogram and a narrative summary. 
              Returns an empty list if no plot was created.
    """

    if relevant_numeric_column:
        mathplt.figure(figsize=(512/100, 512/100), dpi=100)
        sns.histplot(loadedcsv[relevant_numeric_column], kde=True, bins=30)

        # Construct the output file path
        dist_path = os.path.join(output_path, f"Histogram_distribution.png")  

        mathplt.title(f"Distribution of {relevant_numeric_column}")
        mathplt.xlabel(relevant_numeric_column)
        mathplt.savefig(dist_path, dpi=100)
        mathplt.close()

        # Embed image tag
        hist_tag = embed_image_in_markdown(dist_path, alt_text=f"Distribution of {relevant_numeric_column}")

        # Narrative summary (similar to plot_categorical_data)
        hplot = []  # Initialize as an empty list
        hplot.append("### Unmasking the Distribution: A Numerical Narrative\n")  # Similar title style
        hplot.append("I embarked on a journey to understand the distribution of numerical data using a histogram. ")
        hplot.append("This visual tool revealed the frequency of different values, ")
        hplot.append("providing insights into the data's central tendencies, spread, and potential outliers.\n")
        hplot.append(hist_tag)  # Embedded image
        hplot.append("\n\n")
        hplot.append("The histogram showcased the range of values, ")
        hplot.append("the concentration of data points around specific areas, ")
        hplot.append("and the presence of any unusual or extreme values.\n")

        return hplot
    else:
        return []

# Calling functions to identify/select/format the input and output data
def identify_date_or_year_column_timeseries(loadedcsv):
    """
    Identifies the date or year column in a pandas DataFrame.

    Args:
        loadedcsv: The pandas DataFrame.

    Returns:
        The name of the date or year column, or None if not found.
    """

    # Check for columns with 'date' or 'year' in their name (case-insensitive)
    date_year_cols = [
        col for col in loadedcsv.columns
        if any(keyword in col.lower() for keyword in ['date', 'year'])
    ]

    if date_year_cols:
        # If multiple columns are found, prioritize 'date' over 'year'
        if any('date' in col.lower() for col in date_year_cols):
            return next(col for col in date_year_cols if 'date' in col.lower())
        else:
            return date_year_cols[0]  # Return the first 'year' column

    # If no obvious date/year columns are found, check data types
    for col in loadedcsv.columns:
        if loadedcsv[col].dtype == 'datetime64[ns]':  # Check for datetime data type
            return col
        # If values are numeric and within a reasonable year range, consider as year
        elif loadedcsv[col].dtype in ['int64', 'float64'] and loadedcsv[col].between(1900, 2100).all():
            return col

    return None  # Return None if no date or year column is found

def select_best_categorical_column(loadedcsv):
    '''
    Selects the best categorical column for analysis.

    This function analyzes the categorical columns in a pandas DataFrame and 
    attempts to identify the most suitable column for further analysis, 
    typically based on criteria such as the number of unique values, 
    data distribution, and potential relevance to the analysis goals.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.

    Returns:
        str or None: The name of the selected best categorical column. 
                     Returns None if no suitable column is found.
    '''  
    categorical_cols = loadedcsv.select_dtypes(include=['object', 'category']).columns
    best_column = None
    max_cardinality = 0

    for indx, column in enumerate(categorical_cols):
        cardinality = loadedcsv[column].nunique()
        if cardinality > max_cardinality:
            max_cardinality = cardinality
            best_column = column
            best_column_index = indx
    return best_column

def select_best_numeric_column(loadedcsv):
    '''
    Selects the best numeric column for analysis.

    This function analyzes the numeric columns in a pandas DataFrame and 
    attempts to identify the most suitable column for further analysis, 
    typically based on criteria such as data distribution, variance, 
    and potential relevance to the analysis goals.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.

    Returns:
        str or None: The name of the selected best numeric column. 
                     Returns None if no suitable column is found.
    '''
    numeric_cols = loadedcsv.select_dtypes(include=['number']).columns
    excluded_keywords = ['id', 'code', 'identifier', 'key']
    numeric_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in excluded_keywords)]

    best_column = None
    max_std = 0

    for column in numeric_cols:
        std_dev = loadedcsv[column].std()
        if std_dev > max_std:
            max_std = std_dev
            best_column = column

    return best_column

def select_relevant_numeric_column(loadedcsv):
    '''
    Selects all relevant numeric column for analysis.

    This function analyzes the numeric columns in a pandas DataFrame and 
    attempts to identify the all suitable column for further analysis, 
    and potential relevance to the analysis goals.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.

    Returns:
        str or None: The name of the selected all relevant numeric column. 
                     Returns None if no numberic column is found.
    '''
    numeric_cols = loadedcsv.select_dtypes(include=['number']).columns
    excluded_keywords = ['id', 'code', 'identifier', 'key']
    relevant_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in excluded_keywords)]
    return relevant_cols

def format_describe_table(dataframe, relevant_columns):
    '''
    Formats the descriptive statistics table for better readability.

    This function takes a pandas DataFrame and a list of relevant columns, 
    calculates descriptive statistics using the `describe()` method, and 
    formats the resulting table into a more readable string representation, 
    often using libraries like `tabulate`.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        relevant_columns (list): A list of column names to include 
                                  in the descriptive statistics.

    Returns:
        str: A formatted string representation of the descriptive 
            statistics table.  
    '''
    filtered_dataframe = dataframe[relevant_columns]

    # Get descriptive statistics
    desc_stats = filtered_dataframe.describe(include='all').transpose()

    # Round 'mean' and 'std' columns to 2 decimal places (if they exist)
    for col in ['mean', 'std']:
        if col in desc_stats.columns:
            desc_stats[col] = desc_stats[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else str(x))

    # Fill missing values with 'NaN' for clarity
    desc_stats = desc_stats.fillna('NaN')

    # Explicitly replace missing values with 'NaN' using modern pandas practices
    pd.set_option('future.no_silent_downcasting', True)  # Ensure compatibility with future pandas behavior
    desc_stats = desc_stats.fillna("NaN").infer_objects(copy=False).map(str)

    # Define headers
    headers = ["Feature", "Count", "Unique", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    # Helper function to safely format numerical values
    def safe_format(value):
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)  # Convert non-numerical values to strings
    # Collect rows including headers for width calculation
    rows = [
        headers  # Header row
    ] + [
        [
            feature,
            safe_format(row.get('count', 'NaN')),
            safe_format(row.get('unique', 'NaN')),
            safe_format(row.get('mean', 'NaN')),
            safe_format(row.get('std', 'NaN')),
            safe_format(row.get('min', 'NaN')),
            safe_format(row.get('25%', 'NaN')),
            safe_format(row.get('50%', 'NaN')),
            safe_format(row.get('75%', 'NaN')),
            safe_format(row.get('max', 'NaN'))
        ]
        for feature, row in desc_stats.iterrows()
    ]
    # Determine the maximum width for each column
    col_widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
    # Create the Markdown table
    markdown_table = ""
    for i, row in enumerate(rows):
        # Format each row based on column widths
        formatted_row = " | ".join(f"{str(item):<{col_widths[j]}}" for j, item in enumerate(row))
        markdown_table += f"| {formatted_row} |\n"
        # Add a separator after the header row
        if i == 0:
            separator = "-|-".join("-" * col_width for col_width in col_widths)
            markdown_table += f"|{separator}|\n"
    return markdown_table

def reduce_image_detail(image_path, output_path, max_size=(256, 256), quality=50):

  '''
  This function takes an image, resizes it to a smaller dimension while maintaining aspect ratio, 
  and saves it with a lower quality setting to reduce file size and detail.

  Args:
      image_path (str): The path to the input image file.
      output_path (str): The path where the reduced-detail image will be saved.
      max_size (tuple, optional): The maximum dimensions (width, height) of the resized image. 
                                  Defaults to (256, 256).
      quality (int, optional): The quality setting for saving the image (0-100). 
                               Lower values result in smaller file sizes and lower quality. 
                               Defaults to 50.

  Returns:
      None. The function saves the reduced-detail image to the specified output path.
  '''
  try:
    img = Image.open(image_path)
    img.thumbnail(max_size)  # Resize while maintaining aspect ratio
    img.save(output_path, quality=quality)  # Save with lower quality
  except Exception as e:
    print(f"Error processing image {image_path}: {e}")

# Function to convert saved images into Base64 images for Readme.MD | LLM
def embed_image_in_markdown(image_path, alt_text="Image"):
  '''
  Embeds an image as a Base64 string in Markdown.

  Args:
    image_path: The path to the image file.
    alt_text: The alternative text for the image.

  Returns:
    A Markdown image tag with the embedded image.
  '''
  with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

  image_tag = f"![{alt_text}](data:image/png;base64,{encoded_string})"
  return image_tag

def encode_image_base64(image_path):
    '''
    Encodes an image file into a Base64 string.

    This function reads an image file from the specified path, encodes 
    its content into a Base64 string representation, and returns the 
    encoded string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The Base64 encoded string representing the image.
    '''
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def summarize_text(text, max_length):
    '''
    Truncates or summarizes text to a maximum length.

    Args:
        text (str): The input text.
        max_length (int): The maximum allowed length of the summarized text.

    Returns:
        str: The summarized text.
    '''
    if len(text) <= max_length:
        return text  # Return original text if already within limit
    else:
        # Truncate with ellipsis (...)
        return text[:max_length - 3] + "..."

def extract_key_points(outlier_data):
    '''
    Extracts key points from outlier data.

    Args:
        outlier_data (list or dict): The outlier data (e.g., from the `outliers` key in `data_info`).

    Returns:
        str: A string summarizing the key points about outliers.
    '''
    if isinstance(outlier_data, list):
        # Assuming list of strings (e.g., from standard_analysis_csv)
        key_points = ", ".join(outlier_data) 
    elif isinstance(outlier_data, dict):
        # Assuming dict with column: outlier_count (e.g., from anomoly_detection)
        key_points = "; ".join([f"{col}: {count} outliers" for col, count in outlier_data.items() if count > 0])
    else:
        key_points = "No outlier information available."

    return key_points

# Save results in README
def save_readme(overall_analysis_summary, output_path):
    
    '''
    Saves the overall analysis summary to a README.md file.

    Args:
        overall_analysis_summary: The list containing the analysis summary.
        output_path: The directory where the README.md file should be saved.
    '''

    with open(os.path.join(output_path, "README.md"), "w") as f:
        for item in overall_analysis_summary:
            if isinstance(item, list):
                # If item is a list, join its elements with newlines
                f.write("\n".join(map(str, item)))
                f.write("\n")
            elif item is not None:
                # If item is not None and not a list, write it directly
                f.write(str(item))
                f.write("\n")

def delete_folder(folder_path):
    '''
    Deletes a folder and its contents.

    This function attempts to delete the specified folder and all its 
    contents using the `shutil.rmtree` function. It prints a success 
    message if the deletion is successful or an error message if an 
    exception occurs.

    Args:
        folder_path (str): The path to the folder to delete.

    Returns:
        None
    '''
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting folder '{folder_path}': {e}")

# Function to get recommended columns to be used for further analysis
def column_selector(loadedcsv, inputprompt, num_columns=None, column_type="all"):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Queries an LLM to select relevant columns from a DataFrame.

    This function uses a Large Language Model (LLM) to recommend a subset 
    of columns from a pandas DataFrame based on a provided prompt template, 
    the desired number of columns, and the type of columns (numeric, 
    categorical, or all). It constructs a prompt using the template and 
    parameters, queries the LLM, and extracts the recommended column 
    names from the LLM's response.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        inputprompt (str): A template string for the prompt to send to the LLM.
        num_columns (int, optional): The desired number of columns to select. 
                                      If None, the LLM selects all relevant columns.
                                      Defaults to None.
        column_type (str, optional): The type of columns to focus on 
                                     ("numeric", "categorical", or "all"). 
                                     Defaults to "all".

    Returns:
        list: A list of column names recommended by the LLM. 
              Returns an empty list in case of error or no recommendations.
    '''
    if API_KEY:
        # Prepare the prompt
        prompt = inputprompt.format(columns=loadedcsv.columns.tolist())

        # Add instructions for number of columns and column type if specified
        if num_columns is not None:
            prompt += f"\nPlease provide {num_columns} column names."
        if column_type != "all":
            prompt += f"\nFocus on {column_type} columns."

        # Prepare the payload for AIPROXY
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.5,
        }

        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        # Extract recommended columns
        if response.status_code == 200:
            response_data = json.loads(response.text)
            recommended_columns_str = response_data["choices"][0]["message"]["content"].strip()

            # Split the response string into a list of columns (handling commas and newlines)
            recommended_columns = [col.strip() for col in recommended_columns_str.replace('\n', ',').split(',') if col.strip()]
            
            #''TODELETE LATER''
            result = response.json()  
            # Extract the usage data (tokens)
            usage = result['usage']
            prompt_tokens = usage['prompt_tokens']
            completion_tokens = usage['completion_tokens']
            total_tokens = usage['total_tokens']

            # Print the token usage
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Total Tokens: {total_tokens}")
            #''TODELETE LATER''
        else:
            print(f"Error: {response.status_code} - {response.text}")
            recommended_columns = []  # Return an empty list in case of error

        return recommended_columns
    else:
        return []

# Function to get recommended right clustering algorithm
def clustering_algorithm_selector(loadedcsv, inputprompt, selected_columns):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Selects a clustering algorithm based on LLM recommendations.

    This function queries a Large Language Model (LLM) to recommend a 
    suitable clustering algorithm for the given dataset and selected 
    columns. It constructs a prompt using the provided template and 
    selected columns, sends the prompt to the LLM, and extracts the 
    recommended algorithm from the LLM's response.

    Args:
        loadedcsv (pandas.DataFrame): The loaded CSV data as a pandas DataFrame.
        inputprompt (str): A template string for the prompt to send to the LLM.
        selected_columns (list): A list of column names selected for clustering.

    Returns:
        str: The recommended clustering algorithm (e.g., "DBSCAN", "Hierarchical").
            Defaults to "DBSCAN" in case of error or no recommendation.
    '''
    if API_KEY:
        # Prepare the prompt
        prompt = inputprompt.format(columns=selected_columns)

        # Add instruction for clustering algorithm recommendation (restricted to DBSCAN or Hierarchical)
        prompt += "\nRecommend either DBSCAN or Hierarchical clustering for these columns. Only respond with 'DBSCAN' or 'Hierarchical'."

        # Prepare the payload for AIPROXY
        payload = {
            "model": "gpt-4o-mini",  # Or another suitable model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        response = requests.post(API_URL, headers=headers, json=payload)

        # Extract recommended algorithm
        if response.status_code == 200:
            response_data = json.loads(response.text)
            recommended_algorithm = response_data["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            recommended_algorithm = "DBSCAN"  # Default to DBSCAN in case of error

        return recommended_algorithm
    else:
        return []

# Function to send the data info to the LLM and request LLM to narrate story with provided analysis and findings
def explain_headers(data_info):
    global img_cnt, max_img, API_KEY, API_URL
    '''
    Queries an LLM to explain the meaning of data headers (column names).

    This function takes data information (likely a dictionary or DataFrame)
    and extracts the header names (column names). It constructs a prompt to 
    query a Large Language Model (LLM) about the meaning or interpretation 
    of these headers. It returns the LLM's response, which ideally provides 
    explanations or insights about the data fields.

    Args:
        data_info (dict or pandas.DataFrame): Data information, which could 
                                               be a dictionary containing header 
                                               names or a pandas DataFrame.

    Returns:
        str: The LLM's response explaining the meaning of the headers. 
             Returns an error message if LLM query fails or 
             API_KEY is not available.
    '''
    if not API_KEY:
        print("ERROR:- AIPROXY_TOKEN not available.  System will provide data realted insights only.")
    else:
        # API request with authentication header
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        } 
        prompt = (
            "You are a creative storyteller with a knack for turning data analysis into compelling narratives. "
            "Based on the dataset and statistical analysis provided, craft a comprehensive and insightful story.\n\n"
            f"Summary of Missing Values (Truncated): {summarize_text(data_info['summary_missing_values'], max_length=200)}\n\n"
            f"Outlier Analysis (Key Points): {extract_key_points(data_info['outliers'])}\n\n"
            f"{data_info['images_base64']}\n" 

            "Your task is to weave together the following insights into a clear and engaging narrative, making "
            "complex data accessible to a wide audience. Consider the following key points:\n\n"
            "- **Missing Values**: Explain the nature of the missing data. Are they random or indicative of underlying patterns? "
            "How might these missing values impact the analysis and what strategies could be used to handle them?\n\n"
            "- **Outliers**: Discuss the identified outliers. What might these extreme values represent? "
            "How might they influence the overall analysis, and should they be addressed or considered as valid data points?\n\n"
            "- **Images**: Analyze the images provided and describe the patterns and insights revealed by the charts. "
            "How do these visuals support or challenge the findings from the statistical analysis?\n\n"

            "Remember to maintain a storytelling approach, drawing the reader into the narrative and making the data analysis insightful and engaging."
        )
        print(prompt)
        # Formulate the prompt to query the LLM Model
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        retry = 10  # Retries before giving up
        backoff = 2  # Backoff factor
        max_wait = 60  # Wait time in seconds to prevent indefinite retries

        for tryattempt in range(retry):
            try:
                response = httpx.post(API_URL, headers=headers, json=payload, timeout=30)
                # Raise error if API call fails
                response.raise_for_status()

                # Print the LLM's response content
                print("LLM Response:", response.json()["choices"][0]["message"]["content"].strip())  

                return response.json()["choices"][0]["message"]["content"].strip()
                print(response.status_code)
                # If response is successful, process the response
                if response.status_code == 200:
                    result = response.json()

                    # Extract the usage data (tokens)
                    usage = result['usage']
                    prompt_tokens = usage['prompt_tokens']
                    completion_tokens = usage['completion_tokens']
                    total_tokens = usage['total_tokens']

                    # Print the token usage
                    print(f"Prompt Tokens: {prompt_tokens}")
                    print(f"Completion Tokens: {completion_tokens}")
                    print(f"Total Tokens: {total_tokens}")

                    # Also, print the model's response (optional)
                    print(f"Response: {result['choices'][0]['message']['content']}")
                    break  # Exit the retry loop if successful
                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    time.sleep(backoff)  # Exponential backoff on failure
                    attempt += 1
            except httpx.HTTPStatusError as exp:
                if exp.response.status_code == 429:
                    # Backoff with a cap on wait
                    wait = min(backoff ** tryattempt, max_wait)
                    print(f"Rate limit hit, retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(f"LLM query error: {exp}")
                    break
            except httpx.RequestError as exp:
                print(f"LLM query error: {exp}")
                break

        print("Retry exhausted, please try later.")
        sys.exit(1)  #Terminate as retries failed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py media.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).is_file():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    analyze_and_generate_output(csv_file)