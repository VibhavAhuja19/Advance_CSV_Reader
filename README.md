# Advanced CSV Data Explorer

A powerful Streamlit application for exploring, analyzing, and visualizing CSV data with natural language querying capabilities using Groq's LLM.


## Features

- **Natural Language Querying**: Ask questions about your data in plain English using Groq's LLM
- **Comprehensive Data Analysis**:
  - Dataset overview with statistics and missing value analysis
  - Column-specific analysis (value counts, distributions, unique values)
  - Advanced filtering and sorting capabilities
  - Group by and aggregation operations
- **Data Visualization**:
  - Histograms, scatter plots, bar charts
  - Line charts, box plots, correlation heatmaps
- **Custom Pandas Code Execution**: Write and execute your own pandas code
- **Encoding Detection**: Automatic detection of file encodings
- **Responsive Design**: Works on both desktop and mobile devices



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-csv-explorer.git
   cd advanced-csv-explorer
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

## Run Commands for different .py files 
   ```bash
   streamlit run app.py
   streamlit run main.py
   streamlit run reader.py
   ```