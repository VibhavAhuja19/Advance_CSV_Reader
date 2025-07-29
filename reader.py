import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
import os
import io

# Set page configuration
st.set_page_config(page_title="Advanced CSV Data Explorer", layout="wide")

def load_agent(csv_file_path, groq_api_key):
    """Initialize the CSV agent with Groq LLM"""
    llm = ChatGroq(
        temperature=0, 
        model="llama3-70b-8192", 
        api_key=groq_api_key
    )
    agent = create_csv_agent(
        llm, 
        csv_file_path, 
        verbose=True, 
        allow_dangerous_code=True
    )
    return agent

def query_data(agent, query):
    response = agent.invoke(query)
    return response

def main():
    st.title("Advanced CSV Data Explorer")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key", 
        type="password", 
        help="You need a Groq API key to use this app"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type="csv"
    )
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✓ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Data Overview", "LLM Agent Query", "Advanced Pandas Processing"])
    
    if df is not None:

        # Tab 1: Data Overview
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            with col2:
                st.subheader("Statistical Summary")
                st.write(df.describe())
            
            st.subheader("First 10 rows")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing_data,
                'Percentage (%)': missing_percentage
            })
            st.dataframe(missing_df, use_container_width=True)
        
        # Tab 2: LLM Agent Query
        with tab2:
            st.header("Query Your Data with Natural Language")
            
            query = st.text_input("Enter your query about the dataset")
            
            if st.button("Query Data"):
                if not groq_api_key:
                    st.error("Please enter a Groq API key")
                else:
                    try:
                        # Save the uploaded file temporarily
                        temp_file_path = "temp_uploaded_file.csv"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        agent = load_agent(temp_file_path, groq_api_key)
                        
                        with st.spinner('Processing your query...'):
                            # Add context to the query
                            enhanced_query = f"""
                            This CSV dataset contains {len(df)} rows and {len(df.columns)} columns.
                            The columns are: {', '.join(df.columns.tolist())}
                            
                            Consider the ENTIRE dataset (all {len(df)} rows) to answer this question:
                            {query}
                            """
                            response = query_data(agent, enhanced_query)
                        
                        st.success("Query Result:")
                        st.write(response)
                        
                        # Clean up
                        try:
                            os.remove(temp_file_path)
                        except:
                            pass
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
        # Tab 3: Advanced Pandas Processing
        with tab3:
            st.header("Advanced Data Processing with Pandas")
            
            operation_type = st.selectbox(
                "Select operation type:",
                ["Column Analysis", "Filtering & Sorting", "Group By Analysis", "Custom Pandas Code", "Data Visualization"]
            )
            
            if operation_type == "Column Analysis":
                st.subheader("Column Analysis")
                
                col_select = st.selectbox("Select column:", df.columns)
                analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Value Counts", "Basic Statistics", "Unique Values", "Missing Values"]
                )
                
                if analysis_type == "Value Counts":
                    top_n = st.slider("Show top N values:", 5, 50, 10)
                    normalize = st.checkbox("Show percentages")
                    
                    if st.button("Run Analysis"):
                        result = df[col_select].value_counts(normalize=normalize).head(top_n)
                        st.write(result)
                        
                        # Add visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        result.plot(kind='bar', ax=ax)
                        plt.title(f"Top {top_n} values for {col_select}")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                elif analysis_type == "Basic Statistics":
                    if st.button("Show Statistics"):
                        if pd.api.types.is_numeric_dtype(df[col_select]):
                            st.write(df[col_select].describe())
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(df[col_select].dropna(), kde=True, ax=ax)
                            plt.title(f"Distribution of {col_select}")
                            st.pyplot(fig)
                        else:
                            st.write(f"Column '{col_select}' is not numeric. Basic statistics available:")
                            st.write({
                                "Count": len(df[col_select]),
                                "Unique Values": df[col_select].nunique(),
                                "Top Value": df[col_select].mode()[0] if not df[col_select].mode().empty else "N/A",
                                "Missing Values": df[col_select].isnull().sum()
                            })
                
                elif analysis_type == "Unique Values":
                    if st.button("Show Unique Values"):
                        unique_vals = df[col_select].unique()
                        st.write(f"Total unique values: {len(unique_vals)}")
                        st.write("Unique values:", unique_vals)
                
                elif analysis_type == "Missing Values":
                    if st.button("Analyze Missing Values"):
                        missing_count = df[col_select].isnull().sum()
                        missing_percent = (missing_count / len(df)) * 100
                        
                        st.write(f"Missing values: {missing_count} ({missing_percent:.2f}%)")
                        
                        if missing_count > 0:
                            st.subheader("Rows with missing values in this column:")
                            st.dataframe(df[df[col_select].isnull()].head(10))
            
            elif operation_type == "Filtering & Sorting":
                st.subheader("Filter and Sort Data")
                
                # Filtering
                st.write("Filter data by condition:")
                filter_col = st.selectbox("Select column for filtering:", df.columns)
                
                # Dynamically determine filter options based on column type
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    filter_type = st.selectbox(
                        "Filter type:",
                        ["Greater than", "Less than", "Equal to", "Between"]
                    )
                    
                    if filter_type == "Between":
                        min_val, max_val = st.slider(
                            f"Select range for {filter_col}:",
                            float(df[filter_col].min()),
                            float(df[filter_col].max()),
                            (float(df[filter_col].min()), float(df[filter_col].max()))
                        )
                    else:
                        filter_value = st.number_input(
                            f"Enter value for {filter_col}:",
                            value=float(df[filter_col].median())
                        )
                else:
                    # For non-numeric columns
                    unique_values = df[filter_col].dropna().unique()
                    if len(unique_values) <= 50:  # For columns with reasonable number of unique values
                        filter_type = "Equal to"
                        filter_value = st.selectbox(
                            f"Select value for {filter_col}:",
                            ["All"] + list(unique_values)
                        )
                    else:
                        filter_type = "Contains"
                        filter_value = st.text_input(f"Enter text to search in {filter_col}:")
                
                # Sorting
                st.write("Sort data:")
                sort_col = st.selectbox("Select column for sorting:", df.columns)
                sort_order = st.radio("Sort order:", ["Ascending", "Descending"])
                
                if st.button("Apply Filters and Sorting"):
                    # Apply filtering
                    filtered_df = df.copy()
                    
                    if pd.api.types.is_numeric_dtype(df[filter_col]):
                        if filter_type == "Greater than":
                            filtered_df = filtered_df[filtered_df[filter_col] > filter_value]
                        elif filter_type == "Less than":
                            filtered_df = filtered_df[filtered_df[filter_col] < filter_value]
                        elif filter_type == "Equal to":
                            filtered_df = filtered_df[filtered_df[filter_col] == filter_value]
                        elif filter_type == "Between":
                            filtered_df = filtered_df[(filtered_df[filter_col] >= min_val) & 
                                                     (filtered_df[filter_col] <= max_val)]
                    else:
                        if filter_type == "Equal to" and filter_value != "All":
                            filtered_df = filtered_df[filtered_df[filter_col] == filter_value]
                        elif filter_type == "Contains" and filter_value:
                            filtered_df = filtered_df[filtered_df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)]
                    
                    # Apply sorting
                    filtered_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == "Ascending")
                    )
                    
                    # Display results
                    st.write(f"Results: {len(filtered_df)} rows")
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Option to download filtered data
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download filtered data as CSV",
                        data=csv,
                        file_name="filtered_data.csv",
                        mime="text/csv"
                    )
            
            elif operation_type == "Group By Analysis":
                st.subheader("Group By Analysis")
                
                # Select columns for grouping
                group_cols = st.multiselect(
                    "Select columns to group by:",
                    df.columns.tolist(),
                    max_selections=3
                )
                
                if group_cols:
                    # Select columns for aggregation
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        agg_cols = st.multiselect(
                            "Select numeric columns to aggregate:",
                            numeric_cols
                        )
                        
                        if agg_cols:
                            # Select aggregation functions
                            agg_funcs = st.multiselect(
                                "Select aggregation functions:",
                                ["count", "sum", "mean", "median", "min", "max", "std"],
                                default=["count", "mean"]
                            )
                            
                            if agg_funcs and st.button("Perform Group By"):
                                # Create aggregation dictionary
                                agg_dict = {col: agg_funcs for col in agg_cols}
                                
                                # Perform group by
                                try:
                                    result = df.groupby(group_cols).agg(agg_dict)
                                    st.dataframe(result, use_container_width=True)
                                    
                                    # Option to download
                                    csv = result.to_csv()
                                    st.download_button(
                                        label="Download grouped data as CSV",
                                        data=csv,
                                        file_name="grouped_data.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Visualization if not too complex
                                    if len(group_cols) <= 2 and len(result) <= 20:
                                        st.subheader("Visualization")
                                        # For simple one-level groupby with one agg function
                                        if len(group_cols) == 1 and len(agg_cols) == 1 and len(agg_funcs) == 1:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            result.plot(kind='bar', ax=ax)
                                            plt.title(f"{agg_funcs[0]} of {agg_cols[0]} by {group_cols[0]}")
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error in group by operation: {str(e)}")
                        else:
                            st.warning("Please select at least one column to aggregate.")
                    else:
                        st.warning("No numeric columns available for aggregation.")
                else:
                    st.warning("Please select at least one column to group by.")
            
            elif operation_type == "Custom Pandas Code":
                st.subheader("Custom Pandas Code")
                st.write("Write your own pandas code to analyze the full dataset.")
                st.write("The DataFrame is available as `df` in your code.")
                
                default_code = """# Example: Get summary statistics for numeric columns
result = df.describe()

# Return the result to display it
result"""
                
                custom_code = st.text_area("Enter pandas code:", default_code, height=200)
                
                if st.button("Execute Code"):
                    try:
                        # Create a restricted local scope with only necessary variables
                        local_dict = {"df": df, "pd": pd, "np": np}
                        
                        # Execute the code
                        exec_result = {}
                        exec(f"result = {custom_code}", {"__builtins__": {}}, local_dict)
                        result = local_dict.get("result")
                        
                        st.subheader("Result:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Code execution error: {str(e)}")
            
            elif operation_type == "Data Visualization":
                st.subheader("Data Visualization")
                
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Histogram", "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Correlation Heatmap"]
                )
                
                if viz_type == "Histogram":
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        hist_col = st.selectbox("Select column for histogram:", num_cols)
                        bins = st.slider("Number of bins:", 5, 100, 20)
                        
                        if st.button("Generate Histogram"):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(df[hist_col].dropna(), bins=bins, kde=True, ax=ax)
                            plt.title(f"Histogram of {hist_col}")
                            st.pyplot(fig)
                    else:
                        st.warning("No numeric columns available for histogram.")
                
                elif viz_type == "Scatter Plot":
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if len(num_cols) >= 2:
                        x_col = st.selectbox("Select X-axis column:", num_cols)
                        y_col = st.selectbox("Select Y-axis column:", [c for c in num_cols if c != x_col])
                        
                        color_option = st.checkbox("Add color dimension")
                        if color_option:
                            color_col = st.selectbox("Select column for color:", df.columns)
                        
                        if st.button("Generate Scatter Plot"):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if color_option:
                                scatter = sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            else:
                                scatter = sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                            plt.title(f"Scatter Plot: {y_col} vs {x_col}")
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("Need at least two numeric columns for scatter plot.")
                
                elif viz_type == "Bar Chart":
                    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                    if cat_cols:
                        x_col = st.selectbox("Select category column (X-axis):", cat_cols)
                        
                        # For Y-axis, we need numeric
                        num_cols = df.select_dtypes(include=np.number).columns.tolist()
                        if num_cols:
                            y_col = st.selectbox("Select value column (Y-axis):", num_cols)
                            agg_func = st.selectbox(
                                "Select aggregation function:",
                                ["sum", "mean", "count", "median", "min", "max"]
                            )
                            
                            top_n = st.slider("Show top N categories:", 5, 50, 10)
                            
                            if st.button("Generate Bar Chart"):
                                # Prepare data
                                if agg_func == "count":
                                    plot_data = df[x_col].value_counts().head(top_n)
                                    title = f"Count of {x_col} (Top {top_n})"
                                else:
                                    plot_data = df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False).head(top_n)
                                    title = f"{agg_func.capitalize()} of {y_col} by {x_col} (Top {top_n})"
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_data.plot(kind='bar', ax=ax)
                                plt.title(title)
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.warning("No numeric columns available for Y-axis.")
                    else:
                        st.warning("No categorical columns available for bar chart.")
                
                elif viz_type == "Line Chart":
                    # Check if there's a date column
                    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' 
                                or (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all())]
                    
                    if date_cols or df.index.dtype == 'datetime64[ns]':
                        if date_cols:
                            x_col = st.selectbox("Select date column (X-axis):", date_cols)
                        else:
                            x_col = None
                            st.info("Using DataFrame index as X-axis (time)")
                        
                        num_cols = df.select_dtypes(include=np.number).columns.tolist()
                        if num_cols:
                            y_cols = st.multiselect("Select value columns (Y-axis):", num_cols)
                            
                            if y_cols and st.button("Generate Line Chart"):
                                # Prepare data
                                if x_col:
                                    # Convert to datetime if not already
                                    if df[x_col].dtype != 'datetime64[ns]':
                                        plot_df = df.copy()
                                        plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                                    else:
                                        plot_df = df
                                    
                                    # Sort by date
                                    plot_df = plot_df.sort_values(x_col)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    for col in y_cols:
                                        ax.plot(plot_df[x_col], plot_df[col], label=col)
                                else:
                                    # Use index as x-axis
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    for col in y_cols:
                                        ax.plot(df.index, df[col], label=col)
                                
                                plt.legend()
                                plt.title("Line Chart")
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.warning("No numeric columns available for Y-axis.")
                    else:
                        st.warning("No date columns detected for line chart. Consider converting a column to datetime.")
                
                elif viz_type == "Box Plot":
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        y_col = st.selectbox("Select numeric column for box plot:", num_cols)
                        
                        use_category = st.checkbox("Group by category")
                        if use_category:
                            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                            if cat_cols:
                                x_col = st.selectbox("Select category column for grouping:", cat_cols)
                                
                                # Limit categories if too many
                                top_n = st.slider("Show top N categories:", 3, 20, 10)
                                
                                if st.button("Generate Box Plot"):
                                    # Get top categories
                                    top_cats = df[x_col].value_counts().head(top_n).index
                                    plot_df = df[df[x_col].isin(top_cats)]
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
                                    plt.title(f"Box Plot of {y_col} by {x_col}")
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.warning("No categorical columns available for grouping.")
                        else:
                            if st.button("Generate Box Plot"):
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.boxplot(y=df[y_col].dropna(), ax=ax)
                                plt.title(f"Box Plot of {y_col}")
                                st.pyplot(fig)
                    else:
                        st.warning("No numeric columns available for box plot.")
                
                elif viz_type == "Correlation Heatmap":
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if len(num_cols) >= 2:
                        selected_cols = st.multiselect(
                            "Select columns for correlation matrix (default: all numeric):",
                            num_cols,
                            default=num_cols[:min(10, len(num_cols))]
                        )
                        
                        if not selected_cols:
                            selected_cols = num_cols[:min(10, len(num_cols))]
                            st.info(f"Using first {len(selected_cols)} numeric columns")
                        
                        if st.button("Generate Correlation Heatmap"):
                            # Calculate correlation matrix
                            corr = df[selected_cols].corr()
                            
                            # Plot heatmap
                            fig, ax = plt.subplots(figsize=(10, 8))
                            mask = np.triu(np.ones_like(corr, dtype=bool))
                            cmap = sns.diverging_palette(230, 20, as_cmap=True)
                            
                            sns.heatmap(
                                corr, 
                                mask=mask, 
                                cmap=cmap, 
                                vmax=1, 
                                vmin=-1, 
                                center=0,
                                square=True, 
                                linewidths=.5, 
                                cbar_kws={"shrink": .5},
                                annot=True,
                                fmt=".2f"
                            )
                            
                            plt.title("Correlation Matrix")
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("Need at least two numeric columns for correlation heatmap.")
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()