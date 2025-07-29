# import streamlit as st
# import pandas as pd
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain_groq import ChatGroq

# def load_agent(csv_file_path, groq_api_key):
#     """Initialize the CSV agent with Groq LLM"""
#     llm = ChatGroq(
#         temperature=0, 
#         model="llama3-70b-8192", 
#         api_key=groq_api_key
#     )
#     agent = create_csv_agent(
#         llm, 
#         csv_file_path, 
#         verbose=True, 
#         allow_dangerous_code=True
#     )
#     return agent

# def query_data(agent, query):
#     response = agent.invoke(query)
#     return response

# def main():
#     st.title("Dataset Query App")
    
#     st.sidebar.header("Configuration")
#     groq_api_key = st.sidebar.text_input(
#         "Enter Groq API Key", 
#         type="password", 
#         help="You need a Groq API key to use this app"
#     )
    
#     uploaded_file = st.sidebar.file_uploader(
#         "Choose a CSV file", 
#         type="csv"
#     )
    
#     query = st.text_input("Enter your query about the dataset")
    
#     if st.button("Query Data"):
#         if not groq_api_key:
#             st.error("Please enter a Groq API key")
#             return
        
#         if uploaded_file is None:
#             st.error("Please upload a CSV file")
#             return
        
#         try:
#             with open("temp_uploaded_file.csv", "wb") as f:
#                 f.write(uploaded_file.getvalue())
            
#             agent = load_agent("temp_uploaded_file.csv", groq_api_key)
            
#             with st.spinner('Processing your query...'):
#                 response = query_data(agent, query)
            
#             st.success("Query Result:")
#             st.write(response)
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq

def load_agent(csv_file_path, groq_api_key):
    """Initialize the CSV agent with Groq LLM with modifications to process the full dataset"""
    df = pd.read_csv(csv_file_path)
    
    llm = ChatGroq(
        temperature=0, 
        model="llama3-70b-8192", 
        api_key=groq_api_key
    )
    
    agent = create_csv_agent(
        llm, 
        csv_file_path, 
        verbose=True, 
        allow_dangerous_code=True,
        max_iterations=1000,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    
    return agent, df

def query_data(agent, query, df):
    enhanced_query = f"""
    IMPORTANT: This CSV dataset contains EXACTLY {len(df)} rows and {len(df.columns)} columns.
    The columns are: {', '.join(df.columns.tolist())}
    
    You MUST use the actual data count when answering questions about dataset size.
    There are EXACTLY {len(df)} rows in this dataset, not a sample.
    
    Please answer this question based on the FULL dataset:
    {query}
    
    If asked about the number of rows, always report {len(df)} rows.
    """
    
    response = agent.invoke(enhanced_query)
    return response

def main():
    st.title("Dataset Query App")
    
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
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        with st.expander("Preview Dataset"):
            st.write(f"Total rows: {len(df_preview)}")
            st.write(f"Columns: {', '.join(df_preview.columns.tolist())}")
            st.dataframe(df_preview.head(10))
    
    query = st.text_input("Enter your query about the dataset")
    
    if st.button("Query Data"):
        if not groq_api_key:
            st.error("Please enter a Groq API key")
            return
        
        if uploaded_file is None:
            st.error("Please upload a CSV file")
            return
        
        try:
            temp_file_path = "temp_uploaded_file.csv"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            agent, df = load_agent(temp_file_path, groq_api_key)
            
            row_count_keywords = ["how many rows", "number of rows", "count rows", "total rows", "rows are there", "rows in the dataset"]
            if any(keyword in query.lower() for keyword in row_count_keywords):
                st.success("Query Result:")
                st.write(f"There are exactly {len(df)} rows in this dataset.")
            else:
                with st.spinner('Processing your query...'):
                    response = query_data(agent, query, df)
                
                st.success("Query Result:")
                st.write(response)
            
            with st.expander("Advanced: Direct Pandas Processing"):
                st.write("If the LLM-based agent cannot process your full dataset adequately, you can try specific pandas operations:")
                
                pandas_ops = st.selectbox(
                    "Choose a pandas operation:",
                    ["Count values in a column", "Get statistical summary", "Find top N values", "Custom pandas code", "Dataset Information"]
                )
                
                if pandas_ops == "Count values in a column":
                    col = st.selectbox("Select column:", df.columns)
                    if st.button("Run count"):
                        st.write(df[col].value_counts())
                
                elif pandas_ops == "Get statistical summary":
                    col = st.selectbox("Select column:", df.columns)
                    if st.button("Run stats"):
                        st.write(df[col].describe())
                
                elif pandas_ops == "Find top N values":
                    col = st.selectbox("Select column:", df.columns)
                    n = st.number_input("Number of values:", min_value=1, max_value=100, value=10)
                    if st.button("Find top values"):
                        st.write(df[col].value_counts().head(n))
                
                elif pandas_ops == "Dataset Information":
                    if st.button("Show dataset info"):
                        buffer = []
                        buffer.append(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                        buffer.append(f"Column Data Types:\n{df.dtypes}")
                        buffer.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        buffer.append(f"Missing Values by Column:\n{df.isnull().sum()}")
                        st.text("\n\n".join(buffer))
                
                elif pandas_ops == "Custom pandas code":
                    custom_code = st.text_area("Enter pandas code (df is your dataframe):", "df.head()")
                    if st.button("Execute"):
                        try:
                            result = eval(custom_code)
                            st.write(result)
                        except Exception as e:
                            st.error(f"Code execution error: {str(e)}")
            
            try:
                os.remove(temp_file_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("If you're experiencing issues with the agent, try using the 'Advanced: Direct Pandas Processing' section instead.")

if __name__ == "__main__":
    main()