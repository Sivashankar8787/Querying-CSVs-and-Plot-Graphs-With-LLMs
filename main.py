import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_csv_agent
import io
import re
import json

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Data Insight Explorer")

# Function to load and preprocess data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    # Remove rows with NULL/NONE values in numeric columns
    df = df.dropna(subset=[col for col in df.select_dtypes(include=['number']).columns])
    return df

# Function to display data summary
def display_data_summary(df):
    st.subheader("First 20 Rows")
    st.write(df.head(20))
    st.subheader("Data Summary")
    st.write(df.describe())

# Function for statistical analysis
def statistical_analysis(df, agent):
    st.subheader("Numeric Column Statistics")
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats = df[numeric_cols].agg(['mean', 'median', 'std'])
    st.write(stats)
    
    st.subheader("Mode (Most Frequent Value)")
    mode = df.mode().iloc[0]
    st.write(mode)
    
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    
    # Convert correlation matrix to string for the agent
    corr_str = corr.to_string()
    
    # Get agent insights on the correlation matrix
    # heatmap_insight = agent.run(f"Analyze the following correlation matrix:\n\n{corr_str}\n\nWhat are the key insights and patterns you can identify from the correlations between variables?")
    # st.write("*Heatmap Analysis:*", heatmap_insight)
    
    plt.close()
    
    st.subheader("Categorical Column Analysis")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        st.write(f"Unique values in {col}:", df[col].nunique())
        st.write(df[col].value_counts())

def clean_json_string(json_string):
    json_match = re.search(r'\{[\s\S]*\}', json_string)
    if json_match:
        json_str = json_match.group(0)
        try:
            json_data = json.loads(json_str)
            return json.dumps(json_data)
        except json.JSONDecodeError:
            return None
    return None

def generate_plots(agent, df, num_plots=5):
    prompt = f"""Analyze the given dataset and suggest {num_plots} most informative and relevant plots. For each plot, provide:
    1. A title for the plot
    2. Python code to generate the plot using matplotlib and seaborn
    3. A brief explanation of what the plot shows and why it's informative

    Return your response as a JSON string with the following structure:
    {{
        "plots": [
            {{
                "title": "Plot title",
                "code": "Python code to generate the plot",
                "explanation": "Brief explanation of the plot"
            }},
            ...
        ]
    }}
    IMPORTANT: Your response should only contain the JSON string, nothing else."""

    response = agent.run(prompt)
    
    cleaned_json = clean_json_string(response)
    
    if cleaned_json:
        try:
            plot_data = json.loads(cleaned_json)
            return plot_data
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse the cleaned JSON. Error: {str(e)}")
    else:
        st.error("Failed to extract valid JSON from the agent's response.")
    
    return None

def display_insights(insights, df):
    for i, plot in enumerate(insights['plots'], 1):
        st.subheader(f"Plot {i}: {plot['title']}")
        fig_col1, fig_col2 = st.columns([3, 1])
        with fig_col1:
            st.code(plot['code'], language='python')
            try:
                exec(plot['code'], globals(), {'df': df, 'plt': plt, 'sns': sns})
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"An error occurred while generating plot {i}: {str(e)}")
        with fig_col2:
            st.write("*Explanation:*", plot['explanation'])

def create_custom_plot(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    x_col = st.selectbox("Select X-axis column", numeric_cols)
    y_col = st.selectbox("Select Y-axis column", numeric_cols)
    
    plot_types = ["Scatter", "Line", "Bar", "Box", "Violin", "Histogram"]
    plot_type = st.selectbox("Select plot type", plot_types)
    
    if st.button("Create Custom Plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        elif plot_type == "Line":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif plot_type == "Bar":
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        elif plot_type == "Box":
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        elif plot_type == "Violin":
            sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
        elif plot_type == "Histogram":
            sns.histplot(data=df, x=x_col, ax=ax)
            
        plt.title(f"{plot_type} Plot: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        # Store the plot information in session state
        st.session_state.custom_plot = {
            'type': plot_type,
            'x': x_col,
            'y': y_col,
            'fig': fig
        }
        
        plt.close()

def main():
    st.title("Querying CSVs and Plot Graphs with LLMs")
    
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.8em;'>
        This app was developed as part of an assignment using LangChain and the Gemini 1.5 Flash model. 
        Right now, it's running on the free version of the Gemini API. 
        Upgrading to a more advanced model can improve the quality of responses from the language model and provide even better insights.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create a container for the main content
    main_container = st.container()

    # Create two columns: one for the sidebar, one for the main content
    col1, col2 = st.columns([1, 4])

    with col1:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        with main_container:
            df = load_and_preprocess_data(uploaded_file)
            
            # Initialize the ChatGoogleGenerativeAI with the appropriate model
            model_name = "models/gemini-1.5-pro"
            chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            
            # Create the CSV agent
            csv_file = io.StringIO(df.to_csv(index=False))
            agent = create_csv_agent(chat_model, csv_file, verbose=True, allow_dangerous_code=True)

            with col2:
                # Data Summary and Statistical Analysis
                data_summary_container = st.expander("Data Summary and Statistical Analysis", expanded=True)
                with data_summary_container:
                    display_data_summary(df)
                    statistical_analysis(df, agent)

                # Create tabs for different sections
                tab1, tab2, tab3 = st.tabs(["Auto-generate Insights by the LLM", "Custom Visualization", "Data Q&A"])

                with tab1:
                    st.header("🖼 Auto-generated Insights")
                    num_plots = st.slider("Number of plots to generate", min_value=1, max_value=10, value=5)
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = generate_plots(agent, df, num_plots)
                            st.session_state.insights = insights

                    # Display insights if they exist in session state
                    if 'insights' in st.session_state:
                        display_insights(st.session_state.insights, df)

                with tab2:
                    st.header("Custom Visualization")
                    create_custom_plot(df)
                    
                    # Display the custom plot if it exists in session state
                    if 'custom_plot' in st.session_state:
                        fig_col1, fig_col2 = st.columns([3, 1])
                        with fig_col1:
                            st.pyplot(st.session_state.custom_plot['fig'])
                        with fig_col2:
                            # Add explanation for custom plot
                            custom_plot_explanation = agent.run(f"Analyze the {st.session_state.custom_plot['type']} plot of {st.session_state.custom_plot['y']} vs {st.session_state.custom_plot['x']}. What insights can we gain from this plot?")
                            st.write("*Plot Analysis:*", custom_plot_explanation)

                with tab3:
                    st.header("❔ Data Q&A")
                    # Initialize session state for storing query history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    
                    user_query = st.text_input("Ask a question about the data")
                    if st.button("Get Answer"):
                        answer = agent.run(user_query)
                        st.session_state.query_history.append((user_query, answer))
                    
                    # Display query history
                    for i, (query, answer) in enumerate(st.session_state.query_history, 1):
                        st.subheader(f"Question {i}")
                        st.write(f"Q: {query}")
                        st.write(f"A: {answer}")

    # Apply custom CSS to make tab titles bold and larger
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 24px;
            font-weight: bold;
        }
        .element-container img {
            max-width: 100%;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()