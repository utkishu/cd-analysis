import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
import json
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
import hashlib

# Initialize the OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

client = openai.OpenAI(api_key=openai_api_key)

def chat_with_csv(df, prompt):
    """Function to interact with the CSV data using LLM."""
    from pandasai import SmartDataframe
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    try:
        result = pandas_ai.chat(prompt)
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            # Convert DataFrame or Series to a string representation
            return result.to_string()
        elif isinstance(result, dict) or isinstance(result, list):
            # Handling cases where the result might be in JSON-like structures
            return json.dumps(result)
        else:
            return str(result)  # Ensure all outputs are string for expand_analysis
    except Exception as e:
        return f"Error during analysis: {str(e)}"  # Provide a fallback error message


def load_data():
    """Load and display the CSV file."""
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def generate_prompt(df):
    """Generate a detailed prompt for the GPT based on the dataframe's comprehensive data overview."""
    description_stats = df.describe(include='all').to_string()
    description_types = df.dtypes.to_string()
    vis_types = ['scatter plot', 'bar chart', 'pie chart', 'heatmap', 'histogram']
    prompt = (
        f"Dataset Overview:\n{description_stats}\n\n"
        f"Data Types:\n{description_types}\n\n"
        "Given the data, recommend up to three visualizations from the following list: "
        f"{', '.join(vis_types)}. For each recommended visualization, specify which columns should be used. "
        "Specify the visualization type and the columns to be used. Format the recommendations as JSON. "
        "Example format: "
        "[{'type': 'histogram', 'columns': ['Age', 'Income']}, "
        "{'type': 'scatter plot', 'columns': ['Age', 'Spending Score']}]"
    )
    return prompt

def get_recommendations(prompt):
    """Use OpenAI's GPT to get recommendations on visualizations."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def expand_analysis(initial_analysis, column_name):
    """Expand the initial analysis into a more detailed explanation."""
    initial_analysis_str = str(initial_analysis)  # Ensuring input is string
    prompt = (
        f"Given the initial analysis: '{initial_analysis_str}', "
        f"Column Name: '{column_name}', expand this into a detailed, "
        "easy-to-understand explanation. Summarize key insights, trends, and any noteworthy outliers in about 50 words."
    )
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


def display_visualizations(df, json_recommendations):
    """Display visualizations based on the GPT's recommendations."""
    vis_funcs = {
        'histogram': generate_histogram,
        'scatter plot': generate_scatter_plot,
        'bar chart': generate_bar_chart,
        'line chart': generate_line_chart,
        'pie chart': generate_pie_chart,
        'heatmap': generate_heatmap
    }

    try:
        recommendations = json.loads(json_recommendations)
        for rec in recommendations:
            vis_type = rec['type']
            columns = rec['columns']
            if vis_type in vis_funcs and all(col in df.columns for col in columns):
                vis_funcs[vis_type](df, columns)
            else:
                st.error(f"Cannot generate {vis_type} for columns {columns}")
    except json.JSONDecodeError:
        st.error("Failed to decode JSON recommendations. Check the format.")

# Assume other functions are defined correctly
def generate_bar_chart(df, columns):
    for column in columns:
        if column in df.columns:
            fig = px.bar(df, x=column, title=f'Bar Chart of {column}')
            analysis_prompt = f"Provide an analysis of the distribution and any outliers in the '{column}' column."
            initial_analysis = chat_with_csv(df, analysis_prompt)
            expanded_analysis = expand_analysis(initial_analysis, column)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig)
            with col2:
                st.write(f"**Analysis of {column}:**")
                st.write(expanded_analysis)


def generate_line_chart(df, columns):
    if len(columns) == 2: # Assuming there are 2 columns needed
        fig = px.line(df, x=columns[0], y=columns[1], title=f'Line Chart of {columns[0]} vs {columns[1]}')
        analysis_prompt = f"Provide an analysis of the trend between '{columns[0]}' and '{columns[1]}'."
        initial_analysis = chat_with_csv(df, analysis_prompt)
        expanded_analysis = expand_analysis(initial_analysis, columns[0])
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.write(f"**Analysis of {columns[0]} vs {columns[1]}:**")
            st.write(expanded_analysis)


def generate_scatter_plot(df, columns):
    if len(columns) == 2:
        fig = px.scatter(df, x=columns[0], y=columns[1], title=f'Scatter Plot of {columns[0]} vs {columns[1]}')
        analysis_prompt = f"Provide an analysis of the relationship between '{columns[0]}' and '{columns[1]}'."
        initial_analysis = chat_with_csv(df, analysis_prompt)
        expanded_analysis = expand_analysis(initial_analysis, columns[0])
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.write(f"**Analysis of {columns[0]} vs {columns[1]}:**")
            st.write(expanded_analysis)

def generate_pie_chart(df, columns):
    for column in columns:
        if column in df.columns:
            fig = px.pie(df, names=column, title=f'Pie Chart of {column}')
            analysis_prompt = f"Provide an analysis of the distribution in the '{column}' column."
            initial_analysis = chat_with_csv(df, analysis_prompt)
            expanded_analysis = expand_analysis(initial_analysis, column)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig)
            with col2:
                st.write(f"**Analysis of {column}:**")
                st.write(expanded_analysis)


def generate_histogram(df, columns):
    for column in columns:
        if column in df.columns:
            # Generate the histogram
            fig = px.histogram(df, x=column, title=f'Histogram of {column}')
            # Create a prompt for the analysis
            analysis_prompt = f"Provide an analysis of the distribution and any outliers in the '{column}' column."
            initial_analysis = chat_with_csv(df, analysis_prompt)
            expanded_analysis = expand_analysis(initial_analysis, column)
            # Use columns layout to display chart next to its explanation
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig)
            with col2:
                st.write(f"**Analysis of {column}:**")
                st.write(expanded_analysis)

def generate_heatmap(df, columns):
    if len(columns) >= 2: # Updated to handle more general cases
        fig = px.density_heatmap(df, x=columns[0], y=columns[1], title=f'Heatmap of {columns[0]} vs {columns[1]}')
        analysis_prompt = f"Provide an analysis of the density and distribution between '{columns[0]}' and '{columns[1]}'."
        initial_analysis = chat_with_csv(df, analysis_prompt)
        expanded_analysis = expand_analysis(initial_analysis, columns[0])
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.write(f"**Analysis of {columns[0]} vs {columns[1]}:**")
            st.write(expanded_analysis)


def main():
    st.set_page_config(layout='wide')
    st.title("Consumer Data Analysis using AI")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built for All Data Analysis and Visualizations')

    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        df = pd.read_csv(uploaded_file)
        if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
            st.session_state.file_hash = file_hash
 
            st.session_state.df = df
            st.write(df.head())  # Display data snippet for reference
            prompt = generate_prompt(df)
            json_recommendations = get_recommendations(prompt)
            st.session_state.json_recommendations = json_recommendations
            display_visualizations(df, json_recommendations)
        else:
            st.write(st.session_state.df.head())  # Display cached data snippet
            if 'json_recommendations' in st.session_state:
                display_visualizations(st.session_state.df, st.session_state.json_recommendations)

        input_text = st.text_area("Enter your query here")
        if st.button("Chat with CSV"):
            with st.spinner("Analyzing..."):
                initial_analysis = chat_with_csv(df, input_text)
                if initial_analysis:
                    # Assuming column_name is determined somehow, or set to a default
                    column_name = "DefaultColumn"
                    expanded_analysis = expand_analysis(initial_analysis, column_name)
                    st.write(expanded_analysis)  # Display expanded analysis next to the visualizations
                else:
                    st.error("Failed to get initial analysis.")

if __name__ == "__main__":
    main()
