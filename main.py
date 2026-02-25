import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="SamajhAI", layout="wide")

st.title("📊 SamajhAI")
st.markdown("### ✨ SamajhAI turns confusion into clarity")

# ---------- SIDEBAR ----------
st.sidebar.title("SamajhAI Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sheet_url = st.sidebar.text_input("Or paste Google Sheet link")

df = None

# ---------- FILE INPUT ----------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

# ---------- GOOGLE SHEET INPUT ----------
elif sheet_url:
    csv_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_url)

# ---------- IF NO DATA ----------
if df is None:
    st.info("⬅️ Upload data from sidebar to begin")
    st.stop()

# ---------- NAVIGATION ----------
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "AI Insights", "Visualizations"]
)

# ---------- BEAUTY MODE ----------
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #0E1117;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #262730;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATA PREVIEW ----------
st.subheader("📄 Data Preview")
st.dataframe(df)

# ---------- OPENROUTER SETUP ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

# ---------- OVERVIEW ----------
if section == "Overview":
    st.subheader("📌 Dataset Profile")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Numeric Columns", len(df.select_dtypes(include=['int64','float64']).columns))

# ---------- AI INSIGHTS ----------
elif section == "AI Insights":

    colA, colB = st.columns(2)

    if colA.button("🧠 Generate AI Summary"):
        with st.spinner("Analyzing dataset..."):
            data_sample = df.head(20).to_string()

            prompt = f"""
            This is a dataset:
            {data_sample}

            Explain:
            - What this data represents
            - Major patterns
            """

            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "SamajhAI"
                }
            )

            st.success("AI Summary Generated")
            st.markdown(f"### 📄 AI Summary\n{response.choices[0].message.content}")

    if colB.button("🔍 Find Insights"):
        with st.spinner("Finding insights..."):
            data_sample = df.describe().to_string()

            prompt = f"""
            From this statistical summary:
            {data_sample}

            Provide 5 actionable insights.
            """

            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "SamajhAI"
                }
            )

            st.success("Insights Ready")
            st.markdown(f"### 🔎 Key Insights\n{response.choices[0].message.content}")

# ---------- VISUALIZATIONS ----------
elif section == "Visualizations":

    st.subheader("📊 Smart Visualizations")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"]
    )

    if chart_type == "Histogram":
        col = st.selectbox("Select Numeric Column", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        col = st.selectbox("Select Numeric Column", numeric_cols)
        fig = px.box(df, y=col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols)
        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        cat = st.selectbox("Category", categorical_cols)
        num = st.selectbox("Value", numeric_cols)
        fig = px.bar(df, x=cat, y=num)
        st.plotly_chart(fig, use_container_width=True)