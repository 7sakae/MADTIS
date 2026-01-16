import streamlit as st
import pandas as pd
import plotly.express as px

# App title and description
st.set_page_config(page_title="CSV Data Analyzer", page_icon="📊", layout="wide")

st.title("📊 CSV Data Analyzer")
st.markdown("Upload a CSV file to explore and visualize your data")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Display basic info
        st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Statistics", "📊 Visualizations", "🔍 Filter Data"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download processed CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            with col2:
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicate Rows", df.duplicated().sum())
        
        with tab3:
            st.subheader("Data Visualizations")
            
            # Get numeric columns for plotting
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select column for histogram", numeric_cols)
                    if selected_col:
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("Select X axis", numeric_cols, key="x")
                        y_col = st.selectbox("Select Y axis", numeric_cols, index=1, key="y")
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found for visualization")
        
        with tab4:
            st.subheader("Filter Your Data")
            
            # Column filter
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns)
            
            if selected_columns:
                filtered_df = df[selected_columns]
                
                # Row filter by index
                max_rows = len(df)
                row_range = st.slider("Select row range", 0, max_rows, (0, min(100, max_rows)))
                
                filtered_df = filtered_df.iloc[row_range[0]:row_range[1]]
                
                st.dataframe(filtered_df, use_container_width=True)
                st.info(f"Showing {len(filtered_df)} rows")
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please make sure you uploaded a valid CSV file")
else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload a CSV file to get started")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. Click the **Browse files** button above
        2. Select a CSV file from your computer
        3. Explore your data using the different tabs:
           - **Data Preview**: View your raw data
           - **Statistics**: See summary statistics
           - **Visualizations**: Create charts and graphs
           - **Filter Data**: Filter and customize your view
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈")
