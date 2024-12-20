import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_stationarity(series):
    result = adfuller(series)
    return pd.Series({
        'ADF Statistic': result[0],
        'p-value': result[1],
        '1% critical value': result[4]['1%'],
        '5% critical value': result[4]['5%'],
        '10% critical value': result[4]['10%']
    })

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def main():
    st.title("Economic Data Analysis Dashboard")
    
    # Sidebar with file upload
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file (G2.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_excel(uploaded_file)
            
            # Main area dropdown
            st.write("Select an analysis to view:")
            analysis_option = st.selectbox(
                "",  # Empty label to remove duplicate text
                ["Raw Data", "Stationarity Tests", "Regression Analysis", "Model Comparison", "Visualizations"]
            )
            
            # Horizontal line for better separation
            st.markdown("---")
            
            if analysis_option == "Raw Data":
                st.header("Raw Data")
                st.dataframe(data)
                
                st.header("Data Statistics")
                st.write(data.describe())
            
            elif analysis_option == "Stationarity Tests":
                st.header("Stationarity Analysis")
                for column in ['M', 'Gr', 'Inf']:
                    st.subheader(f"Stationarity Test for {column}")
                    results = check_stationarity(data[column])
                    st.write(results)
            
            elif analysis_option == "Regression Analysis":
                st.header("Regression Analysis")
                
                # Standardize variables
                X = data[['Inf', 'Gr']]
                y = data['M']
                
                X_standardized = (X - X.mean()) / X.std()
                y_standardized = (y - y.mean()) / y.std()
                
                # Add constant
                X_with_const = sm.add_constant(X_standardized)
                
                # Fit model
                model = sm.OLS(y_standardized, X_with_const).fit()
                
                # Display results
                st.write("Regression Summary:")
                st.text(model.summary().as_text())
                
                # VIF Analysis
                st.subheader("Multicollinearity Check (VIF)")
                vif_results = calculate_vif(X)
                st.write(vif_results)
            
            elif analysis_option == "Model Comparison":
                st.header("Model Comparison")
                
                # Full model
                X = data[['Inf', 'Gr']]
                y = data['M']
                model_full = LinearRegression()
                model_full.fit(X, y)
                r2_full = model_full.score(X, y)
                
                # Model without inflation
                X_no_inf = data[['Gr']]
                model_no_inf = LinearRegression()
                model_no_inf.fit(X_no_inf, y)
                r2_no_inf = model_no_inf.score(X_no_inf, y)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Full Model R²", f"{r2_full:.3f}")
                with col2:
                    st.metric("Model without Inflation R²", f"{r2_no_inf:.3f}")
                
                # Add explanation
                st.markdown("""
                **Interpretation:**
                - The Full Model includes both Inflation and Growth as predictors
                - The Reduced Model includes only Growth as a predictor
                - The difference in R² shows the additional explanatory power of including Inflation
                """)
            
            elif analysis_option == "Visualizations":
                st.header("Data Visualizations")
                
                # Add visualization options
                viz_type = st.radio(
                    "Choose visualization type:",
                    ["Time Series", "Scatter Plots", "Both"]
                )
                
                if viz_type in ["Time Series", "Both"]:
                    st.subheader("Time Series Plot")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(data.index, data['M'], label='Money Supply')
                    ax1.set_title('Money Supply Over Time')
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel('Money Supply')
                    plt.xticks(rotation=45)
                    st.pyplot(fig1)
                
                if viz_type in ["Scatter Plots", "Both"]:
                    st.subheader("Relationship Plots")
                    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    ax2.scatter(data['Inf'], data['M'])
                    ax2.set_title('Money Supply vs Inflation')
                    ax2.set_xlabel('Inflation')
                    ax2.set_ylabel('Money Supply')
                    
                    ax3.scatter(data['Gr'], data['M'])
                    ax3.set_title('Money Supply vs Growth')
                    ax3.set_xlabel('Growth')
                    ax3.set_ylabel('Money Supply')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
            
            # Add a status indicator in the sidebar
            st.sidebar.success("Analysis ready! Select options above to view results.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure your Excel file has columns named 'M', 'Gr', and 'Inf'")
    else:
        st.info("Please upload your data file using the sidebar to begin the analysis.")

if __name__ == "__main__":
    main()