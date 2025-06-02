import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import json

# Configure page
st.set_page_config(
    page_title="Seattle Disability & Health Insurance Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

def calculate_advanced_metrics(df):
    """Calculate advanced metrics and correlations"""
    metrics_df = df.copy()
    
    # Calculate rates and ratios
    metrics_df['Disability_Rate'] = (metrics_df['Population 18 years and Over with a Disability'] / 
                                   metrics_df['Population 18 years and Over'] * 100).fillna(0)
    
    metrics_df['Uninsured_Rate'] = (metrics_df['Population without Health Insurance'] /
                                  metrics_df['Population Civilian Noninstitutionalized'] * 100).fillna(0)
    
    metrics_df['Household_Disability_Rate'] = (metrics_df['Households with 1 or more persons with a disability'] /
                                             metrics_df['Households'] * 100).fillna(0)
    
    metrics_df['Elderly_Disability_Rate'] = (metrics_df['Population 65 years and Over with a Disability'] /
                                           metrics_df['Population 65 years and Over'] * 100).fillna(0)
    
    metrics_df['Working_Age_Disability_Rate'] = (metrics_df['Population 18 to 64 years with a Disability'] /
                                               metrics_df['Population 18 to 64 years'] * 100).fillna(0)
    
    metrics_df['Poverty_Disability_Rate'] = ((metrics_df['Population 18 to 64 years Below Poverty with a Disability'] +
                                            metrics_df['Population 65 years and Over Below Poverty with a Disability']) /
                                           (metrics_df['Population 18 to 64 years Below Poverty with a Disability'] +
                                            metrics_df['Population 18 to 64 years Below Poverty without a Disability'] +
                                            metrics_df['Population 65 years and Over Below Poverty with a Disability'] +
                                            metrics_df['Population 65 years and Over Below Poverty without a Disability']) * 100).fillna(0)
    
    metrics_df['Youth_Uninsured_Rate'] = (metrics_df['Population under 19 years without Health Insurance'] /
                                        (metrics_df['Population under 19 years with Health Insurance'] +
                                         metrics_df['Population under 19 years without Health Insurance']) * 100).fillna(0)
    
    metrics_df['Working_Age_Uninsured_Rate'] = ((metrics_df['Population 19 to 34 years without Health Insurance'] +
                                               metrics_df['Population 35 to 64 years without Health Insurance']) /
                                              (metrics_df['Population 19 to 34 years with Health Insurance'] +
                                               metrics_df['Population 19 to 34 years without Health Insurance'] +
                                               metrics_df['Population 35 to 64 years with Health Insurance'] +
                                               metrics_df['Population 35 to 64 years without Health Insurance']) * 100).fillna(0)
    
    metrics_df['Population_Density_Score'] = metrics_df['Population 18 years and Over'] / metrics_df['Population 18 years and Over'].max()
    
    return metrics_df

def perform_clustering_analysis(df):
    """Perform K-means clustering on neighborhoods"""
    metrics_df = calculate_advanced_metrics(df)
    
    # Add cluster column with default value in case clustering fails
    metrics_df['Cluster'] = 0
    
    # Only proceed if we have enough data
    if len(metrics_df) <= 1:
        print("Not enough data for clustering")
        return metrics_df, None
    
    # Select features for clustering
    features = ['Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate', 
               'Elderly_Disability_Rate', 'Working_Age_Disability_Rate', 'Poverty_Disability_Rate']
    
    # Check if all features exist
    missing_features = [f for f in features if f not in metrics_df.columns]
    if missing_features:
        print(f"Missing features for clustering: {missing_features}")
        return metrics_df, None
    
    try:
        # Remove rows with any NaN values in features
        clustering_data = metrics_df[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Determine optimal number of clusters
        n_clusters = min(4, len(metrics_df))  # Max 4 clusters, or fewer if limited data
        
        if n_clusters > 1:
            # Perform K-means clustering with explicit n_init parameter
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            metrics_df['Cluster'] = cluster_labels
            print(f"Clustering complete. Found {n_clusters} clusters.")
        else:
            print("Not enough data for meaningful clustering")
            metrics_df['Cluster'] = 0
        
        return metrics_df, scaler
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        # Return default clustering with all points in one cluster
        metrics_df['Cluster'] = 0
        return metrics_df, None

def calculate_correlations(df):
    """Calculate correlation matrix for key metrics"""
    metrics_df = calculate_advanced_metrics(df)
    
    correlation_features = ['Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate',
                          'Elderly_Disability_Rate', 'Working_Age_Disability_Rate', 
                          'Poverty_Disability_Rate', 'Youth_Uninsured_Rate', 'Working_Age_Uninsured_Rate']
    correlation_matrix = metrics_df[correlation_features].corr()
    return correlation_matrix

def perform_regression_analysis(df):
    """Perform comprehensive regression analysis"""
    metrics_df = calculate_advanced_metrics(df)
    
    # Prepare data for regression
    regression_features = ['Household_Disability_Rate', 'Elderly_Disability_Rate', 
                          'Working_Age_Disability_Rate', 'Poverty_Disability_Rate',
                          'Population 18 years and Over']
    
    # Remove rows with NaN values
    clean_df = metrics_df[regression_features + ['Disability_Rate', 'Uninsured_Rate']].dropna()
    
    results = {}
    
    # Regression 1: Predict Overall Disability Rate
    X1 = clean_df[['Elderly_Disability_Rate', 'Working_Age_Disability_Rate', 
                   'Poverty_Disability_Rate']].fillna(0)
    y1 = clean_df['Disability_Rate'].fillna(0)
    
    model1 = LinearRegression()
    model1.fit(X1, y1)
    y1_pred = model1.predict(X1)
    
    results['disability_model'] = {
        'model': model1,
        'features': ['Elderly_Disability_Rate', 'Working_Age_Disability_Rate', 'Poverty_Disability_Rate'],
        'r2': r2_score(y1, y1_pred),
        'mae': mean_absolute_error(y1, y1_pred),
        'rmse': np.sqrt(mean_squared_error(y1, y1_pred)),
        'coefficients': model1.coef_,
        'intercept': model1.intercept_,
        'X': X1,
        'y_true': y1,
        'y_pred': y1_pred
    }
    
    # Regression 2: Predict Uninsured Rate
    X2 = clean_df[['Disability_Rate', 'Poverty_Disability_Rate', 
                   'Population 18 years and Over']].fillna(0)
    y2 = clean_df['Uninsured_Rate'].fillna(0)
    
    model2 = LinearRegression()
    model2.fit(X2, y2)
    y2_pred = model2.predict(X2)
    
    results['uninsured_model'] = {
        'model': model2,
        'features': ['Disability_Rate', 'Poverty_Disability_Rate', 'Population 18 years and Over'],
        'r2': r2_score(y2, y2_pred),
        'mae': mean_absolute_error(y2, y2_pred),
        'rmse': np.sqrt(mean_squared_error(y2, y2_pred)),
        'coefficients': model2.coef_,
        'intercept': model2.intercept_,
        'X': X2,
        'y_true': y2,
        'y_pred': y2_pred
    }
    
    # Simple correlation regression: Disability vs Uninsured
    X3 = clean_df[['Disability_Rate']].fillna(0)
    y3 = clean_df['Uninsured_Rate'].fillna(0)
    
    model3 = LinearRegression()
    model3.fit(X3, y3)
    y3_pred = model3.predict(X3)
    
    results['correlation_model'] = {
        'model': model3,
        'features': ['Disability_Rate'],
        'r2': r2_score(y3, y3_pred),
        'mae': mean_absolute_error(y3, y3_pred),
        'rmse': np.sqrt(mean_squared_error(y3, y3_pred)),
        'coefficients': model3.coef_,
        'intercept': model3.intercept_,
        'X': X3,
        'y_true': y3,
        'y_pred': y3_pred
    }
    
    return results, clean_df

@st.cache_data
def load_data():
    """Load and cache the CSV data"""
    try:
        df = pd.read_csv('disability_health_insurance_Neighborhoods_-524063700918977155.csv')
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure the file 'disability_health_insurance_Neighborhoods_-524063700918977155.csv' is in the same directory as this app.")
        return None

@st.cache_data
def load_geojson_data():
    """Load and cache the GeoJSON data"""
    try:
        with open('disability_health_insurance_Neighborhoods_-6398636734886665350.geojson', 'r') as f:
            geojson_data = json.load(f)
        return geojson_data
    except FileNotFoundError:
        st.error("GeoJSON file not found. Please ensure the file 'disability_health_insurance_Neighborhoods_-6398636734886665350.geojson' is in the same directory as this app.")
        return None

def create_choropleth_map(df, metric_col, title="Choropleth Map"):
    """Create a choropleth map using the neighborhood data"""
    geojson_data = load_geojson_data()
    if geojson_data is None:
        return None
    
    # Calculate the metric if it's not already in the dataframe
    if metric_col not in df.columns:
        df_with_metrics = calculate_advanced_metrics(df)
    else:
        df_with_metrics = df.copy()
      # Create a mapping from neighborhood name to metric value
    neighborhood_values = {}
    for _, row in df_with_metrics.iterrows():
        neighborhood_values[row['Neighborhood Name']] = row[metric_col] if metric_col in row else 0
    
    # Add the metric values to the GeoJSON features
    for feature in geojson_data['features']:
        neigh_name = feature['properties'].get('NEIGH_NAME', '')
        feature['properties']['metric_value'] = neighborhood_values.get(neigh_name, 0)
    
    # Since the GeoJSON doesn't have geometry, we'll create a simple scatter plot on a map
    # Extract coordinates from the data if available, or use a simple layout
    fig = go.Figure()
    
    # Create a scatter plot with color coding
    neighborhoods = []
    values = []
    for _, row in df_with_metrics.iterrows():
        if metric_col in row:
            neighborhoods.append(row['Neighborhood Name'])
            values.append(row[metric_col])
    
    # Create a simple bar chart as an alternative to the choropleth
    fig = px.bar(
        x=neighborhoods,
        y=values,
        title=f"{title} by Neighborhood",
        labels={'x': 'Neighborhood', 'y': metric_col},
        color=values,
        color_continuous_scale='RdYlBu_r'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        title_font_size=16,
        showlegend=False
    )
    
    return fig

def main():    # Title and description
    st.markdown('<h1 class="main-header">Seattle Disability & Health Insurance Geographic Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Comprehensive Analysis of Disability and Health Insurance Coverage Across Seattle's Geographic Divisions**

    [Ada Carter](https://adacarter.org), University of Washington
                
     **Geographic Coverage:**
    - **53 Community Reporting Areas (CRAs)** - Primary administrative boundaries for city services and planning
    - **Urban Centers & Urban Villages (UCUVs)** - High-density growth areas under Seattle's Comprehensive Plan  
    - **7 City Council Districts** - Electoral boundaries ensuring geographic representation
    
    **Analysis Methods:**
    - Statistical correlation and distribution analysis
    - Machine learning clustering for neighborhood classification  
    - Predictive modeling and regression analysis
    - Geographic visualization and neighborhood comparisons
    - Risk assessment matrix for health service planning
    """)
    
    # Key geographic context
    st.info("""
    **Geographic Context:** Seattle's 53 Community Reporting Areas serve as the foundation for this analysis, 
    representing distinct neighborhoods and communities. These areas align with city planning efforts and service delivery, 
    while Urban Centers/Villages represent focused growth areas, and Council Districts ensure balanced political representation.
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
      # === SEATTLE GEOGRAPHIC DIVISIONS ===
    st.markdown('<h2 class="section-header">Seattle Geographic Organization</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Community Reporting Areas (CRAs)**
        - **53 Total CRAs** across Seattle
        - Primary administrative boundaries
        - Used for city planning and service delivery
        - Each CRA represents distinct communities
        - Population ranges from 1,000 to 15,000+ residents
        """)
    
    with col2:
        st.markdown("""
        **Urban Centers & Urban Villages (UCUVs)**
        - **Growth Strategy Areas** under Seattle's Comprehensive Plan
        - High-density residential and commercial zones
        - Focus areas for transit-oriented development
        - Include Downtown, Capitol Hill, Ballard, etc.
        - Support 70% of city's population growth
        """)
    
    with col3:
        st.markdown("""
        **City Council Districts**
        - **7 Council Districts** for representation
        - Each district ~100,000 residents
        - Established in 2013 for geographic representation
        - Districts 1-7 cover different Seattle regions
        - Balanced for population and geographic diversity
        """)
    
    # Advanced filtering section
    st.markdown('<h3 style="color: #2c3e50;">Data Filtering Options</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Neighborhood type filter
        neighborhood_types = ['All'] + sorted(df['Neighborhood Type'].unique())
        selected_type = st.selectbox("Filter by Geographic Type", neighborhood_types, 
                                   help="Filter analysis by Community Reporting Area, Urban Center/Village, or Council District")
    
    with col2:
        # Filter data based on selection
        if selected_type != 'All':
            filtered_df = df[df['Neighborhood Type'] == selected_type]
        else:
            filtered_df = df
        
        # Population size filter
        min_pop = int(filtered_df['Population 18 years and Over'].min())
        max_pop = int(filtered_df['Population 18 years and Over'].max())
        pop_range = st.slider("Population Range (18+)", min_pop, max_pop, (min_pop, max_pop),
                             help="Adjust population range to focus on specific community sizes")
        
        filtered_df = filtered_df[
            (filtered_df['Population 18 years and Over'] >= pop_range[0]) & 
            (filtered_df['Population 18 years and Over'] <= pop_range[1])
        ]
    
    with col3:
        st.metric("Areas in Analysis", len(filtered_df))
        st.metric("Total Population (18+)", f"{filtered_df['Population 18 years and Over'].sum():,}")
        geographic_types = filtered_df['Neighborhood Type'].nunique()
        st.metric("Geographic Types", geographic_types)
      # Calculate advanced metrics
    metrics_df = calculate_advanced_metrics(filtered_df)
    
    # Calculate correlations
    correlation_matrix = calculate_correlations(filtered_df)
    
    regression_results, regression_df = perform_regression_analysis(filtered_df)
      # === EXECUTIVE SUMMARY DASHBOARD ===
    st.markdown('<h2 class="section-header">Descriptive Statistics</h2>', unsafe_allow_html=True)
    
    # Geographic distribution summary
    geographic_summary = filtered_df['Neighborhood Type'].value_counts()
    
    st.markdown("**Geographic Distribution in Current Analysis:**")
    col1, col2, col3 = st.columns(3)
    
    for i, (geo_type, count) in enumerate(geographic_summary.items()):
        with [col1, col2, col3][i % 3]:
            percentage = (count / len(filtered_df)) * 100
            st.metric(f"{geo_type}", f"{count} areas ({percentage:.1f}%)")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_pop = metrics_df['Population 18 years and Over'].sum()
        st.metric("Total Population (18+)", f"{total_pop:,}")
    
    with col2:
        avg_disability_rate = metrics_df['Disability_Rate'].mean()
        st.metric("Avg Disability Rate", f"{avg_disability_rate:.1f}%")
    
    with col3:
        avg_uninsured_rate = metrics_df['Uninsured_Rate'].mean()
        st.metric("Avg Uninsured Rate", f"{avg_uninsured_rate:.1f}%")
    
    with col4:
        high_risk_areas = len(metrics_df[(metrics_df['Disability_Rate'] > metrics_df['Disability_Rate'].quantile(0.75)) & 
                                       (metrics_df['Uninsured_Rate'] > metrics_df['Uninsured_Rate'].quantile(0.75))])
        risk_percentage = (high_risk_areas / len(metrics_df)) * 100
        st.metric("High-Risk Areas", f"{high_risk_areas} ({risk_percentage:.1f}%)")
    
    with col5:
        correlation_strength = abs(correlation_matrix.loc['Disability_Rate', 'Uninsured_Rate'])
        st.metric("Disability-Uninsured Correlation", f"{correlation_strength:.3f}")
    
    st.markdown('<h2 class="section-header"> Exploratory Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Key Health Metrics",
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Distribution analysis
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Disability Rate Distribution', 'Uninsured Rate Distribution',
                          'Elderly Disability Rate', 'Working Age Disability Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
          # Add histograms
        fig_dist.add_trace(
            go.Histogram(x=metrics_df['Disability_Rate'], name='Disability Rate', nbinsx=20),
            row=1, col=1
        )
        fig_dist.add_trace(
            go.Histogram(x=metrics_df['Uninsured_Rate'], name='Uninsured Rate', nbinsx=20),
            row=1, col=2
        )
        fig_dist.add_trace(
            go.Histogram(x=metrics_df['Elderly_Disability_Rate'], name='Elderly Disability', nbinsx=20),
            row=2, col=1
        )
        fig_dist.add_trace(
            go.Histogram(x=metrics_df['Working_Age_Disability_Rate'], name='Working Age Disability', nbinsx=20),
            row=2, col=2
        )
        
        fig_dist.update_layout(height=500, title_text="Statistical Distributions", showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    # === MACHINE LEARNING CLUSTERING ANALYSIS ===
    # Perform clustering analysis for graphs
    clustered_df, scaler = perform_clustering_analysis(filtered_df)
    st.markdown('<h2 class="section-header">Cluster Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""    
    This clustering analysis identifies natural groupings of Seattle's geographic areas based on health indicators, 
    showing patterns that can inform what granularity of analysis is most effective for policy planning and help detect outliers or anomalous communities in need of extra aid.
    """)
    # Cluster visualizations per geographic division
    if 'Cluster' in clustered_df.columns and len(clustered_df) > 1:
        cluster_colors = px.colors.qualitative.Plotly
        clustered_df['Cluster_Label'] = clustered_df['Cluster'].astype(str).apply(lambda x: f'Cluster {x}')
        # Define mapping of type codes to labels (use 'CD' per CSV for City Council Districts)
        type_map = {
            'CRA': 'Community Reporting Areas (CRAs)',
            'UCUV': 'Urban Centers & Urban Villages (UCUVs)',
            'CD': 'City Council Districts'
        }
        cols = st.columns(3)
        for idx, (t, label) in enumerate(type_map.items()):
            df_t = clustered_df[clustered_df['Neighborhood Type'] == t]
            with cols[idx]:
                st.subheader(label)
                if df_t.empty:
                    st.write("No data available")
                else:
                    fig = px.scatter(
                        df_t,
                        x='Disability_Rate',
                        y='Uninsured_Rate',
                        color='Cluster_Label',
                        size='Population 18 years and Over',
                        hover_name='Neighborhood Name',
                        title=' ',
                        labels={
                            'Disability_Rate': 'Disability Rate (%)',
                            'Uninsured_Rate': 'Uninsured Rate (%)',
                            'Cluster_Label': 'Cluster'
                        },
                        color_discrete_sequence=cluster_colors,
                        height=600
                    )
                    fig.update_layout(
                        xaxis_title="Disability Rate (%)",
                        yaxis_title="Uninsured Rate (%)",
                        legend_title="Cluster",
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
          # Enhanced cluster summary with geographic breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Check if we have the columns needed for the summary
                required_summary_columns = ['Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate', 
                                          'Poverty_Disability_Rate', 'Population 18 years and Over']
                
                missing_summary_columns = [col for col in required_summary_columns if col not in clustered_df.columns]
                
                if missing_summary_columns:
                    st.warning(f"Missing columns for cluster summary: {', '.join(missing_summary_columns)}")
                else:
                    # Cluster summary statistics
                    cluster_summary = clustered_df.groupby('Cluster').agg(
                        Avg_Disability_Rate=('Disability_Rate', 'mean'),
                        Avg_Uninsured_Rate=('Uninsured_Rate', 'mean'),
                        Avg_Household_Disability_Rate=('Household_Disability_Rate', 'mean'),
                        Avg_Poverty_Disability_Rate=('Poverty_Disability_Rate', 'mean'),
                        Total_Population=('Population 18 years and Over', 'sum'),
                        Area_Count=('Neighborhood Name', 'count')
                    ).reset_index()

                    cluster_summary.columns = [
                        'Cluster', 'Avg Disability Rate', 'Avg Uninsured Rate',
                        'Avg Household Disability Rate', 'Avg Poverty Disability Rate',
                        'Total Population', 'Area Count'
                    ]
                    st.dataframe(cluster_summary)
                    
            except Exception as e:
                st.error(f"Error in cluster summary: {str(e)}")
        
        with col2:
            try:
                # Geographic composition of clusters
                st.markdown("**Geographic Boundary Composition by Cluster**")
                
                if 'Neighborhood Type' in clustered_df.columns and clustered_df['Neighborhood Type'].nunique() > 0:
                    pass # Added pass
                else:
                    pass # Added pass
            except Exception as e:
                pass # Added pass
        
        # Policy implications for clusters
        st.markdown("### Strategic Planning Implications by Cluster")
        
        cluster_policies = []
        for cluster_id in sorted(clustered_df['Cluster'].unique()):
            cluster_data = cluster_summary.loc[cluster_id]
            cluster_areas = clustered_df[clustered_df['Cluster'] == cluster_id]
            
            disability_avg = cluster_data['Avg Disability Rate']
            uninsured_avg = cluster_data['Avg Uninsured Rate']
            area_count = int(cluster_data['Area Count'])
            dominant_geo_type = cluster_areas['Neighborhood Type'].mode().iloc[0] if len(cluster_areas) > 0 else "Mixed"
            
            if disability_avg > metrics_df['Disability_Rate'].quantile(0.75):
                policy_focus = "Accessibility services, disability support programs"
            elif uninsured_avg > metrics_df['Uninsured_Rate'].quantile(0.75):
                policy_focus = "Health insurance enrollment, healthcare access"
            elif disability_avg < metrics_df['Disability_Rate'].quantile(0.25) and uninsured_avg < metrics_df['Uninsured_Rate'].quantile(0.25):
                policy_focus = "Resource optimization, best practice sharing"
            else:
                policy_focus = "Preventive health programs, community wellness"
            
            cluster_policies.append({
                'Cluster': f"Cluster {cluster_id}",
                'Areas': area_count,
                'Primary Geographic Type': dominant_geo_type,
                'Disability Rate': f"{disability_avg:.1f}%",
                'Uninsured Rate': f"{uninsured_avg:.1f}%",
                'Policy Focus': policy_focus
            })
        
        policy_df = pd.DataFrame(cluster_policies)
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
    
    # === REGRESSION ANALYSIS ===
    st.markdown('<h2 class="section-header">Predictive Regression Analysis</h2>', unsafe_allow_html=True)
    
    # Regression model comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Disability Rate Prediction Model**")
        disability_model = regression_results['disability_model']
        st.metric("R² Score", f"{disability_model['r2']:.3f}")
        st.metric("RMSE", f"{disability_model['rmse']:.2f}%")
        st.write("**Key Predictors:**")
        for i, feature in enumerate(disability_model['features']):
            coef = disability_model['coefficients'][i]
            st.write(f"• {feature.replace('_', ' ')}: {coef:.3f}")
    
    with col2:
        st.markdown("**Uninsured Rate Prediction Model**")
        uninsured_model = regression_results['uninsured_model']
        st.metric("R² Score", f"{uninsured_model['r2']:.3f}")
        st.metric("RMSE", f"{uninsured_model['rmse']:.2f}%")
        st.write("**Key Predictors:**")
        for i, feature in enumerate(uninsured_model['features']):
            coef = uninsured_model['coefficients'][i]
            st.write(f"• {feature.replace('_', ' ')}: {coef:.6f}")
    
    with col3:
        st.markdown("**Disability-Uninsured Correlation**")
        correlation_model = regression_results['correlation_model']
        st.metric("R² Score", f"{correlation_model['r2']:.3f}")
        st.metric("RMSE", f"{correlation_model['rmse']:.2f}%")
        slope = correlation_model['coefficients'][0]
        st.write(f"**Relationship:** For every 1% increase in disability rate, uninsured rate changes by {slope:.3f}%")
    
    # Regression visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted for Disability Rate
        fig_reg1 = px.scatter(
            x=disability_model['y_true'],
            y=disability_model['y_pred'],
            title="Disability Rate: Actual vs Predicted",
            labels={'x': 'Actual Disability Rate (%)', 'y': 'Predicted Disability Rate (%)'},
            trendline="ols"
        )
        
        # Add perfect prediction line
        min_val = min(disability_model['y_true'].min(), disability_model['y_pred'].min())
        max_val = max(disability_model['y_true'].max(), disability_model['y_pred'].max())
        fig_reg1.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig_reg1, use_container_width=True)
    
    with col2:
        # Actual vs Predicted for Uninsured Rate
        fig_reg2 = px.scatter(
            x=uninsured_model['y_true'],
            y=uninsured_model['y_pred'],
            title="Uninsured Rate: Actual vs Predicted",
            labels={'x': 'Actual Uninsured Rate (%)', 'y': 'Predicted Uninsured Rate (%)'},
            trendline="ols"
        )
        
        # Add perfect prediction line
        min_val = min(uninsured_model['y_true'].min(), uninsured_model['y_pred'].min())
        max_val = max(uninsured_model['y_true'].max(), uninsured_model['y_pred'].max())
        fig_reg2.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig_reg2, use_container_width=True)
    
    # Residual analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals for Disability Rate Model
        residuals1 = disability_model['y_true'] - disability_model['y_pred']
        fig_resid1 = px.scatter(
            x=disability_model['y_pred'],
            y=residuals1,
            title="Disability Rate Model: Residuals Analysis",
            labels={'x': 'Predicted Values (%)', 'y': 'Residuals (%)'}
        )
        fig_resid1.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid1, use_container_width=True)
    
    with col2:
        # Feature importance for Disability Rate Model
        feature_importance = pd.DataFrame({
            'Feature': [f.replace('_', ' ') for f in disability_model['features']],
            'Coefficient': np.abs(disability_model['coefficients']),
            'Direction': ['Positive' if c > 0 else 'Negative' for c in disability_model['coefficients']]
        }).sort_values('Coefficient', ascending=True)
        
        fig_importance = px.bar(
            feature_importance,
            x='Coefficient',
            y='Feature',
            color='Direction',
            title="Feature Importance (Disability Rate Model)",
            labels={'Coefficient': 'Absolute Coefficient Value'},
            orientation='h',
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Regression insights
    st.markdown("### Regression Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance Summary:**")
        st.write(f"• **Best performing model:** {'Disability Rate' if disability_model['r2'] > uninsured_model['r2'] else 'Uninsured Rate'} (R² = {max(disability_model['r2'], uninsured_model['r2']):.3f})")
        st.write(f"• **Disability-Uninsured correlation strength:** {correlation_model['r2']:.3f}")
        st.write(f"• **Most predictive factor for disability:** {disability_model['features'][np.argmax(np.abs(disability_model['coefficients']))]}")
        
        # Calculate prediction intervals
        disability_ci = 1.96 * disability_model['rmse']
        uninsured_ci = 1.96 * uninsured_model['rmse']
        st.write(f"• **95% Prediction interval (Disability):** ±{disability_ci:.1f}%")
        st.write(f"• **95% Prediction interval (Uninsured):** ±{uninsured_ci:.1f}%")
    
    with col2:
        st.markdown("**Predictive Relationships:**")
        
        # Strongest positive and negative relationships
        all_coefs = list(disability_model['coefficients']) + list(uninsured_model['coefficients'])
        all_features = disability_model['features'] + uninsured_model['features']
        
        max_coef_idx = np.argmax(all_coefs)
        min_coef_idx = np.argmin(all_coefs)
        
        st.write(f"• **Strongest positive predictor:** {all_features[max_coef_idx].replace('_', ' ')} ({all_coefs[max_coef_idx]:.3f})")
        st.write(f"• **Strongest negative predictor:** {all_features[min_coef_idx].replace('_', ' ')} ({all_coefs[min_coef_idx]:.3f})")
        
        # Model reliability assessment
        avg_r2 = np.mean([disability_model['r2'], uninsured_model['r2']])
        if avg_r2 > 0.7:
            reliability = "High"
        elif avg_r2 > 0.5:
            reliability = "Moderate"
        else:
            reliability = "Low"
        
        st.write(f"• **Overall model reliability:** {reliability} (Avg R² = {avg_r2:.3f})")
        st.write(f"• **Best prediction accuracy:** {max(disability_model['r2'], uninsured_model['r2'])*100:.1f}% of variance explained")
    
    # === COMPREHENSIVE VISUALIZATION SUITE ===
    st.markdown('<h2 class="section-header">Comprehensive Visualization Suite</h2>', unsafe_allow_html=True)
    
    # First row of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Age group disability comparison
        age_disability_data = {
            '18-64 years': metrics_df['Population 18 to 64 years with a Disability'].sum(),
            '65+ years': metrics_df['Population 65 years and Over with a Disability'].sum()
        }
        
        fig_age_disability = px.pie(
            values=list(age_disability_data.values()),
            names=list(age_disability_data.keys()),
            title="Distribution of Disabilities by Age Group",
            color_discrete_sequence=['#ff6b6b', '#4ecdc4']
        )
        st.plotly_chart(fig_age_disability, use_container_width=True)
    
    with col2:
        # Insurance coverage by age groups
        insurance_data = {
            'Under 19': {
                'Insured': metrics_df['Population under 19 years with Health Insurance'].sum(),
                'Uninsured': metrics_df['Population under 19 years without Health Insurance'].sum()
            },
            '19-34': {
                'Insured': metrics_df['Population 19 to 34 years with Health Insurance'].sum(),
                'Uninsured': metrics_df['Population 19 to 34 years without Health Insurance'].sum()
            },
            '35-64': {
                'Insured': metrics_df['Population 35 to 64 years with Health Insurance'].sum(),
                'Uninsured': metrics_df['Population 35 to 64 years without Health Insurance'].sum()
            },
            '65+': {
                'Insured': metrics_df['Population 65 years and Over with Health Insurance'].sum(),
                'Uninsured': metrics_df['Population 65 years and Over without Health Insurance'].sum()
            }
        }
        
        age_groups = list(insurance_data.keys())
        insured = [insurance_data[age]['Insured'] for age in age_groups]
        uninsured = [insurance_data[age]['Uninsured'] for age in age_groups]
        
        fig_insurance = go.Figure(data=[
            go.Bar(name='Insured', x=age_groups, y=insured, marker_color='#2ecc71'),
            go.Bar(name='Uninsured', x=age_groups, y=uninsured, marker_color='#e74c3c')
        ])
        
        fig_insurance.update_layout(
            barmode='stack',
            title="Health Insurance Coverage by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Population"
        )
        st.plotly_chart(fig_insurance, use_container_width=True)
    
    # Second row of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Poverty and disability analysis (18-64)
        poverty_disability_18_64 = {
            'Below Poverty with Disability': metrics_df['Population 18 to 64 years Below Poverty with a Disability'].sum(),
            'Below Poverty without Disability': metrics_df['Population 18 to 64 years Below Poverty without a Disability'].sum(),
            'Above Poverty with Disability': metrics_df['Population 18 to 64 years Above Poverty with a Disability'].sum(),
            'Above Poverty without Disability': metrics_df['Population 18 to 64 years Above Poverty without a Disability'].sum()
        }
        
        fig_poverty_18_64 = px.pie(
            values=list(poverty_disability_18_64.values()),
            names=list(poverty_disability_18_64.keys()),
            title="Poverty & Disability Status (Ages 18-64)",
            color_discrete_sequence=['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']
        )
        st.plotly_chart(fig_poverty_18_64, use_container_width=True)
    
    with col2:
        # Poverty and disability analysis (65+)
        poverty_disability_65_plus = {
            'Below Poverty with Disability': metrics_df['Population 65 years and Over Below Poverty with a Disability'].sum(),
            'Below Poverty without Disability': metrics_df['Population 65 years and Over Below Poverty without a Disability'].sum(),
            'Above Poverty with Disability': metrics_df['Population 65 years and Over Above Poverty with a Disability'].sum(),
            'Above Poverty without Disability': metrics_df['Population 65 years and Over Above Poverty without a Disability'].sum()
        }
        
        fig_poverty_65_plus = px.pie(
            values=list(poverty_disability_65_plus.values()),
            names=list(poverty_disability_65_plus.keys()),
            title="Poverty & Disability Status (Ages 65+)",
            color_discrete_sequence=['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']
        )
        st.plotly_chart(fig_poverty_65_plus, use_container_width=True)
      # === NEIGHBORHOOD RANKINGS AND COMPARISONS ===
    st.markdown('<h2 class="section-header">Geographic Area Performance Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Ranking Analysis Across Seattle's Administrative Boundaries**
    
    This section examines health outcomes across different types of geographic areas to identify patterns and priorities for resource allocation.
    """)
    
    # Multi-metric ranking with geographic context
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 areas by disability rate with geographic context
        if len(metrics_df) > 1:
            top_disability = metrics_df.nlargest(10, 'Disability_Rate')[['Neighborhood Name', 'Neighborhood Type', 'Disability_Rate', 'Population 18 years and Over']]
            
            fig_top_disability = px.bar(
                top_disability,
                x='Disability_Rate',
                y='Neighborhood Name',
                title="Top 10 Areas by Disability Rate (%)",
                color='Neighborhood Type',
                hover_data={'Population 18 years and Over': ':,'},
                orientation='h',
                height=500
            )
            fig_top_disability.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig_top_disability, use_container_width=True)
            
            # Geographic type breakdown for high disability areas
            high_disability_by_type = top_disability['Neighborhood Type'].value_counts()
            st.markdown("**Geographic Distribution of High-Disability Areas:**")
            for geo_type, count in high_disability_by_type.items():
                st.write(f"• {geo_type}: {count} areas ({count/len(top_disability)*100:.0f}%)")
    
    with col2:
        # Top 10 areas by uninsured rate with geographic context
        if len(metrics_df) > 1:
            top_uninsured = metrics_df.nlargest(10, 'Uninsured_Rate')[['Neighborhood Name', 'Neighborhood Type', 'Uninsured_Rate', 'Population 18 years and Over']]
            
            fig_top_uninsured = px.bar(
                top_uninsured,
                x='Uninsured_Rate',
                y='Neighborhood Name',
                title="Top 10 Areas by Uninsured Rate (%)",
                color='Neighborhood Type',
                hover_data={'Population 18 years and Over': ':,'},
                orientation='h',
                height=500
            )
            fig_top_uninsured.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig_top_uninsured, use_container_width=True)
            
            # Geographic type breakdown for high uninsured areas
            high_uninsured_by_type = top_uninsured['Neighborhood Type'].value_counts()
            st.markdown("**Geographic Distribution of High-Uninsured Areas:**")
            for geo_type, count in high_uninsured_by_type.items():
                st.write(f"• {geo_type}: {count} areas ({count/len(top_uninsured)*100:.0f}%)")
    
    # Cross-analysis: Areas appearing in both top disability and uninsured lists
    overlap_areas = set(top_disability['Neighborhood Name']) & set(top_uninsured['Neighborhood Name'])
    if overlap_areas:
        st.warning(f"""
        **Critical Areas Needing Attention:** {len(overlap_areas)} areas appear in both high-disability and high-uninsured rankings:
        {', '.join(sorted(overlap_areas))}
        
        These areas may benefit from targeted health services and insurance enrollment programs.
        """)
      # === RISK ASSESSMENT MATRIX ===
    st.markdown('<h2 class="section-header">Geographic Risk Assessment Matrix</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Health Risk Quadrant Analysis Across Seattle's Administrative Boundaries**
    
    This risk matrix categorizes Seattle's geographic areas based on disability and uninsured rates, 
    helping identify priority areas for health service planning and resource allocation.
    """)
    
    # Create risk quadrants
    disability_median = metrics_df['Disability_Rate'].median()
    uninsured_median = metrics_df['Uninsured_Rate'].median()
    
    fig_risk = px.scatter(
        metrics_df,
        x='Disability_Rate',
        y='Uninsured_Rate',
        size='Population 18 years and Over',
        color='Neighborhood Type',
        hover_name='Neighborhood Name',
        hover_data={
            'Neighborhood Type': True,
            'Population 18 years and Over': ':,',
            'Household_Disability_Rate': ':.1f',
            'Elderly_Disability_Rate': ':.1f'
        },
        title="Geographic Risk Assessment: Disability vs Uninsured Rates by Administrative Boundary Type",
        labels={
            'Disability_Rate': 'Disability Rate (%)', 
            'Uninsured_Rate': 'Uninsured Rate (%)',
            'Neighborhood Type': 'Geographic Type'
        },
        height=600
    )
    
    # Add quadrant lines and annotations
    fig_risk.add_hline(y=uninsured_median, line_dash="dash", line_color="red", 
                      annotation_text=f"Median Uninsured Rate ({uninsured_median:.1f}%)")
    fig_risk.add_vline(x=disability_median, line_dash="dash", line_color="red", 
                      annotation_text=f"Median Disability Rate ({disability_median:.1f}%)")
    
    # Add quadrant labels
    max_x, max_y = metrics_df['Disability_Rate'].max(), metrics_df['Uninsured_Rate'].max()
    fig_risk.add_annotation(x=disability_median + (max_x - disability_median)/2, 
                           y=uninsured_median + (max_y - uninsured_median)/2,
                           text="HIGH RISK<br>Priority for intervention", 
                           showarrow=False, bgcolor="rgba(255,0,0,0.1)")
    fig_risk.add_annotation(x=disability_median/2, y=uninsured_median/2,
                           text="LOW RISK<br>Stable areas", 
                           showarrow=False, bgcolor="rgba(0,255,0,0.1)")
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk categorization with geographic breakdown
    st.markdown("### Risk Category Analysis by Geographic Type")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate risk categories
    high_risk_areas = metrics_df[(metrics_df['Disability_Rate'] > disability_median) & 
                                (metrics_df['Uninsured_Rate'] > uninsured_median)]
    moderate_risk_1 = metrics_df[(metrics_df['Disability_Rate'] > disability_median) & 
                                (metrics_df['Uninsured_Rate'] <= uninsured_median)]
    moderate_risk_2 = metrics_df[(metrics_df['Disability_Rate'] <= disability_median) & 
                                (metrics_df['Uninsured_Rate'] > uninsured_median)]
    low_risk_areas = metrics_df[(metrics_df['Disability_Rate'] <= disability_median) & 
                               (metrics_df['Uninsured_Rate'] <= uninsured_median)]
    
    with col1:
        st.metric("🔴 High Risk", len(high_risk_areas), 
                 delta=f"{len(high_risk_areas)/len(metrics_df)*100:.1f}% of areas")
        if len(high_risk_areas) > 0:
            high_risk_types = high_risk_areas['Neighborhood Type'].value_counts()
            st.write("**By Geographic Type:**")
            for geo_type, count in high_risk_types.items():
                st.write(f"• {geo_type}: {count}")
    
    with col2:
        st.metric("🟠 Moderate Risk 1 (High Disability)", len(moderate_risk_1),
                 delta=f"{len(moderate_risk_1)/len(metrics_df)*100:.1f}% of areas")
        if len(moderate_risk_1) > 0:
            mod1_types = moderate_risk_1['Neighborhood Type'].value_counts()
            st.write("**By Geographic Type:**")
            for geo_type, count in mod1_types.items():
                st.write(f"• {geo_type}: {count}")
    
    with col3:
        st.metric("🟡 Moderate Risk 2 (High Uninsured)", len(moderate_risk_2),
                 delta=f"{len(moderate_risk_2)/len(metrics_df)*100:.1f}% of areas")
        if len(moderate_risk_2) > 0:
            mod2_types = moderate_risk_2['Neighborhood Type'].value_counts()
            st.write("**By Geographic Type:**")
            for geo_type, count in mod2_types.items():
                st.write(f"• {geo_type}: {count}")
    
    with col4:
        st.metric("🟢 Low Risk", len(low_risk_areas),
                 delta=f"{len(low_risk_areas)/len(metrics_df)*100:.1f}% of areas")
        if len(low_risk_areas) > 0:
            low_risk_types = low_risk_areas['Neighborhood Type'].value_counts()
            st.write("**By Geographic Type:**")
            for geo_type, count in low_risk_types.items():
                st.write(f"• {geo_type}: {count}")
    
    # Policy implications
    st.info("""
    **Policy Implications for Seattle's Geographic Planning:**
    
    🔴 **High-Risk Areas** (High Disability + High Uninsured): Immediate priority for comprehensive health services, insurance enrollment programs, and disability support services.
    
    🟠 **Moderate Risk Areas**: May benefit from targeted interventions - disability areas need accessibility improvements, uninsured areas need insurance navigation support.
    
    🟢 **Low-Risk Areas**: Can serve as models for best practices and may have capacity to support neighboring high-risk areas through resource sharing or program partnerships.
    """)
      # === GEOGRAPHIC VISUALIZATION ===
    st.markdown('<h2 class="section-header">Seattle Geographic Analysis</h2>', unsafe_allow_html=True)
    
    # Geographic context information
    st.markdown("""
    **Understanding Seattle's Administrative Boundaries:**
    
    Seattle's data is organized across multiple geographic levels that serve different administrative and planning purposes:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Community Reporting Areas (CRAs):**
        - 53 areas covering all of Seattle
        - Primary unit for city data reporting
        - Align with neighborhood identities
        - Used for resource allocation and service planning
        - Population typically 5,000-20,000 per CRA
        
        **Urban Centers & Urban Villages:**
        - Designated growth areas in Seattle's Comprehensive Plan
        - Higher density zoning and development focus
        - Transit-oriented development priorities
        - Support majority of population and job growth
        """)
    
    with col2:
        st.markdown("""
        **City Council Districts:**
        - 7 districts for electoral representation
        - Established in 2013 for geographic equity
        - Each district ~100,000 residents
        - Balanced for demographic and geographic diversity
        - Ensure all Seattle areas have council representation
        
        **Analysis Implications:**
        - CRAs provide neighborhood-level detail
        - UCUVs show growth area health patterns  
        - Council Districts ensure citywide coverage
        """)
    
    # Choropleth map section
    st.markdown("""
    **Interactive Geographic Mapping**    
    Explore health and disability patterns across Seattle's administrative boundaries and planning areas.
    """)
    
    # Map metric selection
    map_metrics = {
        'Disability_Rate': 'Disability Rate (%) - Overall disability prevalence',
        'Uninsured_Rate': 'Uninsured Rate (%) - Health insurance coverage gaps', 
        'Household_Disability_Rate': 'Household Disability Rate (%) - Households with disabled members',
        'Elderly_Disability_Rate': 'Elderly Disability Rate (%) - Age 65+ disability rates',
        'Working_Age_Disability_Rate': 'Working Age Disability Rate (%) - Ages 18-64 disability rates',
        'Poverty_Disability_Rate': 'Poverty & Disability Rate (%) - Economic vulnerability intersection'
    }
    
    selected_metric = st.selectbox(
        "Select health metric for geographic visualization:",
        options=list(map_metrics.keys()),
        format_func=lambda x: map_metrics[x],
        index=0,
        help="Choose which health indicator to visualize across Seattle's geographic areas"
    )
    
    # Create and display the map
    fig_map = create_choropleth_map(filtered_df, selected_metric, map_metrics[selected_metric])
    if fig_map:
        st.plotly_chart(fig_map, use_container_width=True)
      # Geographic insights with comprehensive administrative boundary analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Neighborhood type comparison with statistical significance
        if 'Neighborhood Type' in metrics_df.columns and len(metrics_df['Neighborhood Type'].unique()) > 1:
            type_comparison = metrics_df.groupby('Neighborhood Type')[selected_metric].agg(['mean', 'median', 'std', 'count']).round(2)
            type_comparison['cv'] = (type_comparison['std'] / type_comparison['mean'] * 100).round(1)  # Coefficient of variation
            
            fig_type_comparison = px.bar(
                x=type_comparison.index,
                y=type_comparison['mean'],
                error_y=type_comparison['std'],
                title=f"Health Outcomes by Seattle Administrative Boundary Type<br>{map_metrics[selected_metric].split(' - ')[0]}",
                labels={'x': 'Administrative Boundary Type', 'y': f'Mean {map_metrics[selected_metric].split(" - ")[0]}'},
                color=type_comparison['mean'],
                color_continuous_scale='RdYlBu_r',
                hover_data={
                    'Mean': type_comparison['mean'],
                    'Median': type_comparison['median'], 
                    'Std Dev': type_comparison['std'],
                    'Count': type_comparison['count'],
                    'Coeff Var %': type_comparison['cv']
                }
            )
            fig_type_comparison.update_layout(
                xaxis_tickangle=45,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_type_comparison, use_container_width=True)
            
            # Statistical summary for geographic types
            st.markdown("**Statistical Summary by Geographic Type:**")
            for geo_type in type_comparison.index:
                mean_val = type_comparison.loc[geo_type, 'mean']
                count_val = int(type_comparison.loc[geo_type, 'count'])
                cv_val = type_comparison.loc[geo_type, 'cv']
                st.write(f"• **{geo_type}**: {mean_val:.1f}% avg, {count_val} areas, {cv_val}% variability")
    
    with col2:
        # Enhanced top/bottom comparison with geographic context
        if len(metrics_df) >= 10:
            top_5 = metrics_df.nlargest(5, selected_metric)[['Neighborhood Name', 'Neighborhood Type', selected_metric]]
            bottom_5 = metrics_df.nsmallest(5, selected_metric)[['Neighborhood Name', 'Neighborhood Type', selected_metric]]
            
            comparison_data = pd.concat([
                top_5.assign(Category='Top 5 (Highest Need)'),
                bottom_5.assign(Category='Bottom 5 (Lowest Need)')
            ])
            
            # Bar plot of highest vs lowest need, color-coded by geographic type
            fig_comparison = px.bar(
                comparison_data,
                x='Neighborhood Name',
                y=selected_metric,
                color='Neighborhood Type',
                title=f"Highest vs Lowest Need Areas: {map_metrics[selected_metric].split(' - ')[0]}",
                hover_name='Neighborhood Name',
                hover_data={selected_metric: ':.1f'},
                height=500
            )
            fig_comparison.update_layout(
                xaxis_tickangle=45,
                legend=dict(
                    title='Geographic Type',
                    orientation='h',
                    yanchor='bottom',
                    y=-0.25,
                    xanchor='center',
                    x=0.5
                )
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Geographic distribution analysis
            st.markdown("**Geographic Distribution of Extreme Values:**")
            top_5_types = top_5['Neighborhood Type'].value_counts()
            bottom_5_types = bottom_5['Neighborhood Type'].value_counts()
            
            st.write("**Highest Need Areas by Type:**")
            for geo_type, count in top_5_types.items():
                st.write(f"• {geo_type}: {count} areas")
            
            st.write("**Lowest Need Areas by Type:**")
            for geo_type, count in bottom_5_types.items():
                st.write(f"• {geo_type}: {count} areas")
    
    # Geographic disparity analysis
    st.markdown("### Geographic Health Disparity Analysis")
    
    # Calculate disparity metrics across geographic types
    if len(metrics_df['Neighborhood Type'].unique()) > 1:
        disparity_analysis = {}
        for metric in ['Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate']:
            metric_by_type = metrics_df.groupby('Neighborhood Type')[metric].mean()
            max_val = metric_by_type.max()
            min_val = metric_by_type.min()
            disparity_ratio = max_val / min_val if min_val > 0 else float('inf')
            disparity_analysis[metric] = {
                'max': max_val,
                'min': min_val,
                'ratio': disparity_ratio,
                'difference': max_val - min_val
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            disability_ratio = disparity_analysis['Disability_Rate']['ratio']
            disability_diff = disparity_analysis['Disability_Rate']['difference']
            st.metric("Disability Rate Disparity", f"{disability_ratio:.1f}x", 
                     delta=f"{disability_diff:.1f}% difference")
        
        with col2:
            uninsured_ratio = disparity_analysis['Uninsured_Rate']['ratio']
            uninsured_diff = disparity_analysis['Uninsured_Rate']['difference']
            st.metric("Uninsured Rate Disparity", f"{uninsured_ratio:.1f}x",
                     delta=f"{uninsured_diff:.1f}% difference")
        
        with col3:
            hh_disability_ratio = disparity_analysis['Household_Disability_Rate']['ratio']
            hh_disability_diff = disparity_analysis['Household_Disability_Rate']['difference']
            st.metric("Household Disability Disparity", f"{hh_disability_ratio:.1f}x",
                     delta=f"{hh_disability_diff:.1f}% difference")
        
        # Interpretation
        avg_disparity = np.mean([disparity_analysis[m]['ratio'] for m in disparity_analysis.keys()])
        if avg_disparity > 2.0:
            disparity_level = "High"
            color = "🔴"
        elif avg_disparity > 1.5:
            disparity_level = "Moderate"
            color = "🟠"
        else:
            disparity_level = "Low"
            color = "🟢"
        
        st.info(f"""
        {color} **Geographic Health Disparity Level: {disparity_level}** (Average ratio: {avg_disparity:.1f}x)
        
        This indicates {'significant' if avg_disparity > 2.0 else 'moderate' if avg_disparity > 1.5 else 'relatively low'} 
        variation in health outcomes across Seattle's different administrative boundary types, suggesting 
        {'targeted interventions may be needed' if avg_disparity > 1.5 else 'relatively equitable distribution'} 
        for resource allocation and policy planning.
        """)
    
    # === DETAILED DATA TABLE ===
    st.markdown('<h2 class="section-header">Extended Data Table</h2>', unsafe_allow_html=True)

    st.markdown(" Download this data for yourself! Refuse fascism and save government data from the gross hands of the government!")
    # Create comprehensive table
    display_columns = ['Neighborhood Name', 'Neighborhood Type', 'Population 18 years and Over',
                      'Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate',
                      'Elderly_Disability_Rate', 'Working_Age_Disability_Rate', 'Poverty_Disability_Rate']
    
    display_df = metrics_df[display_columns].copy()
    display_df.columns = ['Neighborhood', 'Type', 'Population 18+', 'Disability Rate (%)', 
                         'Uninsured Rate (%)', 'HH Disability Rate (%)', 'Elderly Disability (%)',
                         'Working Age Disability (%)', 'Poverty Disability (%)']
    
    # Format numerical columns
    display_df['Population 18+'] = display_df['Population 18+'].apply(lambda x: f"{x:,}")
    for col in ['Disability Rate (%)', 'Uninsured Rate (%)', 'HH Disability Rate (%)',
               'Elderly Disability (%)', 'Working Age Disability (%)', 'Poverty Disability (%)']:
        display_df[col] = display_df[col].round(1)
    
    st.dataframe(display_df, use_container_width=True, height=400)
      # === STATISTICAL INSIGHTS ===
    st.markdown('<h2 class="section-header">Key Statistical & Geographic Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Statistical Analysis:**")
        st.write(f"• Disability-Uninsured correlation: {correlation_matrix.loc['Disability_Rate', 'Uninsured_Rate']:.3f}")
        st.write(f"• Disability-Poverty correlation: {correlation_matrix.loc['Disability_Rate', 'Poverty_Disability_Rate']:.3f}")
        st.write(f"• Strongest positive correlation: {correlation_matrix.max().max():.3f}")
        st.write(f"• Strongest negative correlation: {correlation_matrix.min().min():.3f}")
        
        st.markdown("**Predictive Model Performance:**")
        st.write(f"• Disability model R²: {regression_results['disability_model']['r2']:.3f}")
        st.write(f"• Uninsured model R²: {regression_results['uninsured_model']['r2']:.3f}")
        st.write(f"• Best predictive accuracy: {max(regression_results['disability_model']['r2'], regression_results['uninsured_model']['r2'])*100:.1f}%")
        
        st.markdown("**Data Distribution:**")
        st.write(f"• Disability rate standard deviation: {metrics_df['Disability_Rate'].std():.2f}%")
        st.write(f"• Uninsured rate standard deviation: {metrics_df['Uninsured_Rate'].std():.2f}%")
        most_variable = metrics_df[['Disability_Rate', 'Uninsured_Rate', 'Household_Disability_Rate']].std().idxmax()
        st.write(f"• Most variable health metric: {most_variable.replace('_', ' ')}")
    
    with col2:
        st.markdown("**Geographic & Population Analysis:**")
        total_population = metrics_df['Population 18 years and Over'].sum()
        total_disabled = metrics_df['Population 18 years and Over with a Disability'].sum()
        total_uninsured = metrics_df['Population without Health Insurance'].sum()
        
        st.write(f"• Total adult population: {total_population:,}")
        st.write(f"• Total with disabilities: {total_disabled:,} ({total_disabled/total_population*100:.1f}%)")
        st.write(f"• Total uninsured: {total_uninsured:,} ({total_uninsured/total_population*100:.1f}%)")
        st.write(f"• Average area population: {total_population/len(metrics_df):,.0f}")
        st.markdown("**Geographic Risk Assessment:**")
        high_risk_count = len(metrics_df[(metrics_df['Disability_Rate'] > disability_median) & 
                                        (metrics_df['Uninsured_Rate'] > uninsured_median)])
        st.write(f"• High-risk areas: {high_risk_count} of {len(metrics_df)} ({high_risk_count/len(metrics_df)*100:.1f}%)")
        above_disability_median = len(metrics_df[metrics_df['Disability_Rate'] > disability_median])
        above_uninsured_median = len(metrics_df[metrics_df['Uninsured_Rate'] > uninsured_median])
        st.write(f"• Areas above median disability rate: {above_disability_median} ({above_disability_median/len(metrics_df)*100:.1f}%)")
        st.write(f"• Areas above median uninsured rate: {above_uninsured_median} ({above_uninsured_median/len(metrics_df)*100:.1f}%)")
        
        # Geographic type analysis
        if len(geographic_summary) > 1:
            st.markdown("**Administrative Boundary Analysis:**")
            for geo_type, count in geographic_summary.items():
                avg_disability = metrics_df[metrics_df['Neighborhood Type'] == geo_type]['Disability_Rate'].mean()
                st.write(f"• {geo_type}: {count} areas, avg disability rate {avg_disability:.1f}%")
      # Footer with comprehensive data source and methodology information
    st.markdown("---")
    st.markdown("""
    ### Data Sources & Methodology
    
    **Primary Data Source:** U.S. Census Bureau American Community Survey (ACS) 5-year estimates  
    
    **Key ACS Tables:**
    - **C21007:** Age by Veteran Status by Poverty Status by Disability Status
    - **B27010:** Types of Health Insurance Coverage by Age  
    - **B22010:** Receipt of Food Stamps/SNAP by Disability Status
    
    **Geographic Coverage:**
    - **53 Community Reporting Areas (CRAs)** - Seattle's primary administrative boundaries
    - **Urban Centers & Urban Villages (UCUVs)** - Comprehensive Plan growth strategy areas  
    - **7 City Council Districts** - Electoral representation boundaries
    
    **Analysis Methodology:**
    - Statistical correlation analysis for health indicator relationships
    - K-means clustering for neighborhood classification and pattern recognition
    - Multiple linear regression for predictive modeling and trend analysis
    - Risk matrix assessment for public health planning and resource allocation
    - Geographic visualization with interactive mapping capabilities
    
    **Data Vintage:** 2019-2023 (5-Year Estimates)  
    **Analysis Features:** Interactive geographic mapping, neighborhood comparisons, cluster analysis, risk assessment, statistical modeling
    
    **Attribution:** Ada Carter, University of Washington — [adacarter.org](https://adacarter.org)""")
    
    # Additional context
    st.info("""
    **Geographic Analysis Note:** Seattle's 53 Community Reporting Areas provide the most detailed geographic breakdown for health data analysis. 
    These areas balance statistical reliability with neighborhood-level insights, making them ideal for local health planning and resource allocation decisions.
    """)

if __name__ == "__main__":
    main()
      # Footer with comprehensive data source and methodology information
