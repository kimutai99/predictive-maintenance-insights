import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_OPERATING_PATH = "images/mining_equipment_operating.jpg"
IMAGE_FAILED_PATH = "images/failed_mining_equipment.jpg"
# --- LANGUAGE TRANSLATIONS ---
translations = {
    "en": {
        "page_title": "Machine Failure Prediction",
        "login_title": "Machine Failure Prediction",
        "login_intro": "This application predicts machine failure using sensor data.",
        "username_prompt": "Username",
        "password_prompt": "Password",
        "login_button": "Login",
        "incorrect_credentials_error": "Incorrect username or password.",
        "welcome_message": "Welcome",
        "logout_button": "Log Out",
        "data_source_title": "Data Source",
        "upload_prompt": "Upload your own CSV",
        "upload_success": "Successfully loaded uploaded data!",
        "download_sample_data": "Download Sample Data",
        "home_tab": "Home",
        "dashboard_tab": "Dashboard",
        "prediction_tab": "Prediction",
        "stats_tab": "Statistical Analysis",
        "data_viewer_tab": "Data Viewer",
        "readme_tab": "Read Me",
        "home_title": "Predictive Maintenance for Mining Equipment",
        "project_summary_header": "Project Summary",
        "problem_statement": "**The Problem:** Unplanned equipment failures in the mining industry are a major source of operational disruption, leading to significant financial losses and safety risks. Maintenance is often reactive, performed only after a breakdown occurs.",
        "solution_statement": "**The Solution:** This project implements a predictive maintenance solution using machine learning. By analyzing real-time sensor data from mining equipment, the system can accurately predict which machines are likely to fail within the next 14 days and identify the specific component that is at risk.",
        "impact_statement": "**The Impact:** This proactive approach allows maintenance teams to schedule repairs *before* failures happen, minimizing downtime, reducing maintenance costs, and improving overall operational efficiency and safety.",
        "dashboard_header": "Fleet Health Dashboard",
        "dashboard_filters_header": "Dashboard Filters",
        "machine_type_filter_label": "Filter by Machine Type",
        "manufacturer_filter_label": "Filter by Manufacturer",
        "age_filter_label": "Filter by Age Range",
        "kpi_header": "Key Performance Indicators",
        "total_machines_kpi": "Total Machines",
        "healthy_machines_kpi": "Healthy Machines",
        "at_risk_kpi": "Machines at High/Critical Risk",
        "fleet_overview_header": "Fleet Overview",
        "machine_type_dist_title": "Distribution of Machine Types",
        "machine_age_dist_title": "Distribution of Machine Ages",
        "failure_prob_dist_title": "Distribution of Failure Probability Across the Fleet",
        "no_data_warning": "No data available for the selected filters.",
        "prediction_header": "Machine-Specific Predictions",
        "machines_at_risk_header": "Machines at Risk of Failure",
        "predict_component_header": "Predict Failing Component",
        "select_machine_prompt": "Select Machine ID from Risk List",
        "predicted_component_error": "**Predicted Failing Component:**",
        "critical_severity_warning": "**Predicted Failure Window:** Within the next 7 days. Immediate inspection required.",
        "high_severity_warning": "**Predicted Failure Window:** Within the next 14 days. Schedule maintenance soon.",
        "medium_severity_info": "**Recommendation:** Monitor this machine closely. Increased risk detected.",
        "no_risk_success": "No machines are currently at risk of failure.",
        "stats_header": "Statistical Analysis",
        "select_analysis_type": "Select Analysis Type",
        "univariate_analysis": "Univariate Analysis",
        "bivariate_analysis": "Bivariate Analysis",
        "linear_regression": "Linear Regression",
        "select_variable_prompt": "Select a variable",
        "qualitative_analysis_header": "**Qualitative Variable Analysis**",
        "quantitative_analysis_header": "**Quantitative Variable Analysis**",
        "normality_analysis_header": "Normality Analysis",
        "shapiro_wilk_test": "Shapiro-Wilk Test",
        "normality_test_success": "The data appears to be normally distributed.",
        "normality_test_warning": "The data does not appear to be normally distributed.",
        "bivariate_analysis_header": "Bivariate Analysis",
        "select_first_variable": "Select the first variable",
        "select_second_variable": "Select the second variable",
        "quant_vs_quant_header": "**Quantitative vs. Quantitative Analysis**",
        "qual_vs_qual_header": "**Qualitative vs. Qualitative Analysis**",
        "chi_square_test": "Chi-square Test",
        "qual_vs_quant_analysis_header": "Analysis of {quant_var} by {qual_var}",
        "ttest": "T-test",
        "anova": "ANOVA",
        "linear_regression_header": "Linear Regression Analysis",
        "select_dependent_variable": "Select the dependent variable (must be quantitative)",
        "select_independent_variables": "Select independent variables",
        "regression_interpretation_header": "Interpretation of Regression Results",
        "regression_assumptions_header": "Linear Regression Assumption Checks",
        "linearity_check": "1. Linearity Check",
        "homoscedasticity_check": "2. Homoscedasticity Check",
        "normality_of_residuals_check": "3. Normality of Residuals Check",
        "data_viewer_header": "About the Data & Models",
        "dataset_overview_header": "Dataset Overview",
        "model_performance_header": "Model Performance",
        "data_dictionary_header": "Data Dictionary",
        "raw_data_explorer_header": "Raw Data Explorer",
        "readme_title": "About This Project",
    },
    "fr": {
        "page_title": "Prédiction de Pannes de Machines",
        "login_title": "Prédiction de Pannes de Machines",
        "login_intro": "Cette application prédit les pannes de machines à l'aide des données de capteurs.",
        "username_prompt": "Nom d'utilisateur",
        "password_prompt": "Mot de passe",
        "login_button": "Se connecter",
        "incorrect_credentials_error": "Nom d'utilisateur ou mot de passe incorrect.",
        "welcome_message": "Bienvenue",
        "logout_button": "Se déconnecter",
        "data_source_title": "Source de Données",
        "upload_prompt": "Téléchargez votre propre CSV",
        "upload_success": "Données téléchargées avec succès !",
        "download_sample_data": "Télécharger un jeu de données d'exemple",
        "home_tab": "Accueil",
        "dashboard_tab": "Tableau de Bord",
        "prediction_tab": "Prédiction",
        "stats_tab": "Analyse Statistique",
        "data_viewer_tab": "Visualiseur de Données",
        "readme_tab": "À Propos",
        "home_title": "Maintenance Prédictive pour Équipement Minier",
        "project_summary_header": "Résumé du Projet",
        "problem_statement": "**Le Problème :** Les pannes d'équipement imprévues dans l'industrie minière sont une source majeure de perturbation opérationnelle, entraînant des pertes financières importantes et des risques pour la sécurité. La maintenance est souvent réactive, effectuée seulement après une panne.",
        "solution_statement": "**La Solution :** Ce projet met en œuvre une solution de maintenance prédictive utilisant l'apprentissage automatique. En analysant les données des capteurs en temps réel de l'équipement minier, le système peut prédire avec précision quelles machines sont susceptibles de tomber en panne dans les 14 prochains jours et identifier le composant spécifique à risque.",
        "impact_statement": "**L'Impact :** Cette approche proactive permet aux équipes de maintenance de planifier les réparations *avant* que les pannes ne se produisent, minimisant ainsi les temps d'arrêt, réduisant les coûts de maintenance et améliorant l'efficacité opérationnelle globale et la sécurité.",
        "dashboard_header": "Tableau de Bord de la Santé de la Flotte",
        "dashboard_filters_header": "Filtres du Tableau de Bord",
        "machine_type_filter_label": "Filtrer par Type de Machine",
        "manufacturer_filter_label": "Filtrer par Fabricant",
        "age_filter_label": "Filtrer par Tranche d'Âge",
        "kpi_header": "Indicateurs Clés de Performance",
        "total_machines_kpi": "Total des Machines",
        "healthy_machines_kpi": "Machines en Bonne Santé",
        "at_risk_kpi": "Machines à Risque Élevé/Critique",
        "fleet_overview_header": "Aperçu de la Flotte",
        "machine_type_dist_title": "Distribution des Types de Machines",
        "machine_age_dist_title": "Distribution de l'Âge des Machines",
        "failure_prob_dist_title": "Distribution de la Probabilité de Panne dans la Flotte",
        "no_data_warning": "Aucune donnée disponible pour les filtres sélectionnés.",
        "prediction_header": "Prédictions Spécifiques à la Machine",
        "machines_at_risk_header": "Machines à Risque de Panne",
        "predict_component_header": "Prédire le Composant Défaillant",
        "select_machine_prompt": "Sélectionnez l'ID de la machine dans la liste des risques",
        "predicted_component_error": "**Composant Défaillant Prédit :**",
        "critical_severity_warning": "**Fenêtre de Panne Prédite :** Dans les 7 prochains jours. Inspection immédiate requise.",
        "high_severity_warning": "**Fenêtre de Panne Prédite :** Dans les 14 prochains jours. Planifiez la maintenance bientôt.",
        "medium_severity_info": "**Recommandation :** Surveillez cette machine de près. Risque accru détecté.",
        "no_risk_success": "Aucune machine n'est actuellement à risque de panne.",
        "stats_header": "Analyse Statistique",
        "select_analysis_type": "Sélectionnez le Type d'Analyse",
        "univariate_analysis": "Analyse Univariée",
        "bivariate_analysis": "Analyse Bivariée",
        "linear_regression": "Régression Linéaire",
        "select_variable_prompt": "Sélectionnez une variable",
        "qualitative_analysis_header": "**Analyse de Variable Qualitative**",
        "quantitative_analysis_header": "**Analyse de Variable Quantitative**",
        "normality_analysis_header": "Analyse de Normalité",
        "shapiro_wilk_test": "Test de Shapiro-Wilk",
        "normality_test_success": "Les données semblent être normalement distribuées.",
        "normality_test_warning": "Les données ne semblent pas être normalement distribuées.",
        "bivariate_analysis_header": "Analyse Bivariée",
        "select_first_variable": "Sélectionnez la première variable",
        "select_second_variable": "Sélectionnez la deuxième variable",
        "quant_vs_quant_header": "**Analyse Quantitative vs. Quantitative**",
        "qual_vs_qual_header": "**Analyse Qualitative vs. Qualitative**",
        "chi_square_test": "Test du Chi-carré",
        "qual_vs_quant_analysis_header": "Analyse de {quant_var} par {qual_var}",
        "ttest": "Test T",
        "anova": "ANOVA",
        "linear_regression_header": "Analyse de Régression Linéaire",
        "select_dependent_variable": "Sélectionnez la variable dépendante (doit être quantitative)",
        "select_independent_variables": "Sélectionnez les variables indépendantes",
        "regression_interpretation_header": "Interprétation des Résultats de la Régression",
        "regression_assumptions_header": "Vérification des Hypothèses de la Régression Linéaire",
        "linearity_check": "1. Vérification de la Linéarité",
        "homoscedasticity_check": "2. Vérification de l'Homoscédasticité",
        "normality_of_residuals_check": "3. Vérification de la Normalité des Résidus",
        "data_viewer_header": "À Propos des Données & Modèles",
        "dataset_overview_header": "Aperçu du Jeu de Données",
        "model_performance_header": "Performance du Modèle",
        "data_dictionary_header": "Dictionnaire de Données",
        "raw_data_explorer_header": "Explorateur de Données Brutes",
        "readme_title": "À Propos de Ce Projet",
    }
}

# --- LANGUAGE SELECTION ---
st.sidebar.title("Language")
language = st.sidebar.selectbox("Select Language", ["English", "French"])
lang_code = "fr" if language == "French" else "en"

def _(text_key, **kwargs):
    return translations[lang_code].get(text_key, text_key).format(**kwargs)

# Set page config
st.set_page_config(layout="wide", page_title=_("page_title"))

# --- LOGIN ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title(_("login_title"))
        st.write(_("login_intro"))
        
        st.info("Username: `Data Professionals` | Password: `Hello World`")
        
        username = st.text_input(_("username_prompt"), key="username_input")
        password = st.text_input(_("password_prompt"), type="password", key="password_input")

        if st.button(_("login_button")):
            correct_username = os.environ.get("APP_USERNAME", "Data Professionals")
            correct_password = os.environ.get("APP_PASSWORD", "Hello World")
            if username == correct_username and password == correct_password:
                st.session_state["password_correct"] = True
                st.session_state["username_for_display"] = username
                st.rerun()
            else:
                st.error(_("incorrect_credentials_error"))
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- LOGOUT ---
st.sidebar.title(f'{_("welcome_message")}, {st.session_state.get("username_for_display", "")}')
if st.sidebar.button(_("logout_button")):
    st.session_state["password_correct"] = False
    st.rerun()

# --- Load Model and Encoders ---
@st.cache_data
def load_model_and_encoders():
    try:
        failure_model_path = os.path.join(MODELS_DIR, 'failure_prediction_model.joblib')
        component_model_path = os.path.join(MODELS_DIR, 'component_prediction_model.joblib')
        failure_model_data = joblib.load(failure_model_path)
        component_model_data = joblib.load(component_model_path)
        return failure_model_data, component_model_data
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.info(f"Please ensure the model files are in the `{MODELS_DIR}` directory.")
        st.stop()

failure_model_data, component_model_data = load_model_and_encoders()
failure_model = failure_model_data['model']
encoders = failure_model_data['encoders']
model_features = failure_model_data['features']
component_model = component_model_data['model']
component_label_encoder = component_model_data['label_encoder']

# --- Data Loading and Selection ---
st.sidebar.title(_("data_source_title"))
uploaded_file = st.sidebar.file_uploader(_("upload_prompt"), type=["csv"])

@st.cache_data
def load_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df

if uploaded_file is not None:
    data = load_csv(uploaded_file)
    st.sidebar.success(_("upload_success"))
else:
    default_data_path = os.path.join(DATA_DIR, 'Default Data.csv')
    data = load_csv(default_data_path)

st.sidebar.markdown("--- ")
st.sidebar.markdown(f'### {_("download_sample_data")}')
with open(os.path.join(DATA_DIR, "Sample Data.csv"), "rb") as f:
    st.sidebar.download_button(_("download_sample_data"), f, file_name="Sample Data.csv")


# --- Feature Engineering ---
@st.cache_data
def feature_engineering(df):
    for col in ['temperature_c', 'vibration_mm_s', 'pressure_psi', 'rotational_speed_rpm', 'load_weight_tonnes']:
        df[f'{col}_rolling_avg_7d'] = df.groupby('machine_id')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    features_to_drop = [f for f in model_features if f not in df.columns]
    return df.dropna(subset=[f for f in model_features if f in df.columns], how='any')

data = feature_engineering(data)

# --- Helper function for safe encoding ---
def safe_encoder_transform(df, encoders_dict):
    df_transformed = df.copy()
    for col, encoder in encoders_dict.items():
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].astype(str)
            known_classes = set(encoder.classes_)
            unseen_labels = set(df_transformed[col]) - known_classes
            if unseen_labels:
                encoder.classes_ = np.append(encoder.classes_, list(unseen_labels))
            df_transformed[col] = encoder.transform(df_transformed[col])
    return df_transformed

# --- Tab Definitions ---
tab_keys = ["home_tab", "dashboard_tab", "prediction_tab", "stats_tab", "data_viewer_tab", "readme_tab"]
tabs = st.tabs([_(key) for key in tab_keys])
tab_home, tab_dashboard, tab_prediction, tab_stats, tab_data_viewer, tab_readme = tabs

# --- Home Tab ---
with tab_home:
    st.title(_("home_title"))
    st.image(IMAGE_OPERATING_PATH, use_container_width=True)
    st.header(_("project_summary_header"))
    st.markdown(f"""
    {_("problem_statement")}

    {_("solution_statement")}

    {_("impact_statement")}
    """)

# --- Dashboard Tab ---
with tab_dashboard:
    st.header(_("dashboard_header"))
    
    st.subheader(_("dashboard_filters_header"))
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        machine_type_options = data['machine_type'].unique()
        machine_type_filter = st.multiselect(_("machine_type_filter_label"), options=machine_type_options, default=machine_type_options)
    with filter_col2:
        manufacturer_options = data['manufacturer'].unique()
        manufacturer_filter = st.multiselect(_("manufacturer_filter_label"), options=manufacturer_options, default=manufacturer_options)

    age_min = int(data['age_years'].min())
    age_max = int(data['age_years'].max())
    if age_min < age_max:
        age_filter = st.slider(_("age_filter_label"), min_value=age_min, max_value=age_max, value=(age_min, age_max))
    else:
        age_filter = (age_min, age_max)

    dashboard_filtered_data = data[
        (data['machine_type'].isin(machine_type_filter)) &
        (data['manufacturer'].isin(manufacturer_filter)) &
        (data['age_years'] >= age_filter[0]) &
        (data['age_years'] <= age_filter[1])
    ]

    if not dashboard_filtered_data.empty:
        latest_data = dashboard_filtered_data.loc[dashboard_filtered_data.groupby('machine_id')['timestamp'].idxmax()].copy()
        X_latest_transformed = safe_encoder_transform(latest_data, encoders)
        
        latest_data['failure_probability'] = failure_model.predict_proba(X_latest_transformed[model_features])[:, 1]
        latest_data['severity'] = pd.cut(latest_data['failure_probability'], bins=[0.75, 0.85, 0.95, 1.0], labels=['Medium', 'High', 'Critical'], right=True, include_lowest=True)
        
        machines_at_risk = latest_data[latest_data['severity'].isin(['High', 'Critical'])]

        st.subheader(_("kpi_header"))
        total_machines = dashboard_filtered_data['machine_id'].nunique()
        num_at_risk = machines_at_risk.shape[0]
        num_healthy = total_machines - num_at_risk

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(_("total_machines_kpi"), total_machines)
        kpi2.metric(_("healthy_machines_kpi"), num_healthy)
        kpi3.metric(_("at_risk_kpi"), num_at_risk, delta_color="inverse")

        st.subheader(_("fleet_overview_header"))
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_machine_type = px.pie(latest_data, names='machine_type', title=_("machine_type_dist_title"))
            st.plotly_chart(fig_machine_type, use_container_width=True)
        with chart_col2:
            fig_age_dist = px.histogram(latest_data, x='age_years', nbins=15, title=_("machine_age_dist_title"))
            st.plotly_chart(fig_age_dist, use_container_width=True)
        
        st.subheader(_("failure_prob_dist_title"))
        fig_prob_dist = px.histogram(latest_data, x='failure_probability', nbins=20, title=_("failure_prob_dist_title"), color_discrete_sequence=['#C70039'])
        st.plotly_chart(fig_prob_dist, use_container_width=True)
    else:
        st.warning(_("no_data_warning"))

# --- Prediction Tab ---
with tab_prediction:
    st.header(_("prediction_header"))
    st.image(IMAGE_FAILED_PATH, use_container_width=True)
    latest_data_full = data.loc[data.groupby('machine_id')['timestamp'].idxmax()].copy()
    X_latest_full_transformed = safe_encoder_transform(latest_data_full, encoders)
    latest_data_full['failure_probability'] = failure_model.predict_proba(X_latest_full_transformed[model_features])[:, 1]
    machines_at_risk_full = latest_data_full[latest_data_full['failure_probability'] > 0.75].copy()

    st.subheader(_("machines_at_risk_header"))
    if not machines_at_risk_full.empty:
        machines_at_risk_full['severity'] = pd.cut(machines_at_risk_full['failure_probability'], bins=[0.75, 0.85, 0.95, 1.0], labels=['Medium', 'High', 'Critical'], right=True, include_lowest=True)
        st.dataframe(machines_at_risk_full[['machine_id', 'machine_type', 'model', 'failure_probability', 'severity']])

        st.subheader(_("predict_component_header"))
        machine_id_to_predict = st.selectbox(_("select_machine_prompt"), options=machines_at_risk_full['machine_id'].unique())

        if machine_id_to_predict:
            machine_data = machines_at_risk_full[machines_at_risk_full['machine_id'] == machine_id_to_predict]
            X_failure_transformed = safe_encoder_transform(machine_data, encoders)
            
            component_pred_encoded = component_model.predict(X_failure_transformed[model_features])
            component_pred = component_label_encoder.inverse_transform(component_pred_encoded)
            
            st.error(f'{_("predicted_component_error")} {component_pred[0]}', icon="🚨")

            severity = machine_data.iloc[0]['severity']
            if severity == 'Critical':
                st.warning(_("critical_severity_warning"), icon="⏳")
            elif severity == 'High':
                st.warning(_("high_severity_warning"), icon="⏳")
            else: # Medium
                st.info(_("medium_severity_info"), icon="🔍")
    else:
        st.success(_("no_risk_success"), icon="✅")

# --- Statistical Analysis Tab ---
with tab_stats:
    st.header(_("stats_header"))

    # Exclude datetime columns from analysis
    analysis_columns = data.select_dtypes(exclude=['datetime64[ns]']).columns

    stats_option = st.selectbox(_("select_analysis_type"), [_("univariate_analysis"), _("bivariate_analysis"), _("linear_regression")])

    if stats_option == _("univariate_analysis"):
        st.subheader(_("univariate_analysis"))
        # Set default selection to 'temperature_c' if it exists
        default_variable = 'temperature_c' if 'temperature_c' in analysis_columns else analysis_columns[0]
        variable_to_analyze = st.selectbox(_("select_variable_prompt"), analysis_columns, index=list(analysis_columns).index(default_variable))

        if data[variable_to_analyze].dtype == 'object' or data[variable_to_analyze].nunique() < 20:
            st.write(_("qualitative_analysis_header"))
            freq_table = data[variable_to_analyze].value_counts().reset_index()
            freq_table.columns = [variable_to_analyze, 'Frequency']
            freq_table['Proportion'] = freq_table['Frequency'] / len(data)
            st.write(freq_table)
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(freq_table, x=variable_to_analyze, y='Frequency', title=f'Bar Plot of {variable_to_analyze}')
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(freq_table, names=variable_to_analyze, values='Frequency', title=f'Pie Chart of {variable_to_analyze}')
                st.plotly_chart(fig_pie, use_container_width=True)

        elif np.issubdtype(data[variable_to_analyze].dtype, np.number):
            st.write(_("quantitative_analysis_header"))
            st.write(data[variable_to_analyze].describe())
            st.write(f"**Skewness:** {data[variable_to_analyze].skew():.4f}")
            st.write(f"**Kurtosis:** {data[variable_to_analyze].kurt():.4f}")
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(data, x=variable_to_analyze, title=f'Histogram of {variable_to_analyze}')
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_box = px.box(data, y=variable_to_analyze, title=f'Box Plot of {variable_to_analyze}')
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.subheader(_("normality_analysis_header"))
            qq_fig = sm.qqplot(data[variable_to_analyze].dropna(), line='s')
            st.pyplot(qq_fig)
            
            shapiro_test = stats.shapiro(data[variable_to_analyze].dropna())
            p_value = shapiro_test.pvalue
            p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
            st.write(f'**{_("shapiro_wilk_test")}:** Statistic={shapiro_test.statistic:.4f}, p-value={p_value_display}')
            if p_value > 0.05:
                st.success(_("normality_test_success"))
            else:
                st.warning(_("normality_test_warning"))

    elif stats_option == _("bivariate_analysis"):
        st.subheader(_("bivariate_analysis"))
        
        var1 = st.selectbox(_("select_first_variable"), analysis_columns, key="var1")
        var2 = st.selectbox(_("select_second_variable"), analysis_columns, key="var2")

        if var1 and var2:
            if np.issubdtype(data[var1].dtype, np.number) and np.issubdtype(data[var2].dtype, np.number):
                st.write(_("quant_vs_quant_header"))
                fig_scatter = px.scatter(data, x=var1, y=var2, title=f'Scatter Plot of {var1} vs {var2}')
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.write(f"**Pearson Correlation:** {data[var1].corr(data[var2], method='pearson'):.4f}")
                st.write(f"**Spearman Correlation:** {data[var1].corr(data[var2], method='spearman'):.4f}")
                st.write(f"**Kendall Correlation:** {data[var1].corr(data[var2], method='kendall'):.4f}")

            elif data[var1].dtype == 'object' and data[var2].dtype == 'object':
                st.write(_("qual_vs_qual_header"))
                contingency_table = pd.crosstab(data[var1], data[var2])
                st.write(contingency_table)
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f'**{_("chi_square_test")}:** Chi2={chi2:.4f}, p-value={p:.4f}')

            else:
                qual_var, quant_var = (var1, var2) if data[var1].dtype == 'object' else (var2, var1)
                st.write(_("qual_vs_quant_analysis_header", quant_var=quant_var, qual_var=qual_var))
                fig_box = px.box(data, x=qual_var, y=quant_var, title=f'Box Plot of {quant_var} by {qual_var}')
                st.plotly_chart(fig_box, use_container_width=True)
                groups = data[qual_var].unique()
                if len(groups) == 2:
                    group1 = data[data[qual_var] == groups[0]][quant_var]
                    group2 = data[data[qual_var] == groups[1]][quant_var]
                    ttest = stats.ttest_ind(group1, group2)
                    st.write(f'**{_("ttest")}:** Statistic={ttest.statistic:.4f}, p-value={ttest.pvalue:.4f}')
                elif len(groups) > 2:
                    f_val, p_val = stats.f_oneway(*[data[data[qual_var] == g][quant_var].dropna() for g in groups])
                    st.write(f'**{_("anova")}:** F-statistic={f_val:.4f}, p-value={p_val:.4f}')

    elif stats_option == _("linear_regression"):
        st.subheader(_("linear_regression_header"))
        
        quantitative_cols = [col for col in analysis_columns if np.issubdtype(data[col].dtype, np.number)]
        dependent_var = st.selectbox(_("select_dependent_variable"), quantitative_cols)
        independent_vars = st.multiselect(_("select_independent_variables"), analysis_columns)

        if dependent_var and independent_vars:
            try:
                X = data[independent_vars]
                y = data[dependent_var]
                X = sm.add_constant(X, has_constant='add')
                X = pd.get_dummies(X, drop_first=True)
                model = sm.OLS(y, X.astype(float)).fit()
                st.write(model.summary())
                with st.expander(f'**{_("regression_interpretation_header")}**'):
                    st.markdown(f"""- **R-squared (R²):** The model explains **{model.rsquared:.1%}** of the variance in `{dependent_var}`.
- **Adj. R-squared:** Adjusted for the number of predictors, the value is **{model.rsquared_adj:.1%}**.
- **F-statistic:** The F-test is statistically significant (Prob(F-statistic) = {model.f_pvalue:.3f}), suggesting that at least one independent variable is related to the dependent variable.
- **Coefficients (coef):** These represent the change in the dependent variable for a one-unit change in the independent variable, holding other variables constant.
- **P>|t|:** P-values less than 0.05 indicate a statistically significant relationship between the predictor and the outcome.
""")
                st.subheader(_("regression_assumptions_header"))
                st.write(f'**{_("linearity_check")}**')
                fig_linearity = px.scatter(x=model.fittedvalues, y=y, labels={'x': 'Predicted Values', 'y': 'Actual Values'}, title='Predicted vs. Actual Values')
                fig_linearity.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red"))
                st.plotly_chart(fig_linearity, use_container_width=True)
                st.write(f'**{_("homoscedasticity_check")}**')
                fig_homo = px.scatter(x=model.fittedvalues, y=model.resid, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residuals vs. Predicted Values')
                fig_homo.add_hline(y=0, line_color="Red")
                st.plotly_chart(fig_homo, use_container_width=True)
                st.write(f'**{_("normality_of_residuals_check")}**')
                qq_fig = sm.qqplot(model.resid, line='s')
                st.pyplot(qq_fig)
            except Exception as e:
                st.error(f"An error occurred during regression analysis: {e}")

# --- Data Viewer Tab ---
with tab_data_viewer:
    st.header(_("data_viewer_header"))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(_("dataset_overview_header"))
        st.markdown(f"""- **Source:** {'Uploaded File' if uploaded_file else 'Default Sample Data'}
- **Total Records:** {len(data)}
- **Unique Machines:** {data['machine_id'].nunique()}
- **Date Range:** {data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}
""")
    with col2:
        st.subheader(_("model_performance_header"))
        st.markdown("""- **Failure Prediction Model Accuracy:** 98.6%
- **Component Prediction Model Accuracy:** 83.3%
*Note: Metrics are based on a held-out test set during training.*
""")

    st.subheader(_("data_dictionary_header"))
    st.markdown("""- **timestamp:** The date and time of the data recording.
- **machine_id:** A unique identifier for each machine.
- **machine_type, model, manufacturer, age_years:** Details about the machine.
- **temperature_c, vibration_mm_s, etc.:** Sensor readings from the machine.
""")

    st.header(_("raw_data_explorer_header"))
    st.dataframe(data.head(200))

# --- Read Me Tab ---
with tab_readme:
    st.title(_("readme_title"))
    st.markdown("""
    ### A Proactive Approach to Industrial Maintenance

    This application is more than just a tool; it's a demonstration of a forward-thinking approach to industrial maintenance. Developed by a passionate data scientist, this project showcases how machine learning can be harnessed to prevent costly equipment failures in sectors like mining and manufacturing. By shifting from a reactive to a predictive maintenance strategy, businesses can significantly reduce downtime, optimize resource allocation, and enhance operational safety.

    ### From Raw Data to Actionable Insights: The Technology Stack

    **1. Interactive Front-End (Streamlit):**
    The intuitive user interface is crafted with **Streamlit**, a powerful Python library for creating dynamic, data-driven web applications. This allows for a seamless and interactive user experience, making complex data accessible to a non-technical audience.

    **2. Robust Back-End (Python & Machine Learning):**
    The application's core is a sophisticated back-end that handles everything from data ingestion to predictive modeling:

    *   **Data Manipulation & Analysis (Pandas, NumPy):** The application leverages the power of **Pandas** and **NumPy** for efficient data loading, cleaning, and transformation.
    *   **Statistical Insights (SciPy, Statsmodels, Seaborn):** The "Statistical Analysis" tab provides a comprehensive suite of tools for in-depth data exploration, utilizing libraries like **SciPy**, **Statsmodels**, and **Seaborn** to uncover hidden patterns and correlations.
    *   **Predictive Modeling (Scikit-learn):** The heart of the application lies in its machine learning models, built with the industry-standard **Scikit-learn** library. These models are trained to identify subtle anomalies in sensor data that are indicative of impending equipment failure.

    ### Business Impact & Strategic Value

    This project is a tangible example of how data science can drive significant business value:

    *   **Increased ROI:** By minimizing unplanned downtime and reducing the need for emergency repairs, predictive maintenance can lead to a substantial return on investment.
    *   **Enhanced Safety:** Proactively addressing potential equipment failures can create a safer working environment for all personnel.
    *   **Data-Driven Decision Making:** This tool empowers maintenance teams to make informed, data-driven decisions, moving away from guesswork and intuition.

    ### Core Competencies Demonstrated

    This project is a testament to a diverse and in-demand skill set:

    *   **Advanced Programming (Python):** Expertise in Python and its data science ecosystem (Pandas, NumPy, Scikit-learn, Streamlit).
    *   **End-to-End Machine Learning:** Proficiency in the complete machine learning lifecycle, from data acquisition and feature engineering to model training, evaluation, and deployment.
    *   **Comprehensive Statistical Analysis:** A strong foundation in both descriptive and inferential statistics, enabling a deeper understanding of the underlying data.
    *   **Effective Data Visualization:** The ability to create compelling and informative data visualizations using libraries like **Plotly**, **Seaborn**, and **Matplotlib**.
    *   **Modern Software Development Practices:** A solid understanding of application design, development, and deployment in a real-world context.

    --- 

    **Contact Information:**

    - **Email:** [abessoloxavier45@gmail.com](mailto:abessoloxavier45@gmail.com)
    - **WhatsApp:** [+233592308335](https://wa.me/233592308335)
    - **YouTube:** [http://www.youtube.com/@abhasabessolo1](http://www.youtube.com/@abhasabessolo1)
    - **GitHub:** [https://github.com/dataprofessionals237](https://github.com/dataprofessionals237)
    """)
