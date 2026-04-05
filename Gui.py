# boston_dash_gui_advanced.py
import pandas as pd
import pickle
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# Load dataset and model
data = pd.read_csv(r"D:\Gemy Study\DEPI\Machine learning DEPI round_4\TEC\Amit-1\src\ML\machine-learning\Project_1\p1\housing.csv")
with open(r"D:\Gemy Study\DEPI\Machine learning DEPI round_4\TEC\Amit-1\src\ML\machine-learning\Project_1\p1\boston_housing_model.pkl", "rb") as f:
    model = pickle.load(f)

TOTAL_HOUSES = len(data)

# KPIs function
def calc_kpis(filtered_df):
    avg_price = filtered_df['MEDV'].mean() if not filtered_df.empty else 0
    avg_rm = filtered_df['RM'].mean() if not filtered_df.empty else 0
    avg_lstat = filtered_df['LSTAT'].mean() if not filtered_df.empty else 0
    return avg_price, avg_rm, avg_lstat

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Boston Housing Dashboard"

# Sidebar filters
sidebar = html.Div([
    html.H4("Filters", style={"textAlign":"center", "marginBottom":"20px"}),

    dbc.Card([
        dbc.CardHeader("RM Range"),
        dbc.CardBody([
            dcc.RangeSlider(
                id='rm_filter',
                min=data['RM'].min(),
                max=data['RM'].max(),
                step=0.1,
                value=[data['RM'].min(), data['RM'].max()],
                marks={round(i,1): str(round(i,1)) for i in data['RM'].quantile([0,0.25,0.5,0.75,1])}
            )
        ])
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("LSTAT Range"),
        dbc.CardBody([
            dcc.RangeSlider(
                id='lstat_filter',
                min=data['LSTAT'].min(),
                max=data['LSTAT'].max(),
                step=0.1,
                value=[data['LSTAT'].min(), data['LSTAT'].max()],
                marks={round(i,1): str(round(i,1)) for i in data['LSTAT'].quantile([0,0.25,0.5,0.75,1])}
            )
        ])
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("PTRATIO Range"),
        dbc.CardBody([
            dcc.RangeSlider(
                id='ptratio_filter',
                min=data['PTRATIO'].min(),
                max=data['PTRATIO'].max(),
                step=0.1,
                value=[data['PTRATIO'].min(), data['PTRATIO'].max()],
                marks={round(i,1): str(round(i,1)) for i in data['PTRATIO'].quantile([0,0.25,0.5,0.75,1])}
            )
        ])
    ], className="mb-3"),

], style={"width":"20%", "float":"left", "padding":"20px", "height":"100vh", "backgroundColor":"#1e1e2f"})

# Main Content
content = dbc.Container([
    html.H1("🏠 Boston Housing Dashboard", style={'textAlign':'center', 'marginBottom':'20px'}),

    dbc.Row(id="kpi-cards", className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='rm_vs_medv'), width=6),
        dbc.Col(dcc.Graph(id='lstat_vs_medv'), width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='ptratio_hist'), width=12),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='scatter_matrix'), width=12)
    ], className="mb-4"),

    html.H2("Predict House Price", style={"marginTop":"30px"}),

    dbc.Row([
        dbc.Col([
            html.Label("Total Rooms (RM):"),
            dcc.Input(id="input_rm", type="number", value=5, min=1),
            html.Br(),
            html.Label("Neighborhood Poverty Level (LSTAT %):"),
            dcc.Input(id="input_lstat", type="number", value=20, min=0),
            html.Br(),
            html.Label("Student-Teacher Ratio (PTRATIO):"),
            dcc.Input(id="input_ptratio", type="number", value=15, min=0),
            html.Br(),
            html.Button("Predict Price", id="predict_btn", style={"marginTop":"10px"}),
            html.H3(id="prediction_output", style={"marginTop":"20px"})
        ], width=6)
    ])
], style={"width":"75%", "float":"right"})

app.layout = html.Div([sidebar, content])

# Callback
@app.callback(
    [Output('kpi-cards','children'),
     Output('rm_vs_medv','figure'),
     Output('lstat_vs_medv','figure'),
     Output('ptratio_hist','figure'),
     Output('scatter_matrix','figure'),
     Output("prediction_output", "children")],
    [Input('rm_filter','value'),
     Input('lstat_filter','value'),
     Input('ptratio_filter','value'),
     Input("predict_btn", "n_clicks"),
     Input("input_rm", "value"),
     Input("input_lstat", "value"),
     Input("input_ptratio", "value")]
)
def update_dashboard(rm_range, lstat_range, ptratio_range, n_clicks, rm, lstat, ptratio):
    # Filter data
    filtered_df = data[
        data['RM'].between(rm_range[0], rm_range[1]) &
        data['LSTAT'].between(lstat_range[0], lstat_range[1]) &
        data['PTRATIO'].between(ptratio_range[0], ptratio_range[1])
    ]

    # KPIs
    avg_price, avg_rm, avg_lstat = calc_kpis(filtered_df)
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([html.H5("Total Houses"), html.H2(f"{TOTAL_HOUSES:,}")],
                         body=True, color="primary", inverse=True), width=4),
        dbc.Col(dbc.Card([html.H5("Avg Price ($1000s)"), html.H2(f"{avg_price:.2f}")],
                         body=True, color="info", inverse=True), width=4),
        dbc.Col(dbc.Card([html.H5("Avg Rooms"), html.H2(f"{avg_rm:.2f}")],
                         body=True, color="success", inverse=True), width=4),
    ])

    # Charts
    rm_vs_medv = px.scatter(filtered_df, x='RM', y='MEDV', trendline='ols',
                            title="Average Rooms vs Median Price", color='RM', color_continuous_scale=px.colors.sequential.Viridis)

    lstat_vs_medv = px.scatter(filtered_df, x='LSTAT', y='MEDV', trendline='ols',
                               title="% Lower Status vs Median Price", color='LSTAT', color_continuous_scale=px.colors.sequential.Plasma)

    ptratio_hist = px.histogram(filtered_df, x='PTRATIO', nbins=20, color_discrete_sequence=['#EF553B'],
                                title="Pupil-Teacher Ratio Distribution")

    scatter_matrix = px.scatter_matrix(filtered_df, dimensions=['RM','LSTAT','PTRATIO','MEDV'],
                                       color='MEDV', color_continuous_scale=px.colors.sequential.Plasma,
                                       title="Scatter Matrix of Features")

    # Prediction
    prediction_text = ""
    if n_clicks:
        features = [[rm, lstat, ptratio]]
        price_pred = model.predict(features)[0]
        prediction_text = f"Predicted House Price: ${price_pred:,.2f}"

    return kpi_cards, rm_vs_medv, lstat_vs_medv, ptratio_hist, scatter_matrix, prediction_text

if __name__ == "__main__":
    app.run(debug=True)