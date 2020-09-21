
import pandas as pd
import apyori as ap
from apyori import apriori 
import mlxtend as ml
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv("./exportdata.csv")
def apriori_mining(data,min_support,min_confidence,min_lift):
    df_new = pd.get_dummies(df)
    rules= apriori(df_new,min_support,use_colnames=True)
    frequent_itemsets = apriori(df_new, min_support, use_colnames=True)
    frequent_itemsets.sort_values(by='support',ascending=False).head(10)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)
    final_rules = rules[(rules['lift']>min_lift) & (rules['confidence'] > min_confidence)]
    final_rules_top = final_rules[["Attrition_No" in list(x) for x in final_rules['consequents']]]
    return final_rules

def table_frame(dataframe):
    return html.Table(
        style={
        'display': 'block',
        'height': '500px',      
        'overflow-y': 'auto',  
        'overflow-x': 'auto',
        },
        children=
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(0,10)]
    )

app.layout = html.Div(
    style={'backgroundColor': colors['background'],
    'padding': '1em' },
    children=[
    html.H2(children='Employee Attrition Prediction',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    html.H3(children='Processed Dataset',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    html.Div( style={
            'textAlign': 'center',
            'color': '#ff8000'
        },
        id='top_rules',children = [table_frame(df)]),
     html.Div(children=[
        html.H3(
            style={
            'textAlign': 'center',
            'color': colors['text']
        },
        children="Association Rules Mining"),
        html.Label(        
            style={
            'color': colors['text']
            },children='Attrition'),
        dcc.Dropdown(
                    style={
            'width':'13em',
            'color': colors['text']
        },
                id='attrition',
                options=[{'label': 'Yes', 'value': 'Attrition=Yes'},
                         {'label': 'No', 'value': 'Attrition=No'},
                        ],
                value='Attrition=No'
        ),
        html.Label(        
            style={
            'color': colors['text']
            },children='Support'),
            dcc.Input(
                id='min_support',
                type='number',
                step=0.01,
                value=0.6
            ),
        html.Label(            
            style={
            'color': colors['text']
            },children='Confidence:'),
            dcc.Input(
                id='min_confidence',
                type='number',
                step=0.01,
                value=0.81
            ),
        html.Label(            
            style={
            'color': colors['text']
            },children='Lift'),
            dcc.Input(
                id='min_lift',
                type='number',
                step=0.01,
                value=1.1
            ),
            html.Div(id='rules'),
            html.H3(
                style={
                'textAlign': 'center',
                'color': colors['text']
                },
                children='Scatter Plot Visualization of Employee Attrition'),
            html.Div([dcc.Graph(id='graphdisplay')])
            ])
        ])

@app.callback(
    Output('graphdisplay','figure'),
    [Input('attrition', 'value'), Input('min_support', 'value'),Input('min_confidence','value'),Input('min_lift','value')]
    )
def update_graph(att, x,y,z):
    a_rules=apriori_mining(df, min_support=x, min_confidence=y, min_lift=z)
    ui={
        'data': [go.Scatter(
            x=a_rules['support'],
            y=a_rules['confidence'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'color':a_rules['lift'],
                'colorscale':'Plasma',
                'line': {'width': 0.5, 'color': 'white'},
                'showscale':True
            }
        )],
        'layout': {
            'xaxis':{
                'title': 'support',
                'type': 'linear'
            },
            'yaxis':{
                'title': 'confidence',
                'type': 'linear'
            },
            'margin':{'l': 50, 'b': 50, 't': 20, 'r': 0},
            'hovermode':'closest'
        }
    }
    return ui

#Restart server
if __name__ == '__main__':
    app.run_server(debug=True)