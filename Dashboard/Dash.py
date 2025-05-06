import dash
import json
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np

from braket.ahs.atom_arrangement import AtomArrangement
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from src.aitor_utils import Graph_gen, get_blockade_configurations2, C_from_gr, gaussian


# Crear la aplicación Dash
app = dash.Dash(__name__)

scale = 5e-6
df = pd.read_csv('df_1000_semillas.csv')
clasif = pd.read_csv('drivings_opt_1000_seeds_hp.csv')

b_value = clasif['b'].unique()[15]
curve_value = 'linear'
simul = clasif[(clasif['curva'] == curve_value) & (clasif['b'] == b_value)]
# simul = pd.read_csv('../Estudio opt por grafo/opt_por_seed.csv')
simul2 = simul


# Cargar los espectros de energía
with open("espectros.json", "r") as archivo:
    espectros_dic = json.load(archivo)
with open("e_min.json", "r") as archivo:
    e_min_dic = json.load(archivo)


# Crear algunas figuras de ejemplo
fig1 = px.line(x=[1, 2, 3], y=[3, 1, 6])
fig2 = px.scatter(x=[1, 2, 3], y=[6, 2, 4])
fig3 = px.bar(x=["A", "B", "C"], y=[4, 7, 3])

# Función para generar el grafo como una imagen PNG
def generate_graph_image(seed):
    n_nodos = df[df['seed'] == int(seed)]['n_nodos'].values[0]
    atoms_w = df[df['seed'] == int(seed)]['atoms_w'].values[0]
    atoms_l = df[df['seed'] == int(seed)]['atoms_l'].values[0]

    np.random.seed(int(seed))
    G, register = Graph_gen(n_nodos, atoms_w, atoms_l, scale)

    # Dibujar el grafo usando NetworkX y matplotlib
    pos = nx.get_node_attributes(G, 'pos')
    weight = nx.get_node_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=[v * 5000 for v in weight.values()], node_color='skyblue')
    plt.title('NetworkX Graph')

    # Guardar la figura como imagen en un buffer de memoria
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Convertir la imagen a base64 para usarla en Dash
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import networkx as nx



# Layout de la aplicación
app.layout = html.Div(children=[
    # Contenedor de las gráficas superiores
    html.Div(children=[
        html.Div(children=[dcc.Graph(id='scatter')], style={"width": "50%", "height": "100%"}),
        html.Div(children=[dcc.Graph(id='espectro')], style={"width": "50%", "height": "100%"}),
    ], style={"display": "flex", "width": "100%", "height": "40vh"}),  # Altura de la fila superior
    
    # Contenedor de la gráfica inferior (alineada a la derecha)
    html.Div(children=[
        html.Div([ 
            html.Div([
                'HP: ',
                dcc.Slider(min=1,
                           max=10,
                           step=0.05,
                           value=1.85,
                           marks=None,
                           tooltip={"placement": "bottom", "always_visible": True},
                           id='hp_inv_slider'),
            ], style={"width": "40%", "marginLeft": "10%"}), 
        ],style={"width": "50%", 'marginTop': '100px'}  # Input para la semilla
        ),
        html.Div(children=[dcc.Graph(id='fig_grafo')], style={"width": "50%", "height": "80%", 'marginTop': '100px'})  # Gráfica a la derecha
    ], style={"display": "flex", "width": "100%", "height": "50vh", 'marginTop': '130px'})  # Altura de la fila inferior
])


@callback(
    Output('espectro', 'figure'),
    Input('scatter', 'clickData'))
def espectro(clickeado, b_value=b_value, curva=curve_value):
    if clickeado is None:
        seed=7
    else:
        idx = clickeado['points'][0]['pointIndex']
        # print(clickeado['points'][0]['customdata'])
        seed = simul2.iloc[idx]['seed']

    espectro = espectros_dic[str(seed)]
    df = pd.DataFrame(espectro, columns=['energia'])

    e_min = e_min_dic[str(seed)]

    # Crear figura con subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0)

    # **Subplot 1: Histograma de energías**
    hist_trace = go.Histogram(x=df['energia'], xbins=dict(size=0.031), 
                              marker=dict(color='blue', opacity=0.7),
                              name='Histograma de Energía',
                              showlegend=False)
    fig.add_trace(hist_trace, row=1, col=1)

    # Línea de la Gaussiana en el histograma
    xx = np.linspace(e_min, 0, 100)
    gauss_trace = go.Scatter(x=xx, y=gaussian(xx, e_min, b_value, met=curve_value), 
                             mode='lines', line=dict(color='red'), name=f'Gaussiana, b={b_value:.4f}')
    fig.add_trace(gauss_trace, row=1, col=1)

    # Subplot 2: Espectro de energía con líneas verticales
    for energy in df['energia']:
        fig.add_trace(
            go.Scatter(
                x=[energy, energy],  # Línea vertical
                y=[0, 1],  
                mode='lines',
                line=dict(color='blue', width=1),
                opacity=0.5,  # Aquí es donde se establece la opacidad correctamente
                showlegend=False
            ),
            row=2, col=1
        )
    # Línea horizontal en el espectro desde e_min hasta e_min + 0.1
    fig.add_trace(
        go.Scatter(
            x=[e_min, e_min + 0.1],  
            y=[1, 1],  
            mode='lines+markers',
            line=dict(color='black', width=2),  
            marker=dict(color='black', size=6),
            name='Referencia'
        ),
        row=2, col=1
    )
    # Línea de la Gaussiana en el espectro
    gauss_trace_2 = go.Scatter(x=xx, y=gaussian(xx, e_min, b_value, met=curve_value),
                               mode='lines', line=dict(color='red'),
                               name=f'Gaussiana, b={b_value:.4f}',
                               showlegend=False)
    fig.add_trace(gauss_trace_2, row=2, col=1)

    # Ajustes de layout
    fig.update_layout(
        height=500, width=800, title_text=f'Análisis de espectro (Semilla: {seed})',
        showlegend=True, template='plotly_white'
    )

    # Quitar grid de la figura de abajo
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)

    # Ajustes de ejes
    fig.update_xaxes(title_text='Coste', row=2, col=1)
    fig.update_yaxes(title_text='Nº de estados', row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)  # Oculta etiquetas en el eje y del espectro

    return fig


@callback(
    Output('scatter', 'figure'),
    Input('hp_inv_slider', 'value'))
def update_figure_scatter(umbral):
    global simul2
    simul2 = simul[simul['hp'] > umbral]

    ar_gap = px.scatter(simul2, 
                 x='e_gap', 
                 y='AR_max', 
                 color='hp',
                 hover_data={'seed': True, 'ar': True, 'e_gap': True, 'succ': True, 'hp_inv': False},
                 custom_data=['seed'], 
                 color_continuous_scale='rainbow',
                 labels={'ar': 'AR', 'e_gap': 'Gap', 'hp': 'HP', 'seed': 'Semilla', 'succ': 'SP'},)
    ar_gap.update_yaxes(range=[0.86, 1.005], row=1, col=1)
    ar_gap.update_xaxes(range=[-0.005, 0.1], row=1, col=1)
    ar_gap.update_layout(clickmode='event')
    
    return ar_gap


# Callback para actualizar el grafo cuando cambie la semilla
@callback(
    Output('fig_grafo', 'figure'),
    Input('scatter', 'clickData'))
def update_figure_graph(clickeado):
    if clickeado is None:
        seed=7
    else:
        idx = clickeado['points'][0]['pointIndex']
        seed = simul2.iloc[idx]['seed']
    
    graph_image = generate_graph_image(seed)

    # Crear la figura para mostrar en Dash
    return {
        'data': [],
        'layout': {
            'images': [{
                'source': graph_image,
                'xref': 'paper', 'yref': 'paper',
                'x': 0, 'y': 1.3,
                'sizex': 1, 'sizey': 1,
                'sizing': 'contain',
                'opacity': 1,
                'layer': 'below'
            }],
            'title': f'NetworkX Graph (Semilla: {seed})',  # Título con la semilla
            'titlefont_size': 16,
            'showlegend': False,
            'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
            'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
        }
    }




# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

