import networkx as nx
import numpy as np
import matplotlib.pylab as plt
import itertools
from braket.ahs.atom_arrangement import AtomArrangement

import gurobipy as gp
from gurobipy import GRB
import seaborn as sns


def Graph_gen(n_nodos, atoms_w, atoms_l, scale):
    '''Genera un grafo en una cuadrícula de anchura atoms_w y altura atoms_l,
       conectando cada nodo con sus primeros vecinos y nodos diagonales, y luego elimina
       nodos aleatorios hasta que el número de nodos sea n_nodos.
    
    Args:
        n_nodos (int): Número final de nodos en el grafo.
        atoms_w (int): Anchura de la cuadrícula.
        atoms_l (int): Altura de la cuadrícula.
        scale (int): Escala de la cuadrícula para dibujar el grafo.
    
    Returns:
        G (Graph): Grafo de NetworkX con nodos conectados a sus primeros vecinos y nodos diagonales.
        register (AtomArrangement): Arrangement de átomos en la cuadrícula.
    '''
    # Crear el grafo en una cuadrícula 2D
    G = nx.grid_2d_graph(atoms_w, atoms_l)
    
    # Conectar nodos diagonales
    for x, y in list(G.nodes()):
        diagonales = [
            (x + 1, y + 1), (x + 1, y - 1), 
            (x - 1, y + 1), (x - 1, y - 1)
        ]
        for (nx_d, ny_d) in diagonales:
            if (nx_d, ny_d) in G:
                G.add_edge((x, y), (nx_d, ny_d))
    

    # Eliminar nodos aleatoriamente hasta que el grafo tenga n_nodos nodos
    indices = list(range(len(G)))
    nodes_to_remove = np.random.choice(indices, size=len(G) - n_nodos, replace=False)
    nodes_to_remove = [list(G.nodes())[i] for i in nodes_to_remove]
    G.remove_nodes_from(nodes_to_remove)

    # Asignar posiciones para dibujar el grafo en formato de cuadrícula
    pos = {}
    for i in range(len(G.nodes())):
        pos[i] = list(G.nodes())[i]
    pos = {k: (v[0] * scale, v[1] * scale) for k, v in pos.items()}
    
    # Guardar las posiciones como atributo de los nodos
    G = nx.convert_node_labels_to_integers(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # Generamos los pesos y lo asignamos a los nodos
    weights = np.round(np.random.rand(n_nodos),4)
    for node in G.nodes():
        G.nodes[node]['weight'] = weights[node]
    
    # Generamos el AtomArrangement
    pos_x = [list(pos.values())[i][0] for i in range(len(pos))]
    pos_y = [list(pos.values())[i][1] for i in range(len(pos))]

    register = AtomArrangement()
    for i in range(len(pos)):
        register.add([pos_x[i], pos_y[i]])
    
    return G, register


def get_blockade_configurations2(lattice: AtomArrangement, blockade_radius: float):
    atoms_coordinates = lattice
    min_separation = float("inf")  # The minimum separation between atoms, or filled sites
    for i, atom_coord in enumerate(atoms_coordinates[:-1]):
        dists = np.linalg.norm(atom_coord - atoms_coordinates[i + 1 :], axis=1)
        min_separation = min(min_separation, min(dists))

    configurations = [
        "".join(item) for item in itertools.product(["g", "r"], repeat=len(atoms_coordinates))
    ]

    if blockade_radius < min_separation:  # no need to consider blockade approximation
        return configurations
    return [
        config
        for config in configurations
        if validate_config(config, atoms_coordinates, blockade_radius)
    ]

def validate_config(config: str, atoms_coordinates: np.ndarray, blockade_radius: float) -> bool:
    """Valid if a given configuration complies with the Rydberg approximation

    Args:
        config (str): The configuration to be validated
        atoms_coordinates (ndarray): The coordinates for atoms in the filled sites
        blockade_radius (float): The Rydberg blockade radius

    Returns:
        bool: True if the configuration complies with the Rydberg approximation,
        False otherwise
    """

    # The indices for the Rydberg atoms in the configuration
    rydberg_atoms = [i for i, item in enumerate(config) if item == "r"]

    for i, rydberg_atom in enumerate(rydberg_atoms[:-1]):
        dists = np.linalg.norm(
            atoms_coordinates[rydberg_atom] - atoms_coordinates[rydberg_atoms[i + 1 :]], axis=1
        )
        if min(dists) <= blockade_radius:
            return False
    return True

def C_from_gr(gr_sol, weights, edges):
    node_sol = []
    for i in range(len(gr_sol)):
        if gr_sol[i] == 'r':
            node_sol.append(i)
    C = calculate_C_function(node_sol, weights, edges)
    return C

def calculate_C_function(solution, weights, edges):
    """
    Calcula el valor de la función C para una solución dada.
    Args:
        solution (list): Lista de nodos en la solución.
        weights (np.array): Pesos de los nodos.
        edges (list): Lista de aristas.
    Returns:
        float: Valor de la función C.
    """
    interactions = list(itertools.combinations(solution, 2))
    Int_term = 0
    for int in interactions:
        if int in edges:     
            Int_term += (1+weights[int[0]])*(1+weights[int[1]])
    
    C =  - np.sum([weights[i] for i in solution]) + Int_term
    return float(C)

def Gurobi_to_gr(sol, N):
    """
    Convierte un vector de solución de Gurobi a un string de nodos en formato gr.
    Args:
        sol (list): Lista de nodos en la solución.
        N (int): Número de nodos.
    Returns:
        str: String de nodos en formato gr.
    """ 
    state = []
    for n in range(N):
        state.append('g')
    for index in sol:
        state[index] = 'r'
    state = "".join(state)
    return state

def gaussian(E: float, a: float, b: float=0.5, met:str='linear'):
    # Convertir FWHM (b) a desviación estándar (sigma)
    sigma = b / (2 * np.sqrt(2 * np.log(2)))
    if met == 'linear':
        delta = E - a
    elif met == 'sqrt':
        delta = np.sqrt(E - a)
    elif met == 'square':
        delta = (E - a)**2
    elif met == 'cubic':
        delta = (E - a)**3
    elif met == 'cubic_root':
        delta = (E - a)**(1/3)
    else:
        print(f'Método {met} no válido -> metodo por defecto: linear')
        delta = E - a
    # Calcular el valor de la gaussiana
    return np.exp(-((delta) ** 2) / (2 * sigma ** 2))

def Gurobi_solver(G):
    #OPTIMAL COST FUNCTION
    weights = nx.get_node_attributes(G, 'weight')
    # Crear un modelo de Gurobi
    model = gp.Model("MaximumWeightedIndependentSet")
    
    # Crear variables de decisión
    vars = {}
    for i in G.nodes:
        vars[i] = model.addVar(vtype=GRB.BINARY, obj=G.nodes[i]['weight'], name=f"node_{i}")
    
    # Agregar restricciones para que no haya dos nodos adyacentes en el conjunto independiente
    for i, j in G.edges:
        model.addConstr(vars[i] + vars[j] <= 1)
    
    # Definir que queremos maximizar la suma de los pesos
    model.modelSense = GRB.MAXIMIZE
    model.setParam('OutputFlag', 0)
    # Optimizar el modelo
    model.optimize()
    
    # En el caso de haber encontrado una solución óptima
    if model.status == GRB.OPTIMAL:
        sol = [i for i in G.nodes if vars[i].Xn > 0.5]
        C_opt = calculate_C_function(sol, weights, G.edges())
    else:
        print("No optimal solution found")

    return C_opt, sol

def plot_espectro(df, binwidth=None):
    """
    Muestra el espectro de energía de un grafo.
    Args:
        df (DataFrame): DataFrame con las columnas 'bitstring' y 'energia'.
    """
    # Calcular el gap y energía mínima
    e_gap = df['energia'].values[1] - df['energia'].values[0]
    e_min = df['energia'].values[0]
    n_nodos = len(df['bitstring'].values[0])

    # Definir el binwidth
    if binwidth is None:
        binwidth = e_gap * 0.99

    # Mostrar distribución de energías

    # Parámetro para regular la altura relativa del segundo plot
    height_ratio = 0.3  # Ajusta este valor para cambiar la proporción

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, height_ratio]}, sharex=True)

    # Histograma de distribución de energías
    sns.histplot(df['energia'], binwidth=e_gap * 0.99, ax=axes[0])
    axes[0].set_title(f'Distribución de coste de configuraciones para grafos de {n_nodos} nodos', fontsize=14)
    axes[0].set_ylabel('Frecuencia', fontsize=14)

    # Superponer la curva gaussiana
    xx = np.linspace(e_min, 0, 100)
    axes[0].plot(xx, gaussian(xx, e_min, 0.5, met='linear'), 'r-', label='Gaussian')

    # Espectro de energía
    for energy in df['energia'].values:
        axes[1].vlines(energy, 0, 1, color='blue', alpha=0.4)

    # Curva gaussiana en el espectro
    axes[1].plot(xx, gaussian(xx, e_min, 0.5, met='linear'), 'r-', label='Gaussian, b=0.5')

    # Asegurar que ambos gráficos tienen los mismos ticks en el eje X
    axes[1].set_xticks(axes[0].get_xticks())

    axes[1].legend(fontsize=14)
    axes[1].set_xlabel('Coste', fontsize=14)
    axes[1].set_yticks([])  # Ocultar el eje Y del espectro

    axes[0].tick_params(axis='both', labelsize=14)
    axes[1].tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()

def plot_sp_ar(df):
    """
    Muestra un gráfico de dispersión de la probabilidad de éxito y el ratio de aproximación.
    Args:
        df (DataFrame): DataFrame con las columnas 'e_gap', 'succ' y 'ar'.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))  # 1 fila, 2 columnas

    # Primer gráfico en el primer eje
    sns.scatterplot(data=df, x='e_gap', y='succ', s=20, alpha=0.8, ax=axes[0])

    # Segundo gráfico en el segundo eje
    sns.scatterplot(data=df, x='e_gap', y='ar', s=20, alpha=0.8, ax=axes[1])
    # Ajustar layout
    plt.subplots_adjust(wspace=0.25) 
    axes[0].set_xlabel('Gap', fontsize=14, style='italic')
    axes[0].set_ylabel('SP', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('Gap', fontsize=14, style='italic')
    axes[1].set_ylabel('AR', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=14)
    print(axes[1].get_xticks())
    print(axes[0].get_xticks())
    axes[1].set_ylim(0.885, 1.005)
    plt.show()