import networkx as nx
from rdflib import Graph, URIRef, Literal, BNode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys


def parse_turtle_file(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g


def get_label(entity):
    if isinstance(entity, URIRef):
        return str(entity).split("#")[-1]
    elif isinstance(entity, BNode):
        return f"_:{entity}"
    elif isinstance(entity, Literal):
        return f'"{entity}"'
    return str(entity)


def visualize_rdf_interactive(g):
    G = nx.Graph()

    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_labels = []
    edge_labels = []

    for s, p, o in g:
        s_label = get_label(s)
        o_label = get_label(o)
        p_label = get_label(p)

        G.add_edge(s_label, o_label)
        edge_labels.append(p_label)

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title='Interactive RDF Graph Visualization',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    # Add edge labels
    edge_label_trace = go.Scatter(
        x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
        y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
        mode='text',
        text=edge_labels,
        textposition='top center',
        textfont=dict(size=8),
        hoverinfo='none'
    )
    fig.add_trace(edge_label_trace)

    # Add custom JavaScript for dragging nodes
    fig.update_layout(
        dragmode='select',
        newshape=dict(line_color='black'),
        # Add other layout options here
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{'dragmode': 'select'}],
                        label="Drag",
                        method="relayout"
                    ),
                    dict(
                        args=[{'dragmode': 'pan'}],
                        label="Pan",
                        method="relayout"
                    ),
                    dict(
                        args=[{'dragmode': 'zoom'}],
                        label="Zoom",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Custom JavaScript for node dragging
    fig.update_layout(
        clickmode='event+select',
        dragmode='select',
        hovermode='closest',
    )

    fig.update_layout(
        shapes=[
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=x - 0.05,
                y0=y - 0.05,
                x1=x + 0.05,
                y1=y + 0.05,
                line_color="LightSeaGreen",
            ) for x, y in zip(node_x, node_y)
        ]
    )

    fig.update_layout(
        newshape=dict(line_color="cyan")
    )

    # Add JavaScript code for dragging
    fig.add_annotation(
        text='Drag nodes to reposition',
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        showarrow=False,
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset",
                        method="relayout",
                        args=[{"shapes": []}]
                    )
                ]
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    file_path = "C:\\Users\\cdri\\OneDrive - Buildwise\\Documents\\Work\\FireBIM\\SSoLDAC\\LDAC_Hackathon\\LDAC_Hackathon\\output\\wallsTest.ttl"

    g = parse_turtle_file(file_path)
    visualize_rdf_interactive(g)