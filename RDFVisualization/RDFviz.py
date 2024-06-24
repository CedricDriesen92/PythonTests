import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, Literal, BNode
import re


def parse_turtle(turtle_string):
    g = Graph()
    g.parse(data=turtle_string, format="turtle")
    return g


def get_label(entity):
    if isinstance(entity, URIRef):
        return str(entity).split("#")[-1]
    elif isinstance(entity, BNode):
        return f"_:{entity}"
    elif isinstance(entity, Literal):
        return f'"{entity}"'
    return str(entity)


def visualize_rdf(g):
    G = nx.Graph()

    edge_labels = {}
    for s, p, o in g:
        s_label = get_label(s)
        o_label = get_label(o)
        p_label = get_label(p)

        G.add_edge(s_label, o_label)
        edge_labels[(s_label, o_label)] = p_label

    # Use spring layout for force-directed graph
    pos = nx.spring_layout(G)

    plt.figure(figsize=(14, 10))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("RDF Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Example Turtle data with various syntax features
turtle_data = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <http://example.org/> .

:Person a rdfs:Class .
:name a rdf:Property ;
    rdfs:domain :Person ;
    rdfs:range xsd:string .

:alice a :Person ;
    :name "Alice" ;
    :age 30 ;
    :friend :bob, :charlie .

:bob a :Person ;
    :name "Bob" .

:charlie a :Person ;
    :name "Charlie" .

[] :relatedTo :alice ;
   :through "work" .

:alice :hasAddress [
    :street "123 Main St" ;
    :city "Wonderland"
] .
"""

# Parse and visualize
g = parse_turtle(turtle_data)
visualize_rdf(g)