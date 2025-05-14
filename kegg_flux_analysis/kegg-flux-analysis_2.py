# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 19:45:32 2025

@author: ruthb
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import os
import gc

# Force garbage collection
gc.collect()

# Step 1: Load Pathway Structure from KGML
def load_kgml_as_graph(kgml_path):
    G = nx.DiGraph()
    tree = ET.parse(kgml_path)
    root = tree.getroot()
    entries = {}
    
    for entry in root.findall('entry'):
        eid = entry.attrib['id']
        name = entry.attrib['name']
        entries[eid] = name

    for relation in root.findall('relation'):
        source = relation.attrib['entry1']
        target = relation.attrib['entry2']
        G.add_edge(entries[source], entries[target])
    
    return G

# Step 2: Load Metabolite Status for Each Timepoint
def load_status_per_timepoint(folder):
    status_dict = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            timepoint = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(folder, filename))
            status_dict[timepoint] = dict(zip(df['metabolite'], df['status']))
    return status_dict

# Step 3: Draw Timeline
def plot_pathway_timeline(G, status_dict):
    pos = nx.spring_layout(G, seed=42)  # fixed layout for consistency

    timepoints = list(status_dict.keys())
    fig, axes = plt.subplots(1, len(timepoints), figsize=(20, 5))
    
    if len(timepoints) == 1:
        axes = [axes]

    for idx, tp in enumerate(timepoints):
        ax = axes[idx]
        status = status_dict[tp]
        node_colors = []
        
        for node in G.nodes():
            node_id = node.split(":")[-1]  # Extract metabolite ID
            if node_id in status:
                if status[node_id] == 'upregulated':
                    node_colors.append('red')
                elif status[node_id] == 'downregulated':
                    node_colors.append('cyan')
                else:
                    node_colors.append('lightgray')
            else:
                node_colors.append('white')

        nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color='gray', ax=ax)
        ax.set_title(tp)

    plt.tight_layout()
    plt.show()

# Example Usage
os.chdir('C:/Users/ruthb/Documents/BCM/PhD project/Ruth pipelines/from computer/kegg_flux_analysis')
kgml_path = 'pathway.kgml'
status_folder = 'timepoints_folder'

G = load_kgml_as_graph(kgml_path)
status_dict = load_status_per_timepoint(status_folder)
plot_pathway_timeline(G, status_dict)
