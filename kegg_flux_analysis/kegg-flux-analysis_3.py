#🛠 Upgrades You Asked For

#Feature	Description	Method
#1. Animate movement (GIF)	Show pathway timeline evolving frame-by-frame	Use matplotlib.animation
#2. Thicken arrows if movement is strong	Draw thicker edges if source and target are both upregulated	Set edge width based on metabolite status
#3. Predict preferred path based on metabolite changes	Highlight paths where "upregulation" flows across nodes	Trace sequences of consecutive upregulated nodes
#4. Integrate enzyme data	Adjust node size/color if enzyme is present or missing	Add enzyme presence info, and use it to modify node visuals

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import os
import matplotlib.animation as animation

# ===========
# Loaders
# ===========

def load_kgml_as_graph(kgml_path):
    G = nx.DiGraph()
    tree = ET.parse(kgml_path)
    root = tree.getroot()
    entries = {}
    
    for entry in root.findall('entry'):
        eid = entry.attrib['id']
        name = entry.attrib['name']
        graphics = entry.find('graphics')
        enzyme_present = graphics.attrib.get('fgcolor', '#000000') == '#008000' if graphics is not None else False  # Example logic
        entries[eid] = {'name': name, 'enzyme_present': enzyme_present}

    for relation in root.findall('relation'):
        source = relation.attrib['entry1']
        target = relation.attrib['entry2']
        G.add_edge(entries[source]['name'], entries[target]['name'])
    
    for node_data in entries.values():
        G.nodes[node_data['name']]['enzyme_present'] = node_data['enzyme_present']
    
    return G

def load_status_per_timepoint(folder):
    status_dict = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            timepoint = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(folder, filename))
            status_dict[timepoint] = dict(zip(df['metabolite'], df['status']))
    return status_dict

# ===========
# Plotters
# ===========

def compute_edge_widths(G, status):
    widths = []
    for u, v in G.edges():
        u_id = u.split(":")[-1]
        v_id = v.split(":")[-1]
        if status.get(u_id, '') == 'upregulated' and status.get(v_id, '') == 'upregulated':
            widths.append(3.0)  # thick edge
        else:
            widths.append(0.5)  # thin edge
    return widths

def plot_network(G, status, pos, ax, title=""):
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_id = node.split(":")[-1]
        if node_id in status:
            if status[node_id] == 'upregulated':
                color = 'red'
            elif status[node_id] == 'downregulated':
                color = 'cyan'
            else:
                color = 'lightgray'
        else:
            color = 'white'
        
        if G.nodes[node].get('enzyme_present', False):
            size = 600
        else:
            size = 300
        
        node_colors.append(color)
        node_sizes.append(size)
    
    edge_widths = compute_edge_widths(G, status)
    
    nx.draw(G, pos, ax=ax,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            width=edge_widths)
    ax.set_title(title)
    ax.axis('off')

# ===========
# Animate
# ===========

def animate_pathway(G, status_dict, save_path='pathway_timeline.gif'):
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    timepoints = list(status_dict.keys())
    
    def update(frame):
        ax.clear()
        plot_network(G, status_dict[timepoints[frame]], pos, ax, title=f"Time: {timepoints[frame]}")
    
    ani = animation.FuncAnimation(fig, update, frames=len(timepoints), repeat=True, interval=1500)
    ani.save(save_path, writer='pillow')
    plt.close()

# ===========
# Preferred Paths Tracer
# ===========

def trace_preferred_paths(G, status):
    preferred_paths = []
    for node in G.nodes():
        node_id = node.split(":")[-1]
        if status.get(node_id, '') == 'upregulated':
            path = [node]
            current = node
            while True:
                neighbors = list(G.successors(current))
                if not neighbors:
                    break
                next_node = None
                for n in neighbors:
                    n_id = n.split(":")[-1]
                    if status.get(n_id, '') == 'upregulated':
                        next_node = n
                        break
                if next_node:
                    path.append(next_node)
                    current = next_node
                else:
                    break
            if len(path) > 1:
                preferred_paths.append(path)
    return preferred_paths

# ===========
# Example Usage
# ===========

kgml_path = 'pathway.kgml'
status_folder = 'timepoints_folder'

G = load_kgml_as_graph(kgml_path)
status_dict = load_status_per_timepoint(status_folder)

# Plot Static Timeline
pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, len(status_dict), figsize=(20, 5))
if len(status_dict) == 1:
    axes = [axes]

for idx, tp in enumerate(status_dict):
    plot_network(G, status_dict[tp], pos, axes[idx], title=tp)

plt.tight_layout()
plt.show()

# Save Animation
animate_pathway(G, status_dict, save_path='pathway_timeline.gif')

# Print Preferred Paths for each timepoint
for tp, status in status_dict.items():
    paths = trace_preferred_paths(G, status)
    print(f"Preferred paths at {tp}:")
    for path in paths:
        print(" -> ".join(path))
    print("-" * 40)
