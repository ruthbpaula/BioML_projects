import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy import stats
import os
import gc

# Force garbage collection
gc.collect()

class KEGGFluxAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_graphics = {}
        self.timepoints = []
        self.metabolite_data = {}
        self.enzyme_data = {}
        
    def parse_kgml(self, kgml_file):
        """Parse KGML file and build network structure"""
        tree = ET.parse(kgml_file)
        root = tree.getroot()
        
        # Extract entries (nodes)
        for entry in root.findall('entry'):
            entry_id = entry.get('id')
            entry_name = entry.get('name')
            entry_type = entry.get('type')
            
            # Extract graphics information
            graphics = entry.find('graphics')
            if graphics is not None:
                x = float(graphics.get('x', 0))
                y = float(graphics.get('y', 0))
                name = graphics.get('name', entry_name)
                bgcolor = graphics.get('bgcolor', '#FFFFFF')
                
                self.node_positions[entry_id] = (x, y)
                self.node_graphics[entry_id] = {
                    'name': name,
                    'bgcolor': bgcolor,
                    'type': entry_type
                }
            
            # Add node to graph
            self.graph.add_node(entry_id, name=entry_name, type=entry_type)
        
        # Extract relations (edges)
        for relation in root.findall('relation'):
            entry1 = relation.get('entry1')
            entry2 = relation.get('entry2')
            relation_type = relation.get('type')
            
            # Add edge to graph
            self.graph.add_edge(entry1, entry2, type=relation_type)
        
        # Extract reactions
        for reaction in root.findall('reaction'):
            reaction_id = reaction.get('id')
            reaction_name = reaction.get('name')
            
            # Get substrates and products
            substrates = [substrate.get('id') for substrate in reaction.findall('substrate')]
            products = [product.get('id') for product in reaction.findall('product')]
            
            # Add reaction edges
            for substrate in substrates:
                for product in products:
                    if substrate in self.graph and product in self.graph:
                        self.graph.add_edge(substrate, product, reaction=reaction_name)
        
        print(f"Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def load_metabolite_data(self, data_file):
        """Load metabolite concentration data across timepoints"""
        df = pd.read_csv(data_file)
        
        # Extract timepoints
        self.timepoints = [col for col in df.columns if col != 'metabolite_id' and col != 'name']
        
        # Store metabolite data
        for _, row in df.iterrows():
            metabolite_id = row['metabolite_id']
            self.metabolite_data[metabolite_id] = {
                'name': row['name'],
                'values': [row[tp] for tp in self.timepoints]
            }
        
        print(f"Loaded data for {len(self.metabolite_data)} metabolites across {len(self.timepoints)} timepoints")
    
    def load_enzyme_data(self, data_file):
        """Load enzyme data (conservation status)"""
        df = pd.read_csv(data_file)
        
        # Store enzyme data
        for _, row in df.iterrows():
            enzyme_id = row['enzyme_id']
            self.enzyme_data[enzyme_id] = {
                'name': row['name'],
                'conserved': row['conserved']
            }
        
        print(f"Loaded data for {len(self.enzyme_data)} enzymes")
    
    def calculate_flux_directions(self):
        """Calculate flux directions based on temporal metabolite patterns"""
        flux_directions = {}
        
        for edge in self.graph.edges():
            source, target = edge
            
            # Skip if we don't have data for both source and target
            if source not in self.metabolite_data or target not in self.metabolite_data:
                continue
            
            source_values = self.metabolite_data[source]['values']
            target_values = self.metabolite_data[target]['values']
            
            # Calculate correlation between source and target over time
            correlation, _ = stats.pearsonr(source_values, target_values)
            
            # Calculate trends for source and target
            source_trend = np.polyfit(range(len(source_values)), source_values, 1)[0]
            target_trend = np.polyfit(range(len(target_values)), target_values, 1)[0]
            
            # Determine flux direction:
            # - If source decreases and target increases, flow is from source to target
            # - If source increases and target decreases, flow is opposite
            # - Otherwise use correlation and relative trends
            
            if source_trend < 0 and target_trend > 0:
                direction = 1  # Forward: source -> target
                confidence = abs(source_trend) + abs(target_trend)
            elif source_trend > 0 and target_trend < 0:
                direction = -1  # Reverse: target -> source
                confidence = abs(source_trend) + abs(target_trend)
            else:
                # Use correlation and relative trends
                if correlation > 0:
                    # Positive correlation suggests same direction changes
                    # Determine direction by which has stronger trend
                    if abs(source_trend) > abs(target_trend):
                        direction = 1  # Forward
                    else:
                        direction = -1  # Reverse
                else:
                    # Negative correlation suggests opposite direction changes
                    if source_trend > target_trend:
                        direction = 1  # Forward
                    else:
                        direction = -1  # Reverse
                
                confidence = abs(correlation) * (abs(source_trend) + abs(target_trend))
            
            flux_directions[edge] = {
                'direction': direction,
                'confidence': confidence
            }
        
        return flux_directions
    
    def visualize_pathway_with_flux(self, output_file, flux_directions):
        """Create visualization with flux directions indicated by purple arrows"""
        #plt.figure(figsize=(20, 15))
        plt.figure(figsize=(10, 7.5), dpi=100)
        
        # Draw nodes
        for node_id, pos in self.node_positions.items():
            node_type = self.node_graphics[node_id]['type']
            color = self.node_graphics[node_id]['bgcolor']
            
            # Adjust based on metabolite data
            if node_id in self.metabolite_data:
                values = self.metabolite_data[node_id]['values']
                if values[-1] > values[0] * 1.5:  # Upregulated
                    plt.plot(pos[0], pos[1], 'ro', markersize=8)  # Red dot
                elif values[-1] < values[0] * 0.67:  # Downregulated
                    plt.plot(pos[0], pos[1], 'co', markersize=8)  # Cyan dot
            
            # Draw node
            if node_type == 'gene' and node_id in self.enzyme_data:
                if self.enzyme_data[node_id]['conserved']:
                    color = '#BFFFBF'  # Green for conserved enzymes
            
            plt.text(pos[0], pos[1], self.node_graphics[node_id]['name'], 
                    ha='center', va='center', bbox=dict(facecolor=color, alpha=0.7))
        
        # Draw flux arrows
        for edge, flux_info in flux_directions.items():
            source, target = edge
            if source in self.node_positions and target in self.node_positions:
                source_pos = self.node_positions[source]
                target_pos = self.node_positions[target]
                
                direction = flux_info['direction']
                confidence = flux_info['confidence']
                
                # Normalize confidence for arrow width
                width = 1 + 3 * (confidence / max([f['confidence'] for f in flux_directions.values()]))
                
                if direction == 1:
                    # Source to target
                    arrow = FancyArrowPatch(source_pos, target_pos, 
                                         connectionstyle="arc3,rad=0.1",
                                         color='purple', linewidth=width, alpha=0.7,
                                         arrowstyle='->')
                else:
                    # Target to source
                    arrow = FancyArrowPatch(target_pos, source_pos, 
                                         connectionstyle="arc3,rad=0.1",
                                         color='purple', linewidth=width, alpha=0.7,
                                         arrowstyle='->')
                plt.gca().add_patch(arrow)
        
        plt.axis('off')
        plt.title('Metabolic Pathway with Inferred Flux Directions')
        #plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file, figsize=(10, 7.5), dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    def analyze_temporal_flux(self, kgml_file, metabolite_data_file, enzyme_data_file, output_file):
        """Run the complete analysis pipeline"""
        # Parse KGML and build network
        self.parse_kgml(kgml_file)
        
        # Load data
        self.load_metabolite_data(metabolite_data_file)
        self.load_enzyme_data(enzyme_data_file)
        
        # Calculate flux directions
        flux_directions = self.calculate_flux_directions()
        
        # Visualize results
        self.visualize_pathway_with_flux(output_file, flux_directions)
        
        return flux_directions


# Example usage
os.chdir('C:/Users/ruthb/Documents/BCM/PhD project/Ruth pipelines/from computer/kegg_flux_analysis')

if __name__ == "__main__":
    analyzer = KEGGFluxAnalyzer()
    analyzer.analyze_temporal_flux(
        "./mock-kgml.txt",
        "./mock-metabolite-data.txt",
        "./mock-enzyme-data.txt",
        "./pyrimidine_flux_visualization.png"
    )

# Close all existing figures
plt.close('all')