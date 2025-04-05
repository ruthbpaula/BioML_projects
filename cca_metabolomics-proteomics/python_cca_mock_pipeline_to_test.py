import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import os

os.chdir("C:/Users/ruthb/Downloads/cca_mock_Python")

# Load datasets (assumed preprocessed and matched by sample ID)
metabolomics_data = pd.read_csv("metabolomics.csv", index_col=0)  # samples x metabolites
proteomics_data = pd.read_csv("proteomics.csv", index_col=0)  # samples x proteins
metadata = pd.read_csv("metadata.csv", index_col=0)  # Contains disease status

print(metabolomics_data.head())
print(proteomics_data.head())
print(metadata.head())

#          Metabolite1  Metabolite2  ...  Metabolite49  Metabolite50
# Sample1     0.374540     0.950714  ...      0.546710      0.184854
# Sample2     0.969585     0.775133  ...      0.025419      0.107891
# Sample3     0.031429     0.636410  ...      0.051479      0.278646
# Sample4     0.908266     0.239562  ...      0.887086      0.779876
# Sample5     0.642032     0.084140  ...      0.963620      0.853009
# [5 rows x 50 columns]
# print(proteomics_data.head())
#          Protein1  Protein2  Protein3  ...  Protein38  Protein39  Protein40
# Sample1  0.393636  0.473436  0.854547  ...   0.969819   0.727631   0.922604
# Sample2  0.762212  0.591717  0.192023  ...   0.236610   0.483498   0.429149
# Sample3  0.074896  0.106160  0.837473  ...   0.482017   0.757777   0.281918
# Sample4  0.318040  0.924642  0.056224  ...   0.756451   0.474158   0.225790
# Sample5  0.616622  0.040334  0.326170  ...   0.063795   0.387223   0.282765
# [5 rows x 40 columns]
# print(metadata.head())
#            group
# Sample1  disease
# Sample2  disease
# Sample3  disease
# Sample4  disease
# Sample5  disease

# Standardize the data
scaler = StandardScaler()
metabolomics_scaled = scaler.fit_transform(metabolomics_data)
proteomics_scaled = scaler.fit_transform(proteomics_data)

# Split data into disease and healthy groups
disease_idx = metadata['group'] == "disease"
healthy_idx = metadata['group'] == "healthy"

metabo_disease = metabolomics_scaled[disease_idx]
proteo_disease = proteomics_scaled[disease_idx]
metabo_healthy = metabolomics_scaled[healthy_idx]
proteo_healthy = proteomics_scaled[healthy_idx]

# Run CCA for disease and healthy groups
cca_disease = CCA(n_components=5)
X_c_disease, Y_c_disease = cca_disease.fit_transform(metabo_disease, proteo_disease)

cca_healthy = CCA(n_components=5)
X_c_healthy, Y_c_healthy = cca_healthy.fit_transform(metabo_healthy, proteo_healthy)

# Compute correlation matrices
corr_matrix_disease = np.corrcoef(X_c_disease.T, Y_c_disease.T)
corr_matrix_healthy = np.corrcoef(X_c_healthy.T, Y_c_healthy.T)

# Difference in correlations
corr_diff = corr_matrix_disease - corr_matrix_healthy

# Flatten correlation difference matrix and apply multiple testing correction if enabled
apply_correction = False  # Set to False to disable correction
corr_diff_flat = corr_diff.flatten()

if apply_correction:
    _, pvals_corrected, _, _ = multipletests(corr_diff_flat, method='fdr_bh')
    pvals_corrected_matrix = pvals_corrected.reshape(corr_diff.shape)
    significant_mask = pvals_corrected_matrix < 0.05
else:
    pvals_corrected_matrix = np.ones_like(corr_diff)  # No correction, keep all p-values as 1
    significant_mask = np.ones_like(corr_diff, dtype=bool)

# Identify top changing metabolite-protein pairs with significant differences
num_top_pairs = 10
top_pairs_idx = np.unravel_index(np.argsort(np.abs(corr_diff * significant_mask), axis=None)[-num_top_pairs:], corr_diff.shape)

# Extract corresponding row and column indices
metabolite_names = metabolomics_data.columns
protein_names = proteomics_data.columns
metabolite_indices = top_pairs_idx[0]
protein_indices = top_pairs_idx[1] - len(metabolite_names)  # Shift protein indices correctly

# Ensure indices are within valid ranges
valid_pairs = (protein_indices >= 0) & (protein_indices < len(protein_names))

# Apply valid filter
top_metabolites = [metabolite_names[i] for i in metabolite_indices[valid_pairs]]
top_proteins = [protein_names[i] for i in protein_indices[valid_pairs]]

# Create dataframe of top associations
top_assoc_df = pd.DataFrame(zip(top_metabolites, top_proteins, corr_diff[top_pairs_idx], pvals_corrected_matrix[top_pairs_idx]),
                             columns=['Metabolite', 'Protein', 'Correlation Difference', 'Corrected P-value'])
print(top_assoc_df)

# Plot heatmap of correlation differences with significance mask
cca_labels = [f"CCA Component {i+1}" for i in range(10)]
plt.figure(figsize=(10, 6))
sns.heatmap(corr_diff, cmap='coolwarm', center=0, annot=False,
            xticklabels=cca_labels, yticklabels=cca_labels)
plt.title("Correlation Differences (Disease vs. Healthy)")
plt.xlabel("CCA Components (Proteins)")
plt.ylabel("CCA Components (Metabolites)")
plt.show()
# Red areas → Stronger correlation in disease compared to healthy.
# Blue areas → Stronger correlation in healthy compared to disease.
# White/neutral areas → No significant difference between the groups.