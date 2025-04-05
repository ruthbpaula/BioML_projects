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
#metabolomics_data = pd.read_csv("metabolomics.csv", index_col=0)  # samples x metabolites
#proteomics_data = pd.read_csv("proteomics.csv", index_col=0)  # samples x proteins
#metadata = pd.read_csv("metadata.csv", index_col=0)  # Contains disease status
metabolomics_data = pd.read_csv("residualized_z_scores_chemID_pmi_msex_age-death_ceradsc_imput_as_min_all_subjects.txt", sep="\t", index_col=0)  # samples x metabolites
proteomics_data = pd.read_csv("Round1-2.Unregressed_batch-corrected_TMT_reporter_new_resid_pmi_sex_age-death.txt", sep=" ", index_col=0)  # samples x proteins
metadata = pd.read_csv("dataset_236_basic_02-02-2023_with_individualID.txt", sep="\t", index_col=0)  # Contains disease status

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
#scaler = StandardScaler()
#metabolomics_scaled = scaler.fit_transform(metabolomics_data)
#proteomics_scaled = scaler.fit_transform(proteomics_data)
metabolomics_scaled = metabolomics_data
proteomics_scaled = proteomics_data

# Split data into disease and healthy groups
disease_idx = metadata['dlbany'] == 1
healthy_idx = metadata['dlbany'] == 0

metabo_disease = metabolomics_scaled[disease_idx]
proteo_disease = proteomics_scaled[disease_idx]
metabo_healthy = metabolomics_scaled[healthy_idx]
proteo_healthy = proteomics_scaled[healthy_idx]

common_ids = metabo_disease.index.intersection(proteo_disease.index)
metabo_disease = metabo_disease.loc[common_ids]
proteo_disease = proteo_disease.loc[common_ids]
common_ids = metabo_healthy.index.intersection(proteo_healthy.index)
metabo_healthy = metabo_healthy.loc[common_ids]
proteo_healthy = proteo_healthy.loc[common_ids]

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

# Plot the correlation matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_diff, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Update the labels to reflect meaningful components
plt.title("Difference in Canonical Correlations: Disease vs. Healthy")
plt.xlabel("Canonical Components")
plt.ylabel("Canonical Components")

# Create meaningful labels for X and Y axis (10 components: 5 Metabolites + 5 Proteins)
x_labels = [f"Metabo C{i+1}" for i in range(5)] + [f"Proteo C{i+1}" for i in range(5)]  # 5 metabolite + 5 protein components
y_labels = [f"Metabo C{i+1}" for i in range(5)] + [f"Proteo C{i+1}" for i in range(5)]

plt.xticks(ticks=np.arange(10) + 0.5, labels=x_labels, rotation=45)
plt.yticks(ticks=np.arange(10) + 0.5, labels=y_labels, rotation=0)

plt.tight_layout()
plt.show()
# Red areas → Stronger correlation in disease compared to healthy.
# Blue areas → Stronger correlation in healthy compared to disease.
# White/neutral areas → No significant difference between the groups.


# Now, we need to figure out which proteins and metabolites below to each component of interest
metabolite_disease_loadings = cca_disease.x_weights_  # Shape: (num_metabolites, num_components)
protein_disease_loadings = cca_disease.y_weights_  # Shape: (num_proteins, num_components)
metabolite_healthy_loadings = cca_healthy.x_weights_  # Shape: (num_metabolites, num_components)
protein_healthy_loadings = cca_healthy.y_weights_  # Shape: (num_proteins, num_components)

import numpy as np
import pandas as pd

# Loadings DataFrames for better readability
metabolites_disease_df = pd.DataFrame(metabolite_disease_loadings, index=metabolite_names, columns=[f"CCA Component {i+1}" for i in range(metabolite_disease_loadings.shape[1])])
proteins_disease_df = pd.DataFrame(protein_disease_loadings, index=protein_names, columns=[f"CCA Component {i+1}" for i in range(protein_disease_loadings.shape[1])])
metabolites_healthy_df = pd.DataFrame(metabolite_healthy_loadings, index=metabolite_names, columns=[f"CCA Component {i+1}" for i in range(metabolite_healthy_loadings.shape[1])])
proteins_healthy_df = pd.DataFrame(protein_healthy_loadings, index=protein_names, columns=[f"CCA Component {i+1}" for i in range(protein_healthy_loadings.shape[1])])

# Get the top 20 most influential metabolites per component
top_20_metabolites_disease = metabolites_disease_df.abs().apply(lambda x: x.nlargest(20).index.tolist(), axis=0)
top_20_metabolites_healthy = metabolites_healthy_df.abs().apply(lambda x: x.nlargest(20).index.tolist(), axis=0)

# Get the top 20 most influential proteins per component
top_20_proteins_disease = proteins_disease_df.abs().apply(lambda x: x.nlargest(20).index.tolist(), axis=0)
top_20_proteins_healthy = proteins_healthy_df.abs().apply(lambda x: x.nlargest(20).index.tolist(), axis=0)

# Print results
print("Top 20 Metabolites per Component (Disease):")
print(top_20_metabolites_disease)

print("\nTop 20 Proteins per Component (Disease):")
print(top_20_proteins_disease)

print("\nTop 20 Metabolites per Component (Healthy):")
print(top_20_metabolites_healthy)

print("\nTop 20 Proteins per Component (Healthy):")
print(top_20_proteins_healthy)

# Save as tab-separated .txt files
top_20_metabolites_disease.to_csv("top_20_metabolites_disease.txt", sep="\t", index=True)
top_20_proteins_disease.to_csv("top_20_proteins_disease.txt", sep="\t", index=True)
top_20_metabolites_healthy.to_csv("top_20_metabolites_healthy.txt", sep="\t", index=True)
top_20_proteins_healthy.to_csv("top_20_proteins_healthy.txt", sep="\t", index=True)

print("Files saved successfully.")



#### Some classical CCA graphs
# Compute canonical correlations (approximate using transformed variables)
canonical_corr_disease = [np.corrcoef(X_c_disease[:, i], Y_c_disease[:, i])[0, 1] for i in range(5)]
canonical_corr_healthy = [np.corrcoef(X_c_healthy[:, i], Y_c_healthy[:, i])[0, 1] for i in range(5)]

# Plot canonical correlations
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(range(1, 6), canonical_corr_disease, color='red', alpha=0.7, label='Disease')
ax[1].bar(range(1, 6), canonical_corr_healthy, color='blue', alpha=0.7, label='Healthy')

for a in ax:
    a.set_xlabel("Canonical Component")
    a.set_ylabel("Canonical Correlation")
    a.set_ylim(0, 1)
    a.legend()

ax[0].set_title("Canonical Correlations - Disease")
ax[1].set_title("Canonical Correlations - Healthy")

plt.tight_layout()
plt.show()

# Scatter plot of first canonical variables
plt.figure(figsize=(6, 5))
plt.scatter(X_c_disease[:, 0], Y_c_disease[:, 0], color='red', alpha=0.5, label='Disease')
plt.scatter(X_c_healthy[:, 0], Y_c_healthy[:, 0], color='blue', alpha=0.5, label='Healthy')
plt.xlabel("First Canonical Variable (X)")
plt.ylabel("First Canonical Variable (Y)")
plt.legend()
plt.title("Scatter Plot of First Canonical Variables")
plt.show()

# Loadings plot (first component)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(range(1, metabo_disease.shape[1] + 1), cca_disease.x_weights_[:, 0], color='red', alpha=0.7, label='Metabolites')
ax[0].bar(range(1, proteo_disease.shape[1] + 1), cca_disease.y_weights_[:, 0], color='black', alpha=0.7, label='Proteins')

ax[1].bar(range(1, metabo_healthy.shape[1] + 1), cca_healthy.x_weights_[:, 0], color='blue', alpha=0.7, label='Metabolites')
ax[1].bar(range(1, proteo_healthy.shape[1] + 1), cca_healthy.y_weights_[:, 0], color='green', alpha=0.7, label='Proteins')

for a in ax:
    a.set_xlabel("Feature Index")
    a.set_ylabel("Canonical Weight")
    a.legend()

ax[0].set_title("Canonical Weights - Disease (First Component)")
ax[1].set_title("Canonical Weights - Healthy (First Component)")

plt.tight_layout()
plt.show()


# Scatter plot for Metabolite Component 4 vs. Protein Component 8
plt.figure(figsize=(6, 5))
plt.scatter(X_c_disease[:, 3], Y_c_disease[:, 7], color='red', alpha=0.5, label='Disease')
plt.scatter(X_c_healthy[:, 3], Y_c_healthy[:, 7], color='blue', alpha=0.5, label='Healthy')

plt.xlabel("Metabolite Component 4")
plt.ylabel("Protein Component 8")
plt.legend()
plt.title("Scatter Plot: Metabolite C4 vs. Protein C8")

plt.show()







