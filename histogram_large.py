import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
file_path = './planning-finite/planning_results.xlsx'
output_path = './planning-finite/histogram_large.png'
df = pd.read_excel(file_path)

# Filter rows based on the naming pattern
df['Key'] = df['Key'].astype(str)

# Filter rows based on the naming pattern in the `Key` column
filtered_df = df[df['Key'].str.contains(r'^nt5_ns5_nc5_', regex=True)]

# Define the target column
target_label = 'MEAN-Ri_riskAware_to_neutral'

# Ensure the target column exists
if target_label not in filtered_df.columns:
    raise ValueError(f"The target column '{target_label}' does not exist in the DataFrame.")

# Extract the data for the histogram
y = filtered_df[target_label]

# Print statistics
print(f'Mean = {y.mean()}')
min_val = y.min()
print(f'Min = {min_val}')
max_val = y.max()
print(f'Max = {max_val}')
print(f"Portion below zero: {sum(y.values < 0) / len(y)}")

# Define the bins, ensuring 0 is included
bins = list(np.linspace(min_val, max_val, num=15))

# Plot the histogram
plt.hist(y, bins=bins, edgecolor='black', linewidth=0.5, color='blue')

# Format the x-axis to have one decimal place
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.xticks(bins, fontsize=14, fontweight='bold', rotation=90)
plt.yticks(fontsize=14, fontweight='bold')

# Add labels and grid
plt.grid(axis='y')
plt.xlabel('Relative Improvement', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig(output_path)
print(f"Histogram saved to {output_path}")

# Show the plot
plt.show()
