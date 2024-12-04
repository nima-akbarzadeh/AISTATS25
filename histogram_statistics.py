import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    df = pd.read_excel('./output-finite/Res_m2.xlsx')
    target_labels = ['MEAN-4'] # 'MEAN-4' for REL_RAObjToNeutral, 'MEAN-8' for REL_RARewToNeutral
    
    for target_label in target_labels:
        y = df[target_label]

        print(f'Mean = {y.mean()}')

        # Define the boundaries for the histogram
        min_val = df[target_label].min()
        print(f'Min = {min_val}')

        max_val = df[target_label].max()
        print(f'Max = {max_val}')

        print(f"Portion below zero: {sum(y.values < 0)/len(y)}")

        # Ensure 0 is included in the bins
        bins = list(np.linspace(min_val, max_val, num=15))
        # if 0 not in bins:
        #     closest_to_zero = np.min(np.abs(bins))
        #     bins.remove(closest_to_zero)
        #     bins = np.sort(np.append(bins, 0))

        # Plot the histogram
        plt.hist(df[target_label], bins=bins, edgecolor='black')

        # Format the x-axis to have one decimal place
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        plt.xticks(bins)

        plt.xticks(fontsize=14, fontweight='bold', rotation=90)
        plt.yticks(fontsize=14, fontweight='bold')

        # Reduce the whitespace between bins
        plt.hist(df[target_label], bins=bins, edgecolor='black', linewidth=0.5, color='blue')

        plt.grid(axis='y')
        plt.xlabel('Relative Improvement', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()
