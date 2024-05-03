import matplotlib; matplotlib.get_backend()

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import csv
import sys
import pandas as pd
import math
sys.path.append('../src')

# Import csv as a df
df = pd.read_csv('nbk/out/degrees_1972_to_2019.csv')
# df = pd.read_csv('nbk/out/abs_summed_weights_1972_to_2019.csv')


# Arrange the columns in order of average value
# df = df.reindex(df.mean().sort_values().index, axis=1)

# Arrange the columns in order of value at the last index, reversed
df = df.reindex(df.iloc[-1].sort_values(ascending=False).index, axis=1)

# df = df.reindex(df.iloc[0].sort_values().index, axis=1)


# Category labels are the variables names from the df
category_labels = list(df.columns)

# Converting the df to a list of lists
data = df.T.values.tolist()
# print(data)
# print(len(data[0]), len(data[1]), len(data[2]))



matplotlib.use('Agg')
cm = 1/2.54  # centimeters in inches
plt.figure(figsize=(1*210*cm, 1*297*cm), dpi=300)

for i in range(len(data)):
    for j in range(len(data[i])):
        # print(data[i][j])
        color_value = j / (len(data[0]) - 1)
        color = plt.cm.PiYG(color_value)
        
        plt.eventplot([data[i][j]], color=color[0:3], lineoffsets=[i], linelengths=0.8, linewidths=75, orientation='horizontal')

# # Max value in the data
max_data = max([max(i) for i in data])

# for i in range(len(data)):
#     for j in range(len(data[i])):
#         # print(data[i][j])
#         color_value = data[i][j] / max_data
#         color = plt.cm.PuOr(color_value)
        
#         plt.eventplot([1972+2*j], color=color[0:3], lineoffsets=[i], linelengths=0.8, linewidths=15, orientation='horizontal')


# Adding labels and title
# plt.xlabel('Degree')
# plt.ylabel('Belief node')
# plt.title('Event Timeline')

# Customizing x-axis limits
# plt.xlim(-0.9, 50)

# Adding category labels on y-axis, with bold font
plt.yticks(range(0, len(category_labels)), category_labels, fontsize=62)
plt.xticks(fontsize=84)

# Create a color bar
# sm = cm.ScalarMappable(cmap=plt.cm.PuOr, norm=plt.Normalize(vmin=1972, vmax=2019))
# sm = cm.ScalarMappable(cmap=plt.cm.PuOr, norm=plt.Normalize(vmin=0, vmax=max_data))
sm.set_array([])  # dummy mappable array needed for the colorbar
plt.colorbar(sm, label='Year')

# Displaying the plot
plt.grid(False)
# plt.show()
plt.savefig('nbk/out/tester.svg', bbox_inches='tight', dpi=300)
print("Done!")