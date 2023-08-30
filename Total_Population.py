#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

# Read CSV using pandas
csv_file_path = r"C:\Users\mitta\Downloads\total_fertility_rate.csv"
df = pd.read_csv(csv_file_path)

# Convert pandas DataFrame to numpy array
numpy_array = df.to_numpy()

print("Pandas DataFrame:")
print(data_frame)

print("\nNumpy Array:")
print(numpy_array)


# In[9]:


df.head(500000)


# In[6]:


import numpy as np
import pandas as pd

# Read CSV using pandas
csv_file_path = r"C:\Users\mitta\Downloads\total_median_age.csv"
df = pd.read_csv(csv_file_path)

# Convert pandas DataFrame to numpy array
numpy_array = df.to_numpy()


# In[8]:


df.head(450000)


# In[13]:


import numpy as np
import pandas as pd

# Read CSV using pandas
csv_file_path = r"C:\Users\mitta\Downloads\total_population.csv"
df = pd.read_csv(csv_file_path)

# Convert pandas DataFrame to numpy array
numpy_array = df.to_numpy()


# In[14]:


df


# In[15]:


features_list = df.columns.tolist()

print("List of All Features:")
for feature in features_list:
    print(feature)


# In[16]:


df.describe()


# In[19]:


df.info


# In[20]:



# Get the list of all features (columns)
features_list = df.columns.tolist()

# Store the features in a numpy array
features_array = np.array(features_list)

print("List of All Features:")
for feature in features_array:
    print(feature)


# In[21]:


features_array


# In[25]:


row_0_list = df.iloc[0].tolist()

print("Values from the 0th row as a list:")
print(row_0_list)


# In[27]:


import matplotlib.pyplot as plt
import random

# Example row_0_list (replace this with your actual data)
row_0_list = ['economy', 'YR2000', 'YR2001', 'YR2002', 'YR2003', 'YR2004', 'YR2005', 'YR2006', 'YR2007', 'YR2008', 'YR2009', 'YR2010', 'YR2011', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']

# Extract years (excluding the first element 'economy')
years = row_0_list[1:]

# Add random variations to years
fancy_years = [int(year[2:]) + random.randint(-5, 5) for year in years]

# Create a histogram
plt.hist(fancy_years, bins=10, edgecolor='black')

plt.xlabel('Years')
plt.ylabel('Frequency')
plt.title('Fancy Histogram of Years')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Example row_0_list (replace this with your actual data)
row_0_list = ['economy', 'YR2000', 'YR2001', 'YR2002', 'YR2003', 'YR2004', 'YR2005', 'YR2006', 'YR2007', 'YR2008', 'YR2009', 'YR2010', 'YR2011', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']

# Extract years (excluding the first element 'economy')
years = row_0_list[1:]

# Add random variations to years
fancy_years = [int(year[2:]) + random.randint(-5, 5) for year in years]

# Create a figure and axis
fig, ax = plt.subplots()

def animate(i):
    plt.clf()  # Clear the previous frame
    plt.hist(fancy_years[:i+1], bins=10, edgecolor='black')
    
    plt.xlabel('Years')
    plt.ylabel('Frequency')
    plt.title('Animated Fancy Histogram of Years')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(fancy_years), interval=500, repeat=False)

plt.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
plt.style.use('ggplot')

# Generate example data
np.random.seed(42)
data = np.random.normal(0, 1, size=1000)

# Set up the histogram
n_bins = 20
hist, bins, _ = ax.hist(data, bins=n_bins, alpha=0.7)
ax.set_xlim(min(data), max(data))
ax.set_ylim(0, max(hist) + 10)
ax.set_title('Animated Histogram')

# Animation update function
def update(frame):
    ax.clear()
    ax.hist(data[:frame], bins=n_bins, alpha=0.7)
    ax.set_xlim(min(data), max(data))
    ax.set_ylim(0, max(hist) + 10)
    ax.set_title('Animated Histogram - Frame {}'.format(frame))

# Create the animation
num_frames = len(data)
animation = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Display the animation
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
plt.style.use('ggplot')

# Generate example data
np.random.seed(42)
data = np.random.normal(0, 1, size=1000)

# Set up the histogram
n_bins = 20
hist, bins, _ = ax.hist(data, bins=n_bins, alpha=0.7)
ax.set_xlim(min(data), max(data))
ax.set_ylim(0, max(hist) + 10)
ax.set_title('Animated Histogram')

# Add x and y labels
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')

# Animation update function
def update(frame):
    ax.clear()
    frame_data = data[:frame]
    hist, bins, _ = ax.hist(frame_data, bins=n_bins, alpha=0.7)
    ax.set_xlim(min(data), max(data))
    ax.set_ylim(0, max(hist) + 10)
    ax.set_title('Animated Histogram - Frame {}'.format(frame))
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Highlight the bins for the current frame
    if frame > 0:
        ax.patches[frame - 1].set_facecolor('orange')
        ax.patches[frame - 1].set_alpha(0.9)

# Create the animation
num_frames = len(data)
animation = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Display the animation
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import random

# Example row_0_list (replace this with your actual data)
row_0_list = ['economy', 'YR2000', 'YR2001', 'YR2002', 'YR2003', 'YR2004', 'YR2005', 'YR2006', 'YR2007', 'YR2008', 'YR2009', 'YR2010', 'YR2011', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']

# Extract years (excluding the first element 'economy')
years = row_0_list[1:]

# Add random variations to years
fancy_years = [int(year[2:]) + random.randint(-5, 5) for year in years]

# Create histograms for the first 50 indexes
for i in range(50):
    plt.hist(fancy_years[:i+1], bins=10, edgecolor='black')
    
    plt.xlabel('Years')
    plt.ylabel('Frequency')
    plt.title(f'Fancy Histogram of Years (Indexes 0 to {i})')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()


# In[33]:


df.tail(34)


# In[34]:


economy_column = df['economy']
print("Economy Column:")
print(economy_column)


# In[35]:


economy_column = df['economy'].tolist()

print("Economy Column:")
print(economy_column)


# In[ ]:




#NMIMS IS A BAD COLLEGE 