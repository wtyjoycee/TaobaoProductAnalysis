#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import requests
import spacy
import nltk
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collections import Counter
from PIL import Image
from io import BytesIO
from transformers import BertModel, BertTokenizer

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler

from keras.applications.vgg16 import VGG16


# In[3]:


product_data_cluster = pd.read_csv("product_data_30clusters.xlsx")
product_data_cluster.head(2)


# In[4]:


lemmatizer = WordNetLemmatizer() 
stop_words = set(stopwords.words('english'))


# In[21]:


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized)

# Apply the preprocessing to name and category
product_data_cluster['processed_Name'] = product_data_cluster['product_name'].apply(preprocess_text)
product_data_cluster['processed_Category'] = product_data_cluster['category'].apply(preprocess_text)


# In[6]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data_cluster['processed_Name'].astype(str).tolist())


# In[7]:


num_clusters = 150  
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

product_data_cluster['Cluster'] = clusters


# In[125]:


def download_image(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    if image is not None:
        # Resize and convert to RGB
        image = image.resize(target_size).convert('RGB')
        # Convert to array
        image_array = np.array(image)
        # Normalize the image array
        image_array = image_array.astype('float32') / 255.0
        print(image_array)
        return image_array
    else:
        return None

def process_and_store_images(df, image_column_name):
    image_arrays = []
    
    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        # Download and preprocess the image
        image_url = row[image_column_name].strip("'")
        image = download_image(image_url)
        processed_image = preprocess_image(image)
        
        # Append the processed image to the list, or None if the image couldn't be processed
        image_arrays.append(processed_image if processed_image is not None else None)
    
    # Assign the list of image arrays as a new column in the DataFrame
    df['image_array'] = image_arrays


# In[126]:


process_and_store_images(product_data_cluster, 'product_image')


# In[20]:


product_data_cluster.to_csv('product_data_images.csv')


# In[29]:


product_data_cluster_images = product_data_cluster.dropna(subset=['image_array'])


# In[41]:


product_data_cluster_images['topic'].values


# # Manually check images and see if they are similar in a specific cluster

# In[44]:


product_data_cluster_images[product_data_cluster_images['Cluster'] == 120]['product_image'].values


# In[ ]:





# In[59]:


grouped.sum()


# In[61]:


import pandas as pd
import numpy as np


# Grouping by 'topic' and 'category'
grouped = product_data_cluster_images.groupby(['topic', 'category'])

# Store a basic analysis results
price_stats = pd.DataFrame(columns=['Topic', 'Category', 'Average Price', 'Median Price', 'Price Range', 'Standard Deviation', 'Outliers'])

# Iterate over each group
for (topic, category), group in grouped:
    average_price = group['price'].mean()
    median_price = group['price'].median()
    price_range = group['price'].max() - group['price'].min()
    std_deviation = group['price'].std()
    
    # Identify outliers using the 1.5*IQR rule
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = group[(group['price'] < (Q1 - 1.5 * IQR)) | (group['price'] > (Q3 + 1.5 * IQR))]['price']
    
    # Append the results to the DataFrame
    price_stats = price_stats.append({
        'Topic': topic,
        'Category': category,
        'Average Price': average_price,
        'Median Price': median_price,
        'Price Range': price_range,
        'Standard Deviation': std_deviation,
        'Outliers': outliers.values
    }, ignore_index=True)

price_stats.to_csv('price_analysis.csv', index=False)


# In[62]:


price_stats


# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize average prices for each cluster
plt.figure(figsize=(14, 7))
sns.barplot(x='Topic', y='Average Price', data=price_stats)
plt.title('Average Price per Topic')
plt.xticks(rotation=90)  
plt.show()


# In[80]:


product_data_cluster_images['processed_Name_test'] = product_data_cluster_images['processed_Name'].apply(lambda x: ' '.join([word for word in x.split() if word.isalpha()]))
product_data_cluster_images['processed_Name_test'].values


# In[86]:


# import pandas as pd
# from collections import Counter


# # Group the DataFrame by 'cluster'
# grouped = product_data_cluster_images.groupby('topic')

# # Initialize a dictionary to store the most common terms in each cluster
# common_terms_per_cluster = {}

# # Iterate over each cluster group
# for cluster, group in grouped:
#     # Initialize a list to store all terms from all products in the cluster
#     all_terms = []
    
#     # Iterate through each row in the group
#     for index, row in group.iterrows():
#         # Split the 'processed_text' string into a list of words and extend the all_terms list
#         all_terms.extend(row['processed_Name'].split())
    
#     # Count the occurrences of each term in this cluster
#     term_counts = Counter(all_terms)
    
#     # Store the most common terms (say, top 10) in the dictionary
#     common_terms_per_cluster[cluster] = term_counts.most_common(10)

# for cluster, common_terms in common_terms_per_cluster.items():
#     print(f"Cluster {cluster}:")
#     for term, count in common_terms:
#         print(f" - {term}: {count}")
#     print("\n")


# In[151]:


cluster0 = product_data_cluster_images[product_data_cluster_images['Cluster'] == 0]


# In[167]:


cluster0['product_image'].values


# In[153]:


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)  # exclude top layers
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from an image array
def extract_features(image_array, model):
    if image_array is not None:
        image_array = np.expand_dims(image_array, axis=0)  # Add the batch dimension
        image_array = preprocess_input(image_array)  # Preprocess the image
        features = model.predict(image_array)  # Extract features
        return features.flatten()  # Flatten the features to a 1D array
    else:
        return None

# Function to process images and store them in a DataFrame
def process_and_store_images(df, image_column_name, model):
    image_features = []  # Initialize an empty list for storing image features
    
    for index, row in df.iterrows():
        image_url = row[image_column_name]  # Get the image URL
        image = download_image(image_url)  # Download the image
        image_array = preprocess_image(image)  # Preprocess the image
        features = extract_features(image_array, model) if image_array is not None else None
        image_features.append(features)  # Append the features to the list
    
    df['image_features'] = image_features 


# In[155]:


image_features = []

# Iterate over each row in the DataFrame
for index, row in cluster0.iterrows():
    features = extract_features(row['image_array'], model)
    if features is not None:
        image_features.append(features)
        
# Assign the list of image features as a new column in the DataFrame
cluster0['image_features'] = image_features


# # cosine_similarity on image array 

# In[156]:


from sklearn.metrics.pairwise import cosine_similarity

def calculate_image_similarity(cluster_images, feature_extraction_model):
    # Extract features for all images in the cluster
    features = [extract_features(image, feature_extraction_model) for image in cluster_images]

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(np.vstack(features))
    return similarity_matrix


# In[157]:


# image_features = []

# # Iterate over each row in the DataFrame
# for index, row in cluster24.iterrows():
#     features = extract_features(row['image_array'], model)
#     if features is not None:
#         image_features.append(features)

image_features_array = np.vstack(image_features)  

# cosine similarity 
cos_sim_matrix = cosine_similarity(image_features_array)

# Visualization
plt.figure(figsize=(10, 10))
sns.heatmap(cos_sim_matrix, cmap='viridis')
plt.title('Image Cosine Similarity in Cluster 24')
plt.show()


# In[196]:


import plotly.express as px
import pandas as pd


# Convert the price to numeric and sort by it 
cluster0['price'] = pd.to_numeric(cluster0['price'], errors='coerce')
cluster0 = cluster0.dropna(subset=['price']).sort_values('price')

# Create a figure with custom data and a hovertemplate to show the image on hover
fig = px.scatter(cluster0, x='price', y=cluster0.index, hover_data=['product_image'],
                 title='Product Prices with Images')

# hovertemplate to include the image
fig.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers'),
                  hovertemplate='<b>%{hovertext}</b><br><br>Price: %{x}$<br><img src="%{customdata[0]}" width="200" height="200"><extra></extra>')

# Update layout for a better visual presentation
fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))

fig.show()


# In[238]:


import plotly.express as px
import pandas as pd

# Convert the price to numeric and sort by it (if not already numeric and sorted)
cluster0['price'] = pd.to_numeric(cluster0['price'], errors='coerce')
cluster0 = cluster0.dropna(subset=['price']).sort_values('price')

# Create a figure with custom data and a hovertemplate to show the image on hover
fig = px.scatter(cluster0, x='price', y=cluster0.index, hover_data=['product_image'],
                 title='Product Prices with Images')

# Customize the hovertemplate to include the image
fig.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers'),
                  hovertemplate='<b>%{hovertext}</b><br><br>Price: %{x}$<br><img src="%{customdata[0]}" width="200" height="200"><extra></extra>')

# Update layout for a better visual presentation
fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))

fig.show()


# In[261]:


cluster0['product_link'] = cluster0['product_link'].str.replace("^'", '', regex=True)

df = cluster0.copy()


# In[266]:


import plotly.graph_objs as go
import pandas as pd

df = df[df['product_image'].notna() & df['product_link'].notna()]

# Create a FigureWidget for interactive plotting
fig = go.FigureWidget(data=[
    go.Scatter(
        x=df['price'],
        y=df['processed_subCategory'],
        mode='markers',
        marker=dict(size=10),
        hoverinfo='text',
        hovertext=df.apply(lambda row: f'<img src="{row["product_image"]}" width="160" height="90">', axis=1),
        customdata=df['product_link']
    )
])

#clock points to transfer to the product page
def open_link(trace, points, selector):
    for i in points.point_inds:
        # This will try to open a new tab in the browser with the product link
        url = trace.customdata[i]
        webbrowser.open(url, new=2)

# Link the function to the scatter trace
fig.data[0].on_click(open_link)

# Set the layout for the plot
fig.update_layout(title='Product Prices with Image on Hover', 
                  xaxis_title='Price', 
                  yaxis_title='Subcategory')

fig


# In[278]:


import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook

output_notebook()

cluster0 = cluster0[cluster0['product_image'].notna()]

# Create 'x', 'y', 'w', 'h' for the image position
cluster0['x'] = cluster0['price']  
cluster0['y'] = pd.Categorical(cluster0['processed_subCategory']).codes 
cluster0['w'] = 50 
cluster0['h'] = 0.3  

source = ColumnDataSource(cluster0)

# Determine the bounds for x and y ranges
x_range = (cluster0['price'].min() - 10, cluster0['price'].max() + 300)
y_range = (0, len(cluster0['processed_subCategory'].unique()))

# Create the figure, setting the x_range and y_range accordingly
p = figure(
    x_range=(cluster0['price'].min(), cluster0['price'].max() + 10),
    y_range= y_range,  
    title='Products with Image Preview'
)

# Add the images to the plot
p.image_url(url='product_image', x='x', y='y', w='w', h='h', source=source)

hover = HoverTool()
hover.tooltips = [
    ("Name", "@product_name"), 
    ("Price", "@price"),
    ("Link", "@product_link")  
]

p.add_tools(hover)
p.xaxis.axis_label = "Price"
p.yaxis.axis_label = "Subcategory"

show(p)


# In[249]:


cluster0[cluster0['price'] == cluster0['price'].max()]['product_image'].values


# In[161]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# PCA for Dimensionality Reduction
pca = PCA(n_components=5)  
pca_result = pca.fit_transform(image_features_array)

# t-SNE for further dimensionality reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)

# Visualization
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    palette=sns.color_palette("hsv", 10),
    legend="full",
    alpha=0.3
)

plt.title('t-SNE visualization of image features')
plt.show()


# In[ ]:





# In[ ]:





# In[15]:


model = VGG16(weights='imagenet', include_top=False)

def extract_features(image, model):
    features = model.predict(image.reshape((1, image.shape[0], image.shape[1], image.shape[2])))
    return features.flatten()

cluster_features = {cluster: [extract_features(image, model) for image in images] for cluster, images in clustered_images.items()}


# In[ ]:


cluster_images = clustered_images[0]  
features_list = [extract_features(image, model) for image in cluster_images]

