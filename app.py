import streamlit as st
import os
import pandas as pd
import io


import openai
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict

openai.api_type = "azure"
openai.api_base = "https://yash-ams-bot.openai.azure.com/"
openai.api_key = "dd4c31d04b2c480685d34365c0d167c6"
openai.api_version = "2023-07-01-preview"  # Use the latest supported version
OPENAI_API_VERSION = "2024-05-01-preview"
# GPT_DEPLOYMENT_NAME = "ams_gpt-35-turbo"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
EMBEDDING_MODEL_DEPLOYEMENT_NAME = "text-embedding-ada-002"


# Read input data while keeping Number (ID) and combining multiple text columns if needed
def read_excel(df, text_columns, id_column="Number"):
    # df = pd.read_excel(file_path)
    print(f"✅ Size of Dataframe {len(df)}")
    st.write(f"✅ Size of Dataframe {len(df)}")
    df = df[[id_column] + text_columns].dropna()  # Keep only relevant columns

    # If multiple text columns, combine them into one
    df["Combined_Text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

    return df[[id_column, "Combined_Text"]]

# Get embeddings from Azure GPT
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        engine=EMBEDDING_MODEL_DEPLOYEMENT_NAME
    )
    return [data['embedding'] for data in response['data']]

# Generate cluster names using LLM
def generate_cluster_name(texts):
    prompt = f"These are some text samples from a cluster: {texts[:10]}. What would be a meaningful short name for this category? and only provide short names nothing else"
    
    response = openai.ChatCompletion.create(
        engine=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are an expert in categorizing text data."},
                  {"role": "user", "content": prompt}],
        max_tokens=20
    )
    
    return response['choices'][0]['message']['content'].strip()

# Automatically determine optimal number of clusters
def find_optimal_clusters(embeddings, max_clusters=10):
    scores = []
    cluster_range = range(2, min(max_clusters, len(embeddings)))  # Avoid too many clusters for small data

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((k, score))

    # Choose the number of clusters with the highest silhouette score
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k

# Cluster text data while keeping IDs
def cluster_texts(df, text_column="Combined_Text", id_column="Number", num_clusters=None, batch_size=50):
    all_texts = []
    all_ids = []
    all_embeddings = []

    # Process data in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_texts = batch[text_column].tolist()
        batch_ids = batch[id_column].tolist()

        embeddings = get_embeddings(batch_texts)  # Ensure this function is defined

        all_texts.extend(batch_texts)
        all_ids.extend(batch_ids)
        all_embeddings.extend(embeddings)

    # Convert embeddings to numpy array
    all_embeddings = np.array(all_embeddings)

    # Determine number of clusters automatically if not provided
    if num_clusters is None:
        computed_clusters = find_optimal_clusters(all_embeddings)  # Ensure this function is defined
        num_clusters = max(5, computed_clusters)  # Ensure at least 2 clusters

    print(f"✅ Using {num_clusters} clusters")
    st.write(f"✅ Using {num_clusters} clusters")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_embeddings)

    # Organize text into clusters while preserving IDs
    clusters = defaultdict(list)
    id_map = defaultdict(list)

    for idx, label in enumerate(cluster_labels):
        clusters[label].append(all_texts[idx])
        id_map[label].append(all_ids[idx])

    # Generate meaningful cluster names
    cluster_names = {cluster_id: generate_cluster_name(texts) for cluster_id, texts in clusters.items()}

    return clusters, cluster_names, id_map

# Save clusters to Excel with IDs
def save_clusters_to_excel(clusters, cluster_names, id_map, df_main):
    data = []
    for cluster_id, texts in clusters.items():
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        for i, text in enumerate(texts):
            data.append([id_map[cluster_id][i], cluster_name, text])

    df = pd.DataFrame(data, columns=["Number", "Cluster Name", "Text"])
    df_main = df_main
    df = df_main.merge(df, on="Number")
    df = df.drop(['Text'], axis = 1)
    return df
    # df.to_excel(output_file, index=False)



# # Run the pipeline
# input_file = "incident_15Oct24.xlsx"
# output_file = "clustered_output.xlsx"
# id_column = "Number"

# # User can provide one or multiple text columns
# text_columns = ["Short description"]  # Example: Change to ["Title", "Description"] to combine multiple columns
# num_clusters = 15  # Set to an integer (e.g., 5) to specify clusters, or leave None for auto-detection

# df = read_excel(input_file, text_columns, id_column)
# clusters, cluster_names, id_map = cluster_texts(df, "Combined_Text", id_column, num_clusters)
# save_clusters_to_excel(clusters, cluster_names, id_map, output_file)

# print(f"✅ Clustering complete! Output saved to {output_file}")
# st.write(f"✅ Clustering complete! Output saved to {output_file}")

















#----------------------------------------------------------------------------------------------------------------------------------------------


st.set_page_config(
    page_title="Text Clustering Bot",
    page_icon="🀄️",
)
st.markdown("<h2 style='text-align: center;'> 🤖📜 Text Clustering Bot 📜🤖 </h2>", unsafe_allow_html=True)



data = None
submit = None


# Upload the text file
uploaded_file = st.file_uploader("Choose a file", type = 'xlsx')
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df_main = df.copy()

    # Additional mandatory inputs
    text_columns = st.text_input("Enter the name of the text columns (comma-separated):")

    # Split the input string into a list of column names
    if text_columns:
        column_list = [col.strip() for col in text_columns.split(',')]
        print("You entered:", column_list)


   # Radio button for cluster selection mode
    cluster_mode = st.radio("Select cluster mode:", ('Automatic', 'Manual'))

    if cluster_mode == 'Manual':
        number_clusters = int(st.number_input("Enter number of clusters:", value=2, step=1))
    else:
        number_clusters = None


    submit = st.button(":blue[Convert]", disabled=False) 










# Function to convert DataFrame to Excel
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  
    processed_data = output.getvalue()
    return processed_data





if submit:  # Check if the button is clicked
    with st.spinner('Wait for it...'):



        id_column = "Number"

        # User can provide one or multiple text columns
        text_columns = column_list  # Example: Change to ["Title", "Description"] to combine multiple columns
        print(text_columns)
        num_clusters = number_clusters  # Set to an integer (e.g., 5) to specify clusters, or leave None for auto-detection

        df = read_excel(df, text_columns, id_column)
        clusters, cluster_names, id_map = cluster_texts(df, "Combined_Text", id_column, num_clusters)
        output = save_clusters_to_excel(clusters, cluster_names, id_map, df_main = df_main)

        print(f"✅ Clustering complete")
        st.write(f"✅ Clustering complete!")









    import matplotlib.pyplot as plt

    


    # Count occurrences of each cluster name
    cluster_counts = output['Cluster Name'].value_counts()

    # Create a bar chart with gray plot, all text in gray, count on bar, no plot outline, and no Y axis
    plt.figure(figsize=(10, 4), facecolor='none')
    bars = plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Clusters', color='gray')
    plt.ylabel('Count', color='gray')
    plt.title('Clusters Counts', color='gray')
    plt.xticks(rotation=80, color='gray')
    plt.gca().patch.set_alpha(0)  # Set the background of the plot to be transparent

    # Remove plot outline and Y axis
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().yaxis.set_visible(False)

    # Add count on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', color='gray')

    # Display the bar chart in Streamlit
    st.pyplot(plt)

    # Convert DataFrame to Excel
    excel_data = to_excel(output)

    left, middle, right = st.columns(3)
    middle.download_button(label=":inbox_tray: Download Output :inbox_tray:", data=excel_data, file_name=((uploaded_file.name[:-4]) +'_clustered.xlsx'), mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    