import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")
#st.markdown("<h1 style='text-align: center; color: blue;'>Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("Customer Segmentation Dashboard")

option = st.selectbox(
    "Select Dataset Type:",
    ["Mall Customers", "E-Commerce (RFM)"]
)

uploaded_file = st.file_uploader("Upload Customer Dataset:", type=["csv"])

#if uploaded_file is not None:
if uploaded_file is not None and option == "Mall Customers":
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Convert Gender
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    features = data[['Age','Annual Income (k$)','Spending Score (1-100)']]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # -----------------------------
    # 🔥 ELBOW METHOD
    # -----------------------------
    st.subheader("Elbow Method")

    wcss = []
    K_range = range(1, 11)

    for i in K_range:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(K_range, wcss, marker='o')
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Chart")

    st.pyplot(fig1)

    # -----------------------------
    # K-MEANS CLUSTERING
    # -----------------------------

    k = st.slider("Select Number of Clusters",2,10,5)

    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(scaled)

    data['Cluster'] = clusters

    score = silhouette_score(scaled, data['Cluster'])

    st.subheader("Silhouette Score")
    st.write("Silhouette Score:", round(score, 3))

    st.subheader("Clustered Data")

    # Rename clusters
    data['Segment'] = data['Cluster'].map({
        0: 'Low Value',
        1: 'High Value',
        2: 'Medium Value',
        3: 'Premium',
        4: 'Occasional',
        5: 'Very Rare'
    })

    #st.write("Clustered Data")
    st.dataframe(data)


    #Filtered data here
    st.subheader("Filter by Customer Segment")
    # Filter for download
    selected_segment = st.selectbox(
        "Select Segment",
        data['Segment'].unique()
    )

    filtered_data = data[data['Segment'] == selected_segment]

    st.write("Filtered Customers")
    st.dataframe(filtered_data)





    # Data Visualization
    st.subheader("Data Visualization")
    # -----------------------------
    # BASIC SCATTER PLOT
    # -----------------------------


    fig, ax = plt.subplots()

    ax.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        c=data['Cluster']
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")

    #st.pyplot(fig)

    # -----------------------------
    # 🔥 PCA VISUALIZATION
    # -----------------------------


    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    data['PCA1'] = pca_data[:, 0]
    data['PCA2'] = pca_data[:, 1]

    fig3, ax3 = plt.subplots()

    scatter = ax3.scatter(
        data['PCA1'],
        data['PCA2'],
        c=data['Cluster']
    )

    ax3.set_xlabel("PCA1")
    ax3.set_ylabel("PCA2")
    ax3.set_title("PCA Cluster Visualization")

    #st.pyplot(fig3)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Basic Scatter Plot:")
        st.pyplot(fig)
    with col2:
        st.write("Customer Segmentation using PCA")
        st.pyplot(fig3)

    #Pie Chart (Segment Distribution)
    st.subheader("Customer Segment Distribution")

    segment_counts = data['Segment'].value_counts()

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%')

    ax_pie.set_title("Customer Segments")

    #st.pyplot(fig_pie)

    #Bar Chart (Better for Comparison)
    # st.subheader("Segment Comparison")
    #
    # fig_bar, ax_bar = plt.subplots()
    #
    # segment_counts.plot(kind='bar', ax=ax_bar)
    #
    # ax_bar.set_xlabel("Segment")
    # ax_bar.set_ylabel("Number of Customers")
    # ax_bar.set_title("Customer Segments Count")
    #
    # st.pyplot(fig_bar)


    # Enhanced Bar
    #st.subheader("Segment Comparison")

    fig_bar, ax_bar = plt.subplots()

    segment_counts = data['Segment'].value_counts()

    bars = ax_bar.bar(segment_counts.index, segment_counts.values)

    ax_bar.set_xlabel("Segment")
    ax_bar.set_ylabel("Number of Customers")
    ax_bar.set_title("Customer Segments Count")

    # 🔥 ADD VALUES ON TOP OF BARS
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(int(height)),
            ha='center',
            va='bottom'
        )

    #st.pyplot(fig_bar)

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig_pie)

    with col2:
        st.pyplot(fig_bar)



    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------

    csv = convert_df_to_csv(data)

    st.download_button(
        label="Download Segmented Mall Customers CSV",
        data=csv,
        file_name='mall_segmented_customers.csv',
        mime='text/csv'
    )

    # Download Filtered Data
    csv_filtered = convert_df_to_csv(filtered_data)

    st.download_button(
        label="Download Selected Segment",
        data=csv_filtered,
        file_name='filtered_customers.csv',
        mime='text/csv'
    )





elif uploaded_file is not None and option == "E-Commerce (RFM)":
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.write("Original Dataset Preview")
    st.dataframe(data.head())

    data.dropna(inplace=True)
    data = data[data['Quantity'] > 0]

    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    today_date = data['InvoiceDate'].max()

    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (today_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.write("RFM Table")
    st.dataframe(rfm.head())

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # -----------------------------
    # 🔥 ELBOW METHOD
    # -----------------------------
    st.subheader("Cluster Selection")
    st.write("Elbow Method")

    wcss = []
    K_range = range(1, 11)

    for i in K_range:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    fig2, ax2 = plt.subplots()
    ax2.plot(K_range, wcss, marker='o')
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("WCSS")
    ax2.set_title("Elbow Chart")

    st.pyplot(fig2)

    k = st.slider("Select Clusters", 2, 10, 4)

    # Clustering

    kmeans = KMeans(n_clusters=k)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, rfm['Cluster'])

    st.subheader("Silhouette Score")
    st.write("Silhouette Score:", round(score, 3))

    # Rename clusters
    rfm['Segment'] = rfm['Cluster'].map({
        0: 'VIP Customers',
        1: 'Loyal Customers',
        2: 'At Risk',
        3: 'Lost Customers'
    })

    st.subheader("Segmented Data:")
    st.write("RFM Head:")
    st.write(rfm.head())

    st.write("Segmented Customers")
    st.dataframe(rfm)

    # Filter for download
    selected_segment = st.selectbox(
        "Select Segment",
        rfm['Segment'].unique()
    )

    filtered_data = rfm[rfm['Segment'] == selected_segment]

    st.write("Filtered Customers")
    st.dataframe(filtered_data)



    # Visualization
    st.subheader("Data Visualization")

    fig, ax = plt.subplots()
    ax.scatter(
        rfm['Frequency'],
        rfm['Monetary'],
        c=rfm['Cluster']
    )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Monetary")
    ax.set_title("Scatter Plot")

    #st.pyplot(fig)

    # -----------------------------
    # 🔥 PCA FOR RFM
    # -----------------------------


    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)

    rfm['PCA1'] = rfm_pca[:, 0]
    rfm['PCA2'] = rfm_pca[:, 1]

    fig_pca, ax_pca = plt.subplots()

    ax_pca.scatter(
        rfm['PCA1'],
        rfm['PCA2'],
        c=rfm['Cluster']
    )

    ax_pca.set_xlabel("PCA1")
    ax_pca.set_ylabel("PCA2")
    ax_pca.set_title("RFM Clusters (PCA View)")

    #st.pyplot(fig_pca)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.pyplot(fig_pca)


    # Pie Chart (Segment Distribution)
    st.subheader("Customer Segment Distribution")

    segment_counts = rfm['Segment'].value_counts()

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%')

    ax_pie.set_title("Customer Segments")

    #st.pyplot(fig_pie)

    # Bar Chart (Better for Comparison)
    # st.subheader("Segment Comparison")
    #
    # fig_bar, ax_bar = plt.subplots()
    #
    # segment_counts.plot(kind='bar', ax=ax_bar)
    #
    # ax_bar.set_xlabel("Segment")
    # ax_bar.set_ylabel("Number of Customers")
    # ax_bar.set_title("Customer Segments Count")
    #
    # st.pyplot(fig_bar)

    # Enhanced Bar
    #st.subheader("Segment Comparison")

    fig_bar, ax_bar = plt.subplots()

    segment_counts = rfm['Segment'].value_counts()

    bars = ax_bar.bar(segment_counts.index, segment_counts.values)

    ax_bar.set_xlabel("Segment")
    ax_bar.set_ylabel("Number of Customers")
    ax_bar.set_title("Customer Segments Count")

    # 🔥 ADD VALUES ON TOP OF BARS
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(int(height)),
            ha='center',
            va='bottom'
        )

    #st.pyplot(fig_bar)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_pie)
    with col2:
        st.pyplot(fig_bar)

    #----------
    # Download
    #----------

    csv = convert_df_to_csv(rfm)

    st.download_button(
        label="Download RFM Segmented Customers CSV",
        data=csv,
        file_name='rfm_segmented_customers.csv',
        mime='text/csv'
    )



    # Download Filtered Data
    csv_filtered = convert_df_to_csv(filtered_data)

    st.download_button(
        label="Download Selected Segment",
        data=csv_filtered,
        file_name='filtered_customers.csv',
        mime='text/csv'
    )