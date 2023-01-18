import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import allfunction as all

# visualize confusion matrix with seaborn heatmap
import seaborn as sns
from sklearn.metrics import confusion_matrix
def confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig=plt.figure(figsize=(4,4))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')        
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
# Interface
st.title("Extraction de connaissances")

selected=option_menu(
    menu_title="Main Menu",
    options=["Home","Data Overview","Clustering"],
    icons=["house","bar-chart"],
    menu_icon="cast",  # optional
    default_index=0,
    orientation="horizontal",  
    styles={
        "nav-link-selected": {"background-color": "#4B9DFF"},
    } 

     )
   
#==================================================================================================================

    # Accueil
if selected=="Home":

    #st.write("Cette application permet de faire une visualisation de votre jeu de données et d'appliquer les différents algorithmes de clustering : KMEANS, DBSCAN.")
    # creer une animation
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    lottie_coding = load_lottiefile("pc.json")  # replace link to local lottie file
    st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high

    height=None,
    width=None,
    key=None,
)

if selected=="Data Overview":
    with st.sidebar:
    # creer une animation
    # Creer un slider
        def load_lottiefile(filepath: str):
                with open(filepath, "r") as f:
                    return json.load(f)
        lottie_coding = load_lottiefile("lottie2.json")  # replace link to local lottie file
        st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality="high",height=None, width=None, key=None,)
    # Chose csv file
    st.sidebar.title("Select Your Dataset")
    upload_file=st.sidebar.file_uploader("Select:",type=["csv"])
    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv('data.csv', index=False)
        st.success("Dataset has selected successfully")
         ##### Encodding
        st.info("Data it will be Encoded if it contains nominal values")
        df=all.encoder(data)
        if st.checkbox("Discover your Data") :
            st.write(""" ## Discover your Data :""")
            radiodicover=st.radio("",("Shape","Description","Missing Value"))
            if radiodicover=="Shape":
                st.write(""" ### Results : """)
                st.success(data.shape)
            if radiodicover=="Description":
                st.write(""" ### Results : """)
                st.write(data.describe())
            if radiodicover=="Missing Value":
                st.write(""" ### Results : """)
                st.write(data.isnull().sum())
        if st.checkbox("Data Visualisation"):
            radio_vis=st.radio("Choose :",("Heat Map","Plot"))
            if radio_vis=="Heat Map":
                data=pd.read_csv("data.csv")
                fig, ax = plt.subplots()
                heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
                heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
                st.pyplot(fig)
            if radio_vis=="Plot":
                data=pd.read_csv("data.csv")
                df=data.select_dtypes(include='number')
                st.write(data.shape)
                st.write("""## Visualize the relation between data variables """)
                st.write("""### X_features """)
                x=st.selectbox("Variables !",df.columns)
                st.write("""### Y_features """)
                y=st.selectbox("Target !",df.columns)
                fig, ax = plt.subplots()
                ax.scatter(df[x],df[y],c='b')
                plt.xlabel(x)
                plt.ylabel(y)
                st.pyplot(fig)
        
    else:
        st.info("Select your Dataset")
        
if selected=="Clustering":
    with st.sidebar:
    # creer une animation
    # Creer un slider
        def load_lottiefile(filepath: str):
                with open(filepath, "r") as f:
                    return json.load(f)
        lottie_coding = load_lottiefile("lottie2.json")  # replace link to local lottie file
        st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality="high",height=None, width=None, key=None,)
    # Chose csv file
    st.sidebar.title("Select Your Dataset")
    upload_file=st.sidebar.file_uploader("Select:",type=["csv"])
    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv('data.csv', index=False)
        #### 
        st.success("Dataset has selected successfully")
        
        ##### Encodding
        st.write(""" ### Data Encoding 
        """)
        df=all.encoder(data)
        if st.checkbox("Show the Data after encoding"):
            st.success(df.shape)
            st.write(df)
        ##### Netoyage de données
        st.write(""" ### Data preporcessing 
        """)
        st.info("The idea of this part of this widget bellow is to delete some attrunutes like: (the classes, id...), and drop also missing value ")
        if st.checkbox("Drop columns"):
            supprimer=st.multiselect('Select the attrubuts to drop from the data',df.columns)
            if supprimer:
                df = df.drop(supprimer, axis=1)
                if st.checkbox("Show the Data after droping the attrubuts wanted"):
                    st.success(df.shape)
                    st.write(df)
        if st.checkbox("Drop missing Value"):
            df= df.dropna()
            st.success(df.shape)
        #####

        #### Algorithms  
        st.write(""" ### Clustering
        """)      
        algorithm=st.selectbox("Choose an algorithm",("Select an Algorithm","Kmeans","DBSCAN"))
        if algorithm=="Kmeans":
            st.info("The parameter n_clusters: The number of clusters in your Data")
            n_clusters = st.slider("n_clusters : ", 1, 10, 3)
            #Initialize the class object
            model = KMeans(n_clusters).fit(df)
            if st.checkbox("The centroides of each cluster"):
                st.write(model.cluster_centers_)
            #predict the labels of clusters.
            label = model.fit_predict(df)
            if st.checkbox('Show labels of each instance'):    
                st.success(label)
            
            if st.checkbox("Show the results"):
                a=st.selectbox("X:",df.columns)
                b=st.selectbox("Y:",df.columns)
                x=pd.DataFrame([df[a], df[b]]).transpose()
                x=x.to_numpy()
                #Getting unique labels
                #centroids = model.cluster_centers_ 
                u_labels = np.unique(label)
                #plotting the results:
                if st.checkbox("Show the plot"):
                    fig=plt.figure(figsize=(4,4))
                    for i in u_labels:
                        plt.scatter(x[label == i , 0] , x[label == i , 1] , label = i)
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                    #plt.scatter(centroids[:,1] , centroids[:,2] , s = 80, color = 'k')
                    plt.legend()
                    plt.show()
                    st.pyplot(fig)
                        
            
        if algorithm=="DBSCAN":
            # instantiate classifier with default hyperparameters
            st.write(''' ''')
            

    else:
        st.info("Select your Dataset that has any type of attrubutes")
        
        

        #=====================================================================================================
       