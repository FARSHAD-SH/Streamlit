# streamlit_app.py

import streamlit as st
st.set_page_config(page_title="Mimiric: A tool for ML data", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")



import s3fs
import os
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from loguru import logger
from io import StringIO


# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
logger.info("Loading dataset from file ...")
@st.experimental_memo(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return f.read().decode("utf-8")

content = read_file("mystreamlit/data/full_2016.csv")

content = StringIO(content)
df = pd.read_csv(content)
df['code_departement'] = pd.to_numeric(df['code_departement'], errors='coerce', downcast='integer')
# st.dataframe(df)
# path = f'./Data/full_{op[0]}.csv'

def main():

    ####################
    ##### LoadData #####
    ####################

    # LOAD DATA ONCE
    # logger.info("Loading dataset from file ...")
    # @st.experimental_singleton
    # def load_data(path):
    #     raw_data = pd.read_csv(
    #         path,
    #         usecols=[
    #                 "id_mutation", "date_mutation",
    #                 "valeur_fonciere", "type_local",
    #                 "code_departement", "surface_terrain",
    #                 "latitude", "longitude"
    #                 ], low_memory=False) # onlread these columns
    #     sample_raw_data = raw_data.sample(frac=.1, random_state=1).copy()  # sample 20% of data
    #     return sample_raw_data


    ####################
    ## Pre-Processing ##
    ####################

    # TRANSFORM Date Mutation to datetime
    logger.info("Pre-processing dataset ...")
    @st.experimental_memo
    def trans_data(R_data):
        R_data['date_mutation'] = pd.to_datetime(R_data['date_mutation'])
        return R_data

    # Drop missing values
    @st.experimental_memo
    def Clean_data(df):
        cl_df = df.dropna(axis = 0, how ='any')
        return cl_df

    # Create Month column
    @st.experimental_memo
    def Mon(df):
        df['month'] = df['date_mutation'].dt.month
        return df

    # Calculate the average price of land in a given month
    @st.experimental_memo
    def mean_(df):
        mean_mon = []
        Month = np.arange(1, 13, 1)
        for i in range(12):
            mean_mon.append(df[df['month']==i+1]['valeur_fonciere'].mean())
        return pd.DataFrame({'Month':Month, f'Year {op[0]}': mean_mon}, index=Month)

    ####################
    ###### CHARTS ######
    ####################

    def radio():
        st.markdown('## Histogram Chart')
        row3_1, row3_2 = st.columns((2, 3))

        with row3_1:
            st.markdown("The histogram chart shows the frequency of the properties values and surface terrain. We can see that the most properties values are between 0 and 500000.")
            opt = st.radio('Select Feature you want to visuale its frequency:', ['properties Value', 'Surface Terrain'])
        return opt

    # Histogram Chart Function
    def Histogram(df, opt):

        row3_1, row3_2 = st.columns((2, 3))

        def hist_chart(df):

            rc = {'figure.figsize':(8,4.5),
            'axes.facecolor':'#0e1117',
            'axes.edgecolor': '#0e1117',
            'axes.labelcolor': 'white',
            'figure.facecolor': '#0e1117',
            'patch.edgecolor': '#0e1117',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': 'grey',
            'font.size' : 8,
            'axes.labelsize': 12,
            'xtick.labelsize': 8,
            'ytick.labelsize': 12}

            plt.rcParams.update(rc)
            fig, ax = plt.subplots()
            ax.hist(x=df, bins=30, color='#e63946', alpha=0.7, rwidth=0.85)
            st.pyplot(fig)

        with row3_2:
            if opt=='properties Value':
                hist_chart(new_data2['valeur_fonciere'])
            elif opt=='Surface Terrain':
                hist_chart(df[df['surface_terrain']<= 10000]['surface_terrain'])

    # Pie chart
    logger.info("Creating pie chart ...")
    @st.experimental_memo
    def pie_chart(x2):

        st.markdown('## Pie Chart')
        st.markdown("The pie chart shows the percentage of the property mutation by the type. We can see that the most properties are Maison.")

        fig = plt.figure(figsize=(8, 8))
        fig = px.pie(x2, values=x2.values, names=x2.index,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                     hole=.4, opacity=0.8)
        st.plotly_chart(fig)

    # Altair chart
    logger.info("Creating Altair chart ...")
    @st.experimental_memo
    def altair_chart(xx):

        st.markdown('## Altair Chart')
        st.markdown("The altair chart shows the average price of properties in each areas. We can see that the most expensive properties are in the area of Nord.")

        st.altair_chart(alt.Chart(xx).mark_circle().encode(
            x='code_departement',
            y='valeur_fonciere',
            size='valeur_fonciere',
            tooltip=['code_departement', 'valeur_fonciere'],
        ).interactive(), use_container_width=True)

    # Pydeck chart
    logger.info("Creating Pydeck chart ...")
    @st.experimental_memo
    def Pydeck_map(dat):

        st.markdown('## Pydeck Map')
        st.markdown("The pydeck map shows the location and Frequency of the property mutations. We can see that the most properties are in the Ile-de-France region.")

        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=48.76,
                longitude=2.4,
                zoom=6,
                pitch=50,
                bearing=-27.36,
            ),
            layers=[
                pdk.Layer(
                'HexagonLayer',
                data=dat[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                radius=200,
                auto_highlight=True,
                elevation_scale=1,
                elevation_range=[0, 3000],
                pickable=True,
                extruded=True,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=dat[["latitude", "longitude"]],
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=200,
                ),
            ],
        ))

    # slider bar
    def select_month():

        st.markdown('## Bar Chart')
        st.markdown("The bar chart shows the number of property mutations per day of the month. We can see that the most property mutations are on the Feburary and May.")

        # st.markdown('## Select month')
        if not st.session_state.get("url_synced", False):
            try:
                pickup_mon = int(st.experimental_get_query_params()["pickup_mon"][0])
                st.session_state["pickup_mon"] = pickup_mon
                st.session_state["url_synced"] = True
            except KeyError:
                pass

        def update_query_params():
            day_selected = st.session_state["pickup_mon"]
            st.experimental_set_query_params(pickup_mon=mon_selected)
        row1_1, row1_2 = st.columns((2, 3))
        with row1_1:
            mon_selected = st.slider("Select month:", 1, 12, key="pickup_mon", on_change=update_query_params)# Bar chart

        return mon_selected


    def Mut_day(d0, mon_sel):


        def histdata(df, dy):
            filtered = df[(df["date_mutation"].dt.month >= dy) & (df["date_mutation"].dt.month < (dy + 1))]
            hist = np.histogram(filtered["date_mutation"].dt.day, bins=30, range=(0, 30))[0]
            return pd.DataFrame({"day": range(30), "Frequency": hist})

        # CALCULATING DATA FOR THE HISTOGRAM
        data = histdata(d0, mon_sel)

        # LAYING OUT THE TOP SECTION OF THE APP
        st.altair_chart(
            alt.Chart(data)
            .mark_area(
                interpolate="step-after",
            )
            .encode(
                x=alt.X("day:Q", scale=alt.Scale(nice=False)),
                y=alt.Y("Frequency:Q"),
                tooltip=["day", "Frequency"],
            )
            .configure_mark(opacity=0.6, color="red"),
            use_container_width=True,
        )

    ####################
    ### INTRODUCTION ###
    ####################

    def Introduction():

        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.0, .1))

        with row0_1:
            st.title('Exploration in property Mutation in France')
        with row0_2:
            st.text("")
            st.subheader('Streamlit App by [Farshad Shamlu](https://www.linkedin.com/in/farshadshamlu/)')

        row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
        with row3_1:
            st.markdown("Hello there! The objective of this project is to applying the process of visual data exploration to the dataset Demandes de valeur foncières"
                        " and publish it with streamlit share. The dataset is available on the [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) website."
                        " The dataset contains the requests for properties value in France. The dataset is updated every month. The dataset contains 1.5 million rows and 17 columns."
                        " The dataset is available from 2016 to 2020 and available in CSV format.")
            # st.markdown("You can find the source code in the []()")

    #################
    ### SELECTION ###
    #################

    def Selection():

        st.sidebar.markdown('## Selection')
        st.sidebar.markdown("In this section, we will select the data that we want to explore. We will select the data based on the year, the region, and the type of property: ** 👇")

        st.sidebar.markdown('### Year')
        option = st.sidebar.selectbox(
        'Which year do you want to visualize?',
        ('2016', '2017', '2018', '2019', '2020'))

        st.sidebar.markdown('### Property type')
        option1 = st.sidebar.multiselect(
        'Which property type do you want to visualize?',
        ('Appartement', 'Maison', 'Dépendance', 'Local industriel. commercial ou assimilé'))

        st.sidebar.markdown('### Region')
        option2 = st.sidebar.multiselect(
        'Which region do you want to visualize?',
        ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
        '12', '13', '14', '50', '15', '16', '17', '18', '19', '21', '22',
        '23', '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31',
        '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42',
        '43', '44', '49', '45', '46', '47', '48', '51', '52', '53', '54',
        '55', '56', '58', '59', '60', '61', '62', '63', '64', '65', '66',
        '69', '70', '71', '72', '73', '74', '76', '77', '78', '79', '80',
        '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
        '92', '93', '94', '95', '971', '972', '973', '974', '75'))
        st.sidebar.write('The complete list of codes, including equivalent region names can be found [here](https://fr.wikipedia.org/wiki/Num%C3%A9rotation_des_d%C3%A9partements_fran%C3%A7ais)')

        return option, option1, option2

    def Pre_processing(op):

        # Get Directory of the data
        # path = f'./Data/full_{op[0]}.csv'

        # LOAD DATA
        raw_data = df #load_data(path)

        # PREPROCESSING
        tr_data = trans_data(raw_data)
        data0 = Clean_data(tr_data)

        # # FILTER DATA BY Local type
        if len(op[1]) == 0:
            data0 = data0
        else:
            data0 = data0[data0["type_local"].isin(op[1])]

        # # FILTER DATA BY Region
        if len(op[2]) == 0:
            data0 = data0
        else:
            data0 = data0[data0["code_departement"].isin(op[2])]

        # Create a new column for the month
        data0 = Mon(data0)

        # Define the data
        data1 = data0[['valeur_fonciere', 'type_local']]
        new_data2 = data1[data1['valeur_fonciere']<= 1500000]
        x2 = data0.groupby("type_local")["valeur_fonciere"].count()
        v1=pd.DataFrame(data0.groupby("code_departement")["valeur_fonciere"].count()).reset_index()

        return data0, new_data2, x2, v1, data1

        # Dataframe
    def Dataframe():
        st.markdown('## Dataframe')
        st.markdown(" As the dataset is very rich, we have a lot of possibilites to create the dashboards we want with the visualizations that interest us the most. So, we will load the columns to create the dashboards we want.")
        st.dataframe(data0)


    # Line Chart
    def Line_chart(data):
        st.markdown('## Line chart')
        st.markdown("In this section, we will create a line chart to visualize the evolution of the average price of the properties in France. We can see that tproperties values has increased over the years.")
        st.line_chart(mean_(data)[f'Year {op[0]}'])

    # Map Chart
    def Map_chart():
        st.markdown('## Map chart')
        st.markdown("In this section, the map shows the location of the property mutations in France. We can see that the most mutations are located in the Ile-de-France region.")
        st.map(data0[["latitude", "longitude"]])

    op = Selection()
    data0, new_data2, x2, v1, data1 = Pre_processing(op)
    Introduction()
    Dataframe()
    Line_chart(data0)
    Map_chart()
    Histogram(data0, radio())
    pie_chart(x2)
    altair_chart(v1)
    Pydeck_map(data0)
    Mut_day(data0, select_month())

    st.write("Done!")

if __name__ == "__main__":
    main()



