import streamlit as st

import altair as alt

import pydeck as pdk

train_area = st.empty()



# California Housing Prices


## Let’s first take a look at imports


with st.echo():

    import tensorflow as tf

    import numpy as np

    import pandas as pd


## Loading the Dataset





with st.echo():

    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()



This will load the entire data in the `housing` variable as you can see below



st.subheader(‘Input Features’)

housing.data

st.subheader(‘Output Labels’)

housing.target



## Splitting the data into Train, Test, and Dev sets




with st.echo():

    from sklearn.model_selection import train_test_split

    X_train_full, X_test, y_train_full, y_test = train_test_split(

        housing.data, housing.target

    )

    X_train, X_valid, y_train, y_valid = train_test_split(

        X_train_full, y_train_full

    )



## Taking a look at the train data


st.write(housing.feature_names)




with st.echo():

    map_data = pd.DataFrame(

        X_train,

        columns=[

            ‘MedInc’, 

            ‘HouseAge’, 

            ‘AveRooms’, 

            ‘AveBedrms’, 

            ‘Population’, 

            ‘AveOccup’, 

            ‘latitude’, 

            ‘longitude’

            ])

    midpoint = (np.average(map_data[“latitude”]), np.average(map_data[“longitude”]))

    st.write(pdk.Deck(

    map_style=”mapbox://styles/mapbox/light-v9″,

    initial_view_state={

        “latitude”: midpoint[0],

        “longitude”: midpoint[1],

        “zoom”: 6,

        “pitch”: 75,

    },

    layers=[

        pdk.Layer(

            “HexagonLayer”,

            data=map_data,

            get_position=[“longitude”, “latitude”],

            radius=1000,

            elevation_scale=4,

            elevation_range=[0, 10000],

            pickable=True,

            extruded=True,

        ),

    ],

))




## Preprocessing


with st.echo():

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_valid = scaler.transform(X_valid)

    X_test = scaler.transform(X_test)



## Creating a model ##



st.sidebar.title(‘Hyperparameters’)

n_neurons = st.sidebar.slider(‘Neurons’, 1, 128, 30)

l_rate = st.sidebar.selectbox(‘Learning Rate’, (0.0001, 0.001, 0.01), 1)

n_epochs = st.sidebar.number_input(‘Number of Epochs’, 1, 50, 20)

#The n_neurons, l_rate, and _nepochs are the inputs taken from the user for training the model. The default values for them are also set. Default value for n_neurons is 30, the default value for l_rate is 0.01 and the default value for n_epochs is 20. So at the beginning the model will have 30 neurons in the first layer, the learning rate will be 0.01 and the number of epochs for which the model will train for is 20. 

with st.echo():

    import tensorflow as tf

    

    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(n_neurons, activation=’relu’, input_shape=X_train.shape[1:]),

        tf.keras.layers.Dense(1)

    ])


## Compiling the model


with st.echo():

    model.compile(

        loss=’mean_squared_error’,

        optimizer=tf.keras.optimizers.SGD(l_rate)

    )



## Training the model



train = st.button(‘Train Model’)

if train:

    with st.spinner(‘Training Model…’):

        with st.echo():

            model.summary(print_fn=lambda x: st.write(“{}”.format(x)))

            history = model.fit(

                X_train,

                y_train,

                epochs=n_epochs,

                validation_data=(X_valid, y_valid)

            )

    st.success(‘Model Training Complete!’)

    

    ## Model Performance

    

    with st.echo():

        st.line_chart(pd.DataFrame(history.history))

    

    ## Evaluating the model on the Test set


   

    with st.echo():

        evaluation = model.evaluate(X_test, y_test)

        evaluation

    
    ## Predictions using the Model

    

    with st.echo():

        X_new = X_test[:3]

        predictions = model.predict(X_new)

    

    ### Predictions

    


    

    ### Ground Truth

   

    y_test[:3]

 


