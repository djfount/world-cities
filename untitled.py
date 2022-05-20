import streamlit as st

import altair as alt

import pydeck as pdk

train_area = st.empty()

“””

# California Housing Prices

This is the California Housing Prices dataset which contains data drawn from the 1990 U.S. Census. The following table provides descriptions, data ranges, and data types for each feature in the data set.

## Let’s first take a look at imports

“””

with st.echo():

    import tensorflow as tf

    import numpy as np

    import pandas as pd

“””

## Loading the Dataset

We will use the scikit-learn’s dataset module to lead data which is already cleaned for us and only has the numerical features. 

“””

with st.echo():

    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()

“””

This will load the entire data in the `housing` variable as you can see below

“””

st.subheader(‘Input Features’)

housing.data

st.subheader(‘Output Labels’)

housing.target

“””

## Splitting the data into Train, Test, and Dev sets

This is one of the most important things at the beginning of any Machine Learning solution as the result of any model can highly depend on how well you have distributed the data into these sets. 

Fortunately for us, we have scikit-learn to the rescue where it has become as easy as 2 lines of code.

“””

with st.echo():

    from sklearn.model_selection import train_test_split

    X_train_full, X_test, y_train_full, y_test = train_test_split(

        housing.data, housing.target

    )

    X_train, X_valid, y_train, y_valid = train_test_split(

        X_train_full, y_train_full

    )

“””

The `train_test_split()` function splits the data into 2 sets where the test set is 25% of the total dataset. We have used the same function again on the train_full to split it into train and validation sets. 25% is a default parameter and you can tweak it as per your needs. Take a look at it from the [Scikit-Learn’s Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

## Taking a look at the train data

The columns represent the following data:

“””

st.write(housing.feature_names)

“””

Now let’s look at the location of the houses by plotting it on the map using Latitude and Longitude values:

“””

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

“””

**Feel free to zoom in or drag while pressing ALT key to change the 3D viewing angle of the map, as required.**

## Preprocessing

As pointed out earlier, this dataset is already well preprocessed by scikit-learn for us to use directly without worrying about any NaN values and other stuff.

Although, we are going to scale the values in specific ranges by using `StandardScaler` to help our model work efficiently.

“””

with st.echo():

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_valid = scaler.transform(X_valid)

    X_test = scaler.transform(X_test)

“””

## Creating a model

We will be creating a simple Sequential Model with the first layer containing 30 neurons and the activation function of RELU.

The next layer will be a single neuron layer with no activation function as we want the model to predict a range of values and not just binary or multiclass results like classification problems.

“””

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

“””

## Compiling the model

Tensorflow keras API provides us with the `model.compile()` function to assign the optimizers, loss function and a few other details for the model.

“””

with st.echo():

    model.compile(

        loss=’mean_squared_error’,

        optimizer=tf.keras.optimizers.SGD(l_rate)

    )

“””

## Training the model

In order to train the model you simply have to call the `fit()` function on the model with training and validation set and a number of epochs you want the model to train for.

**Try playing with the hyperparameters from the sidebar on the left side and click on the `Train Model` button given below to start the training.**

“””

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

    “””

    ## Model Performance

    “””

    with st.echo():

        st.line_chart(pd.DataFrame(history.history))

    “””

    ## Evaluating the model on the Test set

Again another important but easy step to do is to evaluate your model on the test data which it has never seen before. Remember that you should only do this after you are sure enough about the model you’ve built and you should resist making any hyperparameter tuning after evaluating the model on the test set as it would just make it better for the test set and again there will be a generalization problem when the model will see new data in the production phase.

    “””

    with st.echo():

        evaluation = model.evaluate(X_test, y_test)

        evaluation

    “””

    > This loss on the test set is a little worse than that on the validation set, which is as expected, as the model has never seen the images from the test set.

    “””

    “””

    ## Predictions using the Model

    “””

    with st.echo():

        X_new = X_test[:3]

        predictions = model.predict(X_new)

    “””

    ### Predictions

    “””

    predictions

    “””

    ### Ground Truth

    “””

    y_test[:3]

 


#process 2

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


# Title - The title and introductory text and images are all written in Markdown format here, using st.write()

st.write("""
[![craigdoesdata logo][logo]][link]
[logo]: https://www.craigdoesdata.de/img/logo/logo_w_sm.gif
[link]: https://www.craigdoesdata.de/
# California Housing Prices
![Some California Houses](https://images.pexels.com/photos/2401665/pexels-photo-2401665.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260)
Photo by [Leon Macapagal](https://www.pexels.com/@imagevain?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/aerial-photography-of-concrete-houses-2401665/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels).
------------
This app predicts California Housing Prices using a machine learning model powered by [Scikit Learn](https://scikit-learn.org/).
The data for the model is the famous [California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices) Dataset.
Play with the values via the sliders on the left panel to generate new predictions.
""")
st.write("---")


# Import Data - the original source has been commented out here, but left in so the CSV files can be sourced again in future, if needed.


# Import the data from Google
# train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
# train_df = train_df.reindex(np.random.permutation(train_df.index)) # randomise the examples
# test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# full_df = pd.concat([train_df, test_df])

train_df = pd.read_csv('data/train_df.csv')
test_df = pd.read_csv('data/test_df.csv')
full_df = pd.read_csv('data/full_df.csv')

# Assign features (X) and target (Y) to DataFrames
X = full_df
Y = X.pop('median_house_value')

# Sidebar - this sidebar allows the user to set the parameters that will be used by the model to create the prediction.
st.sidebar.header('Specify Input Parameters - these will determine the predicted value.')

def features_from_user():
    longitude = st.sidebar.slider('Longitude', float(full_df.longitude.min()), float(full_df.longitude.max()), float(full_df.longitude.mean()))
    latitude = st.sidebar.slider('Latitude', float(full_df.latitude.min()), float(full_df.latitude.max()), float(full_df.latitude.mean()))
    housing_median_age = st.sidebar.slider('Housing Median Age', float(full_df.housing_median_age.min()), float(full_df.housing_median_age.max()), float(full_df.housing_median_age.mean()))
    total_rooms = st.sidebar.slider('Total Rooms', float(full_df.total_rooms.min()), float(full_df.total_rooms.max()), float(full_df.total_rooms.mean()))
    total_bedrooms = st.sidebar.slider('Total Bedrooms', float(full_df.total_bedrooms.min()), float(full_df.total_bedrooms.max()), float(full_df.total_bedrooms.mean()))
    population = st.sidebar.slider('Population', float(full_df.population.min()), float(full_df.population.max()), float(full_df.population.mean()))
    households = st.sidebar.slider('Households', float(full_df.households.min()), float(full_df.households.max()), float(full_df.households.mean()))
    median_income = st.sidebar.slider('Median Income', float(full_df.median_income.min()), float(full_df.median_income.max()), float(full_df.median_income.mean()))
    
    data = {'Longitude': longitude,
            'Latitude': latitude,
            'Housing Median Age': housing_median_age,
            'Total Rooms': total_rooms,
            'Total Bedrooms': total_bedrooms,
            'Population': population,
            'Households': households,
            'Median Income': median_income}

    features = pd.DataFrame(data, index = [0])
    return features

df = features_from_user()

# Display specified input parameters
st.write('Specified Input Parameters:')
st.table(df)
st.write('---')

X = X.drop('Unnamed: 0', axis=1)


# Build Regression Model - the 3 lines below are commented out to save processing time.
# These lines would allow us to re-run the model whenever required.

# model = RandomForestRegressor()
# model.fit(X, Y)
# dump(model, 'model.joblib') 

# Load the saved model
model_new = load('data/model.joblib') 

# Apply Model to Make Prediction
prediction = int(model_new.predict(df))
prediction_nice = f"{prediction:,d}"

# Main Panel - display prediction

st.header('Prediction of Median House Value:')
st.write('Based on your selections, the model predicts a value of %s US Dollars.' % prediction_nice)
st.write('---')

