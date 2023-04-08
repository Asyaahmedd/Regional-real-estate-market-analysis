# Import neccessary libraries
#==================================================================
import warnings
from glob import glob

import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
#=================================================================
#Prepare and Explore Data
#================================================================
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    # drop features with high null counts
    df.drop(columns=["floor","expenses"],inplace=True)
    # Drop low- and high-cardinality categorical variables
    df.drop(columns=["operation","property_type","properati_url","currency"],inplace=True)
    #drop leaky columns
    df.drop(columns= ['price','price_aprox_local_currency','price_per_m2','price_usd_per_m2'],inplace=True)
    # Drop columns with multicollinearity
    df.drop(columns=["surface_total_in_m2","rooms"],inplace=True)
    return df

#Use glob to create a list that contains the filenames for all the Buenos Aires real estate CSV files in the data directory.
files = glob("data/buenos-aires-real-estate-*.csv")

#create a list contains the cleaned DataFrames for the filenames collected in files
frames = [wrangle(file)for file in files]

#Combine the DataFrames in frames into a single df
df = pd.concat(frames,ignore_index=True)
print(df.info())
df.head()
corr = df.select_dtypes("number").drop(columns="price_aprox_usd").corr()
sns.heatmap(corr)
#=========================================================================
#Split Data
#=========================================================================
target = "price_aprox_usd"
y_train = df[target]
X_train = df.drop(columns=["price_aprox_usd"])
#========================================================================
#Build Model
#========================================================================
#Baseline
y_mean = y_train.mean()

y_pred_baseline = [y_mean]*len(y_train)
print("Mean apt price:",round(y_mean,2) )

print("Baseline MAE:", mean_absolute_error(y_train,y_pred_baseline))
#Mean apt price: 132383.84
#Baseline MAE: 44860.10834274133

#Iterate
# Create a pipeline named model that contains a OneHotEncoder, SimpleImputer, and Ridge predictor.
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()

)
model.fit(X_train,y_train)
#===============================================================================
#Evaluate
y_pred = model.predict(X_train)
print("Training MAE:", mean_absolute_error(y_pred,y_train))
#Training MAE: 24207.107190330353
#==============================================================================
#Test
#==============================================================================
X_test = pd.read_csv("data/buenos-aires-test-features.csv")
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()
#=============================================================================
#Communicate Results
#=============================================================================
#Create a function make_prediction that takes four arguments (area, lat, lon, and neighborhood) and returns your model's prediction for an apartment price.
def make_prediction(area, lat, lon, neighborhood):
    data ={
    "surface_covered_in_m2":area,
    "lat":lat,
    "lon":lon,
    "neighborhood":neighborhood
    }
    df = pd.DataFrame(data,index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted apartment price: ${prediction}"
#call function
make_prediction(110, -34.60, -58.46, "Villa Crespo")
#'Predicted apartment price: $250775.11'
# Create an interact function in Jupyter Widgets.

interact(
    make_prediction,
    area=IntSlider(
        min=X_train["surface_covered_in_m2"].min(),
        max=X_train["surface_covered_in_m2"].max(),
        value=X_train["surface_covered_in_m2"].mean(),
    ),
    lat=FloatSlider(
        min=X_train["lat"].min(),
        max=X_train["lat"].max(),
        step=0.01,
        value=X_train["lat"].mean(),
    ),
    lon=FloatSlider(
        min=X_train["lon"].min(),
        max=X_train["lon"].max(),
        step=0.01,
        value=X_train["lon"].mean(),
    ),
    neighborhood=Dropdown(options=sorted(X_train["neighborhood"].unique())),
);
