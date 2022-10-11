# DSI_Project2: Ames Housing Data and Kaggle Challenge

## Group : Happy Three Friends.

## Member : 
1. Supadit Wanotayapitak (Mink)
2. Kant Charoensedtasin (Gun)
3. Warintorn Nawong (Tonzai)

## Problem statement
1. Invent model to evaluate proper price of the property according to appointed features.
2. Matching property features and clients.

## Background
1. We are seeking new opportunity in Ames, Iowa ( aka Happy Trees Town)
2. Studying dataset for property investment to match client’s need
    - Low price-sale
    - Average price-sale
    - High price-sale
    
## 1. Data import
### Data Dictionary
|Feature         |Type   |Description|
|:---------------|:-----:|:-------------------------------------|
|id              |int64  |Unique id|
|ms_subclass     |int64  |The building class|
|ms_zoning       |int64  |The general zoning classification|
|lot_frontage    |object |Linear feet of street connected to property|
|lot_area        |float64|Lot size in square feet|
|street          |int64  |Type of road access
|alley           |object |Type of alley access
|lot_shape       |object |General shape of property
|land_contour    |object |Flatness of the property
|utilities       |object |Type of utilities available
|lot_config      |object |Lot configuration
|land_slope      |object |Slope of property
|neighborhood    |object |Physical locations within Ames city limits
|condition_1     |object |Proximity to main road or railroad
|condition_2     |object |Proximity to main road or railroad (if a second is present)|
|bldg_type       |object |Type of dwelling
|house_style     |object |Style of dwelling
|overall_qual    |int64  |Overall material and finish quality
|overall_cond    |int64  |Overall condition rating
|year_built      |int64  |Original construction date
|year_remod/add  |int64  |Remodel date
|roof_style      |object |Type of roof
|roof_matl       |object |Roof material
|exterior_1st    |object |Exterior covering on house
|exterior_2nd    |object |Exterior covering on house (if more than one material)|
|mas_vnr_type    |object |Masonry veneer type
|mas_vnr_area    |float64|Masonry veneer area in square feet
|exter_qual      |object |Exterior material quality
|exter_cond      |object |Present condition of the material on the exterior|
|foundation      |object |Type of foundation
|bsmt_qual       |object |Height of the basement
|bsmt_cond       |object |General condition of the basement
|bsmt_exposure   |object |Walkout or garden level basement walls
|bsmtfin_type_1  |object |Quality of basement finished area
|bsmtfin_sf_1    |float64|Type 1 finished square feet
|bsmtfin_type_2  |object |Quality of second finished area (if present)
|bsmtfin_sf_2    |float64|Type 2 finished square feet
|bsmt_unf_sf     |float64|Unfinished square feet of basement area
|total_bsmt_sf   |float64|Total square feet of basement area
|heating         |object |Type of heating
|heating_qc      |object |Heating quality and condition
|central_air     |object |Central air conditioning
|electrical      |object |Electrical system
|1st_flr_sf      |int64  |First Floor square feet
|2nd_flr_sf      |int64  |Second floor square feet
|low_qual_fin_sf |int64  |Low quality finished square feet (all floors)
|gr_liv_area     |int64  |Above grade (ground) living area square feet
|bsmt_full_bath  |float64|Basement full bathrooms
|bsmt_half_bath  |float64|Basement half bathrooms
|full_bath       |int64  |Full bathrooms above grade
|half_bath       |int64  |Half baths above grade
|bedroom_abvgr   |int64  |Number of bedrooms above basement level
|kitchen_abvgr   |int64  |Number of kitchens
|kitchen_qual    |object |Kitchen quality
|totrms_abvgrd   |int64  |Total rooms above grade (does not include bathrooms)|
|functional      |object |Home functionality rating
|fireplaces      |int64  |Number of fireplaces
|fireplace_qu    |object |Fireplace quality
|garage_type     |object |Garage location
|garage_yr_blt   |float64|Year garage was built
|garage_finish   |object |Interior finish of the garage
|garage_cars     |float64|Size of garage in car capacity
|garage_area     |float64|Size of garage in square feet
|garage_qual     |object |Garage quality
|garage_cond     |object |Garage condition
|paved_drive     |object |Paved driveway
|wood_deck_sf    |int64  |Wood deck area in square feet
|open_porch_sf   |int64  |Open porch area in square feet
|enclosed_porch  |int64  |Enclosed porch area in square feet
|3ssn_porch      |int64  |Three season porch area in square feet
|screen_porch    |int64  |Screen porch area in square feet
|pool_area       |int64  |Pool area in square feet
|pool_qc         |object |Pool quality
|fence           |object |Fence quality
|misc_feature    |object |Miscellaneous feature not covered in other categories|
|misc_val        |int64  |$Value of miscellaneous feature
|mo_sold         |int64  |Month Sold
|yr_sold         |int64  |Year Sold
|sale_type       |object |Type of sale
|sale_condition  |object |Condition of sale
|saleprice       |int64  |the property's sale price in dollars. This is the target variable that you're trying to predict.|

### Summary 
Total column : 82 columns
Object 43 columns
Numerical data 39 columns

## 2. Data cleaning

There are some of missing data is NMAR type. (From data_description.txt file) The meaning of missing value in each columns is in table below.

|Columns|Desciption|
|:------|:---------|
|pool_qc           | No pool
|misc_feature      | None misc feature
|alley             | No alley access
|fence             | No fence
|fireplace_qu      | No fireplace
|garage_yr_blt     | No Garage 
|garage_qual       | No Garage 
|garage_cond       | No Garage 
|garage_finish     | No Garage 
|garage_type       | No Garage 
| bsmt_exposure    | No basement
| bsmtfin_type_2   | No basement
| bsmtfin_type_1   | No basement
| bsmt_cond        | No basement
|bsmt_qual         | No basement
| mas_vnr_area     | Maybe actually missing
| mas_vnr_type     | Maybe actually missing
| bsmtfin_sf_2     | No basement

### Handing missing value
We divide handling the missing value into 3 group.
1. Large amount of missing value: Drop columns because it may decrease accuracy of the model.

2. Not too large or NMAR type missing value: Impute appropiate value into the column.

**Note** - Some column has text `'No'` represent absense of the value e.g. garage and basement column.

3. Small amount of missing value: Impute appropiate value into the column.

## 3. Exploration data analysis
In this section, we explore data by using correlation and prepare data for training linear regression model.

### Comparison between eliminate and not eliminate outlier
**Compare by using scatterplot**
1. left-handed side graphs are data without eliminate outlier. 
    - Orange line is boundary of saleprice.
    - Red line is boundary of column in x-axis.


2. right-handed side graph are data with eliminate outlier.
    - Outlier are manually eliminated.
    - We decide to eliminate outlier in gr_liv_area, garage_area and total_bsmt_sf columns.
    
![image](https://user-images.githubusercontent.com/104628789/191751468-6006c116-a5ad-4c64-ba9c-fd4beabeb964.png)
![image](https://user-images.githubusercontent.com/104628789/191751544-1b8f4db1-5959-4855-a92a-3bff42003e29.png)
![image](https://user-images.githubusercontent.com/104628789/191751571-2774a5d0-d2a7-4e2c-8be8-16219e946b76.png)

    
**Note** - For 2 right-handed side point of the graph saleprice vs total_bsmt_sf with outlier, we don't eliminate them because they are gone with above elimination.

After elimination of outlier, The relation between `gr_liv_area` `garage_area` `total_bsmt_sf` and sale price is better.


## 4. Feature engineering
In this project, we have 2 section of feature engineering e.g. log transform column and encoding columns

1. Log tranformation
    - We transform the `saleprice` column by using `np.log()` because the difference between scale of x and y.
    - Expectation: we expect to see more linearity after transforming `saleprice` column

![image](https://user-images.githubusercontent.com/104628789/191751689-e1c39c2e-210b-499a-9faf-d7009831990d.png)
![image](https://user-images.githubusercontent.com/104628789/191751732-db843955-0516-4e04-ae55-2282e2b5308e.png)
![image](https://user-images.githubusercontent.com/104628789/191751761-b0797879-913e-4388-af7e-2b6f16c0b08d.png)
![image](https://user-images.githubusercontent.com/104628789/191751804-b49f54e1-1567-4e0e-9e76-a46cf8f11de3.png)
![image](https://user-images.githubusercontent.com/104628789/191751972-e3c4f511-b468-4ac9-bfac-c67a77389769.png)
![image](https://user-images.githubusercontent.com/104628789/191752020-dcd13839-f749-41b1-acb7-c91385e3c1fa.png)
![image](https://user-images.githubusercontent.com/104628789/191752083-49189733-4b37-42a4-ac62-9bb7d3438856.png)
![image](https://user-images.githubusercontent.com/104628789/191752112-ec08c135-adfb-4441-bacd-39cebfdc1df0.png)
![image](https://user-images.githubusercontent.com/104628789/191752137-871cd623-ce48-4b18-a606-02b8f75651bb.png)
![image](https://user-images.githubusercontent.com/104628789/191752165-2504ee4b-97cc-48d7-a35f-496002fe0dbd.png)

    
After take log to `saleprice`, `saleprice` scale is smaller and the correlation should be more linear. It may help to improve the linear regression model.

2. One-hot encoding
    - We use only one-hot encoding on few numeric feature. The idea is applied one-hot encoding for numeric feature that is category/ordinal.
    - Reason: Eventhough, the value in `numeric_category_col` is numeric, each value in the column represent the categotical data instead of the number.
    
## 5. Training model
In this section, we will train the model and check for bias, variance, split reliabilty and model performance.

### Model selection
We don't use the lasso regression because lasso may penalize all of coefficient to zero. Since we don't require model in the sweet spot, we decide to use ridge regression.

## 6. Model result
After we got the model result, we need to check the performance of the model.
1. Check Bias (underfitting): The model has high $R^{2}$ value. Then the model is not underfitting.
2. check Variance (overfitting): The $R^{2}$ of train set is similar to $R^{2}$ of test set. Then the model is not overfitting.
3. Split reliability: The 5 value of $R^{2}$ from cross validation don't diverge. Then the model performance don't rely on spliting data. 
4. Model performance:
    - $R^{2}$ training set = 0.9263
    - $R^{2}$ test set = 0.919
    - RMSE training set = 19489
    - RMSE test set = 20956
    
## 7. Conclusion and recommendation
### Conclusion
1. Model RMSE has decreased 37 % which lead to return on property investment increase by 27 %.
2. House Pricing Prediction could personlize based on customer demand in individual house features i.e. number of Garage, Lot Space etc. 
3. Some feature could be selectively stored to optimize between cost of storage and model performance i.e. pool_qc ,misc_feature etc.

### Recommendation
1. To explore more predictively powerful algorithm.
2. To customize key features based on individual city.
3. To expand new opportunities in other cities in the U.S. and the world.
