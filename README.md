# Cm_Portfolio
data science portfolio

## FEATURED PROJECT 1 : Determining Housing Prices using Data Science

#### Background
   The Iowa State University and the Commerce Committee of the Iowa House of Representatives are looking into the community of Ames Iowa to understand home values in our region. The Real Estate Appraisers Coalition of Ames Iowa and I have taken on the task of creating a simplified model that will predict home values based on multiple qualities of the home. We will be looking at everything from the numbers of bedrooms and baths, locationality, quality and condition, square footage, and materials.
We have a data set containing information from the Ames Assessor’s Office used in calculating the assessed values for individual residential properties sold in Ames, IA from 2006 to 2010. The detailed description in the data documentation is here:  ([DataDoc](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt))
We will be sorting through the eighty-two variables and identifying the most effective variables to include in our model. Then we will create our model and measure its accuracy.

### Problem Statement:
    Let's say: I am a data scientist working for the Real Estate Appraisers Coalition of Ames Iowa.
    We have a lot of homes in the region that need to be quickly appraised for government and educational purposes.
    We have data from the Ames Iowa Assessor’s Office used in computing values for individual residential properties 
    sold in Ames, IA from 2006 to 2010. We are going to create a model that predicts prices off home assessment 
    data entries, and in doing so we will determine what factors make the most significant impacts on the model
    to make the model more effective. Furthermore, we will test and verify these results on a competitive website for 
    data scientists (keggle) and come to an analysis on how effective the model actually is.
### Essentially we are asking the question:
1. What factors really matter in determining a fair market value?
    
### Brief Summary of Analysis and Visualizations:
The plan of attack on cleaning this data changed numerous times through the process of analyzing the data. The data has 82 columns which include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables. 

#### Cleaning the Ordinals:
The 23 ordinal values had a lot of null values, but 'NA' was often an option. So in the interest of simplicity I assumed the assessor did not mark anything there and fulled nulls with 'NA', because the home likely did not have that feature. This worked great for the basement, garage, fireplace, fence, pool and other quality and condition assessments with no values. 
I set up a dictionary to change the values in the data set of all the ordinal columns into ordered numerical values. NA being set at 0 and the quality and condition starting from worst at 1 and best at the maximum. The features did not all have the same numerical quantity of ranking these qualities, however that can be fixed in the modeling process by using a standard scaler on the data.   

#### Cleaning the Nominals:
For the nominal values, all null values were set to 'NA'. Then I individually one-hot encoded them and checked their correlation to determine what was most relevant to price. I'll go further into detail on this process later in this report. 

#### Cleaning the Numerical:
Most of the numerical values did not have many null values. In fact, 10 of the numerical columns only had one single null value. So not to skew the data I set the null value in each of those 10 columns to the mean of their respective columns. This may have had a small but hardly significant dampening effect on the standard deviation. This is something that has interested me for further inspection later. What should one do if there are lots of null values in numerical columns? Replacing with the mean cannot be good when there are large amounts of data with null values. However in this case because these numerical columns are such a minor subset of the larger dataset I'm going to go with this method. 

Lot Frontage is one of those cases, in which it has many null values.  I decided not to include Lot Frontage as a determining variable in my model because of these null values. Initially I measured Lot Frontage correlation to sale price as 34% with those null values as null (so not included in the calculation), and I also calculated it with setting the nulls to the mean and it only slightly changed the correlation down to 32.5%. However, setting the nulls to equal zero significantly decreased the correlation to the sale price down to 18%. Seeing that I had no actual clue why these values were null I figured best to leave it out of the model entirely, just in case. Perhaps with a different method of cleaning null values I could solve this issue. Lot Frontage has a significant impact on price so it is a big bummer to cut it out. 

### Exploratory Data Analysis

#### Heatmap of the absolute value of all variables correlated to sale price
![title](photos/proj_2_absolutevalue_ALL_heatmap.jpg)

 To analyze the data, one of the first actions I took was to check the correlations of the numerical columns and ordered columns and take note of the columns of interest. I considered all the columns with correlation coefficients with an absolute value greater than 0.45 to be worthy of including in the model because of their strong relationship to price.  Furthermore, for simplification, I combined living area above grade with basement total square footage minus the unfinished basement area. This will give us the total living area and that seems to be one of the most important factors in making a clear and coherent model. The best fit line follows a more linear path with these elements combined.
 
 Then I went through the nominal values analyzing the domified version of each and determined if they are worth including in the model. For the nominal values I picked the 5 values that had the largest coefficients (most were lower than 0.45). 
 
Domified Variables of Importance.
|Variable|Coef Max|Coef Min|
|---|---|---|
|central_air| 0.28|-0.28|
|house_style| 0.2|-0.2|
|roof_style| 0.27|-0.25|
|mas_vnr_type| 0.31|-0.43|
|foundation| 0.53|-0.36|

 For numerical and ordinal values, I only included those with coefficients greater than 0.45. I only removed variables if I found the information redundant or repetitive. Listed below are the variables used.  
 
 Numerical and Ordinal Variables of Importance.
|Variable|Coef|Type|
|---|---|---|
|overall_qual	|0.800207| Discrete|
|tot_liv_area	|0.716558| Continuous
|exter_qual	    |0.712146| Discrete|
|kitchen_qual  	|0.692336| Discrete|
|garage_area	|0.650246| Continuous|
|bsmt_qual   	|0.612188| Discrete|
|year_remod/add	|0.550370| Discrete|
|fireplace_qu   |0.538925| Discrete|
|full_bath    	|0.537969| Discrete|
|totrms_abvgrd	|0.504014| Discrete|
|heating_qc   	|0.458354| Discrete|
  
 I compartmentalized together the 'sale_type', 'exterior_1st' and 'neighborhood' variables into 3 buckets of 'good', 'ok', and 'bad' in respect to their coefficients. This was to simplify the model and prevent an overabundance of columns created during the process. The exact cutoffs were +/- 5% for the neighborhoods, +/- 10% for the home exterior and the sale type. 
 
Bucketed Items of Importance
|Variable|Coef Max|Coef Min|
|---|---|---|
|sale_type|0.36|-0.21|
|exterior_1st|0.34|-0.19|
|neighborhood|0.45|-0.21|
 
 Finally, I removed outlines in general living area, total basement square footage and the 1st floor square footage. It was a total of 3 rows removed. As all three of these column's information is contained in total living area. The significance these 3 rows had on the best fit graph of is demonstrated below. The best fit line is set with an order of 2.


#### Before and After Outliers Removed for Total Living Area vs Price.
![title](photos/proj_2_liv_vs_sale_b4.jpg)
![title](photos/proj_2_liv_vs_sale_after.jpg)

### Setting Up a Baseline Model
 The baseline model I set up for my example is a basic linear regression using only the top 5 variables of Overall Quality, Total Living Area, Exterior Quality, Kitchen Quality, and Garage Area. Doing a train test split on our training data, I got back a score of 84.6% on our split train, and 83.3% on our split testing data. This is not bad considering these are only 5 variables out of 80.


### Setting Up the Model
 For the model itself, I used a train test split to verify the data on itself before applying it to the new test data. I utilized polynomial features to the 2nd degree and fit the data to a standard scaler in order transform the data so rankings from 1-5 and rankings from 1-10 would be comparable. 
 After attempting a few different methods, I settled on fitting the data to a ridge cross validation model. The model's r2 score indicated that the model was 93.6% fit to the training data and 87.6% accurate at determining the price in my split test set. I considered this adequate to use even though it is overfit on my training model because it had the largest improvement on the overall test dataset.


### Conclusion/Recommendations:        
We can say with confidence that the overall quality, total living area, exterior quality, kitchen area and garage area have significant impacts on the price of the house. Going from 5 variables to 19 variables and using a more robust model we only gained 4.3% accuracy. 

I would like to dig further into methods of fitting and transforming the data. I'd like to see if we analyze more what columns should be included and what models could increase the accuracy without overfitting the data.


## Project 3 - Star Trek vs Star Trek, Webscrapping & NLP - Executive Summary : 

Hello there. In this notebook I will demonstrate how to scrape data off the web using a web 
application programming interface or web API. In our case we are using the Pushshift's reddit API to extract submissions. 
In addition to this, we will use classification models and natural language processing (NLP) to determine which subreddit the language came from. 
This can be useful in many applications of technology that use text and/or verbal language to understand humans.
It is important to learn if you are going to explore a career in data science like I am. I hope this lesson is, as I am learning as I go, as 
insightful to you as it has been for me. This analysis will take the following steps to classify reddit submissions of two of my 
all-time favorite science fiction series.

Using Pushshift's API, we will collect posts from two subreddits, Star Wars and Star Trek.
Then we will use NLP to train a classifier on determining which subreddit a given post came from. This is a binary classification problem.

## Project 4 - Shelter Animal Outcomes - Executive Summary :

This was my first group project. We used git to and slack to collaborate and in just a few hours we picked a topic, perforemed basic eda and cleaning, 
models, and visualizations. The goal of our project was to build a classifier which will correctly predict the outcome for a shelter animal based on 
select characteristics of animals. 
The problem is a multi-class classification with the labels being: 'Adoption', 'Return_to_owner','Transfer', 'Euthanasia', and 'Died'.

## Project 5 - Wildfire Prediction - Executive Summary :

2020 was the most active fire season in the Western United States’s recorded history. California had the single worst fire season in it’s history, 
while Arizona had the worst in a decade. Oregon had its most destructive fire season meanwhile Washington and Colorado had several of their all time 
largest wildfires. Overall 10.2 million acres of land went up in flame and 46 people lost their lives. 13,887 buildings were destroyed and the total
cost is upwards of 19.88 billion USD. It is evident that fire is a clear and present danger in the western united states.

The global atmospheric monitoring satellite Copernicus has recorded CO2 emissions from the 2020 fires and noted that 
“The fires are also emitting lots of smoke and pollution into the atmosphere; those in California and Oregon have already emitted far more carbon in 2020
than in any other year since CAMS records begin in 2003” - CAMS monitors smoke release from devastating US wildfires | Copernicus. We decided to investigate 
the relationship between weather data (precipitation, temperatures, and drought) and the occurrence of fires, and to attempt building a model which would
predict the destructive sizes of wildfire to help prevent the associated damage for our communities.

## Project 6 - AI generated Star Trek Scripts - Executive Summary :

I'm interested in AI and deep neural networks so for this project I expermented with gpt-2. I trained gpt-2 on Star Trek scripts and created original stories 
from scratch. 
