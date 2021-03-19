# CM Vigil April : Portfolio
my data science portfolio

--------------------------------------------------------------------

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
We can say with confidence that the overall quality, total living area, exterior quality, kitchen area and garage area have the most significant impacts on the price of the house. Going from 5 variables to 19 variables and using a more robust model we only gained 4.3% accuracy on the testing datasets. However excluding Lot Frontage due to the many null values likely had a significant dampening effect on imporving our accuracy.

---------------------------------------------------------------------------------------------------------------------------

## FEATURED PROJECT 2 : Star Trek vs Star Trek, Webscrapping with Web API’s and NLP

### Executive Summary
Hello there. My name is Cm April, nice to make your acquaintance.  In this project I will demonstrate how to scrape data off the web using a web application programming interface or web API. In our case we are using the [Pushshift's](https://github.com/pushshift/api) reddit API to extract submissions. In addition to this, we will use classification models and natural language processing (NLP) to determine which subreddit the language came from. This can be useful in many applications of technology that use text and/or verbal language to understand humans.  It is important to learn if you are going to explore a career in data science like I am. I hope this lesson is, as I am learning as I go, as insightful to you as it has been for me. This analysis will take the following steps to classify reddit submissions of two of my all-time favorite science fiction series.
1.	Using [Pushshift's](https://github.com/pushshift/api) API, we will collect posts from two subreddits, Star Wars and Star Trek.
2.	Then we will use NLP to train a classifier on determining which subreddit a given post came from. This is a binary classification problem.

### Problem Statement
A close friend of mine recently asked 
“What’s the difference between Star Trek and Star Wars?” 
and I nearly let me emotions get the best of me. How could someone close to me go so many years living a life without understanding what Star Wars and Star Trek are about? It hit me that there may be a lot of people out there that do not know the difference between and Vulcan and a Gungan. Horrifying! 
Well, well, well, I could not let my friend live on in darkness, so this here is a little thing I decided to do about that. For all of you reading this that have not seen Star Wars or Star Trek I mean no offence, but I highly recommend you immerse yourself in the experience sooner rather than later. I have empathy for your sad misfortune though and perhaps you do not have the time yet to watch 12 Star Wars and 13 Star Trek movies plus their respective television episodes or to dive deeper into their extended universes through books or video games. In this case I have the solution for you. My classification model here will take data from these sci-fi series subreddits and make it so that you will know if the language is about Star Trek or Star Wars. 

### Essentially we are asking the question:
How can we know if someone is discussing Star Wars or Star Trek?

### Data Collection 
The data was collected using the API pushshift. Utilizing the requests library on python I collected both comments and submissions. The model is created from submissions exclusively. The comments pulled will be for further research. The parameters for web scaping are available on Pushshift API (github.com). The code in the web scraping notebook provided takes 20 loops through the subreddit of choice, taking 100 posts at a time, at the starting time of when you run the code and working backwards, taking a breather between cycles (to not alarm the website) before continuing the loop to completion.  This process gives us 4000 data entries of submissions, an evenly distributed 2000 entries for each subreddit.

### Data Cleaning and EDA
It is unwise to charge forward without doing some exploring of your surroundings. In the data exploration and cleaning I combined the self-text and title features, cleaned the text characters to fill in null values with an empty space and removing any special characters, and further on down the road when I found outliers I came back and here to remove them. I tokenized, lemmatized and stemmed out the words, considering stop words as well to explore the data in depth and even did a word frequency check that indicates the most common words in each subreddit. However this is a lot of overlap in these simmilar subreddits so we will have to see which are important fror the model after we evaluate it.

![10 MOST COMMON STAR TREK](images/startrek_10common.png)

![10 MOST COMMON STAR WARS](images/starwars_10common.png)

I checked for outliners in the extremes of word length and from count vectorizing my text data. Fortunately, there was not many although that may be different depending on when you scrape more data. After these items are completed and that data looks good and we may proceed to the model. 
![GRAPH OF THE EXPLAINED VARIANCE](images/explained_variance.png)

![GRAPH OF THE COUNT VECTERIZED DATA](images/components.png)

### Feature Engineering 
Important features such as word count and a sentiment intensity analyzer are calculated and added to the data frame to include in the models. A function transformer is set to create the columns we care about for the model. Text data is set to the ‘all’ column which is a cleaned and combined version of self-text and title. Our numerical data is our recently calculated word count and sentiment. 

### Model Selection 
I tested on two models, an adaboost and a logistic regression classifier model and ran them through a cross validated grid search. The adaboost was the least over/under fit model however the logistic region performed at a higher accuracy level so I selected it. 

![ADABOOST Confusion MATRIX](images/adaboost.png)
![Logreg Confusion MATRIX](images/logreg.png)

### Evaluation and Conclusion

![GRAPH OF THE 10 MOST COMMON STAR WARS](images/top10starwars.png)
![GRAPH OF THE 10 MOST COMMON STAR TREK](images/top10startrek.png)

In conclusion for those of you that have not seen or heard of Star Wars or Star Trek you could memorize what category these 20 words are in and have a solid idea of which fandom your friend is talking about. This model evaluates textual data to identify what people are discussing in two distinct but similar categories. If you are like me and aspiring to learn more about data science, you should find this quite fascinating. The fact we can teach a computer to interpret a common theme from human language has far reaching potential. Most people do not get the opportunity to explore this type of knowledge and I am excited to be on this journey with you. 

--------------------------------------------------------------------------------

## FEATURED PROJECT 3 : Shelter Animal Outcomes TEAM HACKATHON

This was my first group project and we participated in a hackathon. We used git to and slack to collaborate and in just a few hours we picked a topic, performed basic eda and cleaning, modeled, created visualizations and a powerpoint presentation. The goal of our project was to build a classifier which will correctly predict the outcome for a shelter animal based on the select characteristics of animals, as well as the ultimate goal of teambuilding and learning how to do GIT commits and pushes and merge requests on group projects. Our focus on animal outcomes is to help animal lovers and enthusiast understand which animals are more at risk in shelter enviorments. 
The problem is a multi-class classification with the labels being: 'Adoption', 'Return_to_owner', 'Transfer', 'Euthanasia', and 'Died'. We looked at animal color, the dates of the outcomes for animals on each month, and animal ages. In our analysis we learned there are more adoptions of puppies, and adoptions peak in december and the beginning of summer. 

-----------------------------------------------------------------------------------

## FEATURED PROJECT 4 : Wildfire Size Predictions GROUP PROJECT

### Fire Risk Prediction Analysis

### Project 5: A group project focused on prediction of fire risk based on meteorological data.

## Table of contents
* [Tech Stack](###Tech-Stack)
* [Problem Statement](###Problem-Statment)
* [Summary](###Summary)
* [Data](###Data)
* [Modeling](###Modeling)
* [Conclusions](###Conclusions)
* [Recommendations](###Recommendations)
* [Next Steps](###Next-Steps)

----------
### Tech Stack
This Project is created with:
* Amazon AWS S3
* Amazon CLI
* Boto3
* Matplotlib
* Mpl_toolkits
* Python3
* Seaborn
* Sklearn
* Sqlite
* Statsmodels
* Streamlit
* Tensorflow

---------
### Problem Statement

This project uses weather data in a classification model to determine which of the Western US states are at high risk of large fires, in order to improve local preparedness.

![](/visuals/fire_size_vs_temp_precip_by_month.png)

--------
### Summary

Our project focused on the following 11 States in the US: 
|States| | 
|---|---|
|Arizona|Montana |
|California|New Mexico |
|Colorado|Nevada|
|Idaho|Utah |
|Oregon|Washington |
|Wyoming||

2020 was the most active fire season in the Western United States’s recorded history. California had the single worst fire season in it’s history, while Arizona had the worst in a decade. Oregon had its most destructive fire season meanwhile Washington and Colorado had several of their all time largest wildfires. Overall 10.2 million acres of land went up in flame and 46 people lost their lives. 13,887 buildings were destroyed and the total cost is upwards of 19.88 billion USD. It is evident that fire is a clear and present danger in the western united states. 

The global atmospheric monitoring satellite Copernicus has recorded CO2 emissions from the 2020 fires and noted that “The fires are also emitting lots of smoke and pollution into the atmosphere; those in California and Oregon have already emitted far more carbon in 2020 than in any other year since CAMS records begin in 2003” - [CAMS monitors smoke release from devastating US wildfires | Copernicus](https://atmosphere.copernicus.eu/cams-monitors-smoke-release-devastating-us-wildfires). We decided to investigate the relationship between weather data (precipitation, temperatures, and drought) and the occurrence of fires, and to attempt building a model which would predict the destructive sizes of wildfire to help prevent the associated damage for our communities.

Following National Wildfire Coordinating Group's convention which groups fires into ranges of fire size based on the number of acres within the final fire perimeter, we chose to set up our model as a a multi-classification where class "A" corresponds to fires smaller than 0.01 acres, "B" - 0.225 acres, "C" - 10 acres, "D" - 100 acres , "E" - 300 acres , "F" - 1000 acres, and "G" - fires larger than that.

----------------
### Data

We used two sources of data which were studied through EDA and then combined into a single data frame which informed the modeling phase:
- Meteorological dataset covering 120 years of weather information for the 11 western US states of: AZ, CA, CO, ID, NM, NV, MT, OR, UT, WA, and WY, including metrics and indexes describing precipitation, temperatures, and droughts;
- A spatial database of 1.88 million wildfires that occurred in the United States from 1992 to 2015 and burned 140 million acres burned during the 24-year period. This data was originally generated to support the national Fire Program Analysis (FPA) system and is currently obtainable from Kaggle.com. The data set includes: discovery date, final fire size, and a point location (latitude and longitude) among many other features.

The two datasets were combined by matching weather information and fire data on the combination of month-year-state for each of the fires that burned from 1992 to 2015 in the eleven states of interest. Due to the cumulative nature of meteorological affects on drought severity, we chose to include drought, temperature and precipitation trailing averages over 12-, 9-, 6-, and 3-months.

We also took a deep dive into NOAA wind data but discovered that the combined datasets were far too large to add to our existing dataframe.  Wind direction is a great weather predictor, and because wind speed can feed fires, we believe that adding wind data, such as wind speed, gusts and potentially wind direction would have added significant value to our models.

Another interesting data set we encountered was foliage data from Google Earth Engine. This required setting up an account with Google and being accepted to use their engine, and then exploring data using Javascript. It became too cumbersome for our efforts.

![](/visuals/fire_size_vs_temp_precip_by_month.png)

---------
### Modeling

The project ultimately uses two main models. Neural network for predictive power and Random Forest Classifier for feature importance. We optimized the neural network on recall score focusing on true positive rate and capturing large fires over small fires. Large fires being more destructive and being more in line with the scope of the project at the expense of smaller fires. The final chosen neural network model topology optimizes recall over accuracy. When we focused on accuracy, we were predicting the majority class (small fires) over the minority (large fires) which missed the most destructive wildfires. 

To improve our models, we employed the modeling technique of bootstrapping which gave us a more normal distribution of wildfire classes. This way, we were able to capture our larger fires. It greatly improved recall which is ultimately the target we wanted to pursue, as this helped us predict  larger and more destructive wildfires. 

The second modeling breakthrough we had was harnessing geospatial data through KMeans clustering of longitude and latitudinal data. We then One-Hot-Encoded it which gave us a sparse matrix that was the most important predictive element of our model. We believe that this is because terrain features matter immensely when determining the potential size of a wildfire. (See confusion matrix below)

The third major breakthrough was the trailing averages as noted in the summary above. Adding that data essentially doubled our recall scores for medium sized fires, which improved our overall model recall. Next, since our Neural Network is a blackbox model, we were not able to glean as much insight into features. We utilized Random Forest Classification to compliment insights from our Neural Network model by providing top features and weights.

**Top 3 features (excluding location clusters):**
|Feature|Importance|Feature Description|  
|---|---|---|  
|tavg_t3m|0.05085|Average Temperature Past 3 Months|  
|pcp|0.04720|Month Precipitation|  
|tavg_t6m|0.04633|Average Temperature Past 6 Months|


![](/visuals/confusion_matrix_fire.png)

---------------------------
### Conclusions

1. Wildfires are extremely complex phenomena. While the NOAA data offered a set of independent variables which fairly comprehensively described weather history, we were not able to include in our model other important factors which also affect final fire size, such as wind or terrain features (e.g. land cover or incline).

2. Switching our target variable from a continuous one (fire size, in acres) to a multi-class problem improved the score from an R2 of under 10% to accuracy of over 60%. Further, re-defining the problem as a binary classification of "large" vs "small" fires drove accuracy up to 62-77%, depending on the threshold chose to delineate between the two classes.

3. Because our goal was to improve preparedness and help contain damage from wildfires without putting efforts into preventing fires which weren't likely to spread, our ultimate focus was on increasing the recall of our model, and moreover - to increase its recall with regards to large fires.

4. Each of the seven-class classifications requires sklearn's estimators to perform 7 x (7-1) / 2 = 21 separate classifications - fitting some of the estimators we evaluated (e.g. SVM) was very computationally expensive.

5. Despite being computationally expensive, we were able to create a model that gave us a significant increase in our recall score. We can confidently predict very large fires and some mid range fires with our current model.

--------------------------------
**Classes and model improvements:**
|Class| Size Acres|Baseline (% of dataset)|Final Model|
|---|---|---|---|
|A| >0<=0.25|62%|59%|
|B|0.26-9.9|29%|1%|
|C|10.0-99.9|6%|20%|
|D|100-299|2%|50%||
|E|300 to 999|1%|58%|
|F|1000 to 4999|2%|59%|
|G|5000+|0.07%|86%|

-----------------------
### Recommendations

For further research we recommend extracting NOAA wind data such as wind speed, gusts and potentially wind directionality. Furthermore, vegetation data and environmental composition data which is available on Google’s Earth Engine’s LANDFIRE databases potentially play a significant part in telling a deeper story on a wildfire destructive ability. Merging those features into our datasets was unfortunately out of reach due to the time restraints thus we did not include them. However we believe these features merged into our current dataset could expand in a worthwhile manner.

-----------------------
### Data Dictionary 
[Kaggle Fire Data]([https://www.kaggle.com/rtatman/188-million-us-wildfires/notebooks](https://www.kaggle.com/rtatman/188-million-us-wildfires/notebooks)) 

[NOAA Climate Data]([https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp#](https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp#))

-----------------------------------------------------------------------------------

## FEATURED PROJECT 5 : AI generated Star Trek Scripts

I'm interested in AI and deep neural networks so for this project I expermented with gpt-2. I trained gpt-2 on Star Trek scripts and created original stories 
from scratch. 
