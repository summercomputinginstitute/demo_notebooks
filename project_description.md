# SCI Module 2 Project

The best way to learn data science, like anything, is to do it!  The Module 2 project will give you a chance to put in practice what you have learned so far on a mini-project.  You will pick a problem/question of interest to you, collect data about the problem, analyze the data, build a model to solve the problem, and then analyze the performance of your model.  The final project deliverable will be a video presentation of 10 minutes maximum describing the problem you chose, your approach, and how your solution is working.

## Topic
The topic of your project is up to you! Choose an area/field that you are particularly passionate about, then find a problem/question that you believe you can address through machine learning (think back to the introductory lesson on the types of problems ML can help solve).  You may work with unstructured data (images, text) if you wish, but be aware that the computational resources to do so will likely be higher and may pose a barrier to training a model on your laptop - if you want to do this you may have an easier time using Google Colab or paying the small fee for access to Google Colab Premium to train your model using a GPU.  You will likely find it easier to work with structured data.  Possible example topics might look something like the following:
- Use data collected from my GPS running watch to understand what factors influence my running pace and build a model to predict my pace given variables such as time of day, weather etc
- Use crime data to build a model to predict crime in a neighborhood based on conditions which contribute to crime level
- Use available twitter data to build a model which can identify the sentiment of a tweet (positive or negative) to assist a company in automatically evaluating sentiment of their customers on a day-to-day basis

## Process
You should follow the steps of the CRISP-DM process (Steps 1-5, skipping Step 6 "Deployment") in working on your project.  

## Project Proposal
Part-way through the course you will be asked to submit a project proposal.  This will be in the form of a jupyter notebook and will include the following information (you will be prompted by questions in the notebook):  
- What problem/question you have selected to focus on  
- Your motivation for selecting the particular problem  
- How is the problem being addressed today 
- Where do you expect to get data to use to solve the problem

## Final Presentation
Your deliverable for the project is a recording of a 10-minute maximum video about your project.  The recording should cover the following:  
- Definition of the problem (see CRISP-DM Step 1)  
- Data used (source/sources and features) and approach to data preparation  
- Modeling approach chosen and why  
- A brief demo of your model in action or screenshot of a visual output, so we can understand what it does  
- Selection of metrics, and evaluation of your model based on the metric(s)  
- Conclusion of how well (or poorly) your model addressed the problem you are trying to solve  
- Any legal and ethical considerations that you deem relevant to solving this problem

## Where to Get Data
There are two main ways to get data for a modeling project: you can collect it, or you can find it.  For this project, you may do either.  Collecting data can be done through visual observation (e.g. counting the number of cars at a drive-through at different times of the day/week), or by using a sensor such as a smartphone or smart watch (e.g. to track sleep minute or running time).  

To find data to use, are many sources for free, non-confidential data available online for almost every industry.  A few links which may be helpful to identify publicly available data to use for your project are:
-	Kaggle Datasets https://www.kaggle.com/datasets
-	Google Dataset Search https://datasetsearch.research.google.com
-	Awesome Public Datasets https://github.com/awesomedata/awesome-public-datasets
-	Harvard Dataverse https://dataverse.harvard.edu
-	DataHub https://datahub.io/search
-	US Government Open Data https://www.data.gov
-	US Census Bureau (demographic data) https://data.census.gov/cedsci/
-	US NOAA (weather data) https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted

## Tips
A couple tips for success based on previous student experiences:  
- Make sure to define your problem in a way that is narrow enough to be manageable. Especially since we have a very short time to complete the project, you will need to pick a bite-sized problem to work on.  E.g. "solving climate change" is not a good problem definition, but "analyzing the variability in rainfall in Durham, NC over the last 100 years" might be.  
- Before you settle on a project topic, make sure you can access the data you need to work on it. You don't want to wait to the last minute to realize that you can't find data to build a model for your project. Make sure up-front that you can find sufficient data.  If you're struggling with this, an alternative approach would be to pick a dataset of interest to you (See links above) and analyze it, using a model to gain some interesting insight.  You'll still need to frame your work in terms of a problem to solve, but a problem can be "understand the impact of X on y in order to ____."  
- Your project is not evaluated based on your model's performance. The performance we can expect from a model varies widely depending on the problem, and in real life we rarely see accuracy rates of 95%+.  It might be that for the phenomenon you are trying to model, a good performance would be an accuracy (or whatever metric you select) of 68%.  This is not a "bad" thing - it is important to recognize that different problems have very different thresholds of what we can do with a model, no matter how complex of a model we build.  And generally, if we have lower performance than we might hope for, the answer is often in collecting more data / adding features, rather than using a more complex model.

Have fun!  I look forward to seeing what you all come up with.
