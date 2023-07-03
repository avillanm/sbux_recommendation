<!-- PROJECT LOGO -->
<img src="../images/sbux_logo.png" alt="General view" align="center" width="200"/>

# **Starbucks Offer Recommendation**
## **1. Overview**
 We want to identify which demographic group is more likely to respond to a particular offer based on their demographics characteristics.  

However, what if we want to do this for new customers? We would obtain a powerful tool to evaluate all new customers and provide them with the offer that best suits their needs right from the start.

3 models are used: best channel to contact, spend model and time model to complete the offer.  
The application is displayed in Gradio where the interface is quite practical: you can choose the client's gender, as well as age and income.  
By sending the data, the application provides the most suitable offer from the portfolio according to your profile.

## **2. Input Data**
We have information about the customer profile, offer portfolio, and transactions, all of them in json format:
* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

## **3. Strategy for solving the problem**
The project aims to estimate the best recommendation for a new customer, using only gender, age, and annual income as input data. Based on these three variables, three different models are generated to help us choose the best offer:

- Best channel to contact model; multi-output classifier model
- Spending model; regression model
- Time to complete the offer model; regression model

By obtaining these three results for a new customer, we can approximate the closest offer that aligns with their needs.

## **4. Discussion of the expected solution**
Below is the diagram of the solution that we will detail:
1. New customer information enters the flow
2. Customer information is pre-processed
3. The preprocessed information is entered into each of the models and the result is obtained for each one.
4. The estimated channel, expense and time are compared with the entire available portfolio based on Euclidean distance to identify which offer is closest to the estimate
5. The recommended offer is returned

<img title="a title" alt="General view" src="../images/flow_chart.png">

General view of project:
<img title="a title" alt="General view" src="../images/web_app.png">


## **5. Metrics**
Since these are three different models, the metrics we will use will also be different:

- Best contact channel model: as a multi-output classifier model, for each target (in this case, contact channels), we will obtain the confusion matrix, as well as precision, recall, and F1 score for an overall assessment of the model's quality.
- Spending model: as a regression model, we will use mean squared error and square root as goodness-of-fit measures for the model.
- Time to complete the offer model: similarly, as a regression model, we will use mean squared error and square root as goodness-of-fit measures for the model.

## **6. EDA**

## **7. Data Preprocessing**

## **8. Modeling**

## **9. Hyperparameter Tuning**

## **10. Results**

## **11. Comparision table**

## **12. Conclusion**
The problem began by knowing how the demographic group can influence when offering an offer.  
On my side I raised the bet and asked myself what would happen if he is a new client? I found this question a bit more interesting.  
Being a new client, it is reasonable that the estimates are not the best, but they give us a north of what to offer to these clients. 

## **13. Improvements**
It is possible to improve this project in 2 ways:
- We have only used one type of algorithm for each model. Perhaps carrying out a benchmark with more models will give us a greater number of alternatives
- Try more hyperparameters. For hardware reasons, I only tried with a small set of hyperparameters.

## **14. Acknowledgment**
I want to recognize and recommend this great post from [Freecodecamp](https://www.freecodecamp.org/news/machine-learning-pipeline/) that helped me a lot to improve and clean my code.