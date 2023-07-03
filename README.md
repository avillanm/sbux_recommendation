<!-- PROJECT LOGO -->
<img src="images/sbux_logo.png" alt="drawing" style="display: block; margin: 0 auto" width="100"/>

# Starbucks Offer Recommendation
## 1. Project motivation
Develop a tool to predict the best recommendation, based on the demographic information of a new client.  
3 models are used: best contact channel, spend model and time model to complete the offer.  
The application is displayed in Gradio where the interface is quite practical: you can choose the client's gender, as well as age and income.  
By sending the data, the application provides the most suitable offer from the portfolio according to your profile.

## 2. Installations
- pandas = 1.2.4
- numpy = 1.19.5
- matplotlib = 3.3.4
- seaborn = 0.11.1
- sklearn = 1.1.1
- joblib = 1.1.0
- scipy = 1.5.4
- gradio = 3.35.2

## 3. Files distribution
```
├── app
│   └── run.py  # File that execute the web app
├── data
│   └── portfolio.json  # portfolio of offers
│   └── profile.json  # demographics
│   └── transcript.json # transactions
├── images
├── notebooks  
│   ├── Challenge description.ipynb
│   └── Project Definition
│   └── Analysis  # EDA
│   └── Methodology, results and conclusions
│   └── Predict  # Prediction on demand
├── models  
│   ├── best_channel.pkl # best channel model
│   ├── cuts_spend.npy # cuts of target in spend model
│   ├── spend.pkl # spend model
│   ├── ttc_offer.pkl # time to complete transaction model
│   ├── portfolio_scaler.pkl # scaler of portfolio dataframe
│   └── portfolio_scaled.pkl  # portfolio dataframe scaled
└── README.md     
```
## 4. Diagram of solution
Below is the diagram of the solution that we will detail:
1. New customer information enters the flow
2. Customer information is pre-processed
3. The preprocessed information is entered into each of the models and the result is obtained for each one.
4. The estimated channel, expense and time are compared with the entire available portfolio based on Euclidean distance to identify which offer is closest to the estimate
5. The recommended offer is returned

<img title="a title" alt="General view" style="display: block; margin: 0 auto" src="images/flow_chart.png">

## 4. How to Interact with this project?
Run the following command in the app's directory to run your web app.
```
python run.py
```
a. Go to `http://127.0.0.1:7861`

b. General view of project
<img title="a title" alt="General view" src="images/web_app.png">

c. Complete report in pdf: `notebooks/report.pdf`