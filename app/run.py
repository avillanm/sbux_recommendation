import numpy as np
import pandas as pd
import joblib

from scipy.spatial.distance import euclidean
import gradio as gr

# load data
portfolio = joblib.load('../objects/portfolio_scaled.pkl')
portfolio0 = pd.read_json('../data/portfolio.json', orient = 'records', lines = True)

# load objects
best_channel_model = joblib.load('../objects/best_channel.pkl')
spend_model = joblib.load('../objects/spend.pkl')
cuts = np.load('../objects/cuts_spend.npy')
ttc_offer = joblib.load('../objects/ttc_offer.pkl')
scaler = joblib.load('../objects/portfolio_scaler.pkl')
vec = ['flag_mobile','flag_web','flag_social','difficulty','duration']

# function model
def get_recommendation(gender, age, income):
    """
    Use the three models (channel, spend and time) to calculate the best offer to client

    Args:
        gender (string): gender of client
        age (integer): age of client
        income (float): income of client

    Returns:
        dataframe: complete offer dataframe
    """
    new_client = {'gender': gender, 'age': age, 'income': income}
    new_client_df = pd.DataFrame(new_client, index=[0])
    new_client_df[['flag_mobile','flag_web','flag_social']] = best_channel_model.predict(new_client_df) # mobile, web, social

    position = int(round(spend_model.predict(new_client_df)[0],0))
    if position >= 20:
        spend = cuts[18]
    elif position <= 1:
        spend = cuts[0]
    else:
        spend = (cuts[position - 1] + cuts[position - 2]) / 2
    new_client_df['difficulty'] = spend

    new_client_df['duration'] = ttc_offer.predict(new_client_df)[0]
    new_client_df[vec] = scaler.transform(new_client_df[vec])
    row = new_client_df[vec].values
    target_array = portfolio[vec].values
    distances = np.apply_along_axis(lambda x: euclidean(row, x), axis=1, arr=target_array)
    offer = portfolio0[portfolio0.index == np.argmin(distances)]
    offer = offer.T.reset_index()
    offer.columns = ['item', 'value']
    offer = offer[::-1].reset_index(drop = True)
    return offer

def main():
    """
    Run application with gradio
    """
    gender_input = gr.inputs.Radio(choices=['M', 'F'], label = 'Gender')
    age_input = gr.inputs.Slider(minimum = 18, maximum = 120, step = 1, label = 'Age')
    income_input = gr.inputs.Slider(minimum = 0, maximum = 500000, step = 1000, label = 'Income')
    output = gr.DataFrame(headers=['Selected Offer'], label='Results')#gr.outputs.Textbox(label = 'Recommendation')

    interface = gr.Interface(fn = get_recommendation, inputs = [gender_input, age_input, income_input], outputs = output, title = 'Recommendations ☕️',  allow_flagging = 'never')
    interface.launch()

if __name__ == '__main__':
    main()



