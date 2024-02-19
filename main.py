import sys
import streamlit as st

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils.constants import COLOR_CODES

st.write('''
  # Diamond Price Prediction
  Predict the Price of the *Diamond* ! :crown:
''')

with st.form('diamond_data_form'):
  try:
    carat = st.slider('Carat', min_value=0.0, max_value=3.0, step=0.01, help='Decides the Carat quality of the diamond')

    col1, col2, col3 = st.columns(3)
    cut = col1.selectbox('Cut', ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'], index=2)
    color = col2.selectbox('Color', ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet'])
    clarity = col3.selectbox('Clarity', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], help='Defines the clarity of the diamond')

    table = st.slider('Table %', min_value=30.0, max_value=100.0, step=0.1, help='Percentage of the Table width to its diameter')

    col1, col2, col3 = st.columns(3)
    x = col1.number_input('Length (in mm)', min_value=1.0, step=0.01)
    y = col2.number_input('Width (in mm)', min_value=1.0, step=0.01)
    z = col3.number_input('Depth (in mm)', min_value=1.0, step=0.01)

    if color:
      color = COLOR_CODES[color]
    if y and z:
      depth = (z/y) * 100

    submitted = st.form_submit_button('Submit', type='primary')
  except Exception as err:
    raise CustomException(err, sys)

placeholder = st.empty()
if submitted:
  data = CustomData(carat, cut, color, clarity, depth, table, x, y, z)
  df = data.get_data_as_data_frame()

  pred_pipeline = PredictPipeline()

  with st.spinner(text="Predicting the Price..."):
    result = pred_pipeline.predict_pipeline(df)

    result_str = f"""
      <style>
      p.a {{
        font: bold 30px Arial;
      }}
      </style>
      <p class="a">The Price of the Diamond is ${round(result[0], 2)}</p>
    """

    placeholder.markdown(result_str, unsafe_allow_html=True)