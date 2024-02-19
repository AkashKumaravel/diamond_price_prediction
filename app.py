import sys
from flask import Flask, render_template, request

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'GET':
    return render_template('home.html')
  else:
    try:
      data = CustomData(
        carat = float(request.form.get('carat')),
        cut = request.form.get('cut'),
        color = request.form.get('color'),
        clarity =  request.form.get('clarity'),
        depth =  float(request.form.get('depth')),
        table =  float(request.form.get('table')),
        x =  float(request.form.get('x')),
        y =  float(request.form.get('y')),
        z =  float(request.form.get('z')),
      )

      df = data.get_data_as_data_frame()

      pred_pipeline = PredictPipeline()
      result = pred_pipeline.predict_pipeline(df)

      return render_template('home.html', price = result[0])
    except Exception as err:
      raise CustomException(err, sys)


if __name__ == '__main__':
  app.run(host = '0.0.0.0')