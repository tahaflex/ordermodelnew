from flask import Flask, request, render_template, make_response
from pycaret.classification import *
import datetime
from io import StringIO
import gc
import csv

app = Flask(__name__)
# model = load_model('Final_et_20thJune2022v8.1')
# model = load_model('Final_LGBM_Model_29Jun2022')
model = load_model('Final_RF_Model_5thJuly2022')

# import gcsfs
#
# fs = gcsfs.GCSFileSystem(project='my-google-project')
# fs.ls('flextock_ds_buuket')
# # ['my-file.txt']
# with fs.open('flextock_ds_buuket/Models/my-file.txt', 'rb') as file:
#     print(load_model(file))

# Today's Date
dt = datetime.datetime.now().date()
print(dt)


@app.route('/')
def home():
    # return 'Hello World'
    return render_template('home.html')
    # return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    cols = ['order_date', 'items_cost', 'mapped_area', 'mapped_city', 'fulfillment_center_id', 'merchant_name',
            'phone_number', 'secondary_phone_number', 'apartment_no', 'floor_no', 'fulfilled_date', 'created_date']
    # cols = ['order_date','timestamp','same_day_eligible','items_cost','customer_type','is_fragile','is_dangerous','area','city_code','is_work_address','fulfillment_center_id','merchant_code','phone_number','secondary_phone_number','apartment_no','floor_no','location_id', 'Success_Rate']
    data_unseen = pd.DataFrame([features], columns=cols)
    print(features)
    duration_days = pd.to_datetime(data_unseen['fulfilled_date']).dt.day - pd.to_datetime(
        data_unseen['created_date']).dt.day
    print(duration_days)
    data_unseen['Aging_Days'] = duration_days
    data_unseen['order_date'] = dt
    data_unseen.drop('fulfilled_date', axis='columns', inplace=True)
    data_unseen.drop('created_date', axis='columns', inplace=True)
    print(data_unseen)
    prediction = predict_model(model, data=data_unseen)  # , raw_score=True
    # prediction = prediction[0]
    prediction1 = prediction.Label[0]
    print(prediction)
    Score = prediction.Score[0]
    gc.collect()
    return render_template('home.html',
                           prediction_text='>  Courier to Use : {}, Probability : {:.2%}  <'.format(prediction1, Score))


@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = stream.read()  # transform(stream.read())

    df = pd.read_csv(StringIO(result))
    print(df)
    duration_days = pd.to_datetime(df['fulfilled_date']).dt.day - pd.to_datetime(
        df['created_date']).dt.day
    print(duration_days)
    df['Aging_Days'] = duration_days
    dropped = df['order_code']
    df.drop('created_date', axis='columns', inplace=True)
    df.drop('fulfilled_date', axis='columns', inplace=True)
    df.drop('order_code', axis='columns', inplace=True)
    print(df)

    # load the model from disk
    # pred_model = load_model('Final_GBC_Model_3rdJuly2022')
    prediction_result = predict_model(model, data=df)
    prediction_result.insert(0, 'order_code', dropped)
    print(prediction_result)

    response = make_response(prediction_result.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    gc.collect()
    return response


if __name__ == '__main__':
    # app.run(debug=False, port=8080)
    app.run(host='127.0.0.1', port=8080, debug=True)
