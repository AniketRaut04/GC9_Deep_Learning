from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
crop_yield = pickle.load(open('crop_yield.pkl', 'rb'))

# label_encoded = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
label_encoded = [  0.0,6.42857143,12.85714286,19.28571429,25.71428571,
  32.14285714,38.57142857, 45.0,51.42857143,57.85714286,
  64.28571429,70.71428571, 77.14285714 ,83.57142857,90.0,
  96.42857143,102.85714286,109.28571429,115.71428571,122.14285714,
 128.57142857, 135.0]
crops = ['apple','banana','blackgram','chickpea','coconut','coffee','cotton','grapes','jute','kidneybeans','lentil','maize','mango','mothbeans',
 'mungbean','muskmelon','orange','papaya','pigeonpeas','pomegranate',
 'rice','watermelon']

app = Flask(__name__)

@app.route("/")
def home(): 
    return render_template("index.html")     



@app.route("/crop_yield",methods=['GET','POST'])
def redirect1():
    if request.method == 'GET':
        return render_template("crop_yield.html")
    if request.method == 'POST':
        District = request.form.get("District")
        Crop = request.form.get("Crop")
        Season = request.form.get("Season")
        Area = request.form.get("Area")
        dataset_new = pd.read_csv('crop_yield.csv')
        dataset_new["Area"].fillna(dataset_new["Area"].mean(), inplace = True)
        dataset_new["Production"].fillna(dataset_new["Production"].mean(), inplace = True)
        dataset_new_onehot = pd.get_dummies(dataset_new, columns=['State_Name', 'District_Name', 'Crop', 'Season'], prefix = ['State_name', 'District_Name', 'Crop', 'Season'])
        dummy_cols = dataset_new_onehot.loc[:, dataset_new_onehot.columns != 'Production']
        X=dataset_new_onehot.loc[:, dataset_new_onehot.columns != 'Production']
        Y=dataset_new['Production']
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        new_data = [['Maharashtra', District , '2023', Season,Crop, Area ]]
        new_df = pd.DataFrame(new_data, columns=['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area'])
        new_df = pd.get_dummies(new_df, columns=['State_Name', 'District_Name', 'Crop', 'Season'], prefix = ['State_name', 'District_Name', 'Crop', 'Season'])
        new_df = new_df.reindex(columns=dummy_cols.columns, fill_value=0)
        new_df1 = scaler.transform(new_df)
        new_yield_prediction = crop_yield.predict(new_df1)
        # # print("Predicted Yield:", new_yield_prediction[0]/float(new_df['Area']))
        # print("Predicted Yield:", new_yield_prediction[0])
        return render_template("crop_yield.html",Area = new_yield_prediction[0])



@app.route("/home")
def redirect2():
    return render_template("index.html")


@app.route("/crop_pred",methods=['GET','POST'])
def redirect3():
    if request.method == 'GET':
        return render_template("crop_pred.html")
    if request.method == 'POST':
        arr = np.array([[[103,40,30,27.309018,55.196224,6.348316,141.483164],[118,31,34,27.548230,62.881792,6.123796,181.417081],
       [106,21,35,25.627355,57.041511,7.428524,188.550654],
       [116,38,34,23.292503,50.045570,6.020947,183.468585],
       [97,35,26,24.914610,53.741447,6.334610,166.254931],
       [107,34,32,26.774637,66.413269,6.780064,177.774507],
       [99,15,27,27.417112,56.636362,6.086922,127.924610],
       [118,33,30,24.131797,67.225123,6.362608,173.322839],
       [117,32,34,26.272418,52.127394,6.758793,127.175293],
       [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]]])
        nitrogen = request.form.get("nitrogen")
        phosphorus = request.form.get("phosphorus")
        potassium = request.form.get("potassium")
        temp = request.form.get("temp")
        rain = request.form.get("rain")
        humid = request.form.get("humid")
        ph = request.form.get("ph")
        arr[0][9][0] = nitrogen
        arr[0][9][1] = phosphorus
        arr[0][9][2] = potassium
        arr[0][9][3] = temp
        arr[0][9][4] = rain
        arr[0][9][5] = humid
        arr[0][9][6] = ph
        arr = arr.reshape(10,7)
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        arr=scaler.fit_transform(arr)
        arr = arr.reshape(1,10,7)
        result = model.predict(arr)
        # inv = np.repeat(result,7, axis=-1)
        # inv_scaler = scaler.inverse_transform(inv)[:,0]
        for j in result:
            temp = []
            for k in label_encoded:
                diff = j - k
                diff = int(abs(diff))
                temp.append(diff)
            value = min(temp)
            ind = temp.index(value)
            i = label_encoded[ind]
        return render_template("crop_pred.html",crop = crops[int(i)-1])


@app.route("/pred_yield",methods=['GET','POST'])
def redirect4():
    if request.method == 'GET':
        return render_template("pred_yield.html")
    if request.method == 'POST':
        arr = np.array([[[103,40,30,27.309018,55.196224,6.348316,141.483164],[118,31,34,27.548230,62.881792,6.123796,181.417081],
        [106,21,35,25.627355,57.041511,7.428524,188.550654],
        [116,38,34,23.292503,50.045570,6.020947,183.468585],
        [97,35,26,24.914610,53.741447,6.334610,166.254931],
        [107,34,32,26.774637,66.413269,6.780064,177.774507],
        [99,15,27,27.417112,56.636362,6.086922,127.924610],
        [118,33,30,24.131797,67.225123,6.362608,173.322839],
        [117,32,34,26.272418,52.127394,6.758793,127.175293],
        [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]]])
        nitrogen = request.form.get("nitrogen")
        phosphorus = request.form.get("phosphorus")
        potassium = request.form.get("potassium")
        temp = request.form.get("temp")
        rain = request.form.get("rain")
        humid = request.form.get("humid")
        ph = request.form.get("ph")
        arr[0][9][0] = nitrogen
        arr[0][9][1] = phosphorus
        arr[0][9][2] = potassium
        arr[0][9][3] = temp
        arr[0][9][4] = rain
        arr[0][9][5] = humid
        arr[0][9][6] = ph
        arr = arr.reshape(10,7)
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        arr=scaler.fit_transform(arr)
        arr = arr.reshape(1,10,7)
        result = model.predict(arr)
        inv = np.repeat(result,7, axis=-1)
        inv_scaler = scaler.inverse_transform(inv)[:,0]
        # for j in inv_scaler:
        #     temp = []
        # for k in label_encoded:
        #     diff = j - k
        #     diff = int(abs(diff))
        #     temp.append(diff)
        #     value = min(temp)
        #     ind = temp.index(value)
        #     index = label_encoded[ind]
        for j in result:
            temp = []
            for k in label_encoded:
                diff = j - k
                diff = int(abs(diff))
                temp.append(diff)
            value = min(temp)
            ind = temp.index(value)
            i = label_encoded[ind]
        District = request.form.get("District")
        Crop = crops[int(i)-1]
        Season = request.form.get("Season")
        Area = request.form.get("Area")
        dataset_new = pd.read_csv('crop_yield.csv')
        dataset_new["Area"].fillna(dataset_new["Area"].mean(), inplace = True)
        dataset_new["Production"].fillna(dataset_new["Production"].mean(), inplace = True)
        dataset_new_onehot = pd.get_dummies(dataset_new, columns=['State_Name', 'District_Name', 'Crop', 'Season'], prefix = ['State_name', 'District_Name', 'Crop', 'Season'])
        dummy_cols = dataset_new_onehot.loc[:, dataset_new_onehot.columns != 'Production']
        X=dataset_new_onehot.loc[:, dataset_new_onehot.columns != 'Production']
        Y=dataset_new['Production']
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        new_data = [['Maharashtra', District , '2023', Season,Crop, Area ]]
        new_df = pd.DataFrame(new_data, columns=['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area'])
        new_df = pd.get_dummies(new_df, columns=['State_Name', 'District_Name', 'Crop', 'Season'], prefix = ['State_name', 'District_Name', 'Crop', 'Season'])
        new_df = new_df.reindex(columns=dummy_cols.columns, fill_value=0)
        new_df1 = scaler.transform(new_df)
        new_yield_prediction = crop_yield.predict(new_df1)
# # print("Predicted Yield:", new_yield_prediction[0]/float(new_df['Area']))
# print("Predicted Yield:", new_yield_prediction[0])
        return render_template("pred_yield.html",crop = crops[int(i)-1],Area = new_yield_prediction[0])

@app.route("/about_us")
def redirect5():
    return render_template("about_us.html") 
  
if(__name__) == "__main__":
    app.run()
    