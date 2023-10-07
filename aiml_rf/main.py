from flask import Flask,render_template,request,session,redirect,url_for,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash,check_password_hash
from flask_login import login_user,logout_user,login_manager,LoginManager
from flask_login import login_required,current_user
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns1
import os
import stat
# MY db connection
local_server= True
app = Flask(__name__)
app.secret_key='rahul'

# @app.route('/')
# def index(): 
#     return render_template('index.html')

def preprocess(state,dist,seas,cro,area,x_train):
           test = pd.DataFrame({
           'State_Name': [state],
           'District_Name': [dist],
           'Season': [seas],
           'Crop': [cro],
           'Area': [int(area)]
           })
    # pd.core.indexes.base.Index(list_data)

           test_dummy = pd.get_dummies(test)
           missing_cols = set(x_train.columns) - set(test_dummy.columns)
           extra_cols = set(test_dummy.columns) - set(x_train.columns)
           for col in missing_cols:
            test_dummy[col] = 0
           for col in extra_cols:
            test_dummy.drop(col, axis=1, inplace=True)
           test_dummy = test_dummy[x_train.columns]
# Load the random forest model
           model_path = 'C://Users//rahul//Downloads//New folder (2) (1)//New folder (2)//AIML//AIML//aiml_rf//random_forest_model.pkl'
           model1 = joblib.load(model_path)
           y_pred = model1.predict(test_dummy)
           return y_pred[0]
@app.route('/',methods=['POST','GET'])
def base():
    crop=pd.read_csv('C://Users//rahul//Downloads//New folder (2) (1)//New folder (2)//AIML//AIML//aiml_rf//crop_production.csv')
    lis=list(crop['District_Name'].unique())
    lis1=list(crop['State_Name'].unique())
    lis2=list(crop['Season'].unique())
    lis3=list(crop['Crop'].unique())
    print(lis)
    if request.method=="POST":
        state=request.form.get('state')
        dist=request.form.get('dist')
        season=request.form.get('season')
        cro=request.form.get('crop')
        area=request.form.get('area')
    
        crop_data = crop.dropna()
        crop_data = crop_data.copy()
        crop_data.loc[:, 'Yield'] = crop_data['Production'] / crop_data['Area']
    # data = crop_data.drop(['State_Name'], axis = 1)
        dummy = pd.get_dummies(crop_data)
        from sklearn.model_selection import train_test_split
        x = dummy.drop(["Production","Yield"], axis=1)
        y = dummy["Production"]

        # Splitting data set - 25% test dataset and 75% train dataset

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)
        test_data = pd.DataFrame({
       'State_Name': [state],
       'District_Name': [dist],
       'Season': [season],
       'Crop': [cro],
       'Area': [int(area)]
       })
    # pd.core.indexes.base.Index(list_data)

        test_data_dummy = pd.get_dummies(test_data)
        missing_cols = set(x_train.columns) - set(test_data_dummy.columns)
        extra_cols = set(test_data_dummy.columns) - set(x_train.columns)
        for col in missing_cols:
         test_data_dummy[col] = 0
        for col in extra_cols:
         test_data_dummy.drop(col, axis=1, inplace=True)
        test_data_dummy = test_data_dummy[x_train.columns]
# Load the random forest model
        model_path = 'C://Users//rahul//Downloads//New folder (2) (1)//New folder (2)//AIML//AIML//aiml_rf//random_forest_model.pkl'
        model1 = joblib.load(model_path)
        y_pred = model1.predict(test_data_dummy)
        bio=y_pred[0]
        bio=round(bio, 2)
        print(y_pred) 
        seas1=preprocess(state,dist,lis2[0],cro,area,x_train)
        seas2=preprocess(state,dist,lis2[1],cro,area,x_train)
        seas3=preprocess(state,dist,lis2[2],cro,area,x_train)
        seas4=preprocess(state,dist,lis2[3],cro,area,x_train)
        seas5=preprocess(state,dist,lis2[4],cro,area,x_train)
        seas6=preprocess(state,dist,lis2[5],cro,area,x_train)
        prod=[seas1,seas2,seas3,seas4,seas5,seas6]
        max_value = max(prod)  # Find the maximum value in the list
        max_index = prod.index(max_value)
        best_seas=lis2[max_index]
        seas_df = pd.DataFrame({'Season': lis2, 'Production': prod})
        sns1.barplot(data=seas_df,x="Season", y="Production")
        os.remove('aiml_rf/static/graph1.png')
        image_path = r"aiml_rf/static/graph1.png"  # Replace with the actual path to your image file

# Check if the image file exists
        if os.path.exists(image_path):
    # Get the current file permissions
           current_permissions = stat.S_IMODE(os.lstat(image_path).st_mode)
    
    # Add write and delete permissions to the file
           new_permissions = current_permissions | stat.S_IWRITE
    
    # Modify the file permissions
           os.chmod(image_path, new_permissions)
    
    # Delete the image file
           os.remove(image_path)
           print("Image deleted successfully.")
        plt.savefig('aiml_rf/static/graph1.png')
       
        # crop_df = crop_data[crop_data["Crop"]==cro]
        # sns.barplot(data=crop_df,x="Season", y="Production")
        # plt.savefig('aiml_rf/static/graph.png')
        user_input_season = season
        user_input_state = state
        user_input_district = dist

       #  df = pd.read_csv('your_file_path.csv')

# Now you can use the code to find the crop with the highest production as mentioned before
        data2 = pd.DataFrame({
        'State_Name': [state],
        'District_Name': [dist],
        'Season': [season],
        })

       # Filter data based on user input from the data2 DataFrame
        filtered_data = crop[
        (crop["Season"] == data2["Season"].iloc[0]) &
        (crop["State_Name"] == data2["State_Name"].iloc[0]) &
        (crop["District_Name"] == data2["District_Name"].iloc[0])
        ]

       # Group by "crop" and calculate total production for each crop
        crop_production_grouped = filtered_data.groupby("Crop")["Production"].sum()

       # Sort the crops based on production in descending order
        sorted_crops = crop_production_grouped.sort_values(ascending=False)

       # Get the crop with the highest production
        crop_pr = sorted_crops.head(3)

       # Print the result
        print("Crop with the highest production in the specified season, state, and district:")
        print(crop_pr)

        return render_template('index1.html',bio=bio,lis=lis,lis1=lis1,lis2=lis2,lis3=lis3,best_seas=best_seas,crop_pr=crop_pr)
    return render_template('index1.html',lis=lis,lis1=lis1,lis2=lis2,lis3=lis3)

app.run(debug=True)