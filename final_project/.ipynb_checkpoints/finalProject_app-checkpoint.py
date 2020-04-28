from flask import Flask, flash, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import os
from io import BytesIO
import base64
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import csv

sns.set()

# Folder to store uploads
UPLOAD_FOLDER = 'C:/git_repo/INFO8000/final_project/uploads/'

app = Flask("finalProject_app",template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensions allowed to avoid dangerous files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to remove green pixels from images to speed up image plot processing.
# It is based on a modified version of the Excess Green minus Excess Red Index (ExGR)
def greenRemoval(img):
    # Convert BGR to RGB
    RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #color space splitting
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    #R,G,B = cv.split(RGB)

    Rnorm = cv.normalize(R, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    Gnorm = cv.normalize(G, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    Bnorm = cv.normalize(B, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)

    #original green extraction
    #ExR = 1.4*Rnorm - Gnorm;
    #ExG = 1.5*Gnorm - Rnorm - Bnorm;

    #cotton pixels pre-segmentation
    #optimum cotton pixels preprocessing
    ExR = 2.0*Rnorm - Gnorm;
    ExG = 1.5*Gnorm - Rnorm - Bnorm;
    
    ExR = cv.normalize(ExR, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    thR, ExR = cv.threshold(ExR, 0, 255, cv.THRESH_OTSU);
    ExR = ExR.astype(np.int16)

    ExG = cv.normalize(ExG, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    thG, ExG = cv.threshold(ExG, 0, 255, cv.THRESH_OTSU);
    ExG = ExG.astype(np.int16)
    
    img_noGreen = cv.normalize(cv.normalize(np.subtract(ExG,ExR) * -1, 
                            None, 0, 1, cv.NORM_MINMAX) * 255, None, 0,
                            255, cv.NORM_MINMAX,cv.CV_8U)
    rgb_cotton = cv.bitwise_and(RGB,RGB,mask = img_noGreen)
    bgr_cotton = cv.cvtColor(rgb_cotton, cv.COLOR_RGB2BGR)

    #green pixels segmentation based on Excess Green index
    #optimum green pixels extraction
    ExR_opt = 1.0*Rnorm - 0.8*Gnorm;
    ExG_opt = 1.5*Gnorm - Rnorm - Bnorm;
    ExR_opt = cv.normalize(ExR_opt, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    thR_opt, ExR_opt = cv.threshold(ExR_opt, 0, 255, cv.THRESH_OTSU);
    ExR_opt = ExR_opt.astype(np.int16)

    ExG_opt = cv.normalize(ExG, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    thG_opt, ExG_opt = cv.threshold(ExG_opt, 0, 255, cv.THRESH_OTSU);
    ExG_opt = ExG_opt.astype(np.int16)

    img_green = cv.normalize(cv.normalize(np.subtract(ExG_opt,ExR_opt), 
                            None, 0, 1, cv.NORM_MINMAX) * 255, None, 0,
                            255, cv.NORM_MINMAX,cv.CV_8U)

    rgb_leaves = cv.bitwise_and(RGB,RGB,mask = img_green)

    return bgr_cotton, rgb_leaves

# Function to perform color space conversion
def colorSpaces(img):
# input: BGR image
# output: HSV and Lab color space images
    # Convert BGR to HSV
    HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #display(Image.fromarray(HSV))
    #H,S,V = cv.split(HSV)
    # Convert BGR to CIELab
    Lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
    #display(Image.fromarray(Lab))
    #L,a,b = cv.split(Lab)
    
    return (HSV,Lab)

# Function to create the dataset for SVM pixels classification
def createValidationDF(HSV,Lab):
# inputs: HSV and Lab color space images
# output: daframe for validation, incluing H, S, V, a, b color channel components
    #Create HSV subset
    index = pd.MultiIndex.from_product(
        (*map(range, HSV.shape[:2]), ('H', 'S', 'V')),
        names=('row', 'col', None))
    # HSV color channels extraction
    df2 = pd.Series(HSV.flatten(), index=index)
    df2 = df2.unstack()
    df2 = df2.reset_index().reindex(columns=['row', 'col', 'H', 'S', 'V'])
    
    #Create CIE Lab subset
    index = pd.MultiIndex.from_product(
        (*map(range, Lab.shape[:2]), ('L', 'a', 'b')),
        names=('row', 'col', None))
    # Lab color channels extraction
    df3 = pd.Series(Lab.flatten(), index=index)
    df3 = df3.unstack()
    df3 = df3.reset_index().reindex(columns=['row', 'col', 'L', 'a', 'b'])

    # Merge HSV and Lab subsets using pixel location (row, col)
    df = pd.merge(df2, df3, on=['row', 'col'])
    
    #reduce number of points by deleting not presegmented cotton pixels
    df_temp = df.copy()
    #create pre-segmented cotton pixels datframe
    df_cotton = df_temp[(df_temp['H'] > 0)&(df_temp['S'] > 0)&(df_temp['V'] >= 0)] 

    return df, df_cotton   

# Useful function that gets the current figure as a base 64 image for embedding into websites
def getCurrFigAsBase64HTML(fig):
    im_buf_arr = BytesIO()
    fig.savefig(im_buf_arr,format='png')
    im_buf_arr.seek(0)
    b64data = base64.b64encode(im_buf_arr.read()).decode('utf8');
    return render_template('img.html',img_data=b64data) 

# Function to store in the dataset extracted data from an image
def add_data(data):
    global data_df
    data_df = pd.concat([data_df,data],ignore_index=True)
    ax = sns.barplot(x = "plot_id", y = "boll number", data = data_df, color='green')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    ax.set(xlabel="Plots", ylabel = "Cotton pixels")
    fig=plt.gcf()
    img_html = getCurrFigAsBase64HTML(fig);
    return "Added new sample " + "<pre>"+ data_df.to_string() \
            + "</pre><br> ... <br><br>" + img_html
    
   #if is_test != "no":
   #     obs = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
   #                        columns=["sepal.length","sepal.width","petal.length","petal.width"])
 
#    return "not implemented"

def init():
    global data_df
    data_df = pd.read_csv('data.csv')
      
try:
    # Read data from csv 
    data_df = pd.read_csv('data.csv')
    # Load SVM classification model
    clf = joblib.load('SVM_model_HSVab.pkl')
    
except:
    init()
    
# This method resets/initializes the database when called
@app.route("/reset")
def reset():
    # Delete database content using w+ (truncate file to 0 size)
    f = open('data.csv', "w+")
    f.close()
    # Initialize database headers
    with open('data.csv', 'w+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj,delimiter=',',lineterminator='\n')
        # Add contents of list as last row in the csv file
        csv_writer.writerow(['plot_id','boll number'])
        
    init()
    return "reset model"
    
# show an interface to upload an image and select visualization method
@app.route("/")
def main():
    return render_template("main.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               filename)

@app.route("/img_upload", methods=["POST"])
def upload_file():
    # check if the image has been included in the request
    if "file" not in request.files:
        return error
        
    file = request.files['file']
        
    # check if the image exists
    if file.filename == '':
        return error
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #load SVM classification model
        #clf = joblib.load('./SVM_model_HSVab.pkl')
        
        # Extract the name of the file including its extension
        fileName = os.path.basename(file.filename)
        # Use splitext() to get filename and extension separately.
        (name, ext) = os.path.splitext(fileName)
    
        # Read image
        img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
        # Create dataframe
        #green pixels preprocecssing extraction
        image_cotton, image_leaves = greenRemoval(img)
                     
        # Save green pixels mask (leaves)
        leaves_filename = os.path.splitext(file.filename)[0] + '_leaves.png'
        leaves = Image.fromarray(image_leaves)
        leaves.save(os.path.join(app.config['UPLOAD_FOLDER'], leaves_filename))
            
        #color spaces creation
        HSV,Lab = colorSpaces(image_cotton)
        #create dataframe for validation
        full_data, cotton_data = createValidationDF(HSV,Lab)
            
        # Results verification
        # Drop pixel position columns (row, and col), and L column
        X = cotton_data.drop(['row','col','L'], axis = 1)

        # Apply SVM model for pixels classification    
        y_pred = clf.predict(X)
            
        # Create prediction mask
        # Create prediction mask
        cotton_df = cotton_data.copy()
        #recover pixel indices
        cotton_df.insert(0,'cotton', y_pred, True)
        cotton_df = pd.DataFrame(cotton_df.iloc[:,0])
        cotton_df[cotton_df['cotton']=='no'] = 0
        cotton_df[cotton_df['cotton']=='yes'] = 255

        # Reconstruct image dataframe
        full_mask = full_data.copy()
        full_mask['cotton']=0
        full_mask = pd.DataFrame(full_mask.iloc[:,-1])
        #substitute values with cotton prediction values
        full_mask.update(cotton_df)
        # Change the dtype to 'uint8'
        full_mask = (full_mask.astype('uint8')).to_numpy()
        # Convert from array to image pixels
        h,w,c = image_cotton.shape
        mask_result = np.reshape(full_mask,(h,w))
           
        # Store resulting image mask
        cotton_filename = os.path.splitext(file.filename)[0] + '_cotton.png'
        cotton = Image.fromarray(mask_result)
        cotton.save(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))
        
        # Count number of cotton pixels found in the image
        cottonPx_number = np.count_nonzero(mask_result)
        # Use linear regression model coefficients (intercept and slope) to calculate number of cotton bolls
        boll_number = round(0.009008599788066408*cottonPx_number + 48.19938481068874)
        
        # Store predicted pixel number into the csv file
        with open('data.csv', 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj,delimiter=',',lineterminator='\n')
            csv_writer.writerow([os.path.splitext(file.filename)[0], boll_number])
        
        plot_df = pd.DataFrame([[os.path.splitext(file.filename)[0],boll_number]],columns=['plot_id','boll number'])
        msg = add_data(plot_df)
        return msg
    return error

if __name__ == '__main__':
    app.run(port=8080)