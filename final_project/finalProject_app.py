import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import sys

UPLOAD_FOLDER = 'C:/git_repo/INFO8000/final_project/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def showMessage(code):
    if code == 1 or code == 2:
        msg = '''
                <!doctype html>
                <title>No image selected</title>
                <h1>Please, select a valid Image</h1>
                <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
                </form>
                '''
    else:
        msg = '''
                <!doctype html>
                <title>Upload New Image</title>
                <h1>Select a new image to upload</h1>
                <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
                </form>
                '''
    return msg

def greenExtract(img):
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

def colorSpaces(img):
# This function performs the color space conversion
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

def createValidationDF(HSV,Lab):
# This function creates the dataset for model validation:
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
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the image has been included in the request
        if 'file' not in request.files:
            msg = showMessage(1)
            return error
        
        file = request.files['file']
        
        # check if the image exists
        if file.filename == '':
            msg = showMessage(2)
            return msg
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #load SVM classification model
            clf = joblib.load('./SVM_model_HSVab.pkl')
            
            # Read image
            img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
            # Create dataframe
            #green pixels preprocecssing extraction
            image_cotton, image_leaves = greenExtract(img)
                     
            # Save green pixels mask (leaves)
            leaves_filename = os.path.splitext(file.filename)[0] + '_leaves.png'
            leaves = Image.fromarray(image_leaves)
            leaves.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                     leaves_filename))
            
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

            #reconstruct image dataframe
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
            #np.savetxt(os.path.join(app.config['UPLOAD_FOLDER'],
            #            (os.path.splitext(file.filename)[0] + '.csv')),
            #            mask_result, fmt='%i', delimiter=",")
    
            # Store resulting image mask
            cotton_filename = os.path.splitext(file.filename)[0] + '_seg.png'
            cotton = Image.fromarray(mask_result)
            cotton.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                     cotton_filename))
            
            return redirect(url_for('uploaded_file', filename=filename))
    
    msg = showMessage(0)
    return msg

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
if __name__ == '__main__':
    clf = joblib.load('SVM_model_HSVab.pkl')
    app.run(port=8080)