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
                <title>Upload new File</title>
                <h1>Upload new File</h1>
                <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
                </form>
                '''
    return msg

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

    return df    
    
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
            
            clf = joblib.load('./SVM_model_HSVab.pkl')
            
            # Initialize dataframe
            data = pd.DataFrame(columns=['row', 'col','H','S','V','L','a','b'])

            # Read image
            image = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
            # Create dataframe
            HSV,Lab = colorSpaces(image)
            data = createValidationDF(HSV,Lab)
    
            # Results verification
            # Create copy of the dataframe
            df = data.copy()
            # Drop pixel position columns (row, and col), and L column
            X = df.drop(['row','col','L'], axis = 1)

            # Apply SVM model for pixels classification    
            y_pred = clf.predict(X)
    
            # Create prediction mask
            pred_mask = y_pred.copy()
            pred_mask[pred_mask=='no'] = 0
            pred_mask[pred_mask=='yes'] = 255
            # Change the dtype to 'uint8' 
            mask_ = pred_mask.astype('uint8')
            # Convert from array to image pixels
            h,w,c = image.shape
            mask_result = np.reshape(mask_,(h,w))
            np.savetxt(os.path.join(app.config['UPLOAD_FOLDER'],
                        (os.path.splitext(file.filename)[0] + '.csv')),
                        mask_result, fmt='%i', delimiter=",")
    
            # Store resulting image mask
            mask_result_name = os.path.splitext(file.filename)[0] + '_seg.png'
            cotton = Image.fromarray(mask_result)
            cotton.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                     mask_result_name))
            #mask_result.save(os.path.join(app.config['UPLOAD_FOLDER'], mask_result_name))
            
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