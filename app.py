from flask import Flask
from flask import request , render_template
import imutils
import cv2
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import os
#from app import app
# Adding path to config
#app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
#app.config['EXISTNG_FILE'] = 'app/static/original'
#app.config['GENERATED_FILE'] = 'app/static/generated'

app = Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        file_upload = request.files["file_upload"]
        filename = file_upload.filename

        uploaded_image = Image.open(file_upload).resize((250,160))
        uploaded_image.save('app/static/uploads/upload.jpg')
        
        original_image = Image.open('app/static/original/original.jpg')
        original_image.save('app/static/original/original.jpg')
        original_image = cv2.imread('app/static/original/original.jpg')
        uploaded_image = cv2.imread('app/static/uploads/upload.jpg')


        original_gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

        uploaded_gray = cv2.cvtColor(uploaded_image,cv2.COLOR_BGR2GRAY)

        (score,diff) = structural_similarity(original_gray,uploaded_gray,full=True)
        diff = (diff*255).astype("uint8")

        thresh = cv2.threshold(diff,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.rectangle(uploaded_image,(x,y),(x+w,y+h),(0,0,255),2)

        # Save all output images (if required)
       # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
       # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
       # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
        #cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.jpg'), thresh)
        return render_template('index.html',pred=str(round(score*100,2)) + '%' + ' correct')

if __name__ == "__main__":
    app.run(debug=True)  