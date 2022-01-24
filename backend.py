import os

import cv2
import numpy as np
from flask import Flask
from flask import render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackproofgan_csi2021'

pred_model = load_model('generator_20.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
def prediction(grey_img_path, model, out_img_path, greyscale_path):
    # Conversion of RGB to resized Greyscale
    img_arr = cv2.imread(grey_img_path)
    resized = cv2.resize(img_arr, (256, 256))
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Normalization
    img_gray = (img_gray - 127.5) / 127.5
    img_gray = np.expand_dims(img_gray, -1)
    final = np.expand_dims(img_gray, 0)

    predicted_ab = model.predict(final)
    squeezed = np.squeeze(predicted_ab)

    img_gray = np.float32(img_gray)
    predicted_lab = cv2.merge((img_gray, squeezed))

    # Rescaling LAB Channels
    pred_l = (predicted_lab[:, :, 0] + 1.) * 50.
    pred_ab = (predicted_lab[:, :, 1:] * 127.5)
    pred_l = np.float32(pred_l)
    pred_ab = np.float32(pred_ab)

    pred_lab = cv2.merge((pred_l, pred_ab))
    pred_lab = np.float32(pred_lab)

    # Converting LAB to BGR
    pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_LAB2BGR)

    pred_rgb_final = pred_rgb * 255
    final_pred_rgb = pred_rgb_final.astype(int)

    final_gray_out = (img_gray * 127.5) + 127.5

    # Writing Processed images to corresponding paths
    cv2.imwrite(greyscale_path, final_gray_out)
    cv2.imwrite(out_img_path, final_pred_rgb)


UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class UploadForm(FlaskForm):
    file = FileField()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload():
    # Clearing the uploads directory
    if os.listdir('static/uploads'):
        for f in os.listdir('static/uploads'):
            os.remove('static/uploads/' + f)

    form = UploadForm()

    if form.validate_on_submit():
        # Checking if form data is NoneType
        if form.file.data:
            if allowed_file(form.file.data.filename):
                filename = secure_filename(form.file.data.filename)
                form.file.data.save('static/uploads/' + filename)

                session['image'] = form.file.data.filename

                return redirect(url_for('render_predict'))

            else:
                return render_template('home.html', form=form, msg='File format not supported.Only jpeg, jpg and png '
                                                                   'are '
                                                                   'supported')
        else:
            return render_template('home.html', form=form, msg='Please select a file!')

    return render_template('home.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def render_predict():
    img_path = 'static/uploads/' + session['image']
    gray_path = 'static/uploads/gray.png'
    output_path = 'static/uploads/output.png'
    prediction(img_path, pred_model, output_path, gray_path)

    return render_template('predict.html', image_path=gray_path, color_img=output_path)


if __name__ == '__main__':
    app.run(debug=True)
