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

def prediction(grey_img_path, model, out_img_path):
    img_arr = cv2.imread(grey_img_path)
    resized = cv2.resize(img_arr, (256, 256))
    img_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    lab_channels = cv2.split(img_lab)

    # Normalizing Channels
    # l -> -1 to 1
    # ab -> -1 to 1
    normalized_l = lab_channels[0] / 127.5 - 1.
    normalized_ab = (cv2.merge((lab_channels[1], lab_channels[2])) - 128.) / 127.

    normalized_l = np.expand_dims(normalized_l, -1)
    normalized_l = np.float32(normalized_l)
    normalized_ab = np.float32(normalized_ab)

    final = np.expand_dims(normalized_l, 0)

    predicted_ab = model.predict(final)

    squeezed = np.squeeze(predicted_ab)

    predicted_lab = cv2.merge((normalized_l, squeezed))

    pred_L = (predicted_lab[:, :, 0] + 1.) * 50.
    pred_ab = (predicted_lab[:, :, 1:] * 127.5)
    pred_L = np.float32(pred_L)
    pred_ab = np.float32(pred_ab)

    pred_lab = cv2.merge((pred_L, pred_ab))
    pred_lab = np.float32(pred_lab)

    pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_LAB2BGR)

    pred_rgb_final = pred_rgb * 255
    final_pred_rgb = pred_rgb_final.astype(int)

    cv2.imwrite(grey_img_path, lab_channels[0])
    cv2.imwrite(out_img_path, final_pred_rgb)

    print('Done!')


UPLOAD_FOLDER = './upload'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class UploadForm(FlaskForm):
    file = FileField()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload():
    if os.listdir('static/uploads'):
        for f in os.listdir('static/uploads'):
            os.remove('static/uploads/' + f)

    form = UploadForm()

    if form.validate_on_submit():
        if allowed_file(form.file.data.filename):
            filename = secure_filename(form.file.data.filename)
            form.file.data.save('static/uploads/' + filename)

            session['image'] = form.file.data.filename

            return redirect(url_for('render_predict'))

        else:
            return render_template("home.html", msg='This file format is not valid. Only jpeg, jpg and png is supported', form=[])

    return render_template('home.html', form=form, msg='')


@app.route('/predict', methods=['GET', 'POST'])
def render_predict():
    img_path = 'static/uploads/' + session['image']
    output_path = 'static/uploads/output.png'
    prediction(img_path, pred_model, output_path)

    return render_template('predict.html', image_path=img_path, color_img=output_path)


if __name__ == '__main__':
    app.run(debug=True)
