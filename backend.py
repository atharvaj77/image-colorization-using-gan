import base64
import io

import cv2
import numpy as np
from flask import Flask, request
from flask import render_template, url_for, redirect
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackproofgan_csi2021'

pred_model = load_model('generator_20.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
def prediction(img_data, model):
    np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Conversion of RGB to resized Greyscale
    # img_arr = cv2.imread(img)
    resized = cv2.resize(img, (256, 256))
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

    # cv2.imwrite('static/uploads/gray.png', final_gray_out)
    # cv2.imwrite('static/uploads/output.png', final_pred_rgb)

    # print(final_gray_out.shape)
    # print(final_pred_rgb.shape)

    return final_gray_out, final_pred_rgb


UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
data = io.BytesIO()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']  # get file

        image_b64 = base64.b64encode(image.read()).decode('utf-8')
        return redirect(url_for('render_predict', image_b64=image_b64))

    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def render_predict():
    try:

        gray, color = prediction(request.args.get('image_b64'), pred_model)

        rval, in_gray = cv2.imencode(".png", gray)
        gray_buf = io.BytesIO(in_gray)

        rval, in_color = cv2.imencode(".png", color)
        color_buf = io.BytesIO(in_color)

        encoded_gray = base64.b64encode(gray_buf.getvalue()).decode('utf-8')
        encoded_color = base64.b64encode(color_buf.getvalue()).decode('utf-8')

        img_gray = f"data:image/jpeg;base64,{encoded_gray}"
        img_color = f"data:image/jpeg;base64,{encoded_color}"

        return render_template('predict.html', image_path=img_gray, color_img=img_color, success=True)

    except Exception as e:
        return render_template('predict.html', response=e)


if __name__ == '__main__':
    app.run()
