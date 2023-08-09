from flask import Flask, render_template, request, url_for
import torch
from torch import nn
import torchvision
from torchvision import transforms
from digits_model import HandwrittenDigits
import numpy as np
import imageio
import os
import skimage
from skimage import util
from PIL import Image


# import PIL

PATH = "model/trained_model.pth"

# Instantiate the app
app = Flask(__name__)
upload_folder = os.path.join("static", "Image")
app.config['UPLOAD'] = upload_folder

# Create a route that will render the HTML template
@app.route('/')
def index():
    return render_template("index.html")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# Learn more about get/post methods so that we can connect the frontend and backend
@app.route('/predict', methods=['GET', 'POST'])
def handle_get():
    if request.method == 'POST':
        # Receive the image from the user's input

        file = request.files['image']
        file.save(os.path.join(app.config['UPLOAD'], file.filename))
        img = os.path.join(app.config['UPLOAD'], file.filename)
        img_numpy = skimage.io.imread(img)
        # Convert the image to grayscale
        img_numpy_grayscale = rgb2gray(img_numpy)
        i = Image.fromarray(img_numpy_grayscale)
        i = i.convert("L")
        i.save(os.path.join(app.config['UPLOAD'], "gray_img.png"))
        gray_img_path = os.path.join(app.config['UPLOAD'], "gray_img.png")

        # Convert the grayscale values to intensities between 0 and 1 
        # Also convert the image to a 28x28 pixel image
        img_grayscale_tensor = torch.from_numpy(img_numpy_grayscale).type(torch.float32).unsqueeze(dim = 0)
        resize = transforms.Resize((28,28))
        img_grayscale_transformed_tensor = resize(img_grayscale_tensor)
        

        # Invert the colors
        img_grayscale_transformed_tensor = (255) - img_grayscale_transformed_tensor
    
    

        # img_numpy = img_grayscale_transformed_tensor.numpy()
        # img_numpy_inverted = util.invert(img_numpy)
        # img_grayscale_transformed_tensor = torch.from_numpy(img_numpy_inverted)

        transformed_img = Image.fromarray(img_grayscale_transformed_tensor.squeeze(dim=0).numpy())
        transformed_img = transformed_img.convert("L")
        transformed_img.save(os.path.join(app.config['UPLOAD'], "transformed_img.png"))
        transformed_img_path = os.path.join(app.config['UPLOAD'], "transformed_img.png")

        # Normalize the pixels to be between 0 and 1
        img_grayscale_transformed_tensor /= 255

        # normalize = transforms.Normalize(mean = 0.3, std = 0.1)
        # img_grayscale_transformed_tensor = normalize(img_grayscale_transformed_tensor)

        # Run the tensor through the model and output the prediction on the website

        # Create an instance of the neural network model and then load in the trained model
        model = HandwrittenDigits(784, 32, 10)
        model.load_state_dict(torch.load(f=PATH))

        y_logits = model(img_grayscale_transformed_tensor)
        y_preds = torch.sigmoid(y_logits)
        y_pred = y_preds.argmax(dim = 1).item()


        # return f"tensor shape: {img_grayscale_transformed_tensor.shape} and looks like {img_grayscale_transformed_tensor}"
    

        return render_template("img_render.html", img = transformed_img_path, pred = y_pred)
        
    response = "For receiving the image"
    return response





if __name__ == "__main__":
    app.run(debug = True)


