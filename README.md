# Pool-Object-Detection
##Description:
This Flask web application allows users to upload images, which are then processed by a pre-trained Faster R-CNN model to
detect the presence of a swimming pool. The application provides feedback on whether a pool was detected
in the uploaded image and displays the image with bounding boxes around the detected objects.

## Features:
- Upload an image using a simple web interface.
- The application processes the uploaded image and uses a pre-trained Faster R-CNN model to detect swimming pools.
- The result (weather a pool is detected) is displayed on the web page.
- The uploaded image with bounding boxes around detected objects is displayed on the web page.

## Goal:
To deploy a deep learning model into a production environment via web application.

## Important files:
- app.py
This file contains the main code for the Flask web application.
- main.py
This file contains the main functionality for loading the model, preprocessing the images, making predictions, and visualizing
the results.
- swimming-pool-detection-training.ipynb
Tis jupyter notebook trains a Faster R-CNN model to detect swimming pools in satellite images
using a dataset of images and corresponding XML files containing bounding box annotations.

## Requirements:
- Pytorch
- torchvision
- albumentations
- Pillow (PIL)
- matplotlib
- Flask

## How to Run:
1. Ensure all required packages are installed.
2. Run the `app.py` file to start the Flask server.
3. Open a web browser and navigate to `http://localhost:5000/`
