import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Define the model and load weights
num_classes = 2  # 1 class (Pool) + background
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, progress=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_path = 'model/pool_detection_model_2.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the same image preprocessing used during training
def get_inference_transforms():
    return A.Compose([
        A.Resize(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def show_prediction(image, prediction, save_path):
    """
    Display an image along with its predicted bounding box.

    Args:
        image (Tensor): The image tensor.
        prediction (Dict): The model's prediction.
        save_path (str): Path to save the figure.

    Returns:
        str: Path to the saved image with bounding box.
    """
    # Convert the image to a format that can be displayed
    image = image.permute(1, 2, 0).cpu().numpy()

    # Denormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Create a figure and axis to display the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Display the highest scoring predicted bounding box
    if len(prediction['boxes']) > 0:
        bbox = prediction['boxes'][0]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')  # Hide the axis
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory

    return save_path

def getPrediction(filename):
    # Define classes
    classes = ['Background', 'Pool']  # Ensure background is first

    # Image Preprocessing
    img_path = os.path.join('static', 'images', filename)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    
    # Apply the inference transformations
    transform = get_inference_transforms()
    img_tensor = transform(image=np.array(img))['image']
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract prediction details
    pred_score = predictions[0]['scores'].detach().cpu().numpy()
    pred_labels = predictions[0]['labels'].detach().cpu().numpy()
    pred_boxes = predictions[0]['boxes'].detach().cpu().numpy()

    if len(pred_score) > 0 and pred_score[0] > 0.3:  # If there's a detection with a score above a threshold
        highest_score_idx = np.argmax(pred_score)
        bbox = pred_boxes[highest_score_idx]

        # Draw the highest scoring bounding box
        image_with_boxes_path = show_prediction(img_tensor.squeeze(), {'boxes': [bbox]}, 'static/images/' + filename.replace('.jpg', '_boxed.jpg').replace('.png', '_boxed.png'))

        pred_class = classes[pred_labels[highest_score_idx]]
    else:
        pred_class = 'Background'
        image_with_boxes_path = img_path

    return pred_class, image_with_boxes_path
