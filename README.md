# **Solar Panel Detection and Segmentation**

---

## **Introduction**
In this project, we aim to develop a deep learning model for the detection and segmentation of solar panels in satellite images. The goal is to accurately identify and outline the boundaries of solar panels within the images. This task is crucial for various applications, including solar energy planning, monitoring, and analysis.

---

## **Dataset**
To train our model, we utilize the following datasets:

**1. Solar Panel Detection Dataset: [Roboflow - Francesco Talarico](https://universe.roboflow.com/francesco-talarico/pannelli-8wkam)**

This dataset contains satellite images annotated with bounding boxes around the solar panels. It provides a labeled training set for solar panel detection.

**2. Solar Panel Segmentation Dataset: [Roboflow - SolarPanel](https://universe.roboflow.com/solarpanel-3ku0x/solar_panel-cvecl)**

This dataset includes satellite images with pixel-level segmentation masks for solar panels. It provides a labeled training set for solar panel segmentation.

---

## **Data Preprocessing**

### **Detection Dataset**

#### **1.Auto-Orient:**
The images are automatically oriented to the correct position if needed. ####**2.Resize:**
The images are resized to a fixed size of 416x416 pixels to ensure consistency during training.

### **Augmentations**

#### **1.Outputs per training example:**
Each training example generates 3 augmented images. 
#### **2.Rotate:**
The images are randomly rotated clockwise and counter-clockwise by 90 degrees. 
#### **3.Brightness:**
The brightness of the images is randomly adjusted between -25% and +25% to introduce variations in lighting conditions.

### **Segmentation Dataset**

####**1.Auto-Orient:**
The images are automatically oriented to the correct position if needed. ####**2.Resize:**
The images are resized to a fixed size of 256x256 pixels to ensure consistency during training.

### **Augmentations** 
#### **1.Outputs per training example:**
Each training example generates 3 augmented images. ####**2.Flip:**
The images are randomly flipped horizontally and vertically to introduce variations in orientation.

---

## **Model**
I used pretrained YOLOv8 model has been trained on a large-scale dataset containing diverse images. This allows the model to learn important features and patterns relevant to solar panel detection and segmentation.
Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility.

YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

## **Transfer Learning**
By leveraging transfer learning, we can take advantage of the knowledge learned by the pretrained model and apply it to our specific task. This significantly speeds up the training process and improves the performance of our model.

---

## **Fine-Tuning**
To adapt the pretrained model to our solar panel detection and segmentation task, we perform fine-tuning. During fine-tuning, we further train the model using our specific dataset, fine-tuning the weights and biases to make accurate predictions for solar panel detection and segmentation.

The use of a pretrained YOLOv8 model with transfer learning and fine-tuning allows us to achieve high accuracy and efficiency in detecting and segmenting solar panels in satellite images.

YOLOv8 can be used directly in the Command Line Interface (CLI) with a yolo command:

    !yolo task=**TASK** mode=**MODE** model=**MODEL** data=PATH epochs=EPOCHS_NUMBER imgsz= IMAGE_SIZE

##### TASK:

1.  Segmentation
2.  Pose
3.  Classification

##### MODE:

1. train
2. val
3. predict

#### MODEL:

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

I used YOLOv8s in this project.

---

## Results
After training and evaluating our YOLOv8 model for solar panel detection and segmentation, we achieved impressive results. Here are some key findings:

### Solar Panel Detection Performance

**mAP50: 0.899**

**Precision: 0.862**

**Recall: 0.821**

### Solar Panel Segmentation Performance

1. Instance :
   **mAP50: 0.79**

**Precision: 0.842**

**Recall: 0.732**

2. Mask
   **mAP50: 0.777**

**Precision: 0.859**

**Recall: 0.711**

---

## Model and Plotting Artifacts
All trained models, plotting artifacts, and performance metrics are saved within the project folder. This allows for easy access to the trained models for future use and the ability to reproduce the results.

---

## Configuration Parameters

The following configuration parameters are used in the code:

- **EPOCHS**: The number of training epochs.
- **IMSIZE**: The desired input image size for the YOLOv8 model.
- **CONF**: The confidence threshold, which determines the minimum confidence score required for a detected object or a segmented region to be considered valid.
- **ROOT_DIR**: The root directory for the project, where all the files and folders related to the project are stored.
- **API_KEY**: The API key used to access the dataset, you must get your own api key from [roboflow.com](https://roboflow.com/)
  """
