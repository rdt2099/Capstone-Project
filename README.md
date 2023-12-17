![](UTA-DataScience-Logo.png)

# Bird Species Classification Using Deep Learning

## Project Description
This project employs deep learning techniques to classify bird species from images. We use a dataset comprising various bird species, aiming to automate the identification process. This approach is crucial for ecological research, contributing significantly to biodiversity monitoring and conservation efforts.

## Project Approach

- **Task Definition:** 
  The primary goal is to develop an accurate model for classifying bird species from images using machine learning techniques.

- **Data Collection and Preprocessing:** 
  We collected images of different bird species, resized them to a uniform size (200x200 pixels), and normalized the pixel values for consistency. The data was then labeled with the corresponding bird species.

- **Exploratory Data Analysis (EDA):** 
  EDA was conducted to understand the dataset better, identify patterns, and visualize data distributions.

- **Model Development:** 
  Two models were developed:
  - A custom-built sequential model using convolutional, pooling, and dense layers.
  - A pre-trained ResNet model, adapted for our specific classification task.

- **Model Training and Evaluation:** 
  Both models were trained on the dataset, and their performance was evaluated based on accuracy and loss metrics. We also employed confusion matrices and classification reports for in-depth analysis.

## Files
- `final_Copy_of_Bird_Species_with_model.ipynb`: Main Jupyter notebook containing all code and analysis.
- `bird_dataset.csv`: Contains the bird species image data and labels.
- Additional relevant files (e.g., image files, data preprocessing scripts).

## What is Image Classification?
Image classification involves categorizing and labeling groups of pixels or vectors within an image based on specific rules. It's a core task in computer vision that enables machines to interpret and categorize what they 'see' in images or videos. In our project, it's used to identify different bird species from images.

### Categories and Classes
The dataset contains images from 7 distinct bird species. Each class in the dataset represents a different species, and the model learns to recognize and classify these species based on the training images.

## Our Approach for Classification 
- **Sequential Model**
  This model is built from scratch and consists of convolutional layers for feature extraction, followed by pooling layers and dense layers for classification.

- **ResNet Model**
  We used the pre-trained ResNet50 model, fine-tuned for our bird species classification task. It leverages deep learning to identify intricate features within the images.

## Model Evaluation
- **Results**: 
  - The ResNet model achieved a validation accuracy of 17%.
  - The Sequential model reached a validation accuracy of 13%.

- **Performance Analysis**: 
  Using confusion matrices and classification reports, we analyzed the models' performance in detail, identifying strengths and areas for improvement.

## Next Steps: Segmentation for Cell Counting
In the next semester, we plan to enhance our project by incorporating image segmentation techniques. The goal is to count white cells in the images of the birds, combining classification with advanced segmentation. This step will involve:
- Developing and annotating a dataset specifically for segmentation tasks.
- Employing models like U-Net or Mask R-CNN for precise segmentation and cell counting.
- Integrating these segmentation models with our existing classification models for comprehensive analysis.

## Conclusion
The project demonstrated the potential of deep learning models in classifying bird species from images. While the current accuracy levels indicate room for improvement, the models provided valuable insights into the complexities of image classification tasks. Future work will focus on enhancing the models' accuracy and applying these techniques to broader ecological research applications.

## Contributors
- Revan Thakkar

## Acknowledgments
- Thanks to the various sources of bird species datasets and the deep learning community for the resources and inspiration.
- Thanks to Dr. Shane for his support and the insights provided along with the dataset.


