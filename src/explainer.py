import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from captum.attr import GradientShap
from model_class import CustomCNN
from PIL import Image


tr_transf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()])

# Initialize the model
model = CustomCNN(1)

# Load the state dictionary from the .pth file
state_dict = torch.load('./model/cnn_18.pth')

# Load the state dictionary into the model
model.load_state_dict(state_dict)


def explain_model(img_path: str, model: CustomCNN, tr_transf: transforms.Compose):
    # Define the labels for the model output
    idx_to_labels = {1: ('ok_front'), 0: ('def_front')}
    
    # Load the image and apply the transformations
    img = Image.open(img_path)
    transformed_img = tr_transf(img)
    img_data = np.asarray(transformed_img).squeeze()
    input = transformed_img.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()
    # call our model
    output = model(input)
    # torch.topk returns the k largest elements of the given input tensor along a given #dimension.K here is 1
    prediction_score, pred = torch.topk(output, 1)
    # if the prediction score is greater than 0.5, then the predicted label is 1, else 0
    pred_label_idx = torch.tensor(1 if prediction_score > 0.5 else 0)
    # convert the predicted label index to a string
    predicted_label = idx_to_labels[pred_label_idx.item()]
    
    #convert into a dictionnary of keyvalues pair the predict label, convert it #into a string to get the predicted label
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    # Initialize the attribution algorithm
    torch.manual_seed(10)
    np.random.seed(10)

    # Initialize the GradientShap algorithm 
    gradient_shap = GradientShap(model)

    # Definition of baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    # Attribution computation using the GradientShap algorithm
    attributions_gs = gradient_shap.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred)

    # Plotting
    #subplot(r,c) provide the no. of rows and columns
    _, axarr = plt.subplots(2,1)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(img_data)
    axarr[1].imshow(attributions_gs.squeeze().cpu().detach().numpy())
    plt.show()

# Explain two samples
explain_model('./archive/casting_data/casting_data/test/ok_front/cast_ok_0_2915.jpeg', model, tr_transf)
explain_model('./archive/casting_data/casting_data/test/def_front/new__0_7229.jpeg', model, tr_transf)