 
import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

resnet = models.resnet101(pretrained=True)

def predict(image_path):  
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Classify an image among 1000 categories")
st.write("Example using Convolutional Neural Network by Gunther Bacellar")

file_up = st.file_uploader("Upload a JPG image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
  
    for i, label in enumerate(labels):        
        if i == 0:
            st.write(f"Prediction #1 as {label[0].split(', ')[1]} (score: {label[1]:.4f})")
            st.write("Other smaller possibilities:")
        else:
            st.write(f" {i+1}. {label[0].split(', ')[1]} (score: {label[1]:.4f})")