# CLIP classifier live
# ====================
# Continously read images from the webcam,
# "classify" them via CLIP,
# but without text, online live snapshots of examples
# License: MIT mail@tobias-foertsch.de

# Dependencies and imports
# ------------------------
# Get CLIP from here: https://github.com/openai/CLIP and follow instructions in its README
import torch
import clip
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import threading
import numpy as np
import scipy
import concurrent.futures
import regex as re

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up the workspace
# --------------------

# For sensing:
cap = cv2.VideoCapture(0)
img_pil = None # as global buffer

# For thinking:
model, preprocess = clip.load("ViT-B/32", device=device)
example_list = []

# For acting:
fig, ax = plt.subplots()
fig.canvas.set_window_title('Live CLIP classification') # deprecated, but variants from hint didn't work
visu_state = "img" # "img" (the image) or "bar" (the classification results)


# Functions
# ---------

def classify(img_pil):
    img_clip = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_clip)
        similarity = [1.0-scipy.spatial.distance.cosine(image_features.cpu()[0], example.cpu()[0]) for example in example_list]
        # Note: It's a distance, not a similarity
        
    #similarity = torch.softmax(torch.tensor(similarity), dim=0)
    
    return similarity

def learn():
    """ Extract and keep the features from the provided example image.
    That's the nice thing about classifying with CLIP: You need only
    one example to compeare your measurement to.
    """
    global example_list # not nice, I know
    if img_pil is not None:
        img_clip = preprocess(img_pil).unsqueeze(0).to(device)
        example = model.encode_image(img_clip) # batch
        example_list.append(example)

def forget():
    global example_list # not nice, I know
    if len(example_list) > 0:
        del example_list[-1]

def update_plot(similarity):
    # We clean up. As alternative, on could keep handles on artists and update values
    ax.clear()
    if visu_state == "bar":
        if len(example_list) > 0:
            ax.bar(range(len(example_list)), similarity, color="black")
            for idx, value in enumerate(similarity):
                ax.text(idx,value/2.0,f"{value:.2f}",color="white")
            ax.set_xticks(range(len(example_list)))
            
        else:
            ax.bar("?",1.0,color="gray")
        ax.set_ylabel('Score')
        ax.set_ylim((0.0,1.0))
        ax.set_ylabel("Class")
                
    if visu_state == "img":
        ax.imshow(img_pil)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Press V for switching the view, press ESC to close,\npress + for learning a new class and - for deleting the last one")
    
def handle_keystrokes(event):
    global visu_state # not nice, I know
    if event.key == 'escape':
        plt.close()
    elif event.key == 'v':
        # Toggle
        if visu_state == "img":
            visu_state = "bar"
        else:
            visu_state = "img"
    elif event.key == '+':
        learn()
    elif event.key == '-':
        forget()


# Main action
# ===========

while plt.fignum_exists(fig.number):
    # Sense
    # -----
    _ , img_np = cap.read()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_np)
    
    # Think
    # -----
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(classify, img_pil)
    probs = future.result()

    # Act
    # ---
    update_plot(probs)

    # Also Sense-Think-Act
    # - - - - - - - - - - 
    fig.canvas.mpl_connect('key_press_event', handle_keystrokes)

    # For smooth operation
    # - - - - - - - - - - 
    plt.pause(0.01)

# Cleanup
# -------

cap.release()
plt.close()

