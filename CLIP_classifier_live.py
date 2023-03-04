# CLIP classifier live
# ====================
# Continously read images from the webcam,
# "classify" them via CLIP,
# do learning from 1 example
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

# Parameters
# ----------

# Note: For nices plotting in the figure, you can add newlines \n
#       they will be replaced by whitespaces for the classifier
class_A = "An apple"
class_B = "A pear"
class_C = "No fruit"
class_D = "Not learnt yet" # dummy, won't be used, as not provided verbally


# Set up the workspace
# --------------------

# For sensing:
cap = cv2.VideoCapture(0)
img_pil = None # as global buffer
# For thinking:
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize([class_D]).to(device) # Class that can be learnt
features_D = model.encode_text(text)
learnt_yet = False
# For acting:
fig, ax = plt.subplots()
fig.canvas.set_window_title('Live CLIP classification') # deprecated, but variants from hint didn't work
#pos = ax.get_position() # get the original position 
#pos = [pos.x0, pos.y0+0.1, pos.width, pos.height-0.1] # because xlabel might be long
#ax.set_position(pos)
visu_state = "img" # "img" (the image) or "bar" (the classification results)


# Functions
# ---------

def classify(img_pil):
    img_clip = preprocess(img_pil).unsqueeze(0).to(device)
    text = clip.tokenize([
        re.sub("\n"," ",class_A), 
        re.sub("\n"," ",class_B), 
        re.sub("\n"," ",class_C)
    ]).to(device)
    # Note: We allowed newlines in the description for optical reasons, but maybe they are not good
    
    with torch.no_grad():
        image_features = model.encode_image(img_clip)
        text_features = model.encode_text(text)
        # The magic of CLIP is that it's the same feature space:
        text_features = torch.cat((text_features, features_D), dim=0)
        
        # Cosine similarity via scipy:
        similarity = [1.0-scipy.spatial.distance.cosine(image_features.cpu()[0], text_feature) for text_feature in text_features.cpu()]
        # Note: It's a distance, not a similarity
        if not learnt_yet:
            similarity[-1] = 0.0

    return similarity

def learn():
    """ Extract and keep the features from the provided example image.
    That's the nice thing about classifying with CLIP: You need only
    one example to compeare your measurement to.
    """
    global learnt_yet, features_D # not nice, I know
    if img_pil is not None:
        img_clip = preprocess(img_pil).unsqueeze(0).to(device)
        features_D = model.encode_image(img_clip) # batch
        learnt_yet = True

def update_plot(similarity):
    # We clean up. As alternative, on could keep handles on artists and update values
    ax.clear()
    if visu_state == "bar":
        colors = ['Green', 'Yellow', 'Orange', 'Red']
        colors_text = ['White', 'Black', 'White', 'White']
        ax.bar([class_A, class_B, class_C, "?"], similarity, color=colors)
        for idx, value in enumerate(similarity):
            ax.text(idx,value/2.0,f"{value:.2f}",color=colors_text[idx])
        ax.set_ylabel('Score')
        ax.set_ylim((0.0,1.0))
    if visu_state == "img":
        ax.imshow(img_pil)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Press L for learning the ""?"", press V for switching the view, press ESC to close")
    
def handle_keystrokes(event):
    global visu_state # not nice, I know
    if event.key == 'escape':
        plt.close()
    elif event.key == 'l':
        learn()
    elif event.key == 'v':
        # Toggle
        if visu_state == "img":
            visu_state = "bar"
        else:
            visu_state = "img"


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

