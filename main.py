import numpy as np
import pandas as pd
import streamlit as slt
import helper_funcs
from PIL import Image
import cv2
import keras
import tensorflow as tf
import os

###### sidebar config ######
# select the model want to be used for classification
slt.sidebar.header("Select Model")
model_choice = slt.sidebar.radio("Select the classification model",
                                 ["Recommended Model", "VGG16-1", "VGG16-2", "ResNet50-1", "ResNet50-2"],
                                 captions = ["Overall Best Performance",
                                             "Pretrained ImageNet weights", 
                                             "Fully finetuned on random initialized weights",
                                             "Partial finetuned on ImageNet weights",
                                             "Fully finetuned on ImageNet weights"],
                                             index = None)

# display all model performances in parallel
# show_all_model = slt.sidebar.checkbox("Show all model performances in parallel")

# display model performances so far
slt.sidebar.header("Evaluate Model")
display_model_performance = slt.sidebar.checkbox(f"Display the {model_choice} model performance at current version:")

# upload image
slt.sidebar.header("Upload Image")
uploaded_file = slt.sidebar.file_uploader(
    "Upload microscopic image of the mold", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

###### main page config ######
slt.title("Mold Classification & WIKI")
slt.image("./Banner Header.png")
slt.header("Instruction", divider="rainbow")
slt.markdown("""
             1. Select the classification model you would like to use

             2. You may check the checkbox to view the performance of the current version of the model

             3. Upload the microscopic image you would like to identify

             4. Let the model do the magic!

             5. Post your feedback on the classification result

             6. Learn about knowledge related to the classified strain
             """)
# classification
if model_choice is not None:
    model_path = './Models'
    model_path = os.path.join(model_path, f"{model_choice}.h5")
    classification_model = tf.keras.models.load_model(model_path, custom_objects={'precision':helper_funcs.precision,
                                                                                'recall':helper_funcs.recall,
                                                                                'f1_score':helper_funcs.f1_score})
    class_names = ['AspN', 'AspO', 'Blank', 'Cla', 'PPoly']

    # display model performance -- for model selection
    model_performance_dir = './Model Performance'
    # agg_performance_dir = os.path.join(model_performance_dir, "Agg.csv")
    # if show_all_model:
    #     slt.header("Display performance of all available models!", divider = "rainbow")
    #     slt.subheader("Select criteria to view")
    #     checkboxes = slt.columns(4)
    #     with checkboxes[0]:
    #         show_loss = slt.checkbox('Loss')
    #     with checkboxes[1]:
    #         show_acc = slt.checkbox('Accuracy')
    #     with checkboxes[2]:
    #         show_valloss = slt.checkbox('Validation Loss')
    #     with checkboxes[3]:
    #         show_valacc = slt.checkbox('Validation Accuracy')
    #     agg_performance_csv = pd.read_csv(agg_performance_dir)
    #     display_performance_csv = pd.DataFrame()
    #     display_performance_csv['Epoch'] = agg_performance_csv[['Epoch']]
    #     if show_loss:
    #         display_performance_csv['loss'] = agg_performance_csv[['loss']]
    #     if show_acc:
    #         display_performance_csv['accuracy'] = agg_performance_csv[['accuracy']]
    #     if show_valloss:
    #         display_performance_csv['val_loss'] = agg_performance_csv[['val_loss']]
    #     if show_valacc:
    #         display_performance_csv['val_accuracy'] = agg_performance_csv[['val_accuracy']]
    #     slt.line_chart(display_performance_csv, 
    #                    x = 'Epoch',
    #                    y = list(display_performance_csv.columns))

    # display selected model performance
    selected_model_performance_dir = os.path.join(model_performance_dir, f"{model_choice}.csv")
    performance_csv = pd.read_csv(selected_model_performance_dir)
    performance_csv['loss'] = np.log(performance_csv['loss'])
    performance_csv['val_loss'] = np.log(performance_csv['val_loss'])
    if display_model_performance:
        slt.header(f"{model_choice} performance", divider = "rainbow")
        chosen_model_diagram = slt.columns(2)
        with chosen_model_diagram[0]:
            slt.subheader("Model Accuracy")
            slt.line_chart(performance_csv, x = 'Epoch', 
                        y = ['accuracy', 'val_accuracy'],
                        color = ['#79cb9b', '#ffc48a'])
        with chosen_model_diagram[1]:
            slt.subheader("Model Loss")
            slt.line_chart(performance_csv, x = 'Epoch',
                        y = ['loss', 'val_loss'],
                        color = ['#547ac0', '#a369b0'])

# display image
img_h, img_w = 512, 512
if uploaded_file is not None and model_choice is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_array_RGB = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_array_RGB_resize = cv2.resize(img_array_RGB, (img_h, img_w))
    img_array_RGB_resize = tf.expand_dims(img_array_RGB_resize, axis = 0)
    img_array_RGB_resize = tf.keras.applications.resnet.preprocess_input(img_array_RGB_resize)
    slt.header("Classification of Imported Image", divider="rainbow")
    img_display = slt.columns(3)
    with img_display[0]:
        slt.write('')
    with img_display[1]:
        slt.image(img_array_RGB, caption = "Image 1", channels = "RGB", output_format = "auto")
    with img_display[2]:
        slt.write('')
    #print(img_array_RGB_resize.shape)
    prediction = classification_model.predict(img_array_RGB_resize)
    # print(type(prediction)) # array
    # print(prediction.shape) # (1,5)
    slt.subheader("Prediction:")
    idx_strain = np.argmax(prediction[0, :])
    prediction = pd.DataFrame(prediction, index = pd.Index(['Confidence']), columns = class_names)
    slt.dataframe(prediction.style.highlight_max(axis=1),
                  use_container_width=True)
    predicted_class = class_names[idx_strain]
    slt.write(f"The predicted strain is: **{predicted_class}**")
    opinion_classification = slt.radio("Do you agree with the classification?",
                                       ["Yes", "No"], index = None)
    if opinion_classification == "Yes" and opinion_classification is not None:
        contribute_img = slt.checkbox("Would you like to contribute the image to image collection for future model training?")
        if contribute_img:
            pass # add label and image to designated repo
            slt.write("Thanks for using functionality, looking forward to your next attempt!")
    elif opinion_classification == 'No' and opinion_classification is not None: # doubt on the classification accuracy
        user_label = slt.radio("What do you think the strain of the mold depicted in the uploaded image is?",
                  ["AspN", "AspO", "Gla", "PPoly", "Not listed"], horizontal=True, index = None)
        if user_label is not "Not listed":
            slt.text_area("What is the reasoning behind your doubt?")
        elif user_label == "Not listed":
            add_new_strain = slt.button("Add new strain", 
                                        help = "A new class will be created for the new strain")
            # need to figure out how to setup the version control system

# dropdown box for retrieving hard-coded knowledge
if uploaded_file is not None and model_choice is not None: # only provide knowledge retrieval option when photo is uploaded
    slt.header(f"Wiki about {predicted_class}", divider="rainbow")
    slt.write(helper_funcs.feature_extract(predicted_class, "Basic Info")) # directly writing some basic info
    if predicted_class == 'AspN':
        knowledge_type = slt.selectbox(f"Select the feature you want to learn more about {predicted_class}.", 
                                            ("--select--", 
                                            "Feature & Habit",
                                            "Potential Hazards"))
    elif predicted_class == "AspO":
        knowledge_type = slt.selectbox(f"Select the feature you want to learn more about {predicted_class}.",
                                       ("--select--",
                                       "Health Hazards",
                                       "Feature & Habit"))
    elif predicted_class == "Gla":
        pass
    elif predicted_class == "PPoly":
        knowledge_type = slt.selectbox(f"Select the features you want to learn more about {predicted_class}",
                                       ("Taxonomy",
                                        "Feature & Habit",
                                        "Metabolism",
                                        "Potential Applications"))

    response = helper_funcs.feature_extract(predicted_class, knowledge_type)
    # display text in the main window
    slt.write(response)
    # display text in the sidebar
    #slt.sidebar.write(response)
