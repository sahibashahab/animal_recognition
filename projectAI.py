
import numpy as np
from tkinter.ttk import Progressbar
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Data Augmentation and Preprocessing
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#     'training_set',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical')

# test_set = test_datagen.flow_from_directory(
#     'test_set',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical')

# Model Architecture
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# # Model Compilation
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Model Training
# model.fit(x=training_set,
#           validation_data=test_set,
#           epochs=50)

# # Save the trained model
# model.save('animal_recognition_model.h5')


root = Tk()

frame = tk.Frame(root)
lbl_heading = tk.Label(frame, text='Animal Recognition', padx=25, pady=25, font=('verdana',16))
lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25, font=('verdana',16))
lbl_show_pic = tk.Label(frame)
entry_pic_path = tk.Entry(frame, font=('verdana',16))
btn_browse = tk.Button(frame, text='Selected Image', bg='grey', fg='#ffffff', font=('verdana',16))
lbl_prediction = tk.Label(frame, text='Animal:  ', padx=25, pady=25, font=('verdana',16))
lbl_predict = tk.Label(frame, font=('verdana',16))
lbl_confid = tk.Label(frame, text='Confidence:  ', padx=25, pady=25, font=('verdana',16))
lbl_confidence = tk.Label(frame, font=('verdana',16))


progress = Progressbar(frame, orient=HORIZONTAL, length=300, mode='determinate')
progress.grid(row=6, column=0, columnspan=2, pady=10)

def selectimg():
    global img
    global filename
    filename = filedialog.askopenfilename(initialdir="/my_images", title="Select Image", filetypes=(("png images","*.png"),("jpg images","*.jpg"),("jpeg images","*.jpeg")))
    
    img = Image.open(filename)
    img = img.resize((250,250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img
    entry_pic_path.insert(0, filename)
    
    prediction, confidence = predict_animal(filename)
    lbl_predict.config(text=prediction)
    lbl_confidence.config(text=confidence)
    
    # Update progress bar based on confidence level
    progress_value = int(float(confidence))
    progress['value'] = progress_value
    
    # Determine if it's "good" or "bad"
    if progress_value >= 80:
        progress_color = 'green'
        confidence_label = 'Good'
    else:
        progress_color = 'red'
        confidence_label = 'Bad'
    
    progress.configure(style=f'TProgressbar.{progress_color}Horizontal')
    progress_label.config(text=f'Confidence: {confidence_label}')

progress_label = tk.Label(frame, font=('verdana',16))
progress_label.grid(row=7, column=0, columnspan=2, pady=10)

from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

model = load_model('animal_recognition_model.h5')


class_labels =  ['butterfly', 'cat', 'chicken', 'cow','dog','elephant','goat','horse','spider','squril']
def predict_animal(image_path):
    # Load and resize the image
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    
    # Expand the dimensions to match the model input shape
    img = np.expand_dims(img, axis=0)
    
    # Normalize the pixel values
    img = img / 255.0

    # Make predictions
    prediction = model.predict(img)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(prediction)
    animal = class_labels[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    return animal, confidence



btn_browse['command'] = selectimg
frame.pack()

lbl_heading.grid(row=0, column=0, columnspan="2", padx=10, pady=10)
lbl_pic_path.grid(row=1, column=0)
entry_pic_path.grid(row=1, column=1, padx=(0, 20))
lbl_show_pic.grid(row=2, column=0, columnspan="2")
btn_browse.grid(row=3, column=0, columnspan="2", padx=10, pady=10)
lbl_prediction.grid(row=4, column=0)
lbl_predict.grid(row=4, column=1, padx=2, sticky='w')
lbl_confid.grid(row=5, column=0)
lbl_confidence.grid(row=5, column=1, padx=2, sticky='w')

root.mainloop()








