import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

##########################################
#PART I: Model_AI

def Model_AI():
   (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

   #test
   ''' 
   if X_train.shape == (60000, 28, 28) and Y_train.shape == (60000,):
    print("test ok")
   for i in range(12):  
    plt.subplot(4,3,i+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.show()
    '''
   X_train = tf.keras.utils.normalize(X_train, axis = 1)
   X_test = tf.keras.utils.normalize(X_test, axis = 1)
   
   X_train = X_train.reshape(X_train.shape[0],28,28,1)
   X_test = X_test.reshape(X_test.shape[0],28,28,1)
   
   model = tf.keras.models.Sequential()
   Conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')
   model.add(Conv1)
   MaxPool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
   model.add(MaxPool1)
   Conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')
   model.add(Conv2)
   MaxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
   model.add(MaxPool2)
   model.add(tf.keras.layers.Flatten())
   Dense1 = tf.keras.layers.Dense(units = 128, activation='relu')
   model.add(Dense1)
   Dense2 = tf.keras.layers.Dense(units = 10, activation='softmax')
   model.add(Dense2)
   
   model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
   model.fit(X_train, Y_train, epochs = 2)
   
   model.save('Handwritten_Digit_AI.keras')

##########################################
#PART II: Writing Test
def Writing_Test():
   global model_loaded, image, draw_pil
   model_loaded = tf.keras.models.load_model('Handwritten_Digit_AI.keras')

   window = tk.Tk()
   window.geometry("600x600")
   window.title("Writting Test")
   
   Canvas = tk.Canvas(window, bg = "white", height = 400, width = 400)
   Canvas.pack()
   
   #Create a PIL image
   image = Image.new("L", (300, 300), color=0)
   draw_pil = ImageDraw.Draw(image)

   def draw_digit(event):
      x, y = event.x, event.y
      r = 10
      Canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", width = 20)
      draw_pil.ellipse([x - r, y - r, x + r, y + r], fill=255)
      
   def clear():
      global image, draw_pil
      Canvas.delete("all")
      image = Image.new("L", (400, 400), color=0)
      draw_pil = ImageDraw.Draw(image)
      
   def predict_the_digit():
      img_resized = image.resize((28, 28))
      img_array = np.array(img_resized) / 255.0
      img_array = img_array.reshape(1, 28, 28, 1)
      prediction = model_loaded.predict(img_array)
      digit = np.argmax(prediction)
      result = tk.Toplevel(window)
      result.title("Result")
      label = tk.Label(result, text=f"Predict number : {digit}", font=("Arial", 24))
      label.pack()


   Canvas.bind("<B1-Motion>", draw_digit)

   clear_button = tk.Button(window, text="Delete", font = ('Arial', 18), command=clear)
   clear_button.pack()
   
   predict_button = tk.Button(window, text="Predict", font = ('Arial', 18), command=predict_the_digit)
   predict_button.pack()
   
   window.mainloop()
   
   

##########################################
# PART III:Run  
if __name__ == "__main__":
    Model_AI()
    Writing_Test()