import os

import numpy as np
import tensorflow.keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

model = K.models.load_model("best_model")
types = ["eczema", "carcinoma", "acne", "keratosis", "millia", "rosacea"]



# while True:
#     img_path = input("Enter the path to the image: ")
#     img = image.load_img(img_path, target_size=(299, 299))
#     x = img_to_array(img)
#     x = K.applications.xception.preprocess_input(x)

#     prediction = model.predict(np.array([x]))[0][0:6]
#     print(prediction)
#     test_pred = np.argmax(prediction)

#     result = [(types[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
#     result.sort(reverse=True, key=lambda x: x[1])

#     for j in range(6):
#         (class_name, prob) = result[j]
#         print("Top %d ====================" % (j + 1))
#         print(class_name + ": %.2f%%" % (prob))

#     print("\n")
