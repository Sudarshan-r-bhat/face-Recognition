import keras
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import load_model
import os
import matplotlib.pyplot as plt
"""
# part-1
count = 0
video = cv2.VideoCapture("C:/Users/Microsoft/Desktop/image_recog.mp4")
flag, frame = video.read() # this contains the frames
# cv2.imshow("sudarshan", frame)
#cv2.waitKey(5)
while flag:
    cv2.imwrite(f"C:/Users/Microsoft/Desktop/sudarshan/{count}.jpg", frame)
    flag, frame = video.read()
    count += 1

"""
# part-2
# classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# for i in range(630):
#     image = cv2.imread(f"C:/Users/Microsoft/Desktop/sudarshan/{i}.jpg")
#     face_coordinate = classifier.detectMultiScale(image, 1.2, 5)
#     if str(type(face_coordinate)) == str(type(())):
#         continue
#     print(face_coordinate)
#     x, y, w, h = face_coordinate[0][0], face_coordinate[0][1], face_coordinate[0][2], face_coordinate[0][3]
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
#     # cv2.imshow("sudarshan", image[y:y+h, x:x+w])
#     # cv2.waitKey(5000)
#     cv2.imwrite(f"C:/Users/Microsoft/Desktop/sudarshan_crop/{i}.jpg", image[y:y + h, x:x + w])


''' 
note: the above code part1: generates images form the video frames. part2: detects the face and generates
cropped images as saves them.
and in the below part:
we are using the PIL library to resize the image to out desired dimension
'''
# for i in range(630):
#     try:
#         image2 = Image.open(f"C:/Users/Microsoft/Desktop/sudarshan_crop/{i}.jpg")
#         image2.thumbnail((100, 100))
#         # resized = image2.resize((100, 100)) # not necessary.
#         image2.save(f"C:/Users/Microsoft/Desktop/sudarshan_crop2/{i}.jpg")
#     except Exception as e:
#         continue


def plot_graph(hist):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.plot(hist.history["loss"], label="training loss")
    plt.plot(hist.history["val_loss"], label="testing loss")
    plt.legend()  # it shows the label for the plotted line.

    plt.subplot(1, 2, 2)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.plot(hist.history["acc"], label="training accuracy")
    plt.plot(hist.history["val_acc"], label="testing accuracy")
    plt.legend()
    plt.show()

# os.path.isfile(("C:/workstation/PycharmProjects/Machine Learning workspace/models")


if os.path.isfile(("C:/workstation/PycharmProjects/Machine Learning workspace/models/cnn.hfile")):

    print("model exists already. do you want to retry training the model (y or n)")
    if str(input()).strip().lower() == 'y':
        data_generator_train = ImageDataGenerator()
        data_generator_test = ImageDataGenerator()

        train_generator = data_generator_train.flow_from_directory("C:/Users/Microsoft/Desktop/sudarshan_crop/",
                                                                   target_size=(
                                                                       100, 100),
                                                                   color_mode="grayscale",
                                                                   batch_size=32,
                                                                   class_mode='categorical',
                                                                   shuffle=True)

        test_generator = data_generator_test.flow_from_directory("C:/Users/Microsoft/Desktop/sudarshan_crop/",
                                                                 target_size=(
                                                                     100, 100), color_mode="grayscale", batch_size=32,
                                                                 class_mode='categorical', shuffle=True)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 1)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(7, activation="sigmoid"))

        optimizer = Adam()
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        hist = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                                      epochs=50,
                                      validation_data=test_generator,
                                      validation_steps=test_generator.n // test_generator.batch_size)
        # os.remove("C:/workstation/PycharmProjects/Machine Learning workspace/cnn.hfile")
        model.save("C:/workstation/PycharmProjects/Machine Learning workspace/models/cnn.hfile")
        plot_graph(hist)
    else:
        camera = cv2.VideoCapture(0)
        names = ['Akshay_Kumar',
                 'Nawazuddin_Siddiqui', 'Person_1', 'Salman_Khan', 'Shahrukh_Khan', 'sudarshan_crop2', 'Sunil_Shetty']
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        m = load_model("models/cnn.hfile")
        while True:
            flag, img = camera.read()
            # img = cv2.imread("face_test.jpg")
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# classifier
            faces = classifier.detectMultiScale(gray_frame, 1.3, 5)
            if str(type(faces)) == str(type(())):
                cv2.imshow("sudarshan", img)
                if cv2.waitKey(1) == 27:
                    break
                continue

            for x, y, w, h in faces:
                crop_frame = gray_frame[y:y + h, x:x + w]
                r_img = cv2.resize(crop_frame, (100, 100))
    # m
                prediction = m.predict(r_img[np.newaxis, :, :, np.newaxis])
                name = names[np.argmax(prediction)]
                print(name)
                cv2.putText(img, name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("sudarshan", img)
            if cv2.waitKey(1) == 27:
                break
        camera.release()
        cv2.destroyAllWindows()