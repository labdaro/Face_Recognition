import time
import dlib
import numpy as np
from scipy import misc
import cv2
'''
            #Crop the image
def cropFace(number):
    list_image=[]

    # numer of user in image
    number_user = 1

    # class of face
    args_weights = 'data_file/mmod_human_face_detector.dat'

    # face detector class of dlib
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args_weights)

    # get image from file one by one
    for i in range(1, number):
        print(">>", i)

        args_image = 'Dataset_Image/img' + str(i) + '.jpg'

        # read image
        img =cv2.imread(args_image)
        resize_img =cv2.resize(img,(500,500))
        list_image.append(resize_img)
        
    for image in list_image:
        # if image is not a file move to next
        if image is None:
            print("Could not read input image")
            continue

        # start time of detection
        start = time.time()

        # apply face detection (cnn)
        try:
            faces_cnn = cnn_face_detector(image, 1)

            end = time.time()
            print("CNN : ", format(end - start, '.2f'))

            # loop over detected faces
            for face in faces_cnn:
                x = face.rect.left()
                y = face.rect.top()
                w = face.rect.right() - x
                h = face.rect.bottom() - y

                # side of Face for cropping
                crop_ima = image[y:y + h, x:x + w]
                # save image face
                cv2.imwrite("face_128D/User_" + str(number_user) + '.jpg', crop_ima)
                                # count next number of face
                number_user = number_user + 1

        except ThisIterationTakesTooLong:
            print ("pass")
    return number_user
cropFace(10)'''



                    #Test Encode the image

# Models Loaded
face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('data_file/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('data_file/dlib_face_recognition_resnet_model_v1.dat')


def whirldata_face_detectors(img, number_of_times_to_upsample=1):
    return face_detector(img, number_of_times_to_upsample)


def whirldata_face_encodings(face_image, num_jitters=1):
    face_locations = whirldata_face_detectors(face_image)
    pose_predictor = pose_predictor_68_point
    predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]

list_face =[]
list_encode =[]
i=1
for number_image in range(1,9):
    print(i)
    i+=1
    list_face.append(("face_128D/User_" + str(number_image) + '.jpg'))
reas= cv2.imread('face_128D/User_1.jpg')
cv2.imshow("Image,reas)


