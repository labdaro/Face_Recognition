import dlib
import numpy as np
from scipy import misc
import cv2

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
    return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors]

for i in range(1,11):
    print(i)
    known_image = cv2.imread("face_128D/User_"+ str(i)+ ".jpg")
    if known_image is None:
        continue      
    else:       
        enc = whirldata_face_encodings(known_image)
         #print(type(enc))
        print("imge {}".format(i))
        print(enc)
        np.savetxt('File_NPY/img'+ str(i) +'.npy',enc,delimiter=',')
        
