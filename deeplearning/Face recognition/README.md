# Face recognition using Dlib

This is a 4 step procedure which is as follows:-
* Detect/identify faces in an image (using a face detection model) — for simplicity, we will only use images with one face/person in it, not more/less
* Predict face poses/landmarks (for the faces identified in step 1)
* Using data from step 2 and the actual image, calculate face encodings (numbers that describe the face)
* Compare the face encodings of known faces with those from test images to tell who is in the picture.

# How to execute 

1. Create a folder called 'face_recognition' or any name you like. Create another folder images inside this folder which will hold images of the different people you want to run face recognition on. Remember to number the images starting from 1.
## Note: make sure that all of those images only have ONE face in them (i.e. they can’t be group pictures) and they are all in JPEG format with filenames ending in .jpg.

2. Make another folder named test which will contain different images of the same people whose pictures you stored in the images folder. Again, make sure that each picture only has one person in it.

3. Install dlib and these two dependencies:-
dlib_face_recognition_resnet_model_v1.dat.bz2 from here: ["dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"]
shape_predictor_68_face_landmarks.dat.bz2 fom here: ["dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"]

