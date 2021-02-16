import os
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
from matplotlib import pyplot

#import mtcnn is working, this is a handy repo for face detection
from mtcnn.mtcnn import MTCNN


#inputting data

#dataset directory
directory = 'celebA\\'


#function to import an image as RGB numpy array
def load_image(filename):
    #load from file
    image = Image.open(filename)
    #convert to RGB
    image = image.convert('RGB')
    #store as array
    pixels = asarray(image)
    #line below can be used to examine array size and factors
    #print(pixels.shape)
    #return array
    return pixels
    #close opened file (not necessary but seems good)
    image.close()


#function to load and store multiple image pixel arrays,
#extract and resize facial pixel arrays via MTCNN, and then store
#in a higher order array
def load_faces(directory, n_faces):
    #define the list, to become the array^2
    faces = list()
    #invoke MTCNN model to do facial detection
    model = MTCNN()
    
    for filename in os.listdir(directory):
        #get single image pixel array from above
        pixels = load_image(directory + filename)
        #detect, resize, and extract facial pixel array
        face = extract_face(model, pixels)
        #no-face image handling (see likewise if: in extract_face)
        if face is None:
            continue
        #store individual resized facial pixel array
        faces.append(face)
        print(len(faces), face.shape)
        #stop once we have enough
        if len(faces) >= n_faces:
            break
    #last step converts the list into the array^2
    return asarray(faces)


#finally, the last function we need formalizes the above
def extract_face(model, pixels, required_size=(80,80)):
    #detect faces in the image
    faces = model.detect_faces(pixels)
    #skip cases with no face found
    if len(faces) == 0:
        return None
    #extract details of detected facial pixels; note faces[0] means
    #this will only extract the FIRST FACE DETECTED
    #(with a clean dataset such as celebA this doesn't matter..)
    x1, y1, width, height = faces[0]['box']
    #force detected pixel values to be positive (bug fix per tutorial)
    x1, y1 = abs(x1), abs(y1)
    
    #convert into original image coordinates
    x2, y2 = x1 + width, y1 + height
    #retrieve facial pixels themselves
    face_pixels = pixels[y1:y2, x1:x2]
    #resize pixels to target (GAN model) size
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    #note this function parses ONE pixel array and returns
    #a single, face-detected, resized pixel array
    return face_array


#actually load the images
faces = load_faces(directory, 1)
#print debugging check on loaded array^2
print('loaded: ', faces.shape)

faces_dataset = load_faces(directory, 10000)
print('loaded: ', faces_dataset.shape)

#save in compressed format
savez_compressed('align-80px_celebA_10000set.npz', faces_dataset)