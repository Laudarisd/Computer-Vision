from imutils import paths
import face_recognition
import pickle
import cv2
import os


#create a folder model inside src folder
if not os.path.exists('src/model'):
	os.makedirs('src/model')

class FaceEncoding():
	def __init__(self):
		self.knownEncodings = []
		self.knownNames = []
		self.imagePaths = list(paths.list_images("src/dataset"))
		print("[INFO] start processing faces...")
	# loop over the image paths
	def encodeFaces(self):
		for (i, imagePath) in enumerate(self.imagePaths):
			# extract the person name from the image path
			print("[INFO] processing image {}/{}".format(i + 1,
				len(self.imagePaths)))
			name = imagePath.split(os.path.sep)[-2]
			# load the input image and convert it from RGB (OpenCV ordering)
			# to dlib ordering (RGB)
			image = cv2.imread(imagePath)
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# detect the (x, y)-coordinates of the bounding boxes
			# corresponding to each face in the input image
			boxes = face_recognition.face_locations(rgb,
				model="hog")
			# compute the facial embedding for the face
			encodings = face_recognition.face_encodings(rgb, boxes)
			# loop over the encodings
			for encoding in encodings:
				# add each encoding + name to our set of known names and
				# encodings
				self.knownEncodings.append(encoding)
				self.knownNames.append(name)
		# dump the facial encodings + names to disk
		print("[INFO] serializing encodings...")
		data = {"encodings": self.knownEncodings, "names": self.knownNames}
		f = open("src/model/model.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()

if __name__ == '__main__':
	encode = FaceEncoding()
	encode.encodeFaces()
