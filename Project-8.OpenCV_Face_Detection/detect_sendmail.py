import os
import cv2
import pickle
import imutils
import base64
import time
from datetime import datetime
from imutils.video import VideoStream, FPS
import face_recognition
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from sendgrid.helpers.mail import Attachment, FileContent, FileName, FileType, Disposition
from dotenv import load_dotenv
from save_data_sql import Database
from datetime import datetime, timedelta

load_dotenv()

class DetectAndSendMail():
    def __init__(self):
        self.currentname = "unknown"
        self.encodingsP = "./src/model/model.pickle"
        self.cascade = "./src/xml/haarcascade_frontalface_default.xml"
        try:
            print("[INFO] loading encodings + face detector...")
            self.data = pickle.loads(open(self.encodingsP, "rb").read())
            self.detector = cv2.CascadeClassifier(self.cascade)
            print("[INFO] encodings and face detector loaded successfully!")
        except Exception as e:
            print(f"Error loading encodings or face detector: {e}")
        # print("[INFO] loading encodings + face detector...")
        # self.data = pickle.loads(open(self.encodingsP, "rb").read())
        # self.detector = cv2.CascadeClassifier(self.cascade)
        # print("[INFO] starting video stream...")
        # Additional initialization
        self.sent_emails = set()  # Set to store names for which emails have been sent
        self.unknown_count = 0  # Counter for unknown image filenames

    def Stream_video(self):
        try:
            self.vs = VideoStream(src=0).start()
            # vs = VideoStream(usePiCamera=True).start()
            time.sleep(2.0)
            self.fps = FPS().start()
        except Exception as e:
            print(f"Error initializing camera: {e}")

    def detect(self):
        db = Database() #create a database instances
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1,
                                                   minNeighbors=5, minSize=(30, 30),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)

            # Print date time and detected person names
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("Date Time:", dt_string)
            print("Person Recognized as", names)
            for i, ((top, right, bottom, left), name) in enumerate(zip(boxes, names)):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                # Save to respective directories
                face_image = frame[top:bottom, left:right]  # Extracting the face from the frame
                if name != "Unknown":
                    save_path = f"./src/known/{name}.jpg"
                    if not os.path.exists(save_path):  # Only save if the image doesn't already exist
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, face_image)
                else:
                    self.unknown_count += 1
                    save_path = f"./src/unknown/unknown_{self.unknown_count}.jpg"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, face_image)
                #save data in sql
                now = datetime.now()
                db.insert_detected_face(now, name, save_path)  # Save detected information to the database

                # # Save data in SQL
                # now = datetime.now()
                # last_detected = db.last_detection_time(name)

                # # Set a threshold for the time frame, e.g., 1 hour (can be adjusted)
                # threshold = timedelta(hours=1)

                # # Check if the face was detected outside the threshold time or if it's the first time
                # if not last_detected or now - last_detected > threshold:
                #     db.insert_detected_face(now, name, save_path)  # Save detected face

                # Send email if not sent before for this name 
                if name not in self.sent_emails:
                    self.send_email(name, save_path)  # modified to send the correct image path
                    self.sent_emails.add(name)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        db.close()
        cv2.destroyAllWindows()
        self.vs.stop()
        print("[INFO] elapsed time:", self.fps.elapsed())
        print("[INFO] approx. FPS:", self.fps.fps())
        cv2.destroyAllWindows()
        #self.vs.stop()
    def send_email(self, name, image_path):
        SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
        sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
        sender_email = Email(os.getenv('SENDER_EMAIL'))
        receiver_email = To(os.getenv('RECEIVER_EMAIL'))


        subject = f"You have a visitor: {name}"
        content = Content("text/html", f'<strong>Your webcam detected someone at the door. Do you know {name}?</strong>')

        # Prepare the email with the attached image
        mail = Mail(sender_email, receiver_email, subject, content)

        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_encoded = base64.b64encode(image_data).decode()

        attached_image = Attachment(
            FileContent(image_encoded),
            FileName(f"{name}.jpg"),
            FileType("image/jpeg"),
            Disposition('attachment')
        )
        mail.attachment = attached_image
        try:
            response = sg.client.mail.send.post(request_body=mail.get())
            print(response.status_code)
            print(response.headers)
        except Exception as e:
            print(f"Error occurred: {e}")

		
if __name__ == "__main__":
    dam = DetectAndSendMail()
    dam.Stream_video()
    dam.detect()

