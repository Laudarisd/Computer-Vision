# Face Detection with OpenCV and Haarcascades(implementation in Raspberry Pi)
A real-time face detection system using OpenCV and Haarcascades, further enhanced with the capability to recognize known faces and alert the user via SendGrid email notifications. Detected face information, along with related data, is stored in a MySQL database.

## Features
- Real-time face detection using Haarcascades.
- Face recognition using the `face_recognition` library.
- Sends alert emails for recognized and unrecognized faces using SendGrid.
- Stores detected face information in a MySQL database.

## Installation

### Prerequisites
- Python 3.x
- MySQL Server
- sendgrid

### Steps

1. **Clone the Repository**:
   ```bash
   git clone [your_repository_link]
   cd [repository_name]
   ```


2. **Install Dependencies**:

- Using pip:

    Install the necessary libraries:

```bash
    pip3 install imutils face_recognition sendgrid
```

    
- Using requirements.txt:

Install all dependencies using the provided requirements.txt file:

```bash
    pip3 install -r requirements.txt
``` 
3. Environment Variables:

Rename the .env.example to .env and fill in the required fields:

```bash
    SENDGRID_API_KEY=your_sendgrid_api_key
    SENDER_EMAIL=your_email@example.com
    RECEIVER_EMAIL=receiver_email@example.com
    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=your_mysql_password
    DB_NAME=FaceRecognition
```
***⚠️ Warning: Ensure your .env file is not shared or pushed to public repositories as it contains sensitive information.***

4. Database Setup:

Set up your MySQL server, create a database named FaceRecognition, and then create the necessary tables as outlined in the provided scripts.

5. Run the Program:

```bash
    python3 detect_sendmail.py
```

6. Usage
With the program running, it will utilize the camera to detect and recognize faces in real-time. Recognized and unrecognized faces will trigger an email alert, and relevant details will be saved in the MySQL database.


7. Note: Camera Module which is provided here uses a generic webcam (src=0). If we plan to use the Raspberry Pi camera module, we might need to modify the video stream initialization. Uncomment the line vs = VideoStream(usePiCamera=True).start() and comment out the line self.vs = VideoStream(src=0).start()


***Update***

- Added more error handling approached.
- Cleanup: Removed redundant lines.
- Code Optimization : Refactored repetitive tasks and improved readability. More advanced optimizations methods might require for robust optimizations process.


