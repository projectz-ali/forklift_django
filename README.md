# forklift_django
The Django/eel based project that detects Persons using Yolo Algorithm and triggers alarm when the person is in given threshold set on the login page.

The project uses django along with Yolov5 to detect person and trigger alarms accordingly.


# Environment Setup
After setting up the envitonment according to the requirements.txt with

pip3 install -r requirements.txt

# Browser
Make sure you have Google Chrome browser installed in your machine to get your front-end appear

# Pre-trained Yolo Weights
Download the pretrained Yolo weights from https://drive.google.com/file/d/1eefGFtotnicknQru80449osE0Vy66UXi/view?usp=sharing
and add these under the main directory.

# Run Project
Then Simply run

python3 manage.py runserver

that will pop a window up in your browser. Enter password and You can start the processing after setting given thresholds for level 2 and level 3 co efficients.

# For Jetson:
To Run the code with GPIOs with jetson, You need to set view_with_GPIO in demo/urls

# Webcam
Set '0' in source variable to use webcam instead of a recorded video in line 212 of views.py and corresponding line of views_with_GPIO.py 

