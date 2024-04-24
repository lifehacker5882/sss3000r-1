import base64
import cv2
import os
import numpy as np
import io
import logging
import socketserver
import datetime
import time
#import keyboard
import threading
from gpiozero import DistanceSensor, LED, Button
from http import server
from threading import Condition
 
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import sqlite3

def startStream():
    stream_active = True
    ledGreen = None  # Initialize LED object
    try:
        # Parameters
        id = 0
        font = cv2.FONT_HERSHEY_COMPLEX
        height=1
        boxColor=(0,0,255)      #BGR- GREEN
        nameColor=(255,255,255) #BGR- WHITE
        confColor=(255,255,0)   #BGR- TEAL
 
        face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
 
        # names related to id
        names = []
 
        # Filling the names list with names:
        if os.path.exists('users.txt'):
            # Reading paths from file:
            userfile = open('users.txt', 'r')
 
            # Leser første linje/første felt i første post
            userid = userfile.readline()
 
            while userid != '':
                userid = userid.rstrip('\n')
                username = userfile.readline().rstrip('\n')
     
                names.append((userid, username))
                # Leser studentnr til neste student
                userid = userfile.readline()
 
            userfile.close()
 
            print(names)
 
        # Led lights:
        ledGreen = LED(21)
        ledYellow = LED(20)
        ledRed = LED(16)
 
        # The button:
        button = Button(12)
 
        PAGE = """\
        <html>
        <head>
        <title>Stamping system</title>
        </head>
        <body
            style="
                background-image: url('https://img.freepik.com/free-photo/white-painted-beautiful-brick-wall_53876-163288.jpg?w=1380&t=st=1713870755~exp=1713871355~hmac=7e1b63d71b28ce8af326ed702d51da4c427f0a013859659d22a1079ee91e0009');
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: 100% 100%;
            ">
 
            <div>
                <h1 style="text-align: center; font-size: 64px; text-shadow: 3px 3px #808080;">Stamping System</h1>
 
                <div style="display: flex; justify-content: center;">
 
                    <img style="text-align: center; position: relative; border-style: solid; border-radius: 50% / 10%" src="stream.mjpg" width="855" height="640"/>
 
                </div>
 
            </div>
 
        </body>
        </html>
        """
 
        class StreamingOutput(io.BufferedIOBase):
            def __init__(self):
                self.frame = None
                self.condition = Condition()
 
            def write(self, buf):
                with self.condition:
                    self.frame = buf
                    self.condition.notify_all()
 
 
        class StreamingHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(301)
                    self.send_header('Location', '/index.html')
                    self.end_headers()
                elif self.path == '/index.html':
                    content = PAGE.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                elif self.path == '/stream.mjpg':
                    print("The show is on")
                    self.send_response(200)
                    self.send_header('Age', 0)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                    self.end_headers()
                    try:
                        stamped = False
                        streamContinue = True
                        while streamContinue == True:
                            with output.condition:
                                    output.condition.wait()
                                    frame = output.frame
 
                            
                            ledRed.off()
                            ledYellow.off()
                            ledGreen.off()
                                
                            frame_array = np.frombuffer(frame,dtype=np.uint8)
                            frame_image = cv2.imdecode(frame_array,cv2.IMREAD_COLOR)
                            frame_umat = cv2.UMat(frame_image)



                            # Convert the image to grayscale
                            frameGray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)


                            #Convert fram from BGR to grayscale
                            #frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            #Create a DS faces- array with 4 elements- x,y coordinates top-left corner), width and height
                            faces = face_detector.detectMultiScale(
                                    frameGray,      # The grayscale frame to detect
                                    scaleFactor=1.1,# how much the image size is reduced at each image scale-10% reduction
                                    minNeighbors=5, # how many neighbors each candidate rectangle should have to retain it
                                    minSize=(150, 150) # Minimum possible object size. Objects smaller than this size are ignored.
                                    )


                            for(x,y,w,h) in faces:

                                namepos=(x+5,y-5) #shift right and up/outside the bounding box from top
                                confpos=(x+5,y+h-5) #shift right and up/intside the bounding box from bottom
                                #create a bounding box across the detected face
                                cv2.rectangle(frame_umat, (x,y), (x+w,y+h), boxColor, 3) #5 parameters - frame, topleftcoords,bottomrightcooords,boxcolor,thickness


                                #recognizer.predict() method takes the ROI as input and
                                #returns the predicted label (id) and confidence score for the given face region.

                                id, confidence = recognizer.predict(frameGray[y:y+h,x:x+w])
                                
                                index = 0
                                booli = True
                                count = 0
                                
                                for x in names:
                                    if int(x[0]) == id:
                                        index = count
                                        booli = False
                                    count += 1

                                # If confidence is less than 100, it is considered a perfect match
                                if confidence <= 60:
                                    if booli:
                                        id = names[id][1]
                                    else:
                                        id = names[int(index)][1]
                                    
                                    confidence = f"{100 - confidence:.0f}%" 
                                    cv2.putText(frame_umat, str(confidence), confpos, font, height, confColor, 1)
                                    #cv2.putText(frame, cm, (x+5, y+h+30), font, height, nameColor, 2)
                                    ledRed.off()
                                    ledYellow.on()
                                    
                                    #while True:
                                        #button.wait_for_press()
                                    if button.is_pressed:
                                        def readStatus():
                                            userStatusFile = open('userstatus.txt', 'r')
                                            
                                            userStatusList = {}

                                            # Leser første linje/første felt i første post
                                            userid = userStatusFile.readline()

                                            while userid != '':
                                                userid = userid.rstrip('\n')
                                                status = userStatusFile.readline().rstrip('\n')

                                                #users.append(userid,username)
                                                userStatusList[userid] = status
                                                # Leser studentnr til neste student
                                                userid = userStatusFile.readline()

                                            userStatusFile.close()
                                            return userStatusList
                                        
                                        def changeStatus(id, status):
                                            tempFile = open('tempstatus.txt', 'w')
                                            statusList = readStatus()
                                            for key, value in statusList.items():
                                                if key != id:
                                                    tempFile.write(key + "\n")
                                                    tempFile.write(value + "\n")
                                            tempFile.write(id + "\n")
                                            tempFile.write(status + "\n")
                                            
                                            tempFile.close()
                                            
                                            os.remove("userstatus.txt")
                                            os.rename("tempstatus.txt","userstatus.txt")
                                            
                                        def checkStatus(id):
                                            statusList = readStatus()
                                            status = None
                                            for key, value in statusList.items():
                                                if key == id:
                                                    status = value
                                            if status == None:
                                                status = "in"
                                            return status
                                        
                                        def switchStatus(status):
                                            if status == "out":
                                                newStatus = "in"
                                            else:
                                                newStatus = "out"
                                            return newStatus
                                        
                                        def stampInOrOut(id, status):
                                            #writes a timestamp for the user once their face is recognized
                                            timecard = open('timecard.txt', 'a')
                                            timestamp = datetime.datetime.now()
                                            timecard.write(status + ","+ id + "," + timestamp.strftime("%c") + "\n")
                                            timecard.close()
                                            cv2.putText(frame_umat, "Stamped" + status, confpos, font, height, confColor, 1)
                                            
                                        def sett_inn_data(status, id):
                                            conn = sqlite3.connect('stampData.db')
                                            cursor = conn.cursor()
                                            
                                            cursor.execute('''CREATE TABLE IF NOT EXISTS stampdata
                                                        (status TEXT,
                                                        name TEXT,
                                                        timestamp TEXT)''')
                                                        
                                            cursor.execute('''INSERT INTO stampdata (status, name, timestamp)
                                                        VALUES (?, ?, ?)''', (status, id, datetime.datetime.now()))
                                            
                                            conn.commit()
                                            conn.close()
                                        

                                        status = checkStatus(id)
                                        stampInOrOut(id, status)
                                        newStatus = switchStatus(status) 
                                        changeStatus(id, newStatus)
                                        sett_inn_data(status, id)
                                        ledRed.off()
                                        ledYellow.off()
                                        ledGreen.on()
                                        time.sleep(3)
                                        #event = threading.Event()
                                        #event.wait(5)

                                else:
                                    id = "unknown"
                                    confidence = f"{100 - confidence:.0f}%"
                                    ledRed.on()
                                    ledYellow.off()
                                    ledGreen.off()

                                #Display name and confidence of person who's face is recognized
                                cv2.putText(frame_umat, str(id), namepos, font, height, nameColor, 2)
                                #cv2.putText(frame, cm, (x+5, y+h+30), font, height, nameColor, 2)


                                # Display realtime capture output to the user


                            #cv2.imshow('Raspi Face Recognizer',frame_umat)
                            #ret, jpeg = cv2.imencode('.jpg', frame)
                            #frame_array = np.asarray(frame_umat)  # Convert the UMat to a NumPy array
                            img_8u = cv2.convertScaleAbs(frame_umat)
                            _, buffer = cv2.imencode('.jpg',  img_8u)  # Encode the image as JPEG
                            #jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                            # Convert the NumPy array back to bytes
                            #frame_bytes = frame.tobytes()
                            #print(frame_bytes)

                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(buffer))
                            self.end_headers()
                            self.wfile.write(buffer)
                            self.wfile.write(b'\r\n')


                            # Wait for 30 milliseconds for a key event (extract sigfigs) and exit if 'ESC' or 'q' is pressed
                            key = cv2.waitKey(30) & 0xff
 
 
                    except Exception as e:
                        logging.warning(
                            'Removed streaming client %s: %s',
                            self.client_address, str(e))
                else:
                    self.send_error(404)
                    self.end_headers()
 
        class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True
 
        cam = Picamera2()
        ## Initialize and start realtime video capture
        # Set the resolution of the camera preview
        cam.preview_configuration.main.size = (640, 360)
        cam.preview_configuration.main.format = "RGB888"
        cam.preview_configuration.controls.FrameRate=30
        cam.preview_configuration.align()
        cam.configure("preview")
 
        output = StreamingOutput()
        cam.start_recording(JpegEncoder(), FileOutput(output))
 
        try:
            address = ('0.0.0.0', 8000)
            http_server = StreamingServer(address, StreamingHandler)
            http_server.serve_forever()
        except KeyboardInterrupt:
            cam.stop_recording()
            # Closing everything
            cam.close()
            ledGreen.close()
            ledYellow.close()
            ledRed.close()
            ledRed.close()
            button.close()
            http_server.shutdown()
            http_server.server_close()
            stream_active = False    
    finally:
       print("Stream closing...")


def newFace():
    
    # Paths
    pathsArray = []
    # User-ID:
    userID = []
    
    # Uniqe ID:
    currentID = 0 

    # Constants
    COUNT_LIMIT = 10
    POS=(30,60)  #top-left
    FONT=cv2.FONT_HERSHEY_COMPLEX #font type for text overlay
    HEIGHT=1.5  #font_scale
    TEXTCOLOR=(0,0,255)  #BGR- RED
    BOXCOLOR=(255,0,255) #BGR- BLUE
    WEIGHT=3  #font-thickness
    FACE_DETECTOR=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    # Using LBPH(Local Binary Patterns Histograms) recognizer
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    
    def trainRecognizer(faces, ids):
        recognizer.train(faces, np.array(ids))
        # Create the 'trainer' folder if it doesn't exist
        if not os.path.exists("trainer"):
            os.makedirs("trainer")
        # Save the model into 'trainer/trainer.yml'
        recognizer.write('trainer/trainer.yml')
    
    # function to read the images in the dataset, convert them to grayscale values, return samples
    def getImagesAndLabels():
        
        faceSamples=[]
        ids = []
        
        print()
        print("Time to train the model with faces:")
        print()
        print("This will take some seconds...")
        print()
        
        c = 0
        
        for p in pathsArray:
            for file_name in os.listdir(str(p)):
                if file_name.endswith(".jpg"):
                    id = int(file_name.split(".")[1])
                    img_path = os.path.join(p, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    faces = face_detector.detectMultiScale(img)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img[y:y+h, x:x+w])
                        ids.append(id)
            c += 1
            print("Face number " + str(c) + " is finished")
            print()
        
        print("Total faces trained: " + str(c))
                    
        return faceSamples, ids
        

    def trainModel():
        # Get face samples and their corresponding labels
        faces, ids = getImagesAndLabels()
        #print(ids)
        
        #Train the LBPH recognizer using the face samples and their corresponding labels
        trainRecognizer(faces, ids)

    def collectAndTrainData(face_id, path, currentID):
        
        # Create an instance of the PiCamera2 object
        cam = Picamera2()
        ## Set the resolution of the camera preview
        cam.preview_configuration.main.size = (640, 360)
        cam.preview_configuration.main.format = "RGB888"
        cam.preview_configuration.controls.FrameRate=30
        cam.preview_configuration.align()
        cam.configure("preview")
        cam.start()
        
        count = 0
        
        while True:
            # Capture a frame from the camera
            frame=cam.capture_array()
            # Display count of images taken
            cv2.putText(frame,'Count:'+str(int(count)),POS,FONT,HEIGHT,TEXTCOLOR,WEIGHT)

            # Convert frame from BGR to grayscale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Create a DS faces- array with 4 elements- x,y coordinates (top-left corner), width and height
            faces = FACE_DETECTOR.detectMultiScale( # detectMultiScale has 4 parameters
                    frameGray,      # The grayscale frame to detect
                    scaleFactor=1.1,# how much the image size is reduced at each image scale-10% reduction
                    minNeighbors=5, # how many neighbors each candidate rectangle should have to retain it
                    minSize=(30, 30)# Minimum possible object size. Objects smaller than this size are ignored.
            )
            
            for (x,y,w,h) in faces:
                # Create a bounding box across the detected face
                cv2.rectangle(frame, (x,y), (x+w,y+h), BOXCOLOR, 3) # 5 parameters - frame, topleftcoords,bottomrightcooords,boxcolor,thickness
                count += 1 # increment count
                
                file_path = os.path.join(str(path), f"{face_id}.{currentID}.{count}.jpg") 
                
                # Kommer til å brukes dersom en bruker er registrert to ganger:
                '''
                if os.path.exists(file_path):
                    # Move the existing file to the "old_dataset" folder
                    old_file_path = file_path.replace(str(file_path), "old_dataset")
                    os.rename(file_path, old_file_path)
                    print("usz")
                '''
            
                # Write the newer images after moving the old images
                cv2.imwrite(file_path, frameGray[y:y+h, x:x+w])
            
            

            # Display the original frame to the user
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('FaceCapture', frame)
            # Wait for 30 milliseconds for a key event (extract sigfigs) and exit if 'ESC' or 'q' is pressed
            key = cv2.waitKey(100) & 0xff
            # Checking keycode
            if key == 27:  # ESCAPE key
                break
            elif key == 113:  # q key
                break
            elif count >= COUNT_LIMIT: # Take COUNT_LIMIT face samples and stop video capture
                break

        # Release the camera and close all windows
        print("\n [INFO] Exiting Program and cleaning up stuff")
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()
        
        trainModel()

    
    if os.path.exists('paths.txt'):
        # Reading paths from file:
        pathfile = open('paths.txt', 'r')

        # Leser første linje/første felt i første post
        path = pathfile.readline()

        while path != '':
            path = path.rstrip('\n')
            
            pathsArray.append(path)
           
            # Leser studentnr til neste student
            path = pathfile.readline()

        pathfile.close()
    
    
    boolNavn = True
    
    # if dataset folder doesnt exist create:
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    
    while boolNavn == True:
        face_id = input('\nNavn: ')
        
        path = "dataset" + "/" + face_id
        
        if not os.path.exists(str(path)):
            os.makedirs(str(path))
            # Writing data to file:
            f = open("paths.txt", "a")
            f.write(path + "\n")
            f.close()
            # Writing data to array:
            pathsArray.append(path)
            
            
            # Filling the names list with names:
            if os.path.exists('users.txt'):
                # Reading paths from file:
                userfile = open('users.txt', 'r')
     
                # Leser første linje/første felt i første post
                userid = userfile.readline()
     
                while userid != '':
                    userid = userid.rstrip('\n')
                    username = userfile.readline().rstrip('\n')
     
                    userID.append(userid)
                    # Leser studentnr til neste student
                    userid = userfile.readline()
     
                userfile.close()
                
                # Lengden på brukerlisten:
                userIdLen = len(userID) 
               
                
                
                if int(userIdLen) > 0:
                    print("1")
                    currentID = int(userID[userIdLen - 1]) + 1   
                else:
                    print("2")
                    currentID = 1
            else:
                currentID = 1
 
           
            
            # Writing user information to file:
            f = open("users.txt", "a")
            f.write(str(currentID) + "\n")
            f.write(face_id + "\n")
            f.close()
            
            print(pathsArray)
            
            print("Current id: " + str(currentID))
            boolNavn = False
        else:
            print("Slik bruker eksisterer allerede i systemet")
    
    collectAndTrainData(face_id, path, currentID)

def deleteFace():
    def deleteFiles(userid,name):
        #shutil.rmtree("/home/admin/Desktop/Njeg/HttpFaceRecognition/dataset/" + name)
        moreFiles = True
        index  = 1
        while moreFiles == True:
            fileExists = os.path.isfile("/home/admin/Desktop/Njeg/HttpFaceRecognition/dataset/" + name + "/" + name + "." + str(selectedUserId) + "." + str(index) + ".jpg")
            if fileExists == True:
                os.remove("/home/admin/Desktop/Njeg/HttpFaceRecognition/dataset/" + name + "/" + name + "." + str(selectedUserId) + "." + str(index) + ".jpg")
                index += 1
            else:
                moreFiles = False
        if index != 1:
            os.rmdir("/home/admin/Desktop/Njeg/HttpFaceRecognition/dataset/" + name)
            
    def deleteUserFileEntry(users,userid):
        tempFile = open("temp.txt", "w")
            
        for key, value in users.items():
            if key != userid:
                tempFile.write(key + "\n")
                tempFile.write(value + "\n")
        tempFile.close()
        
        os.remove("users.txt")
        os.rename("temp.txt","users.txt")
            
    def deletePathsFileEntry(users,userid):
        tempFile = open("temp.txt", "w")
            
        for key, value in users.items():
            if key != userid:
                tempFile.write("dataset/"+ value + "\n")
        tempFile.close()
        
        os.remove("paths.txt")
        os.rename("temp.txt","paths.txt")
    
    def updateTrainingData():
        # Paths
        pathsArray = []
        
        face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Using LBPH(Local Binary Patterns Histograms) recognizer
        recognizer=cv2.face.LBPHFaceRecognizer_create()
        
        def trainRecognizer(faces, ids):
            recognizer.train(faces, np.array(ids))
            # Create the 'trainer' folder if it doesn't exist
            if not os.path.exists("trainer"):
                os.makedirs("trainer")
            # Save the model into 'trainer/trainer.yml'
            recognizer.write('trainer/trainer.yml')
        
        # function to read the images in the dataset, convert them to grayscale values, return samples
        def getImagesAndLabels():
            
            faceSamples=[]
            ids = []
            
            print()
            print("Time to train the model with faces:")
            print()
            print("This will take some seconds...")
            print()
            
            c = 0
            
            for p in pathsArray:
                for file_name in os.listdir(str(p)):
                    if file_name.endswith(".jpg"):
                        id = int(file_name.split(".")[1])
                        img_path = os.path.join(p, file_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        faces = face_detector.detectMultiScale(img)

                        for (x, y, w, h) in faces:
                            faceSamples.append(img[y:y+h, x:x+w])
                            ids.append(id)
                c += 1
                print("Face number " + str(c) + " is finish")
                print()
            
            print("Total faces trained: " + str(c))
                        
            return faceSamples, ids
            

        def trainModel():
            # Get face samples and their corresponding labels
            faces, ids = getImagesAndLabels()
            #print(ids)
            
            #Train the LBPH recognizer using the face samples and their corresponding labels
            trainRecognizer(faces, ids)
        
        if os.path.exists('paths.txt'):
            # Reading paths from file:
            pathfile = open('paths.txt', 'r')

            # Leser første linje/første felt i første post
            path = pathfile.readline()

            while path != '':
                path = path.rstrip('\n')
                
                pathsArray.append(path)
               
                # Leser studentnr til neste student
                path = pathfile.readline()

            pathfile.close()
            
            #print(pathsArray)
            listlen = len(pathsArray)
            
            if int(listlen) > 0:
                trainModel()
            else:
               with open('trainer/trainer.yml', 'w') as file:
                   file.write("")
        else:
            print("Path does not exist")
        
        
    # Filling the names list with names:
    if os.path.exists('users.txt'):
        
        users = {}
        # Reading paths from file:
        userfile = open('users.txt', 'r')

        # Leser første linje/første felt i første post
        userid = userfile.readline()

        while userid != '':
            userid = userid.rstrip('\n')
            username = userfile.readline().rstrip('\n')

            #users.append(userid,username)
            users[userid] = username
            # Leser studentnr til neste student
            userid = userfile.readline()

        userfile.close()
        print("The following users have training data registered:\n([userid], [name])")
        for row in users.items():
            print(row)
        selectedUserId = str(input("Input the userid for the face you want to delete "))
        if selectedUserId in users.keys():
            name = users.get(selectedUserId)
            
            deleteFiles(selectedUserId,name)
            deleteUserFileEntry(users,selectedUserId)
            deletePathsFileEntry(users,selectedUserId)
            
            
            
            updateTrainingData()
                
                
            print()
            print("training data deleted")
            print()
        else:
            print("Please select a valid userid")
            print()
                    
        
def main():
    program_active = True
    while program_active == True:
        print("╭━╮╱╭╮╱╱╱╱╱╱╱╱╱╱╱╱╭━━╮╱╱╱╱╱╱╱╱╱╱╱╭╮╭━━━┳━━━╮\n┃┃╰╮┃┃╱╱╱╱╱╱╱╱╱╱╱╭┫╭╮┃╱╱╱╱╱╱╱╱╱╱╱┃┃┃╭━╮┃╭━╮┃\n┃╭╮╰╯┣━━┳━━┳━━┳━━╋┫╰╯╰┳━┳━━┳━╮╭━━┫╰┻┫╭╯┣╯╭╯┃\n┃┃╰╮┃┃┃━┫╭╮┃╭╮┃━━╋┫╭━╮┃╭┫╭╮┃╭╮┫╭━┫╭╮┃┃╭╯╱┃╭╯\n┃┃╱┃┃┃┃━┫╰╯┃╰╯┣━━┃┃╰━╯┃┃┃╭╮┃┃┃┃╰━┫┃┃┃┃┃╱╱┃┃\n╰╯╱╰━┻━━┻━╮┣━━┻━━┫┣━━━┻╯╰╯╰┻╯╰┻━━┻╯╰╯╰╯╱╱╰╯\n╱╱╱╱╱╱╱╱╭━╯┃╱╱╱╱╭╯┃\n╱╱╱╱╱╱╱╱╰━━╯╱╱╱╱╰━╯\nWelcome!\nHere are the different operations:\n1 - Start Stream\n2 - Add new face to model\n3 - Delete face from model\n4 - Quit program")
 
        choice = int(input("Choose one operation: "))
 
        if choice == 1:
            print("Starting stream...")
            file_size = os.path.getsize('trainer/trainer.yml')
            if (file_size != 0):
                startStream()
            else:
                print()
                print("!!!!!!!!!!!")
                print("You must first register a face before starting the stream!")
                print("!!!!!!!!!!!")
                print()
 
        elif choice == 2:
            newFace()
 
        elif choice == 3:
            deleteFace()
 
        elif choice == 4:
            print("Program shutting down...")
            program_active = False
 
        else:
            print("Please provide a valid input!\n")
 
main()
 