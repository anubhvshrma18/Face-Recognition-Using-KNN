import cv2
import numpy as np

#init camera
cap = cv2.VideoCapture(1)

#face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

skip = 0
face_data=[]
dataset_path = './data/'

file_name =input("Enter the name of person : ")
while True:
    ret,frame = cap.read()

    if ret==False:
        continue
    
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    print(len(faces))


    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame , (x,y),(x+w,y+h),(0,255,255),2)

        #extract :region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        # print(len(face_section))

        skip+=1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))


    cv2.imshow("Frame",frame)
    # cv2.imshow("Face section",face_section)
    # cv2.imshow("Face Section",face_section)

    #Store every 10th face
    if(skip%10 == 0):
        #store the 10th face later on
        pass
    
    keypressed = cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break

# Convert our face list array into a numpy array
face_data= np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved at"+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()