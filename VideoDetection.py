import cv2

cap=cv2.VideoCapture("resourses/vidtry.mp4")
faceCascade=cv2.CascadeClassifier("resourses/haarcascade_frontalface_default.xml")
nplateCascade = cv2.CascadeClassifier("resourses/haarcascade_russian_plate_number.xml")

count=0
while True:
    success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        minArea= 850
        grayCrop = imgGray[(y+2*h):(y+6*h),(x-2*w):(x+2*w)]
        # grayCrop=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)

        plates = nplateCascade.detectMultiScale(grayCrop,1.06,4)
        print(plates)

        for (x1,y1,w1,h1) in plates:
            area = w1*h1
            if (area > minArea):
                cv2.rectangle(img, ((x-2*w)+x1, (y+2*h)+y1), ((x-2*w)+x1+w1,(y+2*h)+y1+h1),(0, 255, 255), 2)
                imgNPlate = img[(y+2*h)+y1:(y+2*h)+y1+h1, (x-2*w)+x1:(x-2*w)+x1+w1]
                # print("YOOOOO",count)
                cv2.imwrite("resourses/NoPlates_Images/NoPlate_" + str(count) + ".jpg", imgNPlate)
                count += 1
                cv2.rectangle(grayCrop,(x1,y1),(x1+w1,y1+h1),(0,255,255),2)

                cv2.imshow("Image of Number Plate of the Culprit", imgNPlate)
                cv2.waitKey(0)

                # cv2.rectangle(img,(0,0),(640,50),(255,0,255),cv2.FILLED)
                # cv2.putText(img,"No Helmet",(150,25),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0),2)
                # cv2.imshow("Culprit",img)
                # cv2.waitKey(0)

    nimg=cv2.resize(img,(640,480))

    cv2.imshow("Helmet detection",nimg)
    if(cv2.waitKey(100) and 0xFF==ord("q")):
        break

