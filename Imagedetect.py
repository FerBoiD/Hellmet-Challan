import cv2

frameWidth=640
frameHeight=480

faceCascade = cv2.CascadeClassifier("resourses/haarcascade_frontalface_default.xml")
nplateCascade = cv2.CascadeClassifier("resourses/haarcascade_russian_plate_number.xml")
# nplateCascade = cv2.CascadeClassifier("resourses/cascade.xml")


img = cv2.imread("resourses/helmet3.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

# print(faces)

for (x, y, w, h) in faces:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    minArea=2000;count=0
    grayCrop=imgGray[(y+2*h):(y+6*h),(x-2*w):(x+2*w)]
    # grayCrop=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)

    plates = nplateCascade.detectMultiScale(grayCrop, 1.06, 4)
    print(plates)

    for (x1,y1,w1,h1) in plates:
        area=w1*h1
        if(area>minArea):
            cv2.rectangle(img,((x-2*w)+x1,(y+2*h)+y1),((x-2*w)+x1+w1,(y+2*h)+y1+h1),(0,255,255),2)
            imgNPlate=img[(y+2*h)+y1:(y+2*h)+y1+h1,(x-2*w)+x1:(x-2*w)+x1+w1]

            cv2.imwrite("resourses/NoPlates_Images/NoPlate_"+str(count)+ ".jpg",imgNPlate)
            count+=1
            # cv2.rectangle(grayCrop,(x1,y1),(x1+w1,y1+h1),(0,255,255),2)

            cv2.imshow("Image of Noumber Plate", imgNPlate)
            cv2.waitKey(0)


cv2.imshow("Myself", img)
cv2.waitKey(0)
