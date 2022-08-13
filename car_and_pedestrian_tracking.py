import cv2
video=cv2.VideoCapture('pedestrain_vehicle.mp4')
#pre trained car & pedestrian detector
classifier_file='haarcascade_car.xml'
pedestrian_tracker_file='haarcascade_fullbody.xml'
#creating car & pedestrian classifier
car_tracker=cv2.CascadeClassifier(classifier_file)
pedestrian_tracker=cv2.CascadeClassifier(pedestrian_tracker_file)
while True:
    read_successful, frame=video.read() #read one frame at a time
    if read_successful:
        #convert frames into grayscaled color
        grayscaled_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #detect all the cars & pedestrians in grayscaled frame(still image)
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)
    #drawing squares in cars
    for (x,y,w,h) in cars:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #drawing sqaures in pedestrians
    for (x,y,w,h) in pedestrians:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #to show the detected frames
    cv2.imshow("car & pedestrian detector",frame)
    key=cv2.waitKey(1)
    if key== 81 or key==113:
        break
video.release()

