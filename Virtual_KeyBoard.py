import cv2
import time
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3,1288)
cap.set(4,720)
keyboard = [["Q","W","E",'R','T','Y','U','I','O','P',"[","]"],
            ["A",'S','D','F','G','H','J','K','L',';',"'"],
            ['Z','X','C','V','B','N','M',',','.','/']]
finaltext = ''
x = 0
kb = Controller()

# def drawAll(img,buttonlist):
#     for button in buttonlist:
#         x,y = button.pos
#         w,h = button.size
#         cvzone.cornerRect(img, (x,y,w,h),20, rt=0)
#         img = cv2.rectangle(img,button.pos,(x+w,y+h),(255,0,255),cv2.FILLED)
#         img = cv2.putText(img,button.text,(x+25,y+40),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
#     return img

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                        20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                    (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    # print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


class Button():
    def __init__(self,pos,text,size=[70,70]) -> None:
        self.pos = pos
        self.text = text
        self.size = size
    
    
    
buttonlist = []
shft = 0
for y,rows in enumerate(keyboard):
    for x,key in enumerate(rows):
        buttonlist.append(Button([50+(x*80)+shft,100+(y*80)],key))
    shft += 40
    
detector = HandDetector(detectionCon=0.9)
ptime= 0

while True:
    ret,frame = cap.read()
    # frame=  cv2.flip(frame,1)
    click  = False
    frame = detector.findHands(frame)
    
    lmlist,bboxInfo = detector.findPosition(frame)    
    frame = drawAll(frame,buttonlist)

    if lmlist:
        for button in buttonlist:
            x,y = button.pos
            w,h = button.size
            if x< lmlist[8][0] < x+w and y< lmlist[8][1] < y+h:
                frame = cv2.rectangle(frame,button.pos,(x+w,y+h),(170,0,170),cv2.FILLED)
                frame = cv2.putText(frame,button.text,(x+25,y+40),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
                l,__,_ = detector.findDistance(8,12,frame,draw=False)
                if l <30:
                    kb.press(button.text)
                    click = True
                    frame = cv2.rectangle(frame,button.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
                    frame = cv2.putText(frame,button.text,(x+25,y+40),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
                    finaltext += button.text
       
                    time.sleep(0.15)
    frame = cv2.rectangle(frame,(50,500),(750,600),(200,20,200),cv2.FILLED)
    if len(finaltext)>11:x=1
    frame = cv2.putText(frame,finaltext[x*-11:],(60,580),cv2.FONT_HERSHEY_SIMPLEX,3, (255,255,255),6)
                    
    
    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    frame = cv2.putText(frame,str(fps),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,200),3)
    
    
    
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 13:
        break

cv2.destroyAllWindows()
cap.release()