import random
import cv2
import numpy as np


class obj:
    def __init__(self, frame = np.array((640, 480, 3), dtype = 'uint8')):
        self.points = []
        self.frame = frame
        self.features = np.empty((2,2))
        self.aquired = False
        self.orb = cv2.ORB_create(nfeatures = 10000,scaleFactor = 1.1, nlevels = 8, edgeThreshold = 50, patchSize=40)
        
        
    def get_features(self):
        self.points, self.features = self.orb.detectAndCompute(self.frame, None)
        self.aquired = True

class projectile:
    def __init__(self):
        self.fired = False
        self.locs = []
        self.dirs = []
    
    def new_proj(self, center, direction):
        self.fired = True
        self.locs.append(center)
        if direction == 27:
            self.dirs.append((random.randint(-20,20), random.randint(-20,20)))
        elif direction == 37:
            self.dirs.append((-20, 0)) # left
        elif direction == 38:
            self.dirs.append((0, -20)) # up
        elif direction == 39:
            self.dirs.append((20, 0)) # right
        elif direction == 40:
            self.dirs.append((0, 20)) # down
        else:
            self.dirs.append((random.gauss(0,20), random.gauss(0,20)))
    def update(self):
        for i in range(len(self.locs)):
            self.locs[i][0] += self.dirs[i][0]
            self.locs[i][1] += self.dirs[i][1]
        
        
class Enemy:
    def __init__(self, location = (200,200), mode = 'e'):
        sizes = {'e' : (50,50), 'm' : (35,35), 'h' : (20,20)}
        self.speed_modifier = {'e' : 1, 'm' : 3, 'h' : 5}
        shield = {1: (0,0,255), 2 : (0,165,255), 3 : (0,255,0)}
        self.loc = np.array([location])
        self.mode = mode;
        self.mode_set = False
        self.size = sizes[mode]
        self.dir = np.array((random.randint(-5,5), random.randint(-5,5))) * self.speed_modifier[self.mode]
        self.health = 3
    def initialize(self):
        self.dir = np.array((random.randint(-5,5), random.randint(-5,5))) * self.speed_modifier[self.mode]
        
    def move(self):
        if(self.loc[0][1] < 480 and self.loc[0][1] > 0 and self.loc[0][0] < 640 and self.loc[0][0] > 0):
            self.loc[0][0] += self.dir[0]
            self.loc[0][1] += self.dir[1]
            
        else:
            self.loc[0] = (random.randint(0,640-1), random.randint(0,480-1))
            self.dir = np.array((random.randint(-5,5), random.randint(-5,5))) * self.speed_modifier[self.mode]
            while ( self.dir[0] == 0 and self.dir[1] == 0):
                self.dir = np.array((random.randint(-5,5), random.randint(-5,5))) * self.speed_modifier[self.mode]
            
    def draw(self, frame):
        sizes = {'e' : (50,50), 'm' : (35,35), 'h' : (20,20)}
        cv2.rectangle(frame, (self.loc[0][0], self.loc[0][1]), (self.loc[0][0] + sizes[self.mode][0], self.loc[0][1] + sizes[self.mode][1] ), shield[self.health], -1)
        
    def isHit(self, proj):
        return (proj[0] > self.loc[0][0] and proj[0] < self.loc[0][0] + self.size[0] and proj[1] > self.loc[0][1] and proj[1] < self.loc[0][1] + self.size[1])
    
    
    def setMode(self,difficulty):
        if difficulty == 101: 
            self.mode ='e'
            self.mode_set = True
        elif difficulty == 109:
            self.mode ='m'
            self.mode_set = True
        elif difficulty == 104: 
            self.mode ='h'
            self.mode_set = True
 
        
try:
    cap.release()
    cv2.destroyAllWindows()
except:
    pass
    
cap = cv2.VideoCapture(0)

win_size = 3
MIN_MATCH_COUNT = 40
shooter = projectile()
controller = obj()
Enemy = Enemy()
orb = cv2.ORB_create(nfeatures = 10000,scaleFactor = 1.1, nlevels = 8, edgeThreshold = 50, patchSize=40)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ret = True
count =0
shots = 0
shield = {1: (0,0,255), 2:(0,165,255), 3:(0,255,0)}
frame_count = 0
while(ret):
    frame_count +=1
    ret, frame = cap.read()
    count +=1
    frame_cpy = frame.copy()
    while(not Enemy.mode_set):
            ret, frame = cap.read()
            cv2.rectangle(frame, (100,100), (550,150), (0,255,0), -1)
            cv2.rectangle(frame, (100,250), (550,300), (0,255,255), -1)
            cv2.rectangle(frame, (100,400), (550,450), (0,0,255), -1)
            cv2.putText(frame, "Press 'e' for Easy", (250,120),cv2.FONT_HERSHEY_TRIPLEX, 0.55, (0,0,0) )
            cv2.putText(frame, "Press 'm' for Medium", (250,270),cv2.FONT_HERSHEY_TRIPLEX, 0.55, (0,0,0) )
            cv2.putText(frame, "Press 'h' for Hard", (250,420),cv2.FONT_HERSHEY_TRIPLEX, 0.55, (0,0,0) )
            cv2.imshow("frame" , cv2.resize(frame, (win_size * frame.shape[1], win_size * frame.shape[0])))
            x=cv2.waitKey(1)
            Enemy.setMode(x)
            Enemy.initialize()
            frame_count +=1
    if not controller.aquired:
        # Draw some arrows to guide user...
        cv2.arrowedLine(frame, (frame.shape[1],frame.shape[0]), (int(0.8 * frame.shape[1]), int(0.8 * frame.shape[0])), (248,150, 75), 3)
        cv2.arrowedLine(frame, (0,0), (int(0.2 * frame.shape[1]), int(0.2 * frame.shape[0])), (248,150, 75), 3)
        cv2.arrowedLine(frame, (0,frame.shape[0]), (int(0.2 * frame.shape[1]), int(0.8 * frame.shape[0])), (248,150, 75), 3)
        cv2.arrowedLine(frame, (frame.shape[1],0), (int(0.8 * frame.shape[1]), int(0.2 * frame.shape[0])), (248,150, 75), 3)
        cv2.rectangle(frame, (int(0.2 * frame.shape[1]-5), int(0.2 * frame.shape[0])-25), (int(0.8 * frame.shape[1]), int(0.2 * frame.shape[0])), (255,255,255), -1)
        cv2.putText(frame, "Place object inside of box and press 'G'", (int(0.2*frame.shape[1])-5, int(0.2 * frame.shape[0])-5), cv2.FONT_HERSHEY_TRIPLEX, 0.55, (100,0,150))
        cv2.rectangle(frame, (int(0.2 * frame.shape[1]), int(0.2 * frame.shape[0])), (int(0.8 * frame.shape[1]), int(0.8 * frame.shape[0])), (0,165,255), 2)
        cv2.imshow('frame', cv2.resize(frame, (win_size * frame.shape[1], win_size * frame.shape[0])))
        
        frame_count +=1
    k = cv2.waitKey(1)
    # Press g to aquire new object
    if k == 103:
        #cv2.destroyWindow('frame')
        count = 0
        controller.frame = frame_cpy[int(0.2*frame.shape[0]): int(0.8 * frame.shape[0]), int(0.2*frame.shape[1]) : int(0.8 * frame.shape[1]), :]
        controller.get_features()
    if(controller.aquired):
        
        
        if(Enemy.health > 0):
            kp, feats = orb.detectAndCompute(frame, None)
            
            # Check if any keypoints are detected in the current frame...if not we can't do any matching
            if kp:
                matches = bf.match(controller.features, feats)
                good = [elem for elem in matches if elem.distance < 40]
                
                ####### ~~~ homography  ~~~~ #######
                if len(good)>MIN_MATCH_COUNT and kp:
                    src_pts = np.float32([ controller.points[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist() 
                
                    h,w,_ = controller.frame.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    brect = cv2.minAreaRect(dst)
                    box = cv2.boxPoints(brect)
                    y = max(box[:,0]) - min(box[:,0])
                    x = max(box[:,1]) - min(box[:,1])
                    box = cv2.boxPoints(brect)
                    box = np.int0(box)
                    center, radius = cv2.minEnclosingCircle(box)
                    center = (int(center[0]), int(center[1]))
                    cv2.circle(frame, center, 12, (255,0,0),-1)
                    cv2.circle(frame, center, 8, (0,255,0),-1)
                    cv2.circle(frame, center, 4, (0,0,255),-1)
                    cv2.putText(frame, 'Shoot this box ' + str(Enemy.health) + ' more times to win!', (Enemy.loc[0][0]-50, Enemy.loc[0][1]-50),cv2.FONT_HERSHEY_TRIPLEX, 0.5, shield[Enemy.health] )
                    cv2.arrowedLine(frame, (Enemy.loc[0][0]-50, Enemy.loc[0][1]-50), (Enemy.loc[0][0], Enemy.loc[0][1]), shield[Enemy.health])
                    
#                    cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255,0,0), 2)
#                    cv2.drawContours(frame,[box],0,(0,0,255),Enemy.health) 
#                    frame = cv2.polylines(frame,[np.int32(dst)],True,Enemy.health,3, cv2.LINE_AA)
                    
                    if k != 255 and k!= 103: # as long as 'G' or 'NULL' are not entered, shoot projectile
                        shooter.new_proj([center[0], center[1]], k)
                        shots += 1
                    if shooter.fired: shooter.update()
                    Enemy.draw(frame)
                    
                    for circ in shooter.locs:
                        cv2.circle(frame, (int(circ[0]), int(circ[1])), 4, (0,0,255), -1)
                        
                        if Enemy.isHit(circ):
                            Enemy.health -=1
                            circ[0] = 1000
                            circ[1] = 1000
                    
                    Enemy.move()
                
                
                img3 = cv2.drawMatches(controller.frame, [],frame,[], [], None, flags=2)
                cv2.imshow('frame', cv2.resize(frame, (win_size*frame.shape[1], win_size*frame.shape[0])))   
                
            else:
                print('not enough kp')
                img3 = cv2.drawMatches(controller.frame, [],frame,[], [], None, flags=2)
                cv2.imshow('frame', cv2.resize(frame, (win_size*frame.shape[1], win_size*img3.shape[0])))
                
                frame_count +=1
        if k == 27:         # wait for ESC key to exit
            break
        if Enemy.health < 1:
            
            cv2.putText(frame, 'You won in ' + str(shots) + ' shots!', (200,200), cv2.FONT_HERSHEY_TRIPLEX, 1, (150,20,150))        
            cv2.imshow('frame', cv2.resize(frame, (win_size*frame.shape[1], win_size*frame.shape[0])));
            
            frame_count +=1

cv2.destroyAllWindows()
cap.release()