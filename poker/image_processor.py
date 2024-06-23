import numpy as np
import cv2

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
# Get the current script's path as a Path object
script_dir = Path(__file__).parent
# Define the template directory relative to the script directory
TEMPLATE_DIRECTORY = script_dir / "templates"

OUTPUT_CARD_WIDTH= 200
OUTPUT_CARD_HEIGHT= 300

class ImageCardDetector():

    def __init__(self, filepath=None, np_image=None):
        self.image = None
        self.card_images= []
        self.contours= []

        self.load_image(filepath,np_image)

    def load_image(self,filepath=None, np_image=None):
        self.card_images= []
        self.contours= []

        self.image = np_image
        if filepath is not None:#numpy doesn't like a naked "if filepath:"" if user accidentally throws in an image instead of a string
            self.image = cv2.imread(filepath)

        if self.image is not None:
            return self.isolate_card_images()
    
        return []

    def isolate_card_images(self):
        self.card_images=[]
        self.find_cards()
        for contour in self.contours:
            try:
                flat_card=self.flattener(self.image,contour)
            except:
                print("A card canidate failed to transform")
    
            self.card_images.append(flat_card.copy())
        return self.card_images    

    def find_cards(self, binary_threshold = 150, blur_factor = 100, fractional_card_min_area=.01 ):
        # find the card outlines by looking at a very blurry image, finding the edges, 
        # then simplifying the edge path dramatically
        image= self.image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        kernal_size= (11,11)
        blur = cv2.GaussianBlur(gray, kernal_size, blur_factor)

        set_binary_to, thresh_type= 255, cv2.THRESH_BINARY
        ret,thresh = cv2.threshold(blur, binary_threshold, set_binary_to, thresh_type)

        initial_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        print("Number of Contours found = " + str(len(initial_contours)))  
   
        #filter the noise out
        image_area= image.shape[0] * image.shape[1]
        self.contours=[]
        for contour in initial_contours:
            if cv2.contourArea(contour) > image_area * fractional_card_min_area:
                self.contours.append(contour)
        print("Number of Card Canidates = " + str(len(self.contours)))  
        return self.contours

    def flattener(self,image,contour):
        peri = cv2.arcLength(contour,True)
        corner_points = cv2.approxPolyDP(contour,0.01*peri,True)
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle  = rect
        box_points = cv2.boxPoints(rect)

        #match the rectangle coordinates with the corner points
        matched_corners=[]
        for bp in box_points:
            bx,by = bp
            closest_point=None
            closest_distance = float('inf')
            closest_index = None
            #find the closest corner point to this box point
            for i, cp in enumerate(corner_points):
                distance =np.linalg.norm(bp - cp[0])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = i
                    closest_point=cp
            matched_corners.append(corner_points[closest_index])
        
        
        # points from min area rect are from bottom most point 0, just to the left side is 1, 3 above, 2 across
        # angle from right side
        # see https://theailearner.com/tag/cv2-minarearect/
        # /\
        # \/___ 
        
        bottom= box_points[0]
        lhs=    box_points[1]
        rhs=    box_points[3]
        #sqrt(dx^2+dy^2) = len, but we don't need the sqrt because just doing a comparison to find which is longer
        rms_left_len=  (bottom[0]-lhs[0])**2 + (bottom[1]-lhs[1])**2
        rms_right_len= (bottom[0]-rhs[0])**2 + (bottom[1]-rhs[1])**2

        
        if rms_left_len < rms_right_len:
            #there is a way to use the four points and thier anglesto calculate the angle and distance of items on an image
            #this covers most cases

            #rank/suit are at 0, 2 coordinates
            #matched_corners=destination_orientation
            pass
        else:   
            #rank/suit are at 1,3 coordinates
            #rotate 90 degrees
            tmp=matched_corners.pop(0)
            matched_corners.append(tmp)
        

        point_map=np.float32(np.array(matched_corners))
        # destination points are the corners that these four points should map to 
        destination_points = np.array([[0,0],[OUTPUT_CARD_WIDTH-1,0],[OUTPUT_CARD_WIDTH-1,OUTPUT_CARD_HEIGHT-1],[0, OUTPUT_CARD_HEIGHT-1]], np.float32)
        Matrix = cv2.getPerspectiveTransform(point_map,destination_points)
        warp = cv2.warpPerspective(image, Matrix, (OUTPUT_CARD_WIDTH, OUTPUT_CARD_HEIGHT))
        return warp

    def get_hand_detection_cv_image(self):
        img = self.image.copy() 
        for contour in self.contours:
            self._draw_min_area_rect_info(img,contour)
        img= cv2.resize(img, (600, 600))
        return img

    def _draw_min_area_rect_info(self, image, cnt):
        # Get information about the MAR
        rect=cv2.minAreaRect(cnt)
        (x, y), (w, h), angle  = rect
        box = cv2.boxPoints(rect)  # Get corner points of the rectangle
        box = np.intp(box)  
        cv2.drawContours(image, [box], 0, (0, 255, 0), 10)  # Draw green rectangle


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_color = (255, 0, 0)
        text = f"Angle: {angle:.2f}"

        cv2.putText(image, text, (int(x), int(y)), font, font_scale, text_color, thickness)

        text = f"w:{int(w)},h:{int(h)}"
        cv2.putText(image, text, (int(x), int(y-30)), font, font_scale, text_color, thickness)
        return image

class SimpleSuitRankDetector:
    CORNER_WIDTH:int = 40
    CORNER_HEIGHT:int = 130
    MIDPOINT:int = 45
    TEMPLATE_DIMENSIONS = (32,32)  

    def __init__(self,np_cropped_card):
        self.image = np_cropped_card.copy()
        self.corner = self.image[0:self.CORNER_HEIGHT, 0:self.CORNER_WIDTH].copy()
        self.contours=None
        self.suit_crop=None
        self.suit=None
        self.rank_crop=None
        self.rank=None
        self.color=None
        self.get_rank_suit()

    def get_rank_suit(self):
        self.rank_crop, self.suit_crop, self.color = self.crop_rank_suit()
        try: 
            pass
        except Exception as e:  # Catch any exception
            print(f"Error cropping rank and suit: {e}")  
            self.rank,self.suit= False,False 
            return   False,False 
        #find values for suit and number
        try:
            file = self.find_template_match(self.rank_crop,TEMPLATE_DIRECTORY/"ranks")
            rank = file.name.split('_')[0]
            #Filenames need to be mapped to the ranks correctly
            rank_converter ={ 'ace': 'A', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
                              'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10','jack':'J','queen':'Q','king':'K'}
            self.rank= rank_converter.get(rank,rank)

            file = self.find_template_match(self.suit_crop, TEMPLATE_DIRECTORY/"suits")
            self.suit = file.name.split('_')[0]  
        except Exception as e:  # Catch any exception
            print(f"Failed to find appropriate template match: {e}") 
            self.rank,self.suit= False,False 
        
        print([self.rank,self.suit])  
        return self.rank,self.suit  
    
    def crop_rank_suit(self, corner=None):
        #pulled this function out to make it easier for me to train and label
        if corner is None:
            corner = self.corner.copy()


        gray = cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        binary_threshold, set_binary_to, thresh_type= 100, 255, cv2.THRESH_BINARY
        ret,thresh = cv2.threshold(gray, binary_threshold, set_binary_to, thresh_type)

        #find_contours and crop 
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours= [c for c in contours if cv2.contourArea(c)>50]#hit a 0 area contour which broke another function
        contours = sorted(contours, key=cv2.contourArea, reverse=True)#self.contours potential bug
        corner=cv2.drawContours(corner, contours, -1, (0, 255, 0), 1) 
        for contour in contours:
            print(cv2.contourArea(contour))
        #plt.imshow(corner)
        #plt.waitforbuttonpress()

        if len(contours)<2:
            print("contours not found")
            return None,None,None
        print()
        x0,y0,w0,h0 = cv2.boundingRect(contours[0])
        x1,y1,w1,h1 = cv2.boundingRect(contours[1])
        midpoint_x,midpoint_y=self._find_contour_weighted_center(contours)
        #crop the images around the suit and number
        color,suit_crop,rank_crop=None,None,None
        if y0 > y1:
            color = corner[ int(y0+h0/2), int(x0+w0/2) ]
            suit_crop=thresh[y0:y0+h0, x0:x0+w0].copy()
            rank_crop=thresh[y1:y1+h1, x1:x1+w1].copy()
        else:
            color = corner[int(y1+h1/2),int(x1+w1/2)]
            rank_crop=thresh[y0:y0+h0, x0:x0+w0].copy()
            suit_crop=thresh[y1:y1+h1, x1:x1+w1].copy()
            
        color = "Red" if color[2]>150 else "black"
        #plt.imshow(rank_crop)
        #plt.waitforbuttonpress()
        self.contours=contours
        return  rank_crop, suit_crop, color      
    
    def find_template_match(self, image, path=None):
        if not path:
            path = TEMPLATE_DIRECTORY  # Assuming TEMPLATE_DIRECTORY is a pathlib.Path

        # Check difference between all images in templates to find best match
        lowest_diff = float('inf')
        most_similar_file = None
        
        for file in path.iterdir():
            if file.is_file():  # Check if it's a file (not a directory)
                template_path = path / file
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

                # Resize both to match
                template = cv2.resize(template, self.TEMPLATE_DIMENSIONS, 0, 0)
                image = cv2.resize(image, self.TEMPLATE_DIMENSIONS, 0, 0)

                # Update if lower difference is found
                absdiff = cv2.absdiff(image, template).sum()
                if absdiff < lowest_diff:
                    lowest_diff = absdiff
                    most_similar_file = file

        return most_similar_file

    def get_hand_detection_cv_image(self):
        out=self.image.copy()
        corner= self.corner.copy()
        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(corner, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Green color, thickness 2
            out[0:self.CORNER_HEIGHT, 0:self.CORNER_WIDTH]=corner


            if self.suit and self.rank:
                #show rank and suit
                box_y=[int(OUTPUT_CARD_HEIGHT/2)-25,int(OUTPUT_CARD_HEIGHT/2+10)]
                box_x=[int(OUTPUT_CARD_WIDTH/4)-20, int(OUTPUT_CARD_WIDTH*3/4)]
                bg_color=(200,200,200)
                out[box_y[0]:box_y[1],box_x[0]:box_x[1]]= bg_color
                pretty_suits= {"Hearts": "♥", "Diamonds": "♦", "Spades": "♠", "Clubs": "♣"}
                #ps=pretty_suits.get(self.suit,self.suit)
                text= f'{self.rank} {self.suit[0]}'
                orgin = (int(OUTPUT_CARD_WIDTH/4), int(OUTPUT_CARD_HEIGHT/2))
                fontScale, color, thickness , font= 1 , (200, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX 
                out = cv2.putText(out, text, orgin, font, fontScale, color, thickness, cv2.LINE_AA) 
        return out
    

    def _find_contour_weighted_center(self,contours):
        # Accumulate weighted moments
        weighted_M10, weighted_M01, total_area = 0, 0, 0
        for cnt in contours:
            moments = cv2.moments(cnt)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            area = moments["m00"]

            weighted_M10 += centroid_x * area
            weighted_M01 += centroid_y * area
            total_area += area

        # Calculate weighted center
        if total_area > 0:
            weighted_center_x = int(weighted_M10 / total_area)
            weighted_center_y = int(weighted_M01 / total_area)
            return (weighted_center_x, weighted_center_y)
        else:
            return None
        

def _train_rank(path,out):
    detected_cards = ImageCardDetector(path)
    ssrd=SimpleSuitRankDetector(detected_cards.card_images[0])
    for i, card in enumerate(detected_cards.card_images):
            corner=card[0:150, 0:45].copy()
            rank_crop, suit_crop, color= ssrd.crop_rank_suit(corner)
            cv2.imwrite(f'{out}_r{i}.jpg',rank_crop)
            cv2.imwrite(f'{out}_s{i}.jpg',suit_crop)
        


def _basic_game(image_path): 
    import basic_poker
    #load an image then detect the best hand with the cards available
    simple_hand = basic_poker.Hand()
    detected_cards = ImageCardDetector(image_path)
    for i in range(len(detected_cards.card_images)):
        processed_card = SimpleSuitRankDetector(detected_cards.card_images[i])
        simple_card=basic_poker.Card(processed_card.rank,processed_card.suit)
        simple_hand.add_card(simple_card)
    

    #show the best available hand
    hand, high_rank,high_card= simple_hand.check_hand()
    reverse_value={14:"A",13:"K",12:"Q",11:"J"}
    high_rank= reverse_value.get(high_rank,high_rank)
    high_card= reverse_value.get(high_card,high_card)
    print(f"Hand: {hand}\nHighest Rank: {high_rank}\n2nd Rank / High Card: {high_card}")
    simple_hand.pretty_print()

if __name__=="__main__":
    _basic_game('/home/cjb/poker/hand_samples/clubs_2.jpg')
    #_train_rank('/home/cjb/poker/hand_samples/clubs_2.jpg',"tst")
