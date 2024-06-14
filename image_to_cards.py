import numpy as np
import cv2

import matplotlib.pyplot as plt
import numpy as np
import basic_poker
import os

class ImageCardDetector():

    def __init__(self,image_path=None):
        self.image=None
        self.card_images=[]
        self.contours=[]
        if image_path:
            self.load_image(image_path) 

    def load_image(self,image_path):
        self.image = cv2.imread(image_path)
        self.isolate_card_images()

    def find_cards(self, binary_threshold = 150, blur_factor = 100, fractional_card_min_area=.01 ):
        # find the card outlines by looking at a very blurry image, finding the edges, 
        # then simplifying the edge path dramatically
        image= self.image.copy()
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

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

    def isolate_card_images(self):
        self.card_images=[]
        self.find_cards()
        for contour in self.contours:
            flat_card=self.flattener(self.image,contour)
            self.card_images.append(flat_card.copy())
        return self.card_images

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

        output_card_width,output_card_height=200,300
        
        if rms_left_len < rms_right_len:
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
        destination_points = np.array([[0,0],[output_card_width-1,0],[output_card_width-1,output_card_height-1],[0, output_card_height-1]], np.float32)
        Matrix = cv2.getPerspectiveTransform(point_map,destination_points)
        warp = cv2.warpPerspective(image, Matrix, (output_card_width, output_card_height))
        return warp

    def _show_hand_detection(self):
        img = self.image.copy()
        #img = cv2.drawContours(img, self.contours, -1, (0, 255, 0), 5) 
        for contour in self.contours:
            self.flattener(img,contour)
            self.draw_min_area_rect_info(img,contour)
        img= cv2.resize(img, (600, 600))
        plt.imshow(img)
        plt.waitforbuttonpress()

    def draw_min_area_rect_info(self, image, cnt):
        # Get information about the MAR
        rect=cv2.minAreaRect(cnt)
        (x, y), (w, h), angle  = rect
 

        # Draw the rectangle on the image
        box = cv2.boxPoints(rect)  # Get corner points of the rectangle
        box = np.intp(box)  # Convert floating point coordinates to integer
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)  # Draw green rectangle

        # Draw the center point
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw blue circle at center

        # Put text for angle and coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_color = (255, 0, 0)

        # Display angle in degrees with a label
        angle_text = f"Angle: {angle:.2f}"
        cv2.putText(image, angle_text, (int(x), int(y)), font, font_scale, text_color, thickness)
        # Display angle in degrees with a label
        angle_text = f"w:{int(w)},h:{int(h)}"
        cv2.putText(image, angle_text, (int(x), int(y-30)), font, font_scale, text_color, thickness)
        return image

class SimpleSuitRankDetector:
    CORNER_WIDTH:int = 35
    CORNER_HEIGHT:int = 130
    MIDPOINT:int = 45
    TEMPLATE_DIMENSIONS = (32,32)
    TEMPLATE_DIRECTORY:str = "./templates"

    def __init__(self,np_cropped_card):
        self.image = np_cropped_card.copy()
        self.corner = self.image[0:self.CORNER_HEIGHT, 0:self.CORNER_WIDTH].copy()
        #plt.imshow(image)
        #plt.waitforbuttonpress()
        self.contours=None
        self.suit_crop=None
        self.suit=None
        self.rank_crop=None
        self.rank=None
        self.color=None

    def get_rank_suit(self):
        #pre process
        image= self.corner.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        binary_threshold, set_binary_to, thresh_type= 80, 255, cv2.THRESH_BINARY
        ret,thresh = cv2.threshold(gray, binary_threshold, set_binary_to, thresh_type)

        #find_contours and crop 
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)
        if len(contours)<2:
            print("contours not found")
            return None,None

        x0,y0,w0,h0 = cv2.boundingRect(contours[0])
        x1,y1,w1,h1 = cv2.boundingRect(contours[1])

        
        color,suit_crop,rank_crop=None,None,None
        #suit should be "above" rank
        if y0 > self.MIDPOINT:
            color = image[ int(y0+h0/2), int(x0+w0/2) ]
            self.suit_crop=thresh[y0:y0+h0, x0:x0+w0].copy()
            self.rank_crop=thresh[y1:y1+h1, x1:x1+w1].copy()
        else:
            color = image[int(y1+h1/2),int(x1+w1/2)]
            self.suit_crop=thresh[y1:y1+h1, x1:x1+w1].copy()
            self.rank_crop=thresh[y0:y0+h0, x0:x0+w0].copy()
        self.color= "Red" if color[2]>150 else "black" 
         
        
        try:
            self.rank = self.find_template_match(self.suit_crop, "templates/suits")
            self.rank = self.rank.split('_')[0]
            self.suit = self.find_template_match(self.rank_crop, "templates/ranks")
            self.suit = self.suit.split('_')[0]
        except:
            self.rank,self.suit= False,False 

        return self.rank,self.suit
        
    
    def find_template_match (self,image, folder=None):
        if not folder:
            folder= self.TEMPLATE_DIRECTORY

        #check difference between all images in templates to find best match
        lowest_diff = float('inf') 
        most_similar_file = None
        for filename in os.listdir(folder):
            template_path = os.path.join(folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            #resize both to match size
            template = cv2.resize(template, self.TEMPLATE_DIMENSIONS, 0, 0)
            image = cv2.resize(image, self.TEMPLATE_DIMENSIONS, 0, 0)

            # Update if lower difference is found
            absdiff = cv2.absdiff(image, template).sum()
            if absdiff < lowest_diff:
                lowest_diff = absdiff
                most_similar_file = filename
        return most_similar_file

    def _show_hand_detection(self):
        img = self.corner.copy()
        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Green color, thickness 2
            cv2.circle(img, (x, y),  5, (255,0,0), -1)  # -1 for filled circle
        plt.imshow(img)
        plt.waitforbuttonpress()
    
if __name__=="__main__":
    #load an image then detect the best hand with the cards available
    simple_hand = basic_poker.Hand()
    detected_cards = ImageCardDetector("hand/clubs_2.jpg")
    for i in range(len(detected_cards.card_images)):
        processed_card = SimpleSuitRankDetector(detected_cards.card_images[i])
        suit,rank= processed_card.get_rank_suit()


        #convert what the card processor returns into what the basic_poker.card takes
        rank_converter={'ace': 'A', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
                        'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10','jack':'J','queen':'Q','king':'K'}
        suit_converter={"clubs":"Clubs","Hearts":"Hearts","diamonds":"Diamonds","spades":"Spades"}
        rank=rank_converter.get(rank,None)
        suit=suit_converter.get(suit,None)
        simple_card=basic_poker.Card(suit,rank)
        simple_hand.add_card(simple_card)
    

    #show the best available hand
    hand, high_rank,high_card= simple_hand.check_hand()
    reverse_value={14:"A",13:"K",12:"Q",11:"J"}
    high_rank= reverse_value.get(high_rank,high_rank)
    high_card= reverse_value.get(high_card,high_card)
    print(f"Hand: {hand}\nHighest Rank: {high_rank}\n2nd Rank / High Card: {high_card}")
    simple_hand.pretty_print()