from flask import Flask, request, redirect, url_for,  send_file, render_template, jsonify , flash
import pathlib
from pathlib import Path
import cv2
import time
import numpy as np
import base64

import matplotlib.pyplot as plt


from poker.image_processor import ImageCardDetector, SimpleSuitRankDetector
from poker.basic_poker import Card, Hand
app = Flask(__name__)


# Get the script's directory
script_dir = Path(__file__).parent
UPLOAD_FOLDER = script_dir / 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def byte_encode_cv2_image(img):    
  _, buffer_original = cv2.imencode('.jpg', img)  # Use JPG for Base64 encoding
  return base64.b64encode(buffer_original.tobytes()).decode('utf-8')

def get_cv_intermediate_images(np_image):
    poker_hand=Hand()
    icd = ImageCardDetector(np_image=np_image)
    processed_hand_image=icd.get_hand_detection_cv_image()
    raw_card_images= icd.card_images
    display_cards=[]
    processed_cards=[]
    for i, card in enumerate(icd.card_images):
      ssrd=None
      try:
        ssrd=SimpleSuitRankDetector(card)
        display_cards.append(ssrd.get_hand_detection_cv_image())
      except Exception as e:  # Catch any exception
        print(f"Error processing card: {e}")
        display_cards.append(card)
      
      if ssrd:
        print(ssrd.suit,ssrd.rank)
        if ssrd.suit and ssrd.rank:
          poker_hand.add_card(Card(ssrd.rank, ssrd.suit))
    
    return processed_hand_image, display_cards, poker_hand
  
def allowed_file(filename):
  return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # Check if file part is available
        if 'image' not in request.files:
            flash('No image uploaded. Please select an image to proceed.', 'error')
            return redirect(request.url)

        file = request.files['image']

        # Check if user selected a file
        if file.filename == '':
            flash('No image uploaded. Please select an image to proceed.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Read the image using OpenCV
            image_data = file.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            try:
                hand_image, card_images, hand = get_cv_intermediate_images(image)
                win_str, high_card, secondary_high_card =hand.check_hand()
                rank_converter={11:'J',12:'Q',13:'K',14:'A'}
                high_card= rank_converter.get(high_card,high_card)
                secondary_high_card=rank_converter.get(secondary_high_card,secondary_high_card)
                best_hand=f'{win_str}, Rank:{high_card}, High Card:{secondary_high_card}'

                hand_image_path= UPLOAD_FOLDER / "hand_image.jpg"
                cv2.imwrite(str(hand_image_path), hand_image)
                # Generate unique filenames for intermediate images
                card_paths = []
                for i, card in enumerate(card_images):
                    filename = f"card_{i}.jpg"
                    card_name = UPLOAD_FOLDER / filename  # Use .jpg for clarity
                    cv2.imwrite(str(card_name), card)
                    card_paths.append(f"uploads/{filename}")

                  

                return render_template("index.html",hand_image=hand_image_path,images=card_paths,best_hand=best_hand)

            except Exception as e:
                flash(f'An error occurred during image processing: {str(e)}', 'error')
                return redirect(request.url)

    return render_template("index.html")  # Handle GET requests or non-POST cases


@app.route('/')
def index():
  return render_template('index.html')

def clear_cache(folder_path=UPLOAD_FOLDER):
  try:
    for item in folder_path.iterdir():
      if item.is_file():  # Check if it's a file (not a directory)
        item.unlink()  # Delete the file object
    print(f"Folder contents deleted: {folder_path}")
  except OSError as e:
    print(f"Error deleting files in folder: {folder_path}. Reason: {e}")


if __name__ == '__main__':
  clear_cache()
  app.run(debug=True)
