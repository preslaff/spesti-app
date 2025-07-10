import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from PIL import Image
import io

#INITIAL SETUP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_credentials.json'
app = Flask(__name__)
CORS(app)
vision_client = vision.ImageAnnotatorClient()
BGN_TO_EUR_RATE = 1.95583

# If the price's height is less than 3% of the
# image's total height, we'll consider it too small.
MIN_RELATIVE_HEIGHT_THRESHOLD = 0.03 

#HELPER FUNCTIONS
def calculate_area(vertices):
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

def parse_price_string(s):
    if not s: return None
    try: return float(s.replace(',', '.'))
    except (ValueError, TypeError): return None

#API ENDPOINT

@app.route('/api/verify-prices', methods=['POST'])
def verify_prices():
    if 'file' not in request.files:
        return jsonify({"status": "ГРЕШКА", "message": "Не е качен файл."}), 400
    
    file = request.files['file']
    image_content = file.read()
    
    
    try:
        image_for_dims = Image.open(io.BytesIO(image_content))
        image_width, image_height = image_for_dims.size
    except Exception as e:
        return jsonify({"status": "ГРЕШКА", "message": f"Невалиден файл с изображение: {e}"})

    #Perform OCR
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    annotations = response.text_annotations

    if not annotations:
        return jsonify({"status": "ГРЕШКА", "message": "Не можах да разчета текст."})

    #Find Price Candidates
    price_candidates = []
    price_pattern = re.compile(r'^\d+([.,]\d{1,2})?$')
    for annotation in annotations[1:]:
        if price_pattern.match(annotation.description):
            value = parse_price_string(annotation.description)
            if value is not None:
                price_candidates.append({
                    'value': value,
                    'area': calculate_area(annotation.bounding_poly.vertices),
                    'box': [{'x': v.x, 'y': v.y} for v in annotation.bounding_poly.vertices]
                })

    if len(price_candidates) < 2:
        return jsonify({"status": "ERROR", "message": f"Намерени са по-малко от две цени."})

    price_candidates.sort(key=lambda p: p['area'], reverse=True)
    
    #Proximity Check
    largest_price_box = price_candidates[0]['box']
    box_height = max(v['y'] for v in largest_price_box) - min(v['y'] for v in largest_price_box)
    
    relative_height = box_height / image_height
    
    if relative_height < MIN_RELATIVE_HEIGHT_THRESHOLD:
        return jsonify({
            "status": "TOO-FAR",
            "message": "Приближете се до етикета и опитайте отново."
        })
    

    #Verification
    largest_price_1 = price_candidates[0]
    largest_price_2 = price_candidates[1]

    if largest_price_1['value'] > largest_price_2['value']:
        bgn_price_data, eur_price_data = largest_price_1, largest_price_2
    else:
        bgn_price_data, eur_price_data = largest_price_2, largest_price_1
        
    price_bgn = bgn_price_data['value']
    price_eur = eur_price_data['value']
    expected_eur = price_bgn / BGN_TO_EUR_RATE
    price_difference = price_eur - expected_eur
    rounding_tolerance = 0.01

    if price_difference > rounding_tolerance:
        status = "INCORRECT"
        message = "Цената в EUR изглежда несправедливо завишена."
    else:
        status = "CORRECT"
        message = "Цената е правилна и съответства на официалния курс."

    return jsonify({
        "status": status,
        "message": message,
        "data": { "found_bgn": price_bgn, "found_eur": price_eur, "expected_eur": round(expected_eur, 2), "difference_eur": round(price_difference, 4), "bgn_box": bgn_price_data['box'], "eur_box": eur_price_data['box']}
    })

if __name__ == '__main__':
    app.run(debug=True)