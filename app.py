from flask import Flask, render_template, redirect, url_for
from flask import Flask, request, jsonify
import torch
from yolov5 import utils  # YOLOv5 的工具库w
from PIL import Image
import io
import base64
from PIL import Image, ImageDraw
import os

app = Flask(__name__)

def get_waste_category_map():
    categories = [
        {'name': 'Plastic bowls and pots', 'description': 'Recyclable', 'display': 'Plastic bowls and pots'}, 
        {'name': 'Plastic hangers', 'description': 'Recyclable', 'display': 'Plastic hangers'},
        {'name': 'Chargers', 'description': 'Recyclable', 'display': 'Chargers'},
        {'name': 'Bags', 'description': 'Recyclable', 'display': 'Bags'},
        {'name': 'Cosmetic bottles', 'description': 'Recyclable', 'display': 'Glass bottles'},
        {'name': 'Cans', 'description': 'Recyclable', 'display': 'Cans'},
        {'name': 'Pillows', 'description': 'Recyclable', 'display': 'Cans'},
        {'name': 'Stuffed Animals', 'description': 'Recyclable', 'display': 'Stuffed toys'},
        {'name': 'Shampoo Bottles', 'description': 'Recyclable', 'display': 'Plastic Bottles'},
        {'name': 'Plastic Toys', 'description': 'Recyclable', 'display': 'Plastic Toys'},
        {'name': 'Courier Bags', 'description': 'Recyclable', 'display': 'Bags'},
        {'name': 'Plugs and Cords', 'description': 'Recyclable', 'display': 'Plugs and Cords'},
        {'name': 'Used Clothes', 'description': 'Recyclable', 'display': 'Used Clothes'},
        {'name': 'Cutting Boards', 'description': 'Recyclable', 'display': 'Cutting Boards'},
        {'name': 'Cardboard Boxes', 'description': 'Recyclable', 'display': 'Cardboard Boxes'},
        {'name': 'Seasoning bottles', 'description': 'Recyclable', 'display': 'Glass bottles'},
        {'name': 'Wine Bottles', 'description': 'Recyclable', 'display': 'Glass bottles'},
        {'name': 'Glasses', 'description': 'Recyclable', 'display': 'Glasses'},
        {'name': 'Shoes', 'description': 'Recyclable', 'display': 'Shoes'},
        {'name': 'Metal food cans', 'description': 'Recyclable', 'display': 'Cans'},
        {'name': 'Pots', 'description': 'Recyclable', 'display': 'Pots'},
        {'name': 'Cooking oil drums', 'description': 'Recyclable', 'display': 'Cooking oil drums'},
        {'name': 'bones', 'description': 'Food Waste', 'display': 'Bones'},
        {'name': 'fruit peelings', 'description': 'Food Waste', 'display': 'Fruit peelings'},
        {'name': 'fruit pulp', 'description': 'Food Waste', 'display': 'Fruit pulp'},
        {'name': 'Tea Leaf Dregs', 'description': 'Food Waste', 'display': 'Tea Leaf Dregs'},
        {'name': 'vegetable leaves and roots', 'description': 'Food Waste', 'display': 'Vegetable leaves/roots'},
        {'name': 'leftovers', 'description': 'Food Waste', 'display': 'Leftovers'},
        {'name': 'Eggshells', 'description': 'Food Waste', 'display': 'Eggshells'},
        {'name': 'fish bones', 'description': 'Food Waste', 'display': 'Fish bones'},
        {'name': 'cigarette butts', 'description': 'Other Waste', 'display': 'Cigarette butts'},
        {'name': 'toothpicks', 'description': 'Other Waste', 'display': 'Toothpicks'},
        {'name': 'disposable fast food containers', 'description': 'Other Waste', 'display': 'Disposable fast food containers'},
        {'name': 'stained plastics', 'description': 'Other Waste', 'display': 'Stained plastics'},
        {'name': 'broken flower pots and bowls', 'description': 'Other Waste', 'display': 'Broken flower pots/bowls'},
        {'name': 'bamboo chopsticks', 'description': 'Other Waste', 'display': 'Chopsticks'},
        {'name': 'Bottles', 'description': 'Recyclable', 'display': 'Plastic Bottles'},
        {'name': 'dry batteries', 'description': 'Hazardous Waste', 'display': 'Batteries'},
        {'name': 'Ointment', 'description': 'Hazardous Waste', 'display': 'Medicines'},
        {'name': 'expired medicines', 'description': 'Hazardous Waste', 'display': 'Medicines'}
    ]
    return {item['name']: {'description': item['description'], 'display': item['display']} for item in categories}



def draw_boxes_and_save(image, boxes, save_path):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        label = box['name']
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="red", width=3)
        draw.text((box['xmin'], box['ymin']), label, fill="red")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    image.save(save_path)
    return image

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        label = box['name']
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="red", width=3)
        draw.text((box['xmin'], box['ymin']), label, fill="red")
    return image

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best-400.pt') 

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    img = Image.open(image_bytes)

    # YOLOv5 model prediction
    results = model(img, size=640)
    results_data = results.pandas().xyxy[0].to_dict(orient="records")

    waste_map = get_waste_category_map()
    detected_objects = []
    for result in results_data:
        obj_name = result['name']
        category_info = waste_map.get(obj_name, {'description': 'Unknown', 'display': obj_name})  # Default to name if not found
        detected_objects.append({
            'display': category_info['display'],
            'category': category_info['description']
        })

    # Draw boxes on the image and save it
    save_path = 'temp/detected_image.jpg'
    img_with_boxes = draw_boxes_and_save(img, results_data, save_path)

    img_byte_arr = io.BytesIO()
    img_with_boxes.save(img_byte_arr, format='JPEG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    return jsonify({
        'image': 'data:image/jpeg;base64,' + encoded_img,
        'results': detected_objects
    })



def get_categories():
    return [
        {'name': 'Disposable Fast Food Box', 'description': 'Other Waste'},
        {'name': 'Contaminated Plastic', 'description': 'Other Waste'},
        {'name': 'Cigarette Butts', 'description': 'Other Waste'},
        {'name': 'Toothpicks', 'description': 'Other Waste'},
        {'name': 'Broken Pots and Dishware', 'description': 'Other Waste'},
        {'name': 'Bamboo Chopsticks', 'description': 'Other Waste'},
        {'name': 'Leftover Food', 'description': 'Food Waste'},
        {'name': 'Large Bones', 'description': 'Food Waste'},
        {'name': 'Fruit Peels', 'description': 'Food Waste'},
        {'name': 'Fruit Pulp', 'description': 'Food Waste'},
        {'name': 'Tea Leaves', 'description': 'Food Waste'},
        {'name': 'Vegetable Leaves and Roots', 'description': 'Food Waste'},
        {'name': 'Eggshells', 'description': 'Food Waste'},
        {'name': 'Fish Bones', 'description': 'Food Waste'},
        {'name': 'Power Bank', 'description': 'Recyclable'},
        {'name': 'Bags', 'description': 'Recyclable'},
        {'name': 'Cosmetic Bottles', 'description': 'Recyclable'},
        {'name': 'Plastic Toys', 'description': 'Recyclable'},
        {'name': 'Plastic Bowls and Basins', 'description': 'Recyclable'},
        {'name': 'Plastic Hangers', 'description': 'Recyclable'},
        {'name': 'Express Delivery Paper Bags', 'description': 'Recyclable'},
        {'name': 'Plugs and Wires', 'description': 'Recyclable'},
        {'name': 'Old Clothes', 'description': 'Recyclable'},
        {'name': 'Cans', 'description': 'Recyclable'},
        {'name': 'Pillows', 'description': 'Recyclable'},
        {'name': 'Stuffed Toys', 'description': 'Recyclable'},
        {'name': 'Shampoo Bottles', 'description': 'Recyclable'},
        {'name': 'Glass Cups', 'description': 'Recyclable'},
        {'name': 'Leather Shoes', 'description': 'Recyclable'},
        {'name': 'Chopping Boards', 'description': 'Recyclable'},
        {'name': 'Cardboard Boxes', 'description': 'Recyclable'},
        {'name': 'Seasoning Bottles', 'description': 'Recyclable'},
        {'name': 'Wine Bottles', 'description': 'Recyclable'},
        {'name': 'Metal Food Cans', 'description': 'Recyclable'},
        {'name': 'Pots', 'description': 'Recyclable'},
        {'name': 'Cooking Oil Barrels', 'description': 'Recyclable'},
        {'name': 'Beverage Bottles', 'description': 'Recyclable'},
        {'name': 'Dry Cells', 'description': 'Hazardous Waste'},
        {'name': 'Ointment', 'description': 'Hazardous Waste'},
        {'name': 'Expired Medications', 'description': 'Hazardous Waste'}
    ]




@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    # 用数据库拉清单
    categories = get_categories()
    return render_template('guide.html', categories=categories)

@app.route('/identify')
def identify():
    return render_template('identify.html')

@app.route('/')
def index():
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
