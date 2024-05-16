from flask import Flask, render_template, redirect, url_for
from flask import Flask, request, jsonify
import torch
from yolov5 import utils  # YOLOv5 的工具库
from PIL import Image
import io
import base64
from PIL import Image, ImageDraw
import os

app = Flask(__name__)

def get_waste_category_map():
    categories = [
        {'name': 'Plastic bowls and pots', 'description': 'Recyclable'},
        {'name': 'Plastic hangers', 'description': 'Recyclable'},
        {'name': 'Chargers', 'description': 'Recyclable'},
        {'name': 'Bags', 'description': 'Recyclable'},
        {'name': 'Cosmetic bottles', 'description': 'Recyclable'},
        {'name': 'Cans', 'description': 'Recyclable'},
        {'name': 'Pillows', 'description': 'Recyclable'},
        {'name': 'Stuffed Animals', 'description': 'Recyclable'},
        {'name': 'Shampoo Bottles', 'description': 'Recyclable'},
        {'name': 'Plastic Toys', 'description': 'Recyclable'},
        {'name': 'Courier Bags', 'description': 'Recyclable'},
        {'name': 'Plugs and Cords', 'description': 'Recyclable'},
        {'name': 'Used Clothes', 'description': 'Recyclable'},
        {'name': 'Cutting Boards', 'description': 'Recyclable'},
        {'name': 'Cardboard Boxes', 'description': 'Recyclable'},
        {'name': 'Seasoning bottles', 'description': 'Recyclable'},
        {'name': 'Wine Bottles', 'description': 'Recyclable'},
        {'name': 'Glasses', 'description': 'Recyclable'},
        {'name': 'Shoes', 'description': 'Recyclable'},
        {'name': 'Metal food cans', 'description': 'Recyclable'},
        {'name': 'Pots', 'description': 'Recyclable'},
        {'name': 'Cooking oil drums', 'description': 'Recyclable'},
        {'name': 'bones', 'description': 'Food Waste'},
        {'name': 'fruit peelings', 'description': 'Food Waste'},
        {'name': 'fruit pulp', 'description': 'Food Waste'},
        {'name': 'Tea Leaf Dregs', 'description': 'Food Waste'},
        {'name': 'vegetable leaves and roots', 'description': 'Food Waste'},
        {'name': 'leftovers', 'description': 'Food Waste'},
        {'name': 'Eggshells', 'description': 'Food Waste'},
        {'name': 'fish bones', 'description': 'Food Waste'},
        {'name': 'cigarette butts', 'description': 'Other Waste'},
        {'name': 'toothpicks', 'description': 'Other Waste'},
        {'name': 'disposable fast food containers', 'description': 'Other Waste'},
        {'name': 'stained plastics', 'description': 'Other Waste'},
        {'name': 'broken flower pots and bowls', 'description': 'Other Waste'},
        {'name': 'bamboo chopsticks', 'description': 'Other Waste'},
        {'name': 'Bottles', 'description': 'Recyclable'},
        {'name': 'dry batteries', 'description': 'Hazardous Waste'},
        {'name': 'ointment', 'description': 'Hazardous Waste'},
        {'name': 'expired medicines', 'description': 'Hazardous Waste'}
    ]
    return {item['name']: item['description'] for item in categories}



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
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 确保路径正确

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
        category = waste_map.get(obj_name, 'Unknown')  # Default to 'Unknown' if not found
        detected_objects.append(f"{obj_name} ({category})")

    # Draw boxes on the image and save it
    save_path = 'temp/detected_image.jpg'  # Specify the path where you want to save the image
    img_with_boxes = draw_boxes_and_save(img, results_data, save_path)

    img_byte_arr = io.BytesIO()
    img_with_boxes.save(img_byte_arr, format='JPEG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    print("Encoded image data: ", encoded_img[:100])  # 打印一小部分以确认数据
    return jsonify({
        'image': 'data:image/jpeg;base64,' + encoded_img,
        'result': ', '.join(detected_objects) if detected_objects else "No objects detected"
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
