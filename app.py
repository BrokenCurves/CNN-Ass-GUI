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
    detected_objects = [result['name'] for result in results_data]

    # Draw boxes on the image and save it
    save_path = 'path/to/save/detected_image.jpg'  # Specify the path where you want to save the image
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
    # 模拟数据，接口以后再写
    return [
        {'name': 'test 1', 'description': 'category A'},
        {'name': 'test 2', 'description': 'category A'},
        {'name': 'test 3', 'description': 'category B'},
        {'name': 'test 4', 'description': 'category C'},
        {'name': 'test 5', 'description': 'category C'},
        {'name': 'test 6', 'description': 'category E'},
        {'name': 'test 7', 'description': 'category E'},
        {'name': 'test 8', 'description': 'category F'}
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
