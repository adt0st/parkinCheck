from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import io
import base64
import os
import re
from werkzeug.utils import secure_filename
import json
import time
from datetime import datetime, UTC
from openpyxl import Workbook
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')

# Модель для истории запросов
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # время обработки в секундах
    spots_found = db.Column(db.Integer)    # количество свободных spots
    result_json = db.Column(db.Text)       # json-строка с деталью detections
    filename = db.Column(db.String(256))   # имя загруженного файла (для идентификации)

    def __repr__(self):
        return f'<История {self.id} {self.timestamp} {self.processing_time}с {self.spots_found} мест>'

@app.route('/')
def index():
    return render_template('index.html')

parking_spots = [
    {'id': 1, 'coords': (23, 104, 92, 231)},
    {'id': 2, 'coords': (100, 104, 169, 231)},
    {'id': 3, 'coords': (176, 104, 245, 231)},
    {'id': 4, 'coords': (252, 104, 320, 231)},
    {'id': 5, 'coords': (328, 104, 397, 231)},
    {'id': 6, 'coords': (404, 104, 473, 231)},
    {'id': 7, 'coords': (480, 104, 549, 231)},
    {'id': 8, 'coords': (556, 104, 624, 231)},
    {'id': 9, 'coords': (632, 104, 701, 231)},
    
    {'id': 10, 'coords': (23, 268, 93, 395)},
    {'id': 11, 'coords': (99, 268, 169, 395)},
    {'id': 12, 'coords': (175, 268, 245, 395)},
    {'id': 13, 'coords': (251, 268, 321, 395)},
    {'id': 14, 'coords': (328, 268, 398, 395)},
    {'id': 15, 'coords': (404, 268, 474, 395)},
    {'id': 16, 'coords': (480, 268, 550, 395)},
    {'id': 17, 'coords': (556, 268, 626, 395)},
    {'id': 18, 'coords': (635, 268, 702, 395)},
]


def is_car_in_spot(annotations, coords, threshold=0.4):
    x1, y1, x2, y2 = map(float, coords)
    spot_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if spot_area == 0:
        return False

    max_overlap_ratio = 0.0
    for box in annotations:
        xmin, ymin, xmax, ymax = map(float, box.xyxy[0])

        # Coordinates пересечения
        inter_x1 = max(x1, xmin)
        inter_y1 = max(y1, ymin)
        inter_x2 = min(x2, xmax)
        inter_y2 = min(y2, ymax)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        ratio = inter_area / spot_area
        if ratio > max_overlap_ratio:
            max_overlap_ratio = ratio

    return max_overlap_ratio >= threshold

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Нет файла'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Недопустимое имя файла'}), 400

    # Считываем байты и сохраняем на диск
    file_bytes = file.read()
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)

    # Декодируем изображение из байтов
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Не удалось декодировать изображение'}), 400

    start_time = time.time()

    results = model(img, conf=0.1)

    all_detections = []
    if results and len(results):
        result = results[0]
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            xmin, ymin, xmax, ymax = box.xyxy[0]
            all_detections.append({
                'class_id': class_id,
                'confidence': conf,
                'coords': (xmin, ymin, xmax, ymax)
            })
    else:
        print("Обнаружений не найдено.")

    # Создаем копию изображения для отрисовки
    img_with_drawings = img.copy()

    # Определение занятости зон
    free_spots = []
    if results and len(results):
        result = results[0]
        for spot in parking_spots:
            x1, y1, x2, y2 = spot['coords']
            occupied = is_car_in_spot(result.boxes, spot['coords'])
            color = (0, 255, 0) if not occupied else (0, 0, 255)
            status_text = 1 if not occupied else 0
            if status_text:
                free_spots.append({'id': spot['id']})

            cv2.rectangle(img_with_drawings, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_drawings, '' if status_text else 'Занято',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        for spot in parking_spots:
            x1, y1, x2, y2 = spot['coords']
            cv2.rectangle(img_with_drawings, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_drawings, 'Free', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Кодируем итоговое изображение в память и возвращаем как base64 data URL
    success, buffer = cv2.imencode('.jpg', img_with_drawings)
    if not success:
        return jsonify({'error': 'Не удалось закодировать изображение'}), 500

    img_bytes = buffer.tobytes()
    b64 = base64.b64encode(img_bytes).decode('ascii')
    data_url = 'data:image/jpeg;base64,' + b64

    result_path = os.path.join(RESULT_FOLDER, "result_" + filename)
    with open(result_path, 'wb') as f:
        f.write(img_bytes)

    processing_time = time.time() - start_time

    history_entry = History(
        timestamp=datetime.now(UTC),
        processing_time=processing_time,
        spots_found=len(free_spots),
        result_json=json.dumps(all_detections, default=str),
        filename=filename
    )
    try:
        db.session.add(history_entry)
        db.session.commit()
    except Exception as e:
        # Не ломаем API из-за ошибок записи в БД
        print("Ошибка записи истории в БД:", e)

    return jsonify({
        'result_image': data_url,
        'spots': len(free_spots)
    })	

@app.route('/history', methods=['GET'])
def get_history():
    entries = History.query.order_by(History.timestamp.desc()).limit(100).all()
    result = []
    for e in entries:
        try:
            result.append({
                'id': e.id,
                'timestamp': e.timestamp.isoformat(),
                'processing_time': e.processing_time,
                'spots_found': e.spots_found,
                'filename': e.filename,
                'result': json.loads(e.result_json) if e.result_json else []
            })
        except Exception:
            # на случай разного формата данных
            result.append({
                'id': e.id,
                'timestamp': e.timestamp.isoformat(),
                'processing_time': e.processing_time,
                'spots_found': e.spots_found,
                'filename': e.filename,
                'result': []
            })
    return jsonify(result)

@app.route('/export_excel', methods=['GET'])
def export_excel(): 
    try:
        entries = History.query.order_by(History.timestamp.desc()).all()

        wb = Workbook()
        ws_summary = wb.active
        ws_summary.title = "Summary"
        ws_summary.append(["ID","Timestamp","Processing Time (s)","Spots Found","Filename"])

        for e in entries:
            ws_summary.append([
                e.id,
                e.timestamp.isoformat() if e.timestamp else '',
                e.processing_time,
                e.spots_found,
                e.filename
            ])

        ws_det = wb.create_sheet(title="Detections")
        ws_det.append(["History ID","Detection Index","Class ID","Confidence","Coords"])

        for e in entries:
            dets = []
            if e.result_json:
                try:
                    dets = json.loads(e.result_json)
                except Exception:
                    dets = []
            for idx, d in enumerate(dets):
                coords = d.get('coords')
                coords_str = ''
                # Преобразуем coords в строку для записи в одну ячейку
                if isinstance(coords, (list, tuple)):
                    parsed = []
                    for c in coords:
                        if isinstance(c, (int, float)):
                            parsed.append(float(c))
                        elif isinstance(c, str):
                            # Попытка извлечь число из строки вида 'tensor(182.6383)'
                            m = re.findall(r'[-+]?\d*\.?\d+', c)
                            if m:
                                parsed.append(float(m[0]))
                    if parsed:
                        coords_str = ','.join(str(v) for v in parsed)
                    else:
                        coords_str = ','.join(str(c) for c in coords)
                else:
                    coords_str = str(coords)

                ws_det.append([
                    e.id,
                    idx,
                    d.get('class_id'),
                    d.get('confidence'),
                    coords_str
                ])

        # Отправка файла клиенту
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)

        now = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"export_history_all_{now}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as ex:
        # Безопасный вывод ошибки
        return jsonify({'error': str(ex)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
