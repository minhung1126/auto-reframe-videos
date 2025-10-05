"""
Object detection logic using AI models.
"""
import os
import requests
import config
import cv2
import numpy as np

def download_file(url, path):
    try:
        print(f"正在下載 {os.path.basename(path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for data in response.iter_content(1024):
                f.write(data)
        print(f"下載完成: {path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下載失敗: {e}")
        return False

def ensure_models_exist():
    """Checks if the selected model files exist, and downloads them if not."""
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)

    if config.MODEL_CHOICE == 'YOLO':
        if not os.path.exists(config.COCO_NAMES_PATH) and not download_file(config.COCO_NAMES_URL, config.COCO_NAMES_PATH): return False
        if not os.path.exists(config.YOLO_CFG_PATH) and not download_file(config.YOLO_CFG_URL, config.YOLO_CFG_PATH): return False
        if not os.path.exists(config.YOLO_WEIGHTS_PATH) and not download_file(config.YOLO_WEIGHTS_URL, config.YOLO_WEIGHTS_PATH): return False
    elif config.MODEL_CHOICE == 'SSD':
        if not os.path.exists(config.SSD_PROTOTXT_PATH) and not download_file(config.SSD_PROTOTXT_URL, config.SSD_PROTOTXT_PATH): return False
        if not os.path.exists(config.SSD_MODEL_PATH) and not download_file(config.SSD_MODEL_URL, config.SSD_MODEL_PATH): return False
    else:
        print(f"錯誤: 無效的模型選擇 '{config.MODEL_CHOICE}' in config.py")
        return False
    return True

def load_model():
    """Loads the selected model and returns the network and class names."""
    print(f"正在載入 {config.MODEL_CHOICE} 模型...")
    if config.MODEL_CHOICE == 'YOLO':
        with open(config.COCO_NAMES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG_PATH, config.YOLO_WEIGHTS_PATH)
    elif config.MODEL_CHOICE == 'SSD':
        classes = config.SSD_CLASSES
        net = cv2.dnn.readNetFromCaffe(config.SSD_PROTOTXT_PATH, config.SSD_MODEL_PATH)
    else:
        return None, None

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("偵測到 CUDA，啟用 GPU 加速。")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    return net, classes

def detect_people(frame, net, classes):
    """Detects people in a frame and returns the bounding box (x, y, w, h) of the largest person found."""
    (H, W) = frame.shape[:2]
    person_detections = []

    if config.MODEL_CHOICE == 'YOLO':
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
        boxes, confidences, classIDs = [], [], []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classes[classID] == "person" and confidence > config.CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                if x < 0: x = 0
                if y < 0: y = 0
                person_detections.append((x, y, x + w, y + h))

    elif config.MODEL_CHOICE == 'SSD':
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config.CONFIDENCE_THRESHOLD:
                idx = int(detections[0, 0, i, 1])
                if classes[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    person_detections.append((startX, startY, endX, endY))

    if len(person_detections) > 0:
        largest_person_box = max(person_detections, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
        (startX, startY, endX, endY) = largest_person_box
        return (startX, startY, endX - startX, endY - startY)
    
    return None