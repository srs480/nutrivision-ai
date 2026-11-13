import { ModelProfile } from '../types/detection';

export const COCO_80_CLASSES: Record<number, string> = {
  0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 4: 'Airplane', 5: 'Bus',
  6: 'Train', 7: 'Truck', 8: 'Boat', 9: 'Traffic Light', 10: 'Fire Hydrant',
  11: 'Stop Sign', 12: 'Parking Meter', 13: 'Bench', 14: 'Bird', 15: 'Cat',
  16: 'Dog', 17: 'Horse', 18: 'Sheep', 19: 'Cow', 20: 'Elephant', 21: 'Bear',
  22: 'Zebra', 23: 'Giraffe', 24: 'Backpack', 25: 'Umbrella', 26: 'Handbag',
  27: 'Tie', 28: 'Suitcase', 29: 'Frisbee', 30: 'Skis', 31: 'Snowboard',
  32: 'Sports Ball', 33: 'Kite', 34: 'Baseball Bat', 35: 'Baseball Glove',
  36: 'Skateboard', 37: 'Surfboard', 38: 'Tennis Racket',
  // Dining & Tableware Context (39 - 45)
  39: 'Bottle / Drink', 40: 'Glass / Beverage', 41: 'Cup / Tea / Coffee', 42: 'Fork', 43: 'Knife', 44: 'Spoon', 45: 'Bowl (Rice / Curry / Dal / Soup)',
  // Core Food Items (46 - 55)
  46: 'Banana', 47: 'Apple', 48: 'Sandwich / Burger', 49: 'Orange / Citrus', 50: 'Broccoli / Green Vegetables',
  51: 'Carrot / Mixed Veg', 52: 'Hot Dog / Sausage / Kebab', 53: 'Pizza / Flatbread', 54: 'Donut / Fried Snack', 55: 'Cake / Dessert / Sweet',
  // Remaining Indoor/Household
  56: 'Chair', 57: 'Couch', 58: 'Potted Plant', 59: 'Bed', 60: 'Dining Table',
  61: 'Toilet', 62: 'TV', 63: 'Laptop', 64: 'Mouse', 65: 'Remote', 66: 'Keyboard',
  67: 'Cell Phone', 68: 'Microwave', 69: 'Oven', 70: 'Toaster', 71: 'Sink',
  72: 'Refrigerator', 73: 'Book', 74: 'Clock', 75: 'Vase', 76: 'Scissors',
  77: 'Teddy Bear', 78: 'Hair Drier', 79: 'Toothbrush'
};

export const CUSTOM_FOOD_20_CLASSES: Record<number, string> = {
  0: 'Rice',
  1: 'Fried Chicken',
  2: 'Chicken Curry',
  3: 'Beef',
  4: 'Fish',
  5: 'Egg',
  6: 'Dal (Lentil Soup)',
  7: 'Mixed Vegetables',
  8: 'Bread / Naan',
  9: 'Pizza',
  10: 'Burger',
  11: 'Pasta',
  12: 'Salad',
  13: 'Fruits',
  14: 'Soup',
  15: 'Biryani',
  16: 'Samosa / Snack',
  17: 'French Fries / Potato',
  18: 'Cake / Dessert',
  19: 'Beverage / Drink'
};

// Direct CORS-friendly ONNX static binary mirrors (No API keys, raw static .onnx file)
const PUBLIC_YOLOV8N_ONNX_MIRRORS = [
  'https://huggingface.co/SpotLab/YOLOv8Detection/resolve/main/yolov8n.onnx?download=true',
  'https://cdn.jsdelivr.net/gh/Hyuto/yolov8-onnxruntime-web@master/public/model/yolov8n.onnx'
];

export const MODEL_PROFILES: ModelProfile[] = [
  {
    id: 'yolov8n-coco',
    name: 'YOLOv8n Pretrained (Food + Dining Bowls)',
    description:
      'Works out-of-the-box. Detects 10 core food classes (Pizza, Sandwich/Burger, Vegetables/Broccoli, Carrot, Banana, Apple, Orange, Fried Snack/Donut, Cake) + 7 dining containers (Bowls of Rice/Dal/Curry, Cups, Bottles) via locally executed ONNX.',
    localPath: '/models/yolov8n.onnx',
    fallbackUrls: PUBLIC_YOLOV8N_ONNX_MIRRORS,
    inputSize: 640,
    classes: COCO_80_CLASSES,
    coreFoodClassIds: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    diningContextClassIds: [39, 40, 41, 42, 43, 44, 45]
  },
  {
    id: 'yolov8n-food20',
    name: 'YOLOv8n Custom South Asian & Global Food-20',
    description:
      'Uses fine-tuned 20-class weights if placed at /models/yolov8n-food20.onnx (or loaded via Load .ONNX button), and automatically falls back to base YOLOv8n ONNX weights if custom weights are not yet uploaded.',
    localPath: '/models/yolov8n-food20.onnx',
    fallbackUrls: PUBLIC_YOLOV8N_ONNX_MIRRORS,
    inputSize: 640,
    classes: COCO_80_CLASSES,
    coreFoodClassIds: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    diningContextClassIds: [39, 40, 41, 42, 43, 44, 45]
  }
];

export const MAX_IMAGE_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB
export const ALLOWED_IMAGE_MIME_TYPES = [
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp'
];
