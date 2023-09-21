import cv2
import numpy as np

# Константы.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Параметры текста.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1

# цветовая гамма
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def draw_label(input_image, label, left, top):
	"""наложение текста на изображение."""

	# получает размер текста.
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	# Использует размер текста, что бы создать чёрный прямоугольник.
	cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
	# отображает текст внутри прямоугольника.
	cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Создаёт 4D обьект из кадра.
	blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

	# Устанавливает входные данные для сети.
	net.setInput(blob)

	# выполняет проход, что бы получить выходные данные со слоёв.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	# списки для хранения соответствующих значений при развёртовании.
	class_ids = []
	confidences = []
	boxes = []

	# строки.
	rows = outputs[0].shape[1]

	image_height, image_width = input_image.shape[:2]

	# коифецен изменения размеров.
	x_factor = image_width / INPUT_WIDTH
	y_factor = image_height / INPUT_HEIGHT

	# выполнение 25200 итерация по распознованию.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		# отбрасывает плохие распознования и продолжает процесс.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# получает макс. значение.
			class_id = np.argmax(classes_scores)

			#  получает максимальное значение в классе, превышающее пороговый уровень.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w / 2) * x_factor)
				top = int((cy - h / 2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)

				box = np.array([left, top, width, height])
				boxes.append(box)

	# выполняет сжатие максимального значения, что бы исключить избыточные блоки
	# меньшой достоверности.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)

	return input_image


if __name__ == '__main__':
	# загружает имя класса.
	classesFile = "coco.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# загружает обрабатываемый объект.
	# frame = cv2.imread("img.png")
	cap = cv2.VideoCapture("street.mp4")

	# передаёт вес модели и с их помощью заружает их в сеть.
	modelWeights = "models/yolov5s.onnx"
	net = cv2.dnn.readNet(modelWeights)

	while True:
		_, frame = cap.read()

		#frame = cv2.resize(frame, (700, 700))

		# .
		detections = pre_process(frame, net)
		img = post_process(frame.copy(), detections)

		# Помещает иформацию о эффективности. ункция getPerfProfile возвращает
		# общее время вывода (t) и время для каждого из обрабатываемых слоев (в LayerTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
		print(label)
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

		cv2.imshow('Output', img)

		if cv2.waitKey(1) == ord('q'):
			break



