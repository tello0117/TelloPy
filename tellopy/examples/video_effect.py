import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
from ssd import SSD300
from ssd_utils import BBoxUtility
from PIL import Image
import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)
def getSSDImage(frame):
img2 = image.img_to_array(frame.resize((300, 300)))
img = np.asarray(frame)
inputs = []
inputs.append(img2.copy())
inputs = preprocess_input(np.array(inputs))
preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)
# Parse the outputs.
det_label = results[0][:, 0]
det_conf = results[0][:, 1]
det_xmin = results[0][:, 2]
det_ymin = results[0][:, 3]
det_xmax = results[0][:, 4]
det_ymax = results[0][:, 5]
# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
for i in range(top_conf.shape[0]):
xmin = int(round(top_xmin[i] * img.shape[1]))
ymin = int(round(top_ymin[i] * img.shape[0]))
xmax = int(round(top_xmax[i] * img.shape[1]))
ymax = int(round(top_ymax[i] * img.shape[0]))
score = top_conf[i]
label = int(top_label_indices[i])
label_name = voc_classes[label - 1]
display_txt = '{:0.2f}, {}'.format(score, label_name)
color = colors[label]
cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (int(colors[label][0]*255), int(colors[label][1]*255), int(colors[label][2]*255)), 2)
cv2.rectangle(img, (xmin, ymin-15), (xmin+100, ymin+5), (int(colors[label][0]*255), int(colors[label][1]*255), int(colors[label][2]*255)),-1)
cv2.putText(img, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
imgcv = img[:, :, ::-1].copy()
return imgcv
def main():
drone = tellopy.Tello()
try:
drone.connect()
drone.wait_for_connection(60.0)
drone.set_loglevel(drone.LOG_INFO)
drone.set_exposure(0)
container = av.open(drone.get_video_stream())
frame_count = 0
while True:
for frame in container.decode(video=0):
frame_count = frame_count + 1
if (frame_count > 300) and (frame_count%50 == 0):
imgpil = frame.to_image()
image = getSSDImage(imgpil)
cv2.imshow('Original', image)
cv2.waitKey(1)
except Exception as ex:
exc_type, exc_value, exc_traceback = sys.exc_info()
traceback.print_exception(exc_type, exc_value, exc_traceback)
print(ex)
finally:
drone.quit()
cv2.destroyAllWindows()
if __name__ == '__main__':
main()
