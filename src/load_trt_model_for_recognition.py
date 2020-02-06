output_names = ['dense_4/Softmax']
input_names = ['dense_1_input']


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('/home/nitish/Nitish/work/final_face_racognition_11-1-19/output/tensort/trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')
operations = tf_sess.graph.get_operations()
for operation in operations:
	print(operation.name)
exit()
# Get graph input size
# for node in trt_graph.node:
#     print("inside",node)
#     if 'input_' in node.name:
#         size = node.attr['shape'].shape
#         image_size = [size.dim[i].size for i in range(1, 4)]
#         break
# print("image_size: {}".format(image_size))
# image_size=640,480
def get_prediction(x):
    x=x.reshape(1,128)
    # print(x)
    # exit()

# input and output tensor names.
    input_tensor_name = input_names[0] + ":0"
    output_tensor_name = output_names[0] + ":0"

    print("input_tensor_name: {}\noutput_tensor_name: {}".format(
        input_tensor_name, output_tensor_name))

    output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

# img_path = '/home/nitish/Nitish/work/final_face_racognition_11-1-19/output/data/face_detected/hiren/hiren2.jpg'

# img = image.load_img(img_path, target_size=image_size[:2])
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
    print(x)
    exit()
    feed_dict = {
        input_tensor_name: x
    }
    print(feed_dict)
    exit()
    preds = tf_sess.run(output_tensor, feed_dict)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    return decode_predictions(preds, top=3)[0]
# import time
# times = []
# for i in range(20):
#     start_time = time.time()
#     one_prediction = tf_sess.run(output_tensor, feed_dict)
#     delta = (time.time() - start_time)
#     times.append(delta)
# mean_delta = np.array(times).mean()
# fps = 1 / mean_delta
# print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))