import ctypes
import random
import numpy as np
import cv2
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA)



def _preprocess_yolo(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)
        # Convert the image to row-major order, also known as "C order":
    return np.ascontiguousarray(img)


def _preprocess_mnet(img, input_shape):
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img)


def cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def _postprocess_yolo(trt_outputs, w_scale, h_scale, conf_th, nms_threshold=0.5):
    """Postprocess TensorRT outputs.
    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
    # Returns
        boxes, scores, classes (after NMS)
    """
    # concatenate outputs of all yolo layers

    num = int(trt_outputs[0])
    # Reshape to a two dimentional ndarray
    pred = np.reshape(trt_outputs[1:], (-1, 6))[:num, :]
    boxes = pred[:, :4]
    scores = pred[:, 4]
    classids = pred[:, 5]

    si = scores > conf_th
    boxes = boxes[si, :]
    scores = scores[si]
    classids = classids[si]

    if boxes.shape[0]:
        cx = boxes[:, 0].reshape(-1, 1)
        cy = boxes[:, 1].reshape(-1, 1)
        w = boxes[:, 2].reshape(-1, 1) / 2
        h = boxes[:, 3].reshape(-1, 1) / 2
        boxes = np.concatenate([cx - w, cy - h, cx + w, cy + h], axis=1)

    if boxes.shape[0]:
        detections = np.column_stack([boxes, scores.reshape(-1, 1)])        
        keep = cpu_nms(detections, nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classids = classids[keep]
        if boxes.shape[0]:
            # 还原到原图
            print("bboxes:", boxes)
            boxes[:, [0, 2]] *= w_scale
            boxes[:, [1, 3]] *= h_scale
            boxes = boxes.astype(np.int)
    return boxes, scores, classids


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)
    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# class TrtMnetv2(object):
#     def _load_engine(self, engine_path):
#         with open(engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())

#     def __init__(self, engine_path, input_shape, cuda_ctx=None):
#         """Initialize TensorRT plugins, engine and conetxt."""
#         self.input_shape = input_shape
#         self.cuda_ctx = cuda_ctx
#         if self.cuda_ctx:
#             self.cuda_ctx.push()

#         self.trt_logger = trt.Logger(trt.Logger.INFO)
#         self.engine = self._load_engine(engine_path)

#         try:
#             self.context = self.engine.create_execution_context()
#             self.inputs, self.outputs, self.bindings, self.stream = \
#                 allocate_buffers(self.engine)
#         except Exception as e:
#             raise RuntimeError('fail to allocate CUDA resources') from e
#         finally:
#             if self.cuda_ctx:
#                 self.cuda_ctx.pop()

#     def __del__(self):
#         """Free CUDA memories."""
#         del self.outputs
#         del self.inputs
#         del self.stream

#     def classify(self, img):
#         """Detect objects in the input image."""
#         img_resized = _preprocess_mnet(img, self.input_shape)

#         np.copyto(self.inputs[0].host, img_resized.ravel())
#         if self.cuda_ctx:
#             self.cuda_ctx.push()
#         trt_outputs = do_inference(
#             context=self.context,
#             bindings=self.bindings,
#             inputs=self.inputs,
#             outputs=self.outputs,
#             stream=self.stream)
#         if self.cuda_ctx:
#             self.cuda_ctx.pop()

#         return trt_outputs[0]


class TrtYOLO(object):
    def __init__(self, engine_path, input_shape):
        self.input_shape = input_shape
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def detect(self, img, conf_thr=0.4, nms_thr=0.4):
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image = _preprocess_yolo(img, self.input_shape)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()

        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        raw_h, raw_w = img.shape[:2]
        print("raw shape:", img.shape)
        w_scale = raw_w / self.input_shape[1]
        h_scale = raw_h / self.input_shape[0]
        boxes, scores, classes = _postprocess_yolo(
            output, w_scale, h_scale, conf_thr, nms_thr)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)
        
        for i in range(len(boxes)):
            box = boxes[i]
            plot_one_box(
                box,
                img,
                label="{}:{:.2f}".format(classes[i], scores[i]))
        cv2.imwrite('test.jpg', img)
        return boxes, scores, classes



if __name__ == '__main__':
    yolo_plugin_path = './libmyplugins.so'
    ctypes.cdll.LoadLibrary(yolo_plugin_path)

    yolo_engine_path = './yolov5s.engine'
    mnet_engine_path = './mobilenet.engine'
    yolo_shape = (608, 608)
    mnet_shape = (224, 224)

    yolo_trt = TrtYOLO(yolo_engine_path, yolo_shape)
    # mnet_trt = TrtMnetv2(mnet_engine_path, mnet_shape)

    img_path = 'zidane.jpg'
    image = cv2.imread(img_path)
    det_out = yolo_trt.detect(image)
    print("det res:", det_out)
    # cls_out = mnet_trt.classify(image)
    # print("cls res:", cls_out.shape)