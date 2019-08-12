# Object_Detection implement with OpenVINO 2019 R1

This demo showcases Mutiple Video Object Detection with SSD.
This demo haven't implement NIC part, the data is on the local side.

![image](https://github.com/aazz44ss/OpenVINO_19R1_Object_Detection/blob/master/pic/System.png)

Other demo objectives are:
* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and text labels (from the .labels file) or class number (if no file is provided)
* OpenCV is used to draw resulting bounding boxes, labels, so you can copy paste this code without
need to pull Inference Engine demos helpers to your app


## Running

Running the application with the `-h` option yields the following usage message:
```sh
./object_detection_demo_ssd_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo_ssd_async [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a video file (specify "cam" to work with camera).
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (CPU, GPU, FPGA, HDDL or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -pc                       Optional. Enables per-layer performance report.
    -r                        Optional. Inference results as raw values.
    -t                        Optional. Probability threshold for detections.
    -auto_resize              Optional. Enables resizable input with support of ROI crop & auto resize.
    -stream                   Number of streams.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool]

You can use the following command to do inference on FPGA with a pre-trained object detection model:
```sh
./object_detection_demo_ssd_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/ssd.xml -d HETERO:FPGA,CPU -stream 5
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode the demo reports

![image](https://github.com/aazz44ss/OpenVINO_19R1_Object_Detection/blob/master/pic/demo.png)


## See Also
* [Using Inference Engine Samples]
* [Model Optimizer]
* [Model Downloader]
