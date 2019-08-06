# OpenVINO_19R1_Object_Detection

This demo showcases Mutiple Video Object Detection with SSD.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps two parallel infer requests and while the current is processed, the input frame for the next
is being captured. This essentially hides the latency of capturing, so that the overall framerate is rather
determined by the `MAXIMUM(detection time, input capturing time)` and not the `SUM(detection time, input capturing time)`.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection SSD Demo Async API. For the Python* implementation, refer to [Object Detection SSD Python* Demo, Async API Performance Showcase](./inference-engine/ie_bridges/python/sample/object_detection_demo_ssd_async/README.md).

The technique can be generalized to any available parallel slack, for example, doing inference and simultaneously encoding the resulting
(previous) frames or running further inference, like some emotion detection on top of the face detection results.
There are important performance
caveats though, for example the tasks that run in parallel should try to avoid oversubscribing the shared compute resources.
For example, if the inference is performed on the FPGA, and the CPU is essentially idle, than it makes sense to do things on the CPU
in parallel. But if the inference is performed say on the GPU, than it can take little gain to do the (resulting video) encoding
on the same GPU in parallel, because the device is already busy.

This and other performance implications and tips for the Async API are covered in the [Optimization Guide](https://software.intel.com/en-us/articles/OpenVINO-Inference-Engine-Optimization-Guide)

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

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

You can use the following command to do inference on FPGA with a pre-trained object detection model:
```sh
./object_detection_demo_ssd_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/ssd.xml -d HETERO:FPGA,CPU -stream 5
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode the demo reports
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and displaying the results.
* **Detection time**: inference time for the (object detection) network. It is reported in the "SYNC" mode only.
* **Wallclock time**, which is combined (application level) performance.


## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
