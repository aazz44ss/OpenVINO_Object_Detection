// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <thread> 
#include <mutex>
#include <deque>
#include <condition_variable> 
#include <math.h> 

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "object_detection_demo_ssd_async.hpp"
#include <ext_list.hpp>

using namespace InferenceEngine;

bool is_stop = false;

struct stream_info {
	double ocv_decode_time;
	double frameToBlob_time;
	int num_of_stream;
	int num_of_frame;
	cv::Mat frame;
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
       showUsage();
       return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat& frame,
                 InferRequest& inferRequest,
                 const std::string& inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest.SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest.GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

struct QueueBuffer {
    std::deque<stream_info> deqInfo;
	std::deque<InferRequest> deq;
    size_t capacity;
	ExecutableNetwork network;
	const std::string inputName;
    std::mutex lock;
    std::condition_variable not_full;
    std::condition_variable not_empty;
    QueueBuffer(size_t capacity, ExecutableNetwork& network, const std::string& inputName) : capacity(capacity), network(network), inputName(inputName){}
    void deposit(int num_of_stream, cv::Mat& curr_frame, int num_of_frame, double ocv_decode_time){
		cv::Mat frame;
		curr_frame.copyTo(frame);
        std::unique_lock<std::mutex> lk(lock);
        while(deq.size() == capacity){
            not_full.wait(lk);
			if( is_stop ) 
				return;
        }
		typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
		InferRequest async_infer_request_curr = network.CreateInferRequest();
		auto t0 = std::chrono::high_resolution_clock::now();
		frameToBlob(curr_frame, async_infer_request_curr, inputName);
		auto t1 = std::chrono::high_resolution_clock::now();
		double frameToBlob_time = std::chrono::duration_cast<ms>(t1 - t0).count();
		
		stream_info info = {ocv_decode_time,frameToBlob_time,num_of_stream,num_of_frame,frame};
        deqInfo.push_back(info);
		deq.push_back(async_infer_request_curr);
		//slog::info << num_of_stream << " deposit, buffer size: " << deq.size() << slog::endl;
        lk.unlock();
        not_empty.notify_one();
    }
    stream_info fetch(InferRequest& async_infer_request_curr){
		
        std::unique_lock<std::mutex> lk(lock);
		stream_info info;
        while(deq.size() == 0){
            not_empty.wait(lk);
			if( is_stop ) 
				return info;
        }
		
		//slog::info << deqStr.front() << " fetch, buffer size: " << deq.size() << slog::endl;
		
		info = deqInfo.front();
		deqInfo.pop_front();
		
		async_infer_request_curr = deq.front();
        deq.pop_front();
		
        lk.unlock();
        not_full.notify_one();
        return info;
    }
	void stop_call() {
		not_full.notify_all();
		not_empty.notify_all();
	}
};

void producer(cv::VideoCapture& cap, int num_of_stream, QueueBuffer& buffer){
	int  num_of_frame = 0;
	cv::Mat curr_frame;
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    while(1) {
		auto t0 = std::chrono::high_resolution_clock::now();
		cap.read(curr_frame);
		auto t1 = std::chrono::high_resolution_clock::now();
		double ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
		
		if (curr_frame.empty()) {
			slog::info << "end of frame" << slog::endl;
			break;
		};
        buffer.deposit(num_of_stream,curr_frame,num_of_frame,ocv_decode_time);
		num_of_frame++;
		if ( is_stop ) {
			slog::info << "thread: " << num_of_stream << " is stopped" << slog::endl;
			break;
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
}

int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        std::vector<cv::VideoCapture> cap;
		std::vector<size_t>  width;
		std::vector<size_t>  height;
		cap.resize(FLAGS_stream);
		width.resize(FLAGS_stream);
		height.resize(FLAGS_stream);
		
		
        if (!((FLAGS_i == "cam") ? cap[0].open(0) : cap[0].open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        width[0]  = (size_t) cap[0].get(cv::CAP_PROP_FRAME_WIDTH);
        height[0] = (size_t) cap[0].get(cv::CAP_PROP_FRAME_HEIGHT);
		
		for(size_t i = 1; i < FLAGS_stream ; i++){
			switch(i%4){
				case 0: 
					cap[i].open("data/cars_1920x1080.h264");
					break;
				case 1: 
					cap[i].open("data/cars_768x768.mp4");
					break;
				case 2: 
					cap[i].open("data/4.mp4");
					break;
				case 3: 
					cap[i].open("data/3_720p.mp4");
					break;
				default:
					cap[i].open("data/4.mp4");
			}
			width[i]  = (size_t) cap[i].get(cv::CAP_PROP_FRAME_WIDTH);
			height[i] = (size_t) cap[i].get(cv::CAP_PROP_FRAME_HEIGHT);
		}
        
		
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher().getPluginByDevice(FLAGS_d);
        printPluginVersion(plugin, std::cout);

        /** Load extensions for the plugin **/

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            plugin.AddExtension(extension_ptr);
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one output");
        }
        DataPtr& output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;
        const int num_classes = netReader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
        if (static_cast<int>(labels.size()) != num_classes) {
            if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const SizeVector outputDims = output->getTensorDesc().getDims();
        const int objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        output->setPrecision(Precision::FP32);
        output->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest async_infer_request_curr = network.CreateInferRequest();
		InferRequest async_infer_request_next = network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Do inference ---------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0, frameToBlob_time = 0;
		int num_of_stream = 0;
        std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
		QueueBuffer buffer(16,network,inputName);
		std::vector<std::thread> stream;
		for(size_t i=0; i<FLAGS_stream; i++){
			stream.push_back(std::thread(producer, std::ref(cap[i]), i, std::ref(buffer)));
		}
		// read input (video) frame
		cv::Mat curr_frame;
		cv::Mat show_image;
		if(FLAGS_stream==1){
			show_image.create(cv::Size(400,400), CV_8UC3);
		}else if(FLAGS_stream<16){
			show_image.create(cv::Size(std::ceil(FLAGS_stream/2)*201+401,401), CV_8UC3);
		}else{
			show_image.create(cv::Size(1808,std::ceil((FLAGS_stream-15)/9.0)*201+401), CV_8UC3);
		}
		int location_id_offset = 0;
        while (true) {
			
			auto t0 = std::chrono::high_resolution_clock::now();
            auto t1 = std::chrono::high_resolution_clock::now();
			stream_info info = buffer.fetch(async_infer_request_next);
			
            t0 = std::chrono::high_resolution_clock::now();
            async_infer_request_next.StartAsync();

			if(OK == async_infer_request_curr.Wait(IInferRequest::WaitMode::RESULT_READY)) {
				
				t1 = std::chrono::high_resolution_clock::now();
				ms detection = std::chrono::duration_cast<ms>(t1 - t0);
				
				t0 = std::chrono::high_resolution_clock::now();
				ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
				wallclock = t0;
				
				std::ostringstream out;
				out << "Decode time: " << std::fixed << std::setprecision(2)
					<< (ocv_decode_time) << " ms";
				cv::putText(curr_frame, out.str(), cv::Point2f(0, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0),2);
				out.str("");
				
				out << "frameToBlob time: " << std::fixed << std::setprecision(2)
					<< (frameToBlob_time) << " ms";
				cv::putText(curr_frame, out.str(), cv::Point2f(0, 60), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0),2);
				out.str("");
				
				out << "Render time: " << std::fixed << std::setprecision(2)
					<< (ocv_render_time) << " ms";
				cv::putText(curr_frame, out.str(), cv::Point2f(0, 90), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0),2);
				out.str("");
				
				out << "Detection time  : " << std::fixed << std::setprecision(2) 
					<< detection.count() << " ms (" << 1000.f / detection.count() << " fps)";
				cv::putText(curr_frame, out.str(), cv::Point2f(0, 120), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 0, 0),2);
				out.str("");
				
				out << "Wallclock time " << std::fixed << std::setprecision(2) 
					<< wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
				cv::putText(curr_frame, out.str(), cv::Point2f(0, 150), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255),2);
				out.str("");
				
				// ---------------------------Process output blobs--------------------------------------------------
				// Processing results of the CURRENT request
				t0 = std::chrono::high_resolution_clock::now();
				const float *detections = async_infer_request_curr.GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
				
				// dims(), Returns the tensor dimensions vector with reversed order.
				size_t objectSize = async_infer_request_curr.GetBlob(outputName)->dims()[0];
				size_t maxProposalCount = async_infer_request_curr.GetBlob(outputName)->dims()[1];
				
				for (size_t i = 0; i < maxProposalCount; i++) {
					float image_id = detections[i * objectSize + 0];
					if (image_id < 0) {
						//std::cout << "Only " << i << " proposals found" << std::endl;
						break;
					}

					float confidence = detections[i * objectSize + 2];
					auto label = static_cast<int>(detections[i * objectSize + 1]);
					float xmin = detections[i * objectSize + 3] * width[num_of_stream];
					float ymin = detections[i * objectSize + 4] * height[num_of_stream];
					float xmax = detections[i * objectSize + 5] * width[num_of_stream];
					float ymax = detections[i * objectSize + 6] * height[num_of_stream];

					if (FLAGS_r) {
						std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
								  "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
								  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
					}
					
					if (confidence > FLAGS_t) {
						/** Drawing only objects when > confidence_threshold probability **/
						std::ostringstream conf;
						conf << ":" << std::fixed << std::setprecision(3) << confidence;
						cv::putText(curr_frame,
									(static_cast<size_t>(label) < labels.size() ?
									labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
									cv::Point2f(xmin, ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
									cv::Scalar(0, 0, 255));
						cv::rectangle(curr_frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 0, 255),2);
					}
				}
				
				cv::Mat resized_image(curr_frame);
				int location_id = ( num_of_stream + location_id_offset ) % FLAGS_stream;
				if(location_id==0){
					cv::resize(curr_frame, resized_image, cv::Size(400/*width*/, 400/*height*/));
					resized_image.copyTo(show_image(cv::Rect(  0, 0, 400, 400)));
				} else {
					cv::resize(curr_frame, resized_image, cv::Size(200/*width*/, 200/*height*/));
					if(location_id<15){
						resized_image.copyTo(show_image(cv::Rect( ((location_id-1)/2)*201+402, ((location_id-1)%2)*201, 200, 200)));
					}else{
						resized_image.copyTo(show_image(cv::Rect( ((location_id-15)%9)*201, ((location_id-15)/9)*201+402, 200, 200)));
					}
				}
			}
            cv::imshow("Detection results", show_image);
			
            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
			
			ocv_decode_time = info.ocv_decode_time;
			frameToBlob_time = info.frameToBlob_time;
			num_of_stream = info.num_of_stream;
			curr_frame = info.frame;
			async_infer_request_curr = async_infer_request_next;
			
			const int key = cv::waitKey(1);
            if ( key == 27 ) {  // Esc
				buffer.stop_call();
				is_stop = true;
                break;
			}
			if ( key == 9 ) { // Tab
				location_id_offset++;
			}
        }
		for(size_t i=0; i<FLAGS_stream; i++){
			stream[i].join();
		}
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;
		
        /** Show performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(async_infer_request_curr, std::cout);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}