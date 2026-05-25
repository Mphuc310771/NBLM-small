#include <iostream>
#include <string>

#ifndef MOCK_VISION_ENGINE
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#endif

extern "C" {
    const char* extract_text_from_image(const char* image_path) {
        static std::string result;
        
        #ifdef MOCK_VISION_ENGINE
        // Mock OCR result demonstrating correct binding
        result = "Báo cáo tiến độ: Hệ thống RAG Hub đang vận hành tối ưu. Cấu hình tự động khôi phục mạng hoạt động tốt.";
        #else
        try {
            if (!image_path) {
                result = "Lỗi: Đường dẫn ảnh rỗng.";
                return result.c_str();
            }
            
            // Actual C++ OpenCV image preprocessing
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            if (image.empty()) {
                result = "Lỗi: Không thể đọc tệp ảnh.";
                return result.c_str();
            }
            
            // Convert to grayscale and threshold to enhance text visibility for OCR
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            
            // Save preprocessed image to temporary file
            std::string temp_path = "/tmp/vision_preprocessed.png";
            cv::imwrite(temp_path, gray);
            
            // Initialize Tesseract API
            tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
            if (api->Init(NULL, "vie")) { // Try Vietnamese language data
                // Fallback to English if Vietnamese language package is missing
                if (api->Init(NULL, "eng")) {
                    result = "Lỗi: Không thể khởi tạo Tesseract OCR API.";
                    delete api;
                    return result.c_str();
                }
            }
            
            // Read image via Leptonica pix reader
            Pix* pix_img = pixRead(temp_path.c_str());
            api->SetImage(pix_img);
            
            // Perform high-speed OCR text extraction
            char* outText = api->GetUTF8Text();
            result = outText ? std::string(outText) : "Không nhận diện được ký tự nào.";
            
            // Cleanup memory allocations
            api->End();
            delete api;
            delete[] outText;
            pixDestroy(&pix_img);
            
        } catch (const std::exception& e) {
            result = "Lỗi C++ Exception: " + std::string(e.what());
        }
        #endif
        
        return result.c_str();
    }
}
