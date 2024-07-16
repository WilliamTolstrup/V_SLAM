#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Define the dimensions of the checkerboard
    int CHECKERBOARD[2]{5, 9}; // Number of internal corners in the checkerboard

    // Square size in your checkerboard (in meters)
    float square_size = 0.015; // 15 mm

    // Create vectors to store 3D points and 2D points for each image
    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Define the world coordinates for the 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++) {
        for (int j = 0; j < CHECKERBOARD[0]; j++) {
            objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
        }
    }

    // Start capturing images
    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_pts;
    bool success;

    cv::VideoCapture cap(4); // Change the parameter to the correct camera index if necessary

    if (!cap.isOpened()) {
        std::cerr << "Unable to open the webcam" << std::endl;
        return -1;
    }

    int image_count = 0;
    while (image_count < 30) { // Capture 30 images
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture an image" << std::endl;
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Find the checkerboard corners
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
                                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (success) {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Display the corners
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
        }

        cv::imshow("Webcam", frame);
        int key = cv::waitKey(1);
        if (key == 'c') { // Press 'c' to capture an image
            if (success) {
                imgpoints.push_back(corner_pts);
                objpoints.push_back(objp);
                image_count++;
                std::cout << "Captured image " << image_count << " of 30" << std::endl;
            } else {
                std::cerr << "Checkerboard not detected. Try again." << std::endl;
            }
        }
        if (key == 27) { // Press ESC to break
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    if (imgpoints.size() < 10) {
        std::cerr << "Not enough valid images captured for calibration. Try again with more images." << std::endl;
        return -1;
    }

    // Calibrate the camera
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objpoints, imgpoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    // Output the calibration results
    std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;
    std::cout << "Camera Matrix: " << cameraMatrix << std::endl;
    std::cout << "Distortion Coefficients: " << distCoeffs << std::endl;

    return 0;
}
