#include <opencv2/opencv.hpp>

// Function for pre-processing
void preprocessFrame(cv::Mat& frame) {
    cv::Mat grayscale_image;
    cv::Mat equalized_image;

    cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayscale_image, equalized_image);
    cv::GaussianBlur(equalized_image, frame, cv::Size(7, 7), 3.0, 3.0);
}

// Function for Canny edge detection
void detectEdges(const cv::Mat& frame, cv::Mat& edges) {
    cv::Canny(frame, edges, 30, 100, 3);
}

// Function for ORB feature extraction and matching
void extractAndMatchORBFeatures(const cv::Mat& currentFrame, const cv::Mat& previousFrame, std::vector<cv::DMatch>& matches,
                                std::vector<cv::KeyPoint>& keypointsCurrent, std::vector<cv::KeyPoint>& keypointsPrevious) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    int maxKeypoints = 20;  // Set the maximum number of keypoints
    int maxMatches = 10;    // Set the maximum number of matches

    // Detect keypoints using ORB in both frames in parallel
    cv::parallel_for_(cv::Range(0, 2), [&](const cv::Range& range) {
        if (range.start == 0) {
            orb->detect(currentFrame, keypointsCurrent);
            // Limit the number of keypoints
            if (keypointsCurrent.size() > maxKeypoints) {
                keypointsCurrent.resize(maxKeypoints);
            }
        } else {
            orb->detect(previousFrame, keypointsPrevious);
            // Limit the number of keypoints
            if (keypointsPrevious.size() > maxKeypoints) {
                keypointsPrevious.resize(maxKeypoints);
            }
        }
    });

    // Check if keypoints are valid
    if (keypointsCurrent.empty() || keypointsPrevious.empty()) {
        std::cerr << "Error: No keypoints detected." << std::endl;
        return;
    }

    // Compute ORB descriptors for the detected keypoints
    cv::Mat descriptorsCurrent, descriptorsPrevious;
    orb->compute(currentFrame, keypointsCurrent, descriptorsCurrent);
    orb->compute(previousFrame, keypointsPrevious, descriptorsPrevious);

    // Check if descriptors are valid
    if (descriptorsCurrent.empty() || descriptorsPrevious.empty()) {
        std::cerr << "Error: No descriptors computed." << std::endl;
        return;
    }

    // Perform feature matching using the descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsCurrent, descriptorsPrevious, matches);

    // Limit the number of matches
    if (matches.size() > maxMatches) {
        std::nth_element(matches.begin(), matches.begin() + maxMatches, matches.end());
        matches.erase(matches.begin() + maxMatches, matches.end());
    }

    // Print the number of keypoints matched
    std::cout << "Number of keypoints matched: " << matches.size() << std::endl;
}

// Function to estimate motion of the cmarea
void estimateMotion(const std::vector<cv::KeyPoint>& keypointsCurrent, const std::vector<cv::KeyPoint>& keypointsPrevious,
                    const std::vector<cv::DMatch>& matches, const cv::Mat& K, cv::Mat& R, cv::Mat& t) {
    // Extract matched points
    std::vector<cv::Point2f> pointsCurrent, pointsPrevious;
    for (const auto& match : matches) {
        pointsCurrent.push_back(keypointsCurrent[match.queryIdx].pt);
        pointsPrevious.push_back(keypointsPrevious[match.trainIdx].pt);
    }

    // Compute Essential matrix
    cv::Mat E, mask;
    E = cv::findEssentialMat(pointsCurrent, pointsPrevious, K, cv::RANSAC, 0.999, 1.0, mask);

    // Decompose Essental matrix to get R and t
    cv::recoverPose(E, pointsCurrent, pointsPrevious, K, R, t, mask);
}


int main() {
    cv::VideoCapture cap(4);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::Mat previousFrame;
    std::vector<cv::KeyPoint> keypointsCurrent, keypointsPrevious;  // Define keypoints vectors

    int frameCount = 0;
    int frameSkip = 2;  // Process every second frame

    // Camera intrinsic parameters (example values, should be calibrated for your camera)
    cv::Mat K = (cv::Mat_<double>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);  // Final rotation matrix
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);  // Final translation vector

// intrinsic parameters with a mean square error of less than 1 pixel.
//Re-projection error reported by calibrateCamera: 0.753519
//Camera Matrix: [1069.396120242646, 0, 924.5548796102744;
// 0, 1071.148706255674, 535.1140734775801;
// 0, 0, 1]
//Distortion Coefficients: [-0.005061470202018862, 0.2009284088312256, 0.001632189771557368, -0.000411624430024224, -1.001394428646603]



    while (true) {
        cv::Mat currentFrame;
        cap >> currentFrame;

        if (currentFrame.empty()) {
            std::cerr << "Error: Blank frame captured." << std::endl;
            break;
        }

        frameCount++;

        if (frameCount % frameSkip == 0) {
            if (previousFrame.empty()) {
                previousFrame = currentFrame.clone();
                continue;
            }

            // Preprocess the current frame
            preprocessFrame(currentFrame);

            // Detect edges
            cv::Mat edges;
            detectEdges(currentFrame, edges);

            // Extract and match ORB features between current and previous frames
            std::vector<cv::DMatch> matches;
            extractAndMatchORBFeatures(currentFrame, previousFrame, matches, keypointsCurrent, keypointsPrevious);

            if (!matches.empty()) {
                // Estimate motion of camera
                cv::Mat R, t;
                estimateMotion(keypointsCurrent, keypointsPrevious, matches, K, R, t);

                // Update the previous frame for the next iteration
                previousFrame = currentFrame.clone();

                // Accumulate the transformations
                t_f = t_f + R_f * t;
                R_f = R * R_f;

                // Display the translation and rotation
                std::cout << "Translation: " << t_f.t() << std::endl;
                std::cout << "Rotation: " << R_f << std::endl;
            }

            // Display the original frame, edge map, and ORB matches
            cv::imshow("Original Frame", currentFrame);
            cv::imshow("Edge Detection", edges);

            // Optionally, display the ORB matches as well
            cv::Mat frameWithMatches;
            cv::drawMatches(currentFrame, keypointsCurrent, previousFrame, keypointsPrevious, matches, frameWithMatches);
            cv::imshow("ORB Matches", frameWithMatches);
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}