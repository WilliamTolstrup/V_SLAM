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

            // Update the previous frame for the next iteration
            previousFrame = currentFrame.clone();

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