#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const String WINDOW_NAME = "TP1";

// greyScale check the image is grey scale or not, if not then convert it to grey scale
void greyScale(Mat &f) {
    // Check if the image is already grey scale and print the message
    if (f.channels() == 1) {
        std::cout << "Image is already grey scale" << std::endl;
    }
    // else convert the image to grey scale and print the message "Image was in color and has been converted to grey scale"
    else {
        cvtColor(f, f, COLOR_BGR2GRAY);
        std::cout
            << "Image was in color and has been converted to grey scale" << std::endl;
    }
    imshow(WINDOW_NAME, f);
}
// Computes the histogram of the image
std::vector<double> histogram(Mat image) {
    std::vector<double> histogram(256, 0.0);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            histogram[image.at<uchar>(i, j)] += 1.0;
        }
    }

    // Store the percentage of each intensity value
    for (int i = 0; i < 256; i++) {
        histogram[i] = histogram[i] / (image.rows * image.cols) * 100;
    }

    // Print the size of the image
    std::cout << "Size of the image: " << image.size() << std::endl;

    // Print each element of the histogram
    for (int i = 0; i < 256; i++) {
        std::cout << "histogram[" << i << "]=" << histogram[i] << std::endl;
    }

    return histogram;
}

// histogramCumulative computes the cumulative histogram of the image
std::vector<double> histogramCumulative(std::vector<double> &histogram) {
    std::vector<double> histogramCumulative(256, 0);
    histogramCumulative[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        histogramCumulative[i] = histogramCumulative[i - 1] + histogram[i];
    }
    // Print the size of the image
    std::cout << "Size of the image: " << histogramCumulative.size() << std::endl;

    // Print the last element of the histogram
    std::cout << "Last element of the histogram: " << histogramCumulative[255] << std::endl;

    // Print each element of the histogram
    for (int i = 0; i < 256; i++) {
        std::cout << "histogramCumulative[" << i << "]=" << histogramCumulative[i] << std::endl;
    }

    return histogramCumulative;
}

// imageHistogram computes the histogram of the image and the cumulative histogram of the image
cv::Mat imageHistogram(Mat &image) {
    std::vector<double> histogramV           = histogram(image);
    std::vector<double> histogramCumulativeV = histogramCumulative(histogramV);

    // Image size
    int size = image.size().height * image.size().width;
    std::cout << "Size of the image: " << size << std::endl;

    // Create the image
    Mat imageHistogram(256, 524, CV_8UC1, Scalar(255));

    // find max value of the histogram
    double max = *std::max_element(histogramV.begin(), histogramV.end());

    // Draw both the histogram and the cumulative histogram horizontally
    for (int i = 0; i < 256; i++) {
        line(imageHistogram, Point(i, 255), Point(i, 255 - histogramV[i] * 256 / max), Scalar(0));
        line(imageHistogram, Point(i + 255, 256), Point(i + 255, (256 - histogramCumulativeV[i] * 255 / 100)), Scalar(0));
    }

    return imageHistogram;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    int value            = 128;
    String trackbar_name = "Trackbar";
    String image_path    = argv[1];

    Mat initImage = imread(image_path);

    namedWindow(WINDOW_NAME);
    createTrackbar(trackbar_name, WINDOW_NAME, nullptr, 255, NULL);
    setTrackbarPos(trackbar_name, WINDOW_NAME, value);
    imshow(WINDOW_NAME, initImage);

    // Wait 50ms for a keystroke and get the key code
    int key = waitKeyEx(50);

    // All images variables
    Mat greyImage(initImage);
    Mat histogramImage;

    while (true) {
        // Print the key code
        std::cout << "Key code: " << key << std::endl;
        key = waitKeyEx(500);

        // If the key is ESC, break the loop
        if (key == 27) {
            break;
        }

        // If the key is g, call greyScale
        if (key == 103) {
            std::cout << "Grey scale" << std::endl;
            greyScale(greyImage);
        }

        // If the key is h, call histogram
        if (key == 104) {
            std::cout << "Histogram" << std::endl;
            histogramImage = imageHistogram(greyImage);
            imshow(WINDOW_NAME, histogramImage);
        }

        // If the key is r, reset the image to initial image
        if (key == 114) {
            std::cout << "Reset the image to initial image" << std::endl;
            imshow(WINDOW_NAME, initImage);
        }

        // int new_value = getTrackbarPos(trackbar_name, WINDOW_NAME);
        // if (value != new_value) {
        //     value = new_value;
        //     std::cout << "value=" << value << std::endl;
        // }
    }
}
