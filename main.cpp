#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const String WINDOW_NAME      = "TP1";
const String WINDOW_HISTOGRAM = "Histogram";

// greyScale check the image is grey scale or not, if not then convert it to grey scale
cv::Mat greyScale(Mat f) {
    Mat res;
    // Check if the image is already grey scale and print the message
    if (f.channels() == 1) {
        std::cout << "Image is already grey scale" << std::endl;
        return f;
    }
    // else convert the image to grey scale and print the message "Image was in color and has been converted to grey scale"
    else {
        cvtColor(f, res, COLOR_BGR2GRAY);
        std::cout << "Image was in color and has been converted to grey scale" << std::endl;
        return res;
    }
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
cv::Mat imageHistogram(Mat image) {
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

// colorHistogram convert the image to HSV color space, split the V channel and compute the histogram of the V channel
cv::Mat colorHistogram(Mat image) {
    Mat imageHSV;
    cvtColor(image, imageHSV, COLOR_BGR2HSV);

    // Split the image in 3 channels
    std::vector<Mat> channels;
    split(imageHSV, channels);

    // Compute the histogram of the V channel
    Mat imageHistogram(256, 524, CV_8UC1, Scalar(255));
    std::vector<double> histogramV = histogram(channels[2]);
    double max                     = *std::max_element(histogramV.begin(), histogramV.end());

    // Compute the cumulative histogram of the V channel
    std::vector<double> histogramCumulativeV = histogramCumulative(histogramV);

    // Draw the histogram of the V channel and the cumulative histogram of the V channel
    for (int i = 0; i < 256; i++) {
        line(imageHistogram, Point(i, 255), Point(i, 255 - histogramV[i] * 256 / max), Scalar(0));
        line(imageHistogram, Point(i + 255, 256), Point(i + 255, (256 - histogramCumulativeV[i] * 255 / 100)), Scalar(0));
    }

    return imageHistogram;
}
// equalizeHistogram equalize the histogram of the image and return the image
cv::Mat equalizeHistogram(Mat image) {
    Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);
    return equalizedImage;
}

// equalizeColorHistogram equalize the histogram of the V channel and return the image
cv::Mat equalizeColorHistogram(Mat image) {
    Mat imageHSV;
    cvtColor(image, imageHSV, COLOR_BGR2HSV);

    // Split the image in 3 channels
    std::vector<Mat> channels;
    split(imageHSV, channels);

    // Equalize the histogram of the V channel
    equalizeHist(channels[2], channels[2]);

    // Merge the channels
    merge(channels, imageHSV);

    // Convert the image back to BGR color space
    Mat equalizedImage;
    cvtColor(imageHSV, equalizedImage, COLOR_HSV2BGR);

    return equalizedImage;
}

// floydSteinbergDitheringGrey dither the grey image using the Floyd-Steinberg algorithm with float values
cv::Mat floydSteinbergDitheringGrey(Mat image) {
    Mat imageDithered;
    image.convertTo(imageDithered, CV_32FC1);

    float oldPixelValue;
    float newPixelValue;
    float errorValue;

    // For each pixel
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // Print (x,y)
            oldPixelValue = imageDithered.at<float>(y, x);
            if (oldPixelValue > 127.0) {
                newPixelValue = 255.0;
            } else {
                newPixelValue = 0.0;
            }
            errorValue = oldPixelValue - newPixelValue;

            imageDithered.at<float>(y, x) = newPixelValue;
            if (x != image.cols - 1) {
                imageDithered.at<float>(y, x + 1) += errorValue * 7.0 / 16.0;
                if (y != image.rows - 1) {
                    if (x != 0) {
                        imageDithered.at<float>(y + 1, x - 1) += errorValue * 5.0 / 16.0;
                    }
                    imageDithered.at<float>(y + 1, x + 1) += errorValue / 16.0;
                    imageDithered.at<float>(y + 1, x) += errorValue * 5.0 / 16.0;
                }
            } else {
                if (y != image.rows - 1) {
                    imageDithered.at<float>(y + 1, x) += errorValue * 5.0 / 16.0;
                    if (x != 0) {
                        imageDithered.at<float>(y + 1, x - 1) += errorValue * 3.0 / 16.0;
                    }
                }
            }
        }
    }

    // Print finished
    Mat imageDithered8UC1;

    imageDithered.convertTo(imageDithered8UC1, CV_8UC1);
    return imageDithered8UC1;
}

// halfToningGreyScale convert the image to grey scale and apply the floyd-steinberg dithering algorithm
cv::Mat halfToningGreyScale(Mat image) {
    Mat imageGreyScale(image);

    // Apply the floyd-steinberg dithering algorithm
    for (int i = 0; i < imageGreyScale.rows; i++) {
        for (int j = 0; j < imageGreyScale.cols; j++) {
            imageGreyScale.at<uchar>(i, j) = imageGreyScale.at<uchar>(i, j) / 16 * 16;
        }
    }

    return imageGreyScale;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    int value           = 128;
    String trackbarName = "Trackbar";
    String imagePath    = argv[1];

    Mat initImage = imread(imagePath);

    namedWindow(WINDOW_NAME);
    namedWindow(WINDOW_HISTOGRAM);
    createTrackbar(trackbarName, WINDOW_NAME, nullptr, 255, NULL);
    setTrackbarPos(trackbarName, WINDOW_NAME, value);
    imshow(WINDOW_NAME, initImage);

    // Wait 50ms for a keystroke and get the key code
    int key = waitKeyEx(50);

    // All images variables
    Mat workingImage(initImage);
    Mat histogramImage;

    // Camera
    cv::VideoCapture camera(0);

    while (true) {
        // Print the key code
        key = waitKeyEx(500);

        // If the key is ESC, break the loop
        if (key == 27) {
            break;
        }

        // If the key is g, call greyScale
        if (key == 103) {
            std::cout << "Grey scale" << std::endl;
            workingImage = greyScale(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is h, call histogram
        if (key == 104) {
            std::cout << "Histogram" << std::endl;
            histogramImage = imageHistogram(workingImage);
            imshow(WINDOW_HISTOGRAM, histogramImage);
        }

        // if the key is c, call colorHistogram
        if (key == 99) {
            std::cout << "Color histogram" << std::endl;
            histogramImage = colorHistogram(workingImage);
            imshow(WINDOW_HISTOGRAM, histogramImage);
        }

        // If the key is e, call equalizeHistogram
        if (key == 101) {
            std::cout << "Equalize histogram" << std::endl;
            workingImage = equalizeHistogram(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is f, call equalizeColorHistogram
        if (key == 102) {
            std::cout << "Equalize color histogram" << std::endl;
            workingImage = equalizeColorHistogram(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is t, call halfToningGreyScale
        if (key == 116) {
            std::cout << "Half toning" << std::endl;
            // Print the type of the working image
            std::cout << workingImage.type() << std::endl;
            std::cout << "Type image " << workingImage.type() << std::endl;
            workingImage = floydSteinbergDitheringGrey(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is SPACE, take a picture from the camera
        if (key == 32) {
            std::cout << "Take a picture" << std::endl;
            camera >> workingImage;
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is r, reset the image to initial image
        if (key == 114) {
            std::cout << "Reset the image to initial image" << std::endl;
            workingImage = initImage;
            imshow(WINDOW_NAME, workingImage);
        }

        // int new_value = getTrackbarPos(trackbar_name, WINDOW_NAME);
        // if (value != new_value) {
        //     value = new_value;
        //     std::cout << "value=" << value << std::endl;
        // }
    }
}
