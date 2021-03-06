#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const String WINDOW_NAME      = "TP1";
const String WINDOW_HISTOGRAM = "Histogram";

const String MEDIAN_NAME = "Median";
int BLUR                 = 3;

const String ALPHA_NAME = "Alpha";
int ALPHA               = 20;

const String T_NAME = "Treshold";
int T               = 0;

// greyScale check the image is grey scale or not, if not then convert it to grey scale
Mat greyScale(Mat f) {
    Mat res;
    // Check if the image is already grey scale and print the message
    if (f.channels() == 1) {
        // std::cout << "Image is already grey scale" << std::endl;
        return f;
    }
    // else convert the image to grey scale and print the message "Image was in color and has been converted to grey scale"
    else {
        cvtColor(f, res, COLOR_BGR2GRAY);
        // std::cout << "Image was in color and has been converted to grey scale" << std::endl;
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

    return histogram;
}

// histogramCumulative computes the cumulative histogram of the image
std::vector<double> histogramCumulative(std::vector<double> &histogram) {
    std::vector<double> histogramCumulative(256, 0);
    histogramCumulative[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        histogramCumulative[i] = histogramCumulative[i - 1] + histogram[i];
    }

    return histogramCumulative;
}

// imageHistogram computes the histogram of the image and the cumulative histogram of the image
Mat imageHistogram(Mat image) {
    std::vector<double> histogramV           = histogram(image);
    std::vector<double> histogramCumulativeV = histogramCumulative(histogramV);

    // Image size
    int size = image.size().height * image.size().width;

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
Mat colorHistogram(Mat image) {
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
Mat equalizeHistogram(Mat image) {
    Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);
    return equalizedImage;
}

// equalizeColorHistogram equalize the histogram of the V channel and return the image
Mat equalizeColorHistogram(Mat image) {
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

// distanceColor return the distance between 2 cv::Vec3f
float distanceColor(cv::Vec3f color1, cv::Vec3f color2) {
    return sqrt(pow(color2[0] - color1[0], 2) +
                pow(color2[1] - color1[1], 2) +
                pow(color2[2] - color1[2], 2));
}

// bestColor return the closest color index to the given color in the given vector of colors
int bestColor(cv::Vec3f color, std::vector<cv::Vec3f> colors) {
    float minDistance = distanceColor(color, colors[0]);
    int minIndex      = 0;
    for (int i = 1; i < colors.size(); i++) {
        float distance = distanceColor(color, colors[i]);
        if (distance < minDistance) {
            minDistance = distance;
            minIndex    = i;
        }
    }
    return minIndex;
}

// errorColor return a color vector the error between the 2 given color vectors
cv::Vec3f errorColor(cv::Vec3f color1, cv::Vec3f color2) {
    return cv::Vec3f(color2[0] - color1[0] < 0 ? 0 : color2[0] - color1[0],
                     color2[1] - color1[1] < 0 ? 0 : color2[1] - color1[1],
                     color2[2] - color1[2] < 0 ? 0 : color2[2] - color1[2]);
}

Mat floydSteinbergDithering(Mat image, std::vector<cv::Vec3f> colors) {
    // Convert image into matrix of 3 floats channels
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC3, 1 / 255.0);
    // for each pixel
    for (int y = 0; y < imageFloat.rows; y++) {
        for (int x = 0; x < imageFloat.cols; x++) {
            // Get the current pixel
            cv::Vec3f pixel = imageFloat.at<cv::Vec3f>(y, x);
            // Find the closest color
            int closestColorIndex = bestColor(pixel, colors);
            // Compute the error
            cv::Vec3f error = errorColor(pixel, colors[closestColorIndex]);
            // Set the pixel to the closest color
            imageFloat.at<cv::Vec3f>(y, x) = colors[closestColorIndex];
            // Apply the error to the next pixel
            if (y + 1 < imageFloat.rows) {
                imageFloat.at<cv::Vec3f>(y + 1, x) += error * 5.0 / 16.0;
            }
            if (x + 1 < imageFloat.cols) {
                imageFloat.at<cv::Vec3f>(y, x + 1) += error * 7.0 / 16.0;
            }
            if (y + 1 < imageFloat.rows && x + 1 < imageFloat.cols) {
                imageFloat.at<cv::Vec3f>(y + 1, x + 1) += error * 1.0 / 16.0;
            }
            if (y + 1 < imageFloat.rows && x - 1 >= 0) {
                imageFloat.at<cv::Vec3f>(y + 1, x - 1) += error * 3.0 / 16.0;
            }
        }
    }
    // Convert the image back to 8 bits
    Mat image8bits;
    imageFloat.convertTo(image8bits, CV_8UC3, 255);
    return image8bits;
}

// floydSteinbergDithering dither the matrix using the Floyd-Steinberg algorithm with float values
Mat floydSteinbergDithering(Mat image) {
    Mat imageDithered;
    image.copyTo(imageDithered);
    float oldPixelValue;
    float newPixelValue;
    float errorValue;

    // Get all channels
    std::vector<Mat> channels;
    cv::split(imageDithered, channels);

    // For each channel
    for (int i = 0; i < channels.size(); i++) {
        channels[i].convertTo(channels[i], CV_32FC1);
        // For each pixel
        for (int y = 0; y < channels[i].rows; y++) {
            for (int x = 0; x < channels[i].cols; x++) {
                oldPixelValue = channels[i].at<float>(y, x);
                if (oldPixelValue > 127.0) {
                    newPixelValue = 255.0;
                } else {
                    newPixelValue = 0.0;
                }
                oldPixelValue = std::max(0.0f,oldPixelValue);
                oldPixelValue = std::min(255.0f,oldPixelValue);

                errorValue = oldPixelValue - newPixelValue;

                channels[i].at<float>(y, x) = newPixelValue;
                if (x != image.cols - 1) {
                    channels[i].at<float>(y, x + 1) += errorValue * 7.0 / 16.0;
                    if (y != image.rows - 1) {
                        if (x != 0) {
                            channels[i].at<float>(y + 1, x - 1) += errorValue * 5.0 / 16.0;
                        }
                        channels[i].at<float>(y + 1, x + 1) += errorValue / 16.0;
                        channels[i].at<float>(y + 1, x) += errorValue * 5.0 / 16.0;
                    }
                } else {
                    if (y != image.rows - 1) {
                        channels[i].at<float>(y + 1, x) += errorValue * 5.0 / 16.0;
                        if (x != 0) {
                            channels[i].at<float>(y + 1, x - 1) += errorValue * 3.0 / 16.0;
                        }
                    }
                }
            }
        }
        channels[i].convertTo(channels[i], CV_8UC1);
    }

    // Merge the channels
    cv::merge(channels, imageDithered);

    return imageDithered;
}

// halfToningGreyScale convert the image to grey scale and apply the floyd-steinberg dithering algorithm
Mat halfToningGreyScale(Mat image) {
    Mat imageGreyScale(image);

    // Apply the floyd-steinberg dithering algorithm
    for (int i = 0; i < imageGreyScale.rows; i++) {
        for (int j = 0; j < imageGreyScale.cols; j++) {
            imageGreyScale.at<uchar>(i, j) = imageGreyScale.at<uchar>(i, j) / 16 * 16;
        }
    }

    return imageGreyScale;
}
// filter convulve the image with the given kernel
Mat filter(Mat image, Mat kernel, double delta) {
    Mat imageFiltered;
    cv::filter2D(image, imageFiltered, -1, kernel, cv::Point(-1, -1), delta, cv::BORDER_DEFAULT);
    return imageFiltered;
}

// filter convulve the image with the given kernel
Mat filter(Mat image, Mat kernel) {
    return filter(image, kernel, 0.0);
}

Mat filtreMoyenneur(Mat image) {
    Mat kernel = (Mat_<float>(3, 3) << 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0);
    return filter(image, kernel);
}

Mat contrastEnhancement(Mat image) {
    // std::cout << "Contraste" << std::endl;
    ALPHA        = getTrackbarPos(ALPHA_NAME, WINDOW_NAME);
    float alphaF = ALPHA / 1000.0;
    Mat kernel   = (Mat_<float>(3, 3) << 0.0, alphaF * -1.0, 0.0, alphaF * -1.0, 1.0 + alphaF * 4.0, alphaF * -1.0, 0.0, alphaF * -1.0, 0.0);
    Mat img      = filter(image, kernel);
    return img;
}
// medianBlur blur the image with a median filter
Mat medianBlur(Mat image, int size) {
    // Print size
    Mat imageBlurred;

    cv::medianBlur(image, imageBlurred, size);

    return imageBlurred;
}

Mat medianBlur(Mat image) {
    // std::cout << "Median blur" << std::endl;
    BLUR    = getTrackbarPos(MEDIAN_NAME, WINDOW_NAME);
    Mat img = medianBlur(image, BLUR);
    return img;
}

// laplacianBlur blur the image with a laplacian filter
Mat laplacianBlur(Mat image, int size) {
    // Print size
    Mat imageBlurred;

    cv::Laplacian(image, imageBlurred, -1, size);

    return imageBlurred;
}

Mat laplacianBlur(Mat image) {
    // std::cout << "Laplacian blur" << std::endl;
    BLUR    = getTrackbarPos(MEDIAN_NAME, WINDOW_NAME);
    Mat img = laplacianBlur(image, BLUR);
    return img;
}

Mat sorbelX(Mat image) {
    // std::cout << "Sorbel X" << std::endl;
    Mat kernel = (Mat_<float>(3, 3) << -1.0 / 4.0, 0.0, 1.0 / 4.0, -2.0 / 4.0, 0.0, 2.0 / 4.0, -1.0 / 4.0, 0.0, 1.0 / 4.0);
    Mat img    = filter(image, kernel, 128.0);
    return img;
}

Mat sorbelY(Mat image) {
    // std::cout << "Sorbel Y" << std::endl;
    Mat kernel = (Mat_<float>(3, 3) << -1.0 / 4.0, -2.0 / 4.0, -1.0 / 4.0, 0.0, 0.0, 0.0, 1.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0);
    Mat img    = filter(image, kernel, 128.0);
    return img;
}

Mat gradient(Mat image) {
    // std::cout << "Gradient" << std::endl;
    Mat grad = Mat(image.size(), CV_8UC1, Scalar(0.0));
    Mat imgX = sorbelX(image);
    Mat imgY = sorbelY(image);
    // for each pixel in grad
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            grad.at<uchar>(i, j) = distanceColor(imgX.at<uchar>(i, j), imgY.at<uchar>(i, j));
        }
    }
    return grad;
}

// isSurroundingPixelSameSign if the surrounding pixels of a given pixel are all the same sign
bool isSurroundingPixelSameSign(Mat image, int row, int col) {
    if (row != 0 && col != 0 && row != image.rows - 1 && col != image.cols - 1) {
        if (
            (image.at<uchar>(row - 1, col - 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row - 1, col - 1) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row - 1, col) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row - 1, col) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row - 1, col + 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row - 1, col + 1) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row, col - 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row, col - 1) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row, col + 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row, col + 1) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row + 1, col - 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row + 1, col - 1) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row + 1, col) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row + 1, col) < 0 && image.at<uchar>(row, col) < 0) && (image.at<uchar>(row + 1, col + 1) > 0 && image.at<uchar>(row, col) > 0) || (image.at<uchar>(row + 1, col + 1) < 0 && image.at<uchar>(row, col) < 0)) {
            return true;
        }
    }
    return false;
}

Mat marrHildreth(Mat image) {
    // std::cout << "Marr Hildreth" << std::endl;
    Mat grad = gradient(image);

    Mat laplacian;
    cv::Laplacian(image, laplacian, -1, 1);

    Mat img(image.size(), CV_8UC1, Scalar(255));

    // for each pixel in img
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            // if the pixel is in the image is greater or equal to the gradient and
            // the surrounding pixels in laplacian are changing sign
            if (grad.at<uchar>(i, j) >= T && isSurroundingPixelSameSign(laplacian, i, j)) {
                img.at<uchar>(i, j) = 0;
            }
        }
    }

    return img;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << "./main <image_path>" << std::endl;
        return -1;
    }

    std::cout << "ESC   - Quit" << std::endl;
    std::cout << "r     - reset image" << std::endl;
    std::cout << "g     - grey scale" << std::endl;
    std::cout << "h     - grey scale histogram" << std::endl;
    std::cout << "c     - color histogram" << std::endl;
    std::cout << "e     - equalize grey scale histogram" << std::endl;
    std::cout << "f     - equalize color histogram" << std::endl;
    std::cout << "t     - Floyd-Steinberg dithering (half toning)" << std::endl;
    std::cout << "u     - Floyd-Steinberg dithering (CMYK) (not working)" << std::endl;
    std::cout << "a     - filtre moyenneur" << std::endl;
    std::cout << "b     - median blur" << std::endl;
    std::cout << "o     - contrast enhancement" << std::endl;
    std::cout << "x     - sorbel X" << std::endl;
    std::cout << "y     - sorbel Y" << std::endl;
    std::cout << "l     - Laplacian blur" << std::endl;
    std::cout << "d     - gradient" << std::endl;
    std::cout << "SPACE - live camera" << std::endl;
    std::cout << "\tg - grey scale" << std::endl;
    std::cout << "\tc - color image" << std::endl;
    std::cout << "\te - equalize grey scale histogram" << std::endl;
    std::cout << "\tf - equalize color histogram" << std::endl;
    std::cout << "\tt - Floyd-Steinberg dithering (half toning)" << std::endl;
    std::cout << "\tu - color Floyd-Steinberg dithering (half toning)" << std::endl;
    std::cout << "\ta - filtre moyenneur" << std::endl;
    std::cout << "\tb - median blur" << std::endl;
    std::cout << "\to - contrast enhancement" << std::endl;
    std::cout << "\tl - Laplacian blur" << std::endl;
    std::cout << "\tx - sorbel X" << std::endl;
    std::cout << "\ty - sorbel Y" << std::endl;
    std::cout << "\td - gradient" << std::endl;
    

    String imagePath = argv[1];

    Mat initImage = imread(imagePath);

    namedWindow(WINDOW_NAME);
    namedWindow(WINDOW_HISTOGRAM);

    // trackbar for median blur
    createTrackbar(MEDIAN_NAME, WINDOW_NAME, &BLUR, 10, NULL);
    setTrackbarPos(MEDIAN_NAME, WINDOW_NAME, BLUR);

    // trackbar for alpha
    createTrackbar(ALPHA_NAME, WINDOW_NAME, &ALPHA, 1000, NULL);
    setTrackbarPos(ALPHA_NAME, WINDOW_NAME, ALPHA);

    // trackbar for delta
    createTrackbar(T_NAME, WINDOW_NAME, &T, 255, NULL);
    setTrackbarPos(T_NAME, WINDOW_NAME, T);

    imshow(WINDOW_NAME, initImage);

    // Wait 50ms for a keystroke and get the key code
    int key = waitKeyEx(50);

    // All images variables
    Mat workingImage;
    initImage.copyTo(workingImage);

    Mat histogramImage;

    // Camera
    cv::VideoCapture camera(0);

    while (true) {
        // Get the key code
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
            workingImage = floydSteinbergDithering(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is u, call floydSteinbergDithering with vector
        if (key == 117) {
            std::cout << "Floyd-Steinberg dithering" << std::endl;
            // Vector containing cyan, magenta, yellow, white and black
            std::vector<cv::Vec3f> colors = {
                cv::Vec3f(1.0, 1.0, 1.0),
                cv::Vec3f(1.0, 1.0, 0.0),
                cv::Vec3f(1.0, 0.0, 1.0),
                cv::Vec3f(0.0, 1.0, 1.0),
                cv::Vec3f(0.0, 0.0, 0.0)};
            workingImage = floydSteinbergDithering(workingImage, colors);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is a, call filtreMoyenneur
        if (key == 97) {
            workingImage = filtreMoyenneur(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is b, call medianBlur with size from trackbar MEDIAN_NAME
        if (key == 98) {
            workingImage = medianBlur(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if o, call filterM with the kernel [[0,1,0],[1,-4,1],[0,1,0]]
        if (key == 111) {
            workingImage = contrastEnhancement(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is x, call filter with Sorbel kernel ([[-1/4,0,1/4],[-2/4,0,2/4],[-1/4,0,1/4]])
        if (key == 120) {
            std::cout << "Sorn x" << std::endl;
            workingImage = sorbelX(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is y, call filter with Sorbel kernel ([[-1/4,-2/4,-1/4],[0,0,0],[1/4,2/4,1/4]])
        if (key == 121) {
            std::cout << "Sorn y" << std::endl;
            workingImage = sorbelY(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is l, call laplacianBlur with size from trackbar medianName
        if (key == 108) {
            std::cout << "Laplacian blur" << std::endl;
            BLUR         = getTrackbarPos(MEDIAN_NAME, WINDOW_NAME);
            workingImage = laplacianBlur(workingImage, BLUR);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is d, call gradent
        if (key == 100) {
            std::cout << "Gradient" << std::endl;
            workingImage = gradient(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // if key is m, call marrHildreth
        if (key == 109) {
            std::cout << "Marr-Hildreth" << std::endl;
            workingImage = marrHildreth(workingImage);
            imshow(WINDOW_NAME, workingImage);
        }

        // If the key is SPACE, take a picture from the camera
        if (key == 32) {
            if (!camera.isOpened())
                return -1;
            Mat frameImage, frameHistogram;

            Mat (*imageFunction)(Mat)             = NULL;
            Mat (*histogramFunction)(Mat)         = &colorHistogram;

            Mat kernel;
            double delta;

            bool needGreyScale = false;
            bool isGreyScale   = false;
            int blurLevel      = 0;

            for (;;) {
                camera >> frameImage;

                if (imageFunction != NULL) {

                    if (needGreyScale) {
                        frameImage = greyScale(frameImage);
                    }
                    if (blurLevel > 0) {
                        for (int i = 0; i < blurLevel; i++) {
                            frameImage = imageFunction(frameImage);
                        }
                    }
                    frameImage = imageFunction(frameImage);
                }

                frameImage.copyTo(frameHistogram);

                if (histogramFunction != NULL) {
                    frameHistogram = histogramFunction(frameHistogram);
                }

                imshow(WINDOW_NAME, frameImage);
                imshow(WINDOW_HISTOGRAM, frameHistogram);

                int key_code   = waitKey(30);
                int ascii_code = key_code & 0xff;

                if (ascii_code == 'g') {
                    needGreyScale     = false;
                    isGreyScale       = true;
                    blurLevel         = 0;
                    imageFunction     = &greyScale;
                    histogramFunction = &imageHistogram;
                }
                if (ascii_code == 'c') {
                    needGreyScale     = false;
                    isGreyScale       = false;
                    blurLevel         = 0;
                    imageFunction     = NULL;
                    histogramFunction = &colorHistogram;
                }
                if (ascii_code == 'e') {
                    needGreyScale     = true;
                    isGreyScale       = true;
                    blurLevel         = 0;
                    imageFunction     = &equalizeHistogram;
                    histogramFunction = &imageHistogram;
                }
                if (ascii_code == 'f') {
                    needGreyScale     = false;
                    isGreyScale       = false;
                    blurLevel         = 0;
                    imageFunction     = &equalizeColorHistogram;
                    histogramFunction = &colorHistogram;
                }

                if (ascii_code == 't') {
                    needGreyScale     = true;
                    isGreyScale       = true;
                    imageFunction     = &floydSteinbergDithering;
                    histogramFunction = &imageHistogram;
                }

                if (ascii_code == 'u') {
                    needGreyScale     = false;
                    isGreyScale       = false;
                    blurLevel         = 0;
                    imageFunction     = &floydSteinbergDithering;
                    histogramFunction = &colorHistogram;
                }

                if (ascii_code == 'a') {
                    imageFunction = &filtreMoyenneur;
                    blurLevel++;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'b') {
                    needGreyScale = false;
                    imageFunction = &medianBlur;
                    blurLevel++;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'o') {
                    needGreyScale = false;
                    imageFunction = &contrastEnhancement;
                    blurLevel++;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'l') {
                    imageFunction = &laplacianBlur;
                    blurLevel++;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'x') {
                    imageFunction = &sorbelX;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'y') {
                    imageFunction = &sorbelY;
                    if (isGreyScale) {
                        histogramFunction = &imageHistogram;
                        needGreyScale     = true;
                    } else {
                        histogramFunction = &colorHistogram;
                        needGreyScale     = false;
                    }
                }

                if (ascii_code == 'd') {
                    needGreyScale     = true;
                    imageFunction = &gradient;
                    histogramFunction = &imageHistogram;
                }

                if (ascii_code == 'm') {
                    needGreyScale = true;
                    imageFunction = &marrHildreth;
                    histogramFunction = &imageHistogram;
                }

                if (key_code == 113)
                    break;
            }
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
