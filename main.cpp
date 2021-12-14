#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    int value            = 128;
    String window_name   = "TP1";
    String trackbar_name = "Trackbar";
    String image_path    = argv[1];

    Mat f = imread(image_path);

    namedWindow(window_name);
    createTrackbar(trackbar_name, window_name, nullptr, 255, NULL);
    setTrackbarPos(trackbar_name, window_name, value);
    imshow(window_name, f);

    // Wait 50ms for a keystroke and get the key code
    int key = waitKeyEx(50);

    while (true) {
        // Print the key code
        std::cout << "Key code: " << key << std::endl;
        key = waitKeyEx(500);

        // If the key is ESC, break the loop
        if (key == 27) {
            break;
        }

        // int new_value = getTrackbarPos(trackbar_name, window_name);
        // if (value != new_value) {
        //     value = new_value;
        //     std::cout << "value=" << value << std::endl;
        // }
    }
}