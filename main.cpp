#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main() {
  int value = 128;
  namedWindow("TP1");                                 // crée une fenêtre
  createTrackbar("track", "TP1", nullptr, 255, NULL); // un slider
  setTrackbarPos("track", "TP1", value);
  Mat f = imread("lena.png"); // lit l'image "lena.png"
  imshow("TP1", f);         // l'affiche dans la fenêtre
  while (waitKey(50) < 0)     // attend une touche
  {                           // Affiche la valeur du slider
    int new_value = getTrackbarPos("track", "TP1");
    if (value != new_value) {
      value = new_value;
      std::cout << "value=" << value << std::endl;
    }
  }
}