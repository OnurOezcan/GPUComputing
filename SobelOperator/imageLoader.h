#pragma once
#include <string>

typedef unsigned char byte; // most useful typedef ever

/************************************************************************************************
 * struct imgData(byte*, uint, uint)
 * - a struct to contain all information about our image
 ***********************************************************************************************/
struct imgData {
    imgData(byte* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte* pixels;
    unsigned int width;
    unsigned int height;
};

imgData loadImage(char* filename);

void writeImage(char* filename, std::string appendTxt, imgData img);