
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define _USE_MATH_DEFINES

#define M_PI 3.14159265358979323846

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;

__global__ void XTI_kernel(unsigned short int* image_buffer_1D_fg_GPU_knl, unsigned short int* image_buffer_1D_bg_GPU_knl, unsigned int* no_pixels_GPU_knl, unsigned int* M_fg_GPU_knl, unsigned int* M_bg_GPU_knl, double* phase_step_fg_GPU_knl, double* phase_step_bg_GPU_knl, double* dph_image_GPU_knl, double* abs_image_GPU_knl, double* vis_image_GPU_knl)
{
    int index_pixel = threadIdx.x + blockIdx.x * blockDim.x;

    int no_pixels_GPU_kernel = int(*no_pixels_GPU_knl);
    unsigned int M_fg_GPU_kernel = *M_fg_GPU_knl;
    unsigned int M_bg_GPU_kernel = *M_bg_GPU_knl;
    double phase_step_fg_GPU_kernel = *phase_step_fg_GPU_knl;
    double phase_step_bg_GPU_kernel = *phase_step_bg_GPU_knl;

    double phase_buffer_fg = 0.0f;
    double amp_buffer_fg = 0.0f;
    double offset_buffer_fg = 0.0f;

    double phase_buffer_bg = 0.0f;
    double amp_buffer_bg = 0.0f;
    double offset_buffer_bg = 0.0f;

    // calculate phase image for foreground
    double iterator = 0.0f;
    double intensity_value = 0.0f;
    // initilaize variable for 0-th fourier coefficient
    double sum_intensity = 0.0f;
    // initialize buffer for 1-st order fourier coefficient, sinus part
    double a1 = 0.0f;
    // initialize buffer for 1-st order fourier coefficient, cosinus part
    double b1 = 0.0f;
    for (int index_0 = 0; index_0 < M_fg_GPU_kernel; index_0++) {
        iterator = double(index_0);
        intensity_value = double(image_buffer_1D_fg_GPU_knl[index_pixel + index_0 * no_pixels_GPU_kernel]);
        sum_intensity += intensity_value;
        a1 += intensity_value * std::sin(1 * iterator * phase_step_fg_GPU_kernel);
        b1 += intensity_value * std::cos(1 * iterator * phase_step_fg_GPU_kernel);
    }
    phase_buffer_fg = atan2(a1, b1);
    amp_buffer_fg = 2.0f * std::sqrt(a1 * a1 + b1 * b1) / double(M_fg_GPU_kernel);
    offset_buffer_fg = sum_intensity / double(M_fg_GPU_kernel);
    // calculate phase image for background
    iterator = 0.0f;
    intensity_value = 0.0f;
    sum_intensity = 0.0f;
    // initialize buffer for 1-st order fourier coefficient, sinus part
    a1 = 0.0f;
    // initialize buffer for 1-st order fourier coefficient, cosinus part
    b1 = 0.0f;
    for (int index_0 = 0; index_0 < M_bg_GPU_kernel; index_0++) {
        iterator = double(index_0);
        intensity_value = double(image_buffer_1D_bg_GPU_knl[index_pixel + index_0 * no_pixels_GPU_kernel]);
        sum_intensity += intensity_value;
        a1 += intensity_value * std::sin(1 * iterator * phase_step_bg_GPU_kernel);
        b1 += intensity_value * std::cos(1 * iterator * phase_step_bg_GPU_kernel);
    }
    phase_buffer_bg = atan2(a1, b1);
    amp_buffer_bg = 2.0f * std::sqrt(a1 * a1 + b1 * b1) / double(M_bg_GPU_kernel);
    offset_buffer_bg = sum_intensity / double(M_bg_GPU_kernel);

    // calculate differential phase image or dph image
    dph_image_GPU_knl[index_pixel] = phase_buffer_fg - phase_buffer_bg;
    // calculate absorption image or dph image
    abs_image_GPU_knl[index_pixel] = offset_buffer_fg / offset_buffer_bg;
    // calculate visibility image or dph image
    vis_image_GPU_knl[index_pixel] = (amp_buffer_fg / offset_buffer_fg) / (amp_buffer_bg / offset_buffer_bg);
}

int main()
{
    // define path to folder with all foreground data or all subfolders
    string path_to_fg_folder("d:/XTI_Momose_lab/BL28B2_2017A/sort_data/pp/fg/");
    // define path to folder with all background data or all subfolders
    string path_to_bg_folder("d:/XTI_Momose_lab/BL28B2_2017A/sort_data/bg/");
    // print path to folder with all foreground folders
    std::cout << path_to_fg_folder << "\n";
    // print path to folder with all background folders
    std::cout << path_to_bg_folder << "\n";

    // define path to output folder with output differential phase (dph) images
    string path_to_output_folder("d:/XTI_Momose_lab/BL28B2_2017A/sort_data/pp_dph_abs_vis_Momose_GPU/");

    // output image name root - differential phase image or dph image
    string image_output_dph_name_root = "dph";
    // output image name root - absorption image or abs image
    string image_output_abs_name_root = "abs";
    // output image name root - visibility image or vis image
    string image_output_vis_name_root = "vis";
    // final ouput image name - differential phase image or dph image
    string image_output_dph_name;
    // final ouput image name - absorption image or abs image
    string image_output_abs_name;
    // final ouput image name - visibility image or vis image
    string image_output_vis_name;
    // extension of output image
    string image_output_extension = ".raw";

    // define size of the raw unsigned 16 bit images
    const unsigned int no_cols = 1536; // in pixels, in horizontal direction
    const unsigned int no_rows = 512; // in pixels, in vertical direction
    // total number of pixels in single image
    const unsigned int no_pixels = no_cols * no_rows;
    // total number of bytes in single image, we consider 16 bit values per pixel = 2 bytes
    const unsigned int no_bytes = 2 * no_pixels;

    // define number of initial and final subfolder for foreground
    unsigned int no_subfolder_fg_initial = 1;
    unsigned int no_subfolder_fg_final = 11200;

    // define number of initial and final folder for background
    unsigned int no_subfolder_bg_initial = 1;
    unsigned int no_subfolder_bg_final = 11200;

    // number of digits in subfolder name for foreground
    string::size_type no_subfolder_digits_fg = 6;
    // number of digits in subfolder name for background
    string::size_type no_subfolder_digits_bg = 6;

    // number of steps in fringe scanning technique
    const unsigned int M = 5;

    // calculate differential phase image for foreground
    // fringe scanning defined from initial value
    const unsigned int M_fg_initial = 1;
    // fringe scanning defined to final value
    const unsigned int M_fg_final = M;
    // number of steps in fringe scanning for foreground
    const unsigned int M_fg = M_fg_final - M_fg_initial + 1;
    const unsigned int N_fg = M_fg * no_pixels;

    // calculate differential phase image for background
    // fringe scanning defined from initial value
    const unsigned int M_bg_initial = 1;
    // fringe scanning defined to final value
    const unsigned int M_bg_final = M;
    // number of steps in fringe scanning for background
    const unsigned int M_bg = M_bg_final - M_bg_initial + 1;
    const unsigned int N_bg = M_bg * no_pixels;

    // define root name of images for foreground
    string root_image_name_fg("a");
    // define root name of images for background
    string root_image_name_bg("a");

    // number of digits in image name for foreground
    string::size_type no_image_digits_fg = 6;
    // number of digits in image name for background
    string::size_type no_image_digits_bg = 6;

    // define image extensions
    // image extension for foreground
    string image_extension_fg = ".raw";
    // image extension for background
    string image_extension_bg = ".raw";

    // allocate image buffer for foreground
    auto image_buffer_fg = new unsigned short int[no_pixels][M_fg];
    // allocate image buffer for background
    auto image_buffer_bg = new unsigned short int[no_pixels][M_fg];
    // allocate image buffer as 1D array for foreground
    // to copy M_fg foreground images to the GPU
    unsigned short int* image_buffer_1D_fg = new unsigned short int[M_fg * no_pixels];
    // allocate image buffer as 1D array for background
    // to copy M_bg background images to the GPU
    unsigned short int* image_buffer_1D_bg = new unsigned short int[M_bg * no_pixels];

    // allocate memory for differential phase image
    double* dph_image = new double[no_pixels];
    // allocate memory for absorption image
    double* abs_image = new double[no_pixels];
    // allocate memory for visibility image
    double* vis_image = new double[no_pixels];

    // define phase for foreground
    double phase_step_fg = (2 * M_PI) / double(M_fg);
    // define phase_step for background
    double phase_step_bg = (2 * M_PI) / double(M_bg);

    // auxiliary variables for iteration through subfolder name for foreground
    string subfolder_name(no_subfolder_digits_fg, '0');
    string subfolder_number = "";
    string::size_type counter_digits = 0;
    string::size_type difference = 0;
    string::size_type counter = 0;
    string path_to_fg_subfolder = "";

    // auxiliary variables for iteration through M_fg images
    int counter_image = 0;
    string image_name = root_image_name_fg;
    string image_name_number(no_image_digits_fg, '0');
    string image_number = "";
    // counter_digits, difference and counter variables are taken from iterations through subfolders
    string path_to_fg_image = "";

    // auxiliary variables for iteration through subfolder name for background
    string path_to_bg_subfolder = "";

    // auxiliary variables for iteration through M_bg images
    image_name = root_image_name_bg;
    image_name_number = string(no_image_digits_bg, '0');
    image_number = "";
    // counter_digits, difference and counter variables are taken from iterations through subfolders
    string path_to_bg_image = "";

    // declare auxiliary variable for output image
    string path_to_output_image = "";

    // declare auxiliary variable for output dph image
    string path_to_output_dph_image = "";
    // declare auxiliary variable for output abs image
    string path_to_output_abs_image = "";
    // declare auxiliary variable for output vis image
    string path_to_output_vis_image = "";

    //*****************************************************
    // variables for the GPU CUDA
    //*****************************************************
    // initialize varaible for number of columns of the images on the GPU
    unsigned int* no_cols_GPU = nullptr;
    // initialize varaible for number of rows of the images on the GPU
    unsigned int* no_rows_GPU = nullptr;
    // initialize variable for number of pixels on the GPU
    unsigned int* no_pixels_GPU = nullptr;
    // initialize variable for number of steps in fringe scanning for foreground on the GPU
    unsigned int* M_fg_GPU = nullptr;
    // initialize variable for number of steps in fringe scanning for background on the GPU
    unsigned int* M_bg_GPU = nullptr;
    // initialize 1D array for M_fg foreground images on the GPU
    unsigned short int* image_buffer_1D_fg_GPU = nullptr;
    // initialize 1D array for M_bg background images on the GPU
    unsigned short int* image_buffer_1D_bg_GPU = nullptr;
    // initialize phase step in fringe scanning for foreground on the GPU
    double* phase_step_fg_GPU = nullptr;
    // initialize phase step in fringe scanning for foreground on the GPU
    double* phase_step_bg_GPU = nullptr;
    // initialize output 1D arrays
    // initialize output 1D array for differential phase image (dph)
    double* dph_image_GPU = nullptr;
    // initialize output 1D array for absorption image (abs)
    double* abs_image_GPU = nullptr;
    // initialize output 1D array for visibility image (vis)
    double* vis_image_GPU = nullptr;
    //*****************************************************
    // end
    //*****************************************************

    //*****************************************************
    // fill pointers with constant values
    //*****************************************************
    unsigned int* no_cols_ptr = new unsigned int;
    *no_cols_ptr = no_cols;
    unsigned int* no_rows_ptr = new unsigned int;
    *no_rows_ptr = no_rows;
    unsigned int* no_pixels_ptr = new unsigned int;
    *no_pixels_ptr = no_pixels;
    unsigned int* M_fg_ptr = new unsigned int;
    *M_fg_ptr = M_fg;
    unsigned int* M_bg_ptr = new unsigned int;
    *M_bg_ptr = M_bg;
    double* phase_step_fg_ptr = new double;
    *phase_step_fg_ptr = phase_step_fg;
    double* phase_step_bg_ptr = new double;
    *phase_step_bg_ptr = phase_step_bg;
    //*****************************************************
    // end
    //*****************************************************

    //*****************************************************
    // define number of therads and blocks
    //*****************************************************
    int no_threads = 512;
    int no_blocks = 1536;
    //*****************************************************
    // end
    //*****************************************************

    // go through all foreground subfolders and foreground images
    for (unsigned int index_0 = no_subfolder_fg_initial; index_0 <= no_subfolder_fg_final; index_0++) {
        // start to measure elapsed time at the beginning
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // initialize subfolder name to "000000"
        subfolder_name = string(no_subfolder_digits_fg, '0');
        // typcast integer value to string, convert integer value to string
        subfolder_number = std::to_string(index_0);
        // initialize digits counter
        counter_digits = subfolder_number.size();
        // initialize difference
        difference = no_subfolder_digits_fg - counter_digits;
        // initialize counter
        counter = 0;
        // generate subfolder name
        for (string::size_type index_1 = difference; index_1 < no_subfolder_digits_fg; index_1++) {
            subfolder_name[index_1] = subfolder_number[counter];
            counter++;
        }
        // generate path to foreground subfolder 
        path_to_fg_subfolder = path_to_fg_folder + subfolder_name;
        // print final path to foreground subfolder
        std::cout << path_to_fg_subfolder << "\n";
        // initilize counter for one of M images
        counter_image = 0;
        // open images in foreground subfolder
        for (unsigned int index_2 = M_fg_initial; index_2 <= M_fg_final; index_2++) {
            // initialize image name to root value
            image_name = root_image_name_fg;
            // initialize image number to "000000"
            image_name_number = string(no_image_digits_fg, '0');
            // typcast integer value to string, convert integer value to string
            image_number = std::to_string(index_2);
            // initialize digits counter
            counter_digits = image_number.size();
            // initialize difference
            difference = no_image_digits_fg - counter_digits;
            // initialize counter
            counter = 0;
            // generate image name number
            for (string::size_type index_3 = difference; index_3 < no_image_digits_fg; index_3++) {
                image_name_number[index_3] = image_number[counter];
                counter++;
            }
            // concatenate root image value and image number
            image_name += image_name_number;
            // generate path to foreground image 
            path_to_fg_image = path_to_fg_subfolder + "/" + image_name + image_extension_fg;
            // print path to image for foregrouund
            //std::cout << path_to_fg_image << "\n";
            //************************************************************************************
            // read binary images
            //************************************************************************************
            ifstream raw_image(path_to_fg_image, ios::out | ios::binary);
            /*streampos begin, end;
            begin = raw_image.tellg();
            raw_image.seekg(0, ios::end);
            end = raw_image.tellg();*/
            //std::cout << "Size of the raw image is: " << (end - begin) << " bytes.\n";
            if (raw_image.is_open())
            {
                //unsigned int counter_pixel = 0;
                //raw_image.seekg(0, ios::beg);
                //while (raw_image.read(reinterpret_cast<char*>(&image_buffer_fg[counter_pixel][counter_image]), sizeof(uint16_t))) { // Read 16-bit integer values from file
                //    counter_pixel++;
                //}
                //raw_image.close();
                for (unsigned int counter_pixel = 0; counter_pixel < no_pixels; counter_pixel++) {
                    raw_image.read((char*)&image_buffer_fg[counter_pixel][counter_image], sizeof(unsigned short int));
                }
                raw_image.close();
            }
            else {
                std::cout << "Warning: Unable to open raw image file!!!" << "\n";
            }
            //************************************************************************************
            // end of reading of binary images
            //************************************************************************************
            // increase image counter by one
            counter_image++;
        }

        // go through background subfolder and background images
        // initialize subfolder name to "000000"
        subfolder_name = string(no_subfolder_digits_bg, '0');
        // typcast integer value to string, convert integer value to string
        subfolder_number = std::to_string(index_0);
        // initialize digits counter
        counter_digits = subfolder_number.size();
        // initialize difference
        difference = no_subfolder_digits_bg - counter_digits;
        // initialize counter
        counter = 0;
        // generate subfolder name
        for (string::size_type index_1 = difference; index_1 < no_subfolder_digits_bg; index_1++) {
            subfolder_name[index_1] = subfolder_number[counter];
            counter++;
        }
        // generate path to background subfolder 
        path_to_bg_subfolder = path_to_bg_folder + subfolder_name;
        // print final path to background subfolder
        std::cout << path_to_bg_subfolder << "\n";
        // initilize counter for one of M images
        counter_image = 0;
        // open images in background subfolder
        for (unsigned int index_2 = M_bg_initial; index_2 <= M_bg_final; index_2++) {
            // initialize image name to root value
            image_name = root_image_name_bg;
            // initialize image number to "000000"
            image_name_number = string(no_image_digits_bg, '0');
            // typcast integer value to string, convert integer value to string
            image_number = std::to_string(index_2);
            // initialize digits counter
            counter_digits = image_number.size();
            // initialize difference
            difference = no_image_digits_bg - counter_digits;
            // initialize counter
            counter = 0;
            // generate image name number
            for (string::size_type index_3 = difference; index_3 < no_image_digits_bg; index_3++) {
                image_name_number[index_3] = image_number[counter];
                counter++;
            }
            // concatenate root image value and image number
            image_name += image_name_number;
            // generate path to background image 
            path_to_bg_image = path_to_bg_subfolder + "/" + image_name + image_extension_bg;
            // print path to image for background
            //std::cout << path_to_bg_image << "\n";
            //************************************************************************************
            // read binary images
            //************************************************************************************
            ifstream raw_image(path_to_bg_image, ios::out | ios::binary);
            /*streampos begin, end;
            begin = raw_image.tellg();
            raw_image.seekg(0, ios::end);
            end = raw_image.tellg();*/
            //std::cout << "Size of the raw image is: " << (end - begin) << " bytes.\n";
            if (raw_image.is_open())
            {
                //unsigned int counter_pixel = 0;
                //raw_image.seekg(0, ios::beg);
                //while (raw_image.read(reinterpret_cast<char*>(&image_buffer_bg[counter_pixel][counter_image]), sizeof(uint16_t))) { // Read 16-bit integer values from file
                //    counter_pixel++;
                //}
                //raw_image.close();
                raw_image.seekg(0, ios::beg);
                for (unsigned int counter_pixel = 0; counter_pixel < no_pixels; counter_pixel++) {
                    raw_image.read((char*)&image_buffer_bg[counter_pixel][counter_image], sizeof(unsigned short int));
                }
                raw_image.close();
            }
            else {
                std::cout << "Warning: Unable to open raw image file!!!" << "\n";
            }
            //************************************************************************************
            // end of reading of binary images
            //************************************************************************************
            // increase image counter by one
            counter_image++;
        }

        // transfer images for foreground to 1D array
        // this 1D array will be later transferred to the GPU, it is data preparation for GPU
        for (unsigned int index_4 = 0; index_4 < M_fg; index_4++) {
            for (unsigned int index_5 = 0; index_5 < no_pixels; index_5++) {
                image_buffer_1D_fg[index_5 + index_4 * no_pixels] = image_buffer_fg[index_5][index_4];
            }
        }
        // transfer images for background to 1D array
        // this 1D array will be later transferred to the GPU, it is data preparation for GPU
        // this 1D array will be later transferred to the GPU, it is data preparation for GPU
        for (unsigned int index_4 = 0; index_4 < M_bg; index_4++) {
            for (unsigned int index_5 = 0; index_5 < no_pixels; index_5++) {
                image_buffer_1D_bg[index_5 + index_4 * no_pixels] = image_buffer_bg[index_5][index_4];
            }
        }

        // allocate memory on the GPU
        // allocate memory on the GPU for number of columns in the images
        cudaMalloc((void**)&no_cols_GPU, sizeof(unsigned int));
        // allocate memory on the GPU for number of rows in the images
        cudaMalloc((void**)&no_rows_GPU, sizeof(unsigned int));
        // allocate memory on the GPU for number of pixels in the images
        cudaMalloc((void**)&no_pixels_GPU, sizeof(unsigned int));
        // allocate memory for number of steps in fringe scanning for foreground
        cudaMalloc((void**)&M_fg_GPU, sizeof(unsigned int));
        // allocate memory for number of steps in fringe scanning for background
        cudaMalloc((void**)&M_bg_GPU, sizeof(unsigned int));
        // allocate memory for 1D array to store unsigned 16 bit integer raw images for foreground
        cudaMalloc((void**)&image_buffer_1D_fg_GPU, N_fg * sizeof(unsigned short int));
        // allocate memory for 1D array to store unsigned 16 bit integer raw images for background
        cudaMalloc((void**)&image_buffer_1D_bg_GPU, N_bg * sizeof(unsigned short int));
        // allocate memory for storing phase step for foreground
        cudaMalloc((void**)&phase_step_fg_GPU, sizeof(double));
        // allocate memory for storing phase step for foreground
        cudaMalloc((void**)&phase_step_bg_GPU, sizeof(double));
        // allocate memory for differential phase (dph) image calculated on the GPU
        cudaMalloc((void**)&dph_image_GPU, no_pixels * sizeof(double));
        // allocate memory for absorption (abs) image calculated on the GPU
        cudaMalloc((void**)&abs_image_GPU, no_pixels * sizeof(double));
        // allocate memory for visibility (vis) image calculated on the GPU
        cudaMalloc((void**)&vis_image_GPU, no_pixels * sizeof(double));

        // copy all data to the GPU
        cudaMemcpy(no_cols_GPU, no_cols_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(no_rows_GPU, no_rows_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(no_pixels_GPU, no_pixels_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(M_fg_GPU, M_fg_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(M_bg_GPU, M_bg_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(image_buffer_1D_fg_GPU, image_buffer_1D_fg, N_fg * sizeof(unsigned short int), cudaMemcpyHostToDevice);
        cudaMemcpy(image_buffer_1D_bg_GPU, image_buffer_1D_bg, N_bg * sizeof(unsigned short int), cudaMemcpyHostToDevice);
        cudaMemcpy(phase_step_fg_GPU, phase_step_fg_ptr, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(phase_step_bg_GPU, phase_step_bg_ptr, sizeof(double), cudaMemcpyHostToDevice);

        // perform kernel calculation or calculation on the GPU
        XTI_kernel << < no_blocks, no_threads >> > (image_buffer_1D_fg_GPU, image_buffer_1D_bg_GPU, no_pixels_GPU, M_fg_GPU, M_bg_GPU, phase_step_fg_GPU, phase_step_bg_GPU, dph_image_GPU, abs_image_GPU, vis_image_GPU);

        // copy results from the GPU to the CPU
        cudaMemcpy(dph_image, dph_image_GPU, no_pixels * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(abs_image, abs_image_GPU, no_pixels * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vis_image, vis_image_GPU, no_pixels * sizeof(double), cudaMemcpyDeviceToHost);

        // free the memory allocated on the GPU
        cudaFree(no_cols_GPU);
        cudaFree(no_rows_GPU);
        cudaFree(no_pixels_GPU);
        cudaFree(M_fg_GPU);
        cudaFree(M_bg_GPU);
        cudaFree(image_buffer_1D_fg_GPU);
        cudaFree(image_buffer_1D_bg_GPU);
        cudaFree(phase_step_fg_GPU);
        cudaFree(phase_step_bg_GPU);
        cudaFree(dph_image_GPU);
        cudaFree(abs_image_GPU);
        cudaFree(vis_image_GPU);

        // define name for output dph image for current subfolder
        image_output_dph_name = image_output_dph_name_root + "_" + subfolder_name + image_output_extension;
        // define name for output abs image for current subfolder
        image_output_abs_name = image_output_abs_name_root + "_" + subfolder_name + image_output_extension;
        // define name for output vis image for current subfolder
        image_output_vis_name = image_output_vis_name_root + "_" + subfolder_name + image_output_extension;
        // define path to the output dph image
        path_to_output_dph_image = path_to_output_folder + image_output_dph_name;
        // define path to the output abs image
        path_to_output_abs_image = path_to_output_folder + image_output_abs_name;
        // define path to the output vis image
        path_to_output_vis_image = path_to_output_folder + image_output_vis_name;
        // write differential phase (dph) image
        // set for output, binary data, trunc
        fstream output_dph_image(path_to_output_dph_image, ios::out | ios::binary | ios::trunc);
        if (output_dph_image.is_open())
        {
            // set pointer to the beginning of the image
            output_dph_image.seekg(0, ios::beg);
            for (unsigned int index_11 = 0; index_11 < no_pixels; index_11++) {
                output_dph_image.write((char*)&dph_image[index_11], sizeof(double));
            }
            output_dph_image.close();
        }
        else {
            std::cout << "Warning: Unable to open dph image file!!!" << "\n";
        }
        // write absorption (abs) image
        // set for output, binary data, trunc
        fstream output_abs_image(path_to_output_abs_image, ios::out | ios::binary | ios::trunc);
        if (output_abs_image.is_open())
        {
            // set pointer to the beginning of the image
            output_abs_image.seekg(0, ios::beg);
            for (unsigned int index_11 = 0; index_11 < no_pixels; index_11++) {
                output_abs_image.write((char*)&abs_image[index_11], sizeof(double));
            }
            output_abs_image.close();
        }
        else {
            std::cout << "Warning: Unable to open abs image file!!!" << "\n";
        }
        // write visibility (vis) image
        // set for output, binary data, trunc
        fstream output_vis_image(path_to_output_vis_image, ios::out | ios::binary | ios::trunc);
        if (output_vis_image.is_open())
        {
            // set pointer to the beginning of the image
            output_vis_image.seekg(0, ios::beg);
            for (unsigned int index_11 = 0; index_11 < no_pixels; index_11++) {
                output_vis_image.write((char*)&vis_image[index_11], sizeof(double));
            }
            output_vis_image.close();
        }
        else {
            std::cout << "Warning: Unable to open vis image file!!!" << "\n";
        }

        // stop to measure elapsed time at the end
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // print elapsed time in milliseconds, microseconds and nanoseconds
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[seconds]" << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[millisec]" << std::endl;
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microsec]" << std::endl;
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[nanosec]" << std::endl;
    }

    // delete image buffer for foreground
    delete[] image_buffer_fg;
    // delete image buffer for background
    delete[] image_buffer_bg;
    // delete image buffer for foreground
    delete[] image_buffer_1D_fg;
    // delete image buffer for background
    delete[] image_buffer_1D_bg;
    // delete buffer for differential phase image
    delete[] dph_image;
    // delete buffer for absorption image
    delete[] abs_image;
    // delete buffer for visibility image
    delete[] vis_image;

    // delete host pointers used to transfer data to the GPU
    delete no_cols_ptr;
    delete no_rows_ptr;
    delete no_pixels_ptr;
    delete M_fg_ptr;
    delete M_bg_ptr;
    delete phase_step_fg_ptr;
    delete phase_step_bg_ptr;

    // delete GPU or device pointers
    /*delete no_cols_GPU;
    delete no_rows_GPU;
    delete no_pixels_GPU;
    delete M_fg_GPU;
    delete M_bg_GPU;
    delete image_buffer_1D_fg_GPU;
    delete image_buffer_1D_bg_GPU;
    delete phase_step_fg_GPU;
    delete phase_step_bg_GPU;
    delete dph_image_GPU;
    delete abs_image_GPU;
    delete vis_image_GPU;*/

    return 0;
}