# dph_abs_vis_images_calc_Momose_approach_GPU_03

I posted here C++ Cuda C code to calculate three kind of images: differential phase images, absorption images, and visibility images. The raw data for images calculation are M_fg = 5 foreground (fg) images nad M_bg = 5 background (bg) images. The raw input foreground and background images are encoded as unsigned 16 bit integers. The size of input images is 1536 x 512 pixels (H x V). The program has input path for folder with foreground images <path_to_fg_folder>. The program has input path for folder with background images <path_to_bg_folder>. The final differential phase, absorption, and visibility images are saved to the <path_to_output_folder>. The foreground images are organized in folders from folder 000001 to folder 011200. Therefore, <no_subfolder_fg_initial = 1> and <no_subfolder_fg_final = 11200>. The background images are organized in folders from folder 000001 to folder 011200. Therefore, <no_subfolder_bg_initial = 1> and <no_subfolder_bg_final = 11200>. The input raw images for foregorund  are saved ino the <image_buffer_1D_fg> on the host. The content of <image_buffer_1D_fg> is copied to the <image_buffer_1D_fg_GPU> on the device or GPU. The input raw images for background  are saved ino the <image_buffer_1D_bg> on the host. The content of <image_buffer_1D_bg> is copied to the <image_buffer_1D_bg_GPU> on the device or GPU. The calculation is running on the GPU. The diff
