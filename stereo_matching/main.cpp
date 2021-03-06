#include "main.h"
#include <chrono>

struct Image
{
 std::vector<unsigned char> pixel;
 unsigned width, height, cross;
};

std::string  sampler = "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; ";
std::string LoadKernel(const char* name)
{
 std::ifstream in(name);
 std::string result(
  (std::istreambuf_iterator<char>(in)),
  std::istreambuf_iterator<char>());
 return result;
}

cl_program CreateProgram(const std::string& source, cl_context context)
{
 size_t lengths =  source.size();
 const char* sources = source.data();
 cl_program program = clCreateProgramWithSource(context, 1, &sources, &lengths,nullptr);
 return program;
}
void ErCheck(cl_int error) {
 if (error != CL_SUCCESS)
  printf("OpenCL error executing kernel: %d\n", error);
}

//This function returns execution time[ms] of two events
double compute_time(cl_event* clevent, char* name) 
{
 int size = sizeof(clevent);
 cl_ulong time_start[2], time_end[2];
 double total_time;

 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_START, sizeof(time_start[0]), &time_start[0], nullptr);
 clGetEventProfilingInfo(clevent[1], CL_PROFILING_COMMAND_START, sizeof(time_start[1]), &time_start[1], nullptr);
 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_END, sizeof(time_end[0]), &time_end[0], nullptr);
 clGetEventProfilingInfo(clevent[1], CL_PROFILING_COMMAND_END, sizeof(time_end[1]), &time_end[1], nullptr);

 total_time = (time_end[0] > time_end[1] ? time_end[0] : time_end[1]) - (time_start[0] < time_start[1] ? time_start[0] : time_start[1]);
 //printf("%s %0.3f ms", name, (total_time / 1000000.0));
 return total_time / 1000000.0;
}

double compute_time(cl_event* clevent, char* name, int i)
{
 int size = sizeof(clevent);
 cl_ulong time_start[2], time_end[2];
 double total_time;

 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_START, sizeof(time_start[0]), &time_start[0], nullptr);
 clGetEventProfilingInfo(clevent[i], CL_PROFILING_COMMAND_START, sizeof(time_start[1]), &time_start[1], nullptr);
 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_END, sizeof(time_end[0]), &time_end[0], nullptr);
 clGetEventProfilingInfo(clevent[i], CL_PROFILING_COMMAND_END, sizeof(time_end[1]), &time_end[1], nullptr);

 total_time = (time_end[0] > time_end[1] ? time_end[0] : time_end[1]) - (time_start[0] < time_start[1] ? time_start[0] : time_start[1]);
 //printf("%s %0.3f ms", name, (total_time / 1000000.0));
 return total_time / 1000000.0;
}

double compute_time(cl_event clevent, char* name)
{
 cl_ulong time_start, time_end;
 double total_time;

 clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
 clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);

 total_time = time_end - time_start;
 //printf("%s %0.3f ms", name, (total_time / 1000000.0));
 return total_time / 1000000.0;
}

int main()
{
 FILE* to_file;
 FILE* pictures;
 //CROSS-BASED METHOD
 bool again = true;
 //Reading all kernels
 std::string kernels[21] = {
  sampler + LoadKernel("kernels/median.cl"),
  LoadKernel("kernels/cross.cl"),
  LoadKernel("kernels/aggregation.cl"),
  LoadKernel("kernels/integral_h.cl"),
  LoadKernel("kernels/oii_hcross.cl"),
  LoadKernel("kernels/integral_v.cl"),
  LoadKernel("kernels/oii_vcross.cl"),
  LoadKernel("kernels/init_disparity.cl"),
  LoadKernel("kernels/disparity.cl"),
  LoadKernel("kernels/asw_hsupport.cl"),
  LoadKernel("kernels/asw_vsupport.cl"),
  LoadKernel("kernels/asw_vcost.cl"),
  LoadKernel("kernels/asw_cost.cl"),
  LoadKernel("kernels/asw_wta.cl"),
  LoadKernel("kernels/asw_aggr.cl"),
  LoadKernel("kernels/asw_vcost_aggregation.cl"),
  LoadKernel("kernels/asw_hcost_aggregation.cl"),
  LoadKernel("kernels/asw_refinement_v.cl"),
  LoadKernel("kernels/asw_refinement_h.cl"),
  LoadKernel("kernels/asw_wta_ref.cl"),
  LoadKernel("kernels/consist.cl")
 };

 int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

 const char* load_kernel[21];
 for (int i = 0; i < num_kernels; i++)
 {
  load_kernel[i] = kernels[i].data();
 }

 while (again) {
  //menu
  cl_uint total_device_number = 0;
  cl_uint platformIdCount = 0;
  clGetPlatformIDs(0, nullptr, &platformIdCount);
  cl_uint* deviceIdCount = new cl_uint[platformIdCount];
  std::vector<cl_platform_id> platformIds(platformIdCount);
  clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);
  for (int i = 0; i < platformIdCount; i++)
  {
   clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount[i]);
   total_device_number += deviceIdCount[i];
  }
  std::vector<cl_device_id> deviceIds(total_device_number);
  char buffer[10240];
  std::string s_path_out;
  //read image path
  char left_path[20][100];
  char right_path[20][100];
  fopen_s(&pictures, "pics.txt", "r");
  int lines;
  int img_i = 0;
  while (1)
  {
   lines = fscanf_s(pictures, "%s\n", &left_path[img_i]);
   lines = fscanf_s(pictures, "%s\n", &right_path[img_i]);

   if (lines == EOF)
    break;
   img_i++;
  }
  fclose(pictures);

  //get folder name from path
  std::string* folder_name = new std::string[img_i];
  for (int i = 0; i < img_i; i++) {
   std::string str(left_path[i]);
   std::size_t pos = str.find('/');
   folder_name[i] = str.substr(0, pos);
  }

  for (int platform_id = 0; platform_id < platformIdCount; platform_id++)
  {
   clGetDeviceIDs(platformIds[platform_id], CL_DEVICE_TYPE_ALL, deviceIdCount[platform_id], deviceIds.data(), nullptr);

   for (int device_id = 0; device_id < deviceIdCount[platform_id]; device_id++)
   {
    clGetDeviceInfo(deviceIds.data()[device_id], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    printf("\t- Device name: %s\n", buffer);
    fopen_s(&to_file, buffer, "w");

    cl_int error = CL_SUCCESS;
    clGetDeviceIDs(platformIds[platform_id], CL_DEVICE_TYPE_ALL, deviceIdCount[platform_id], deviceIds.data(), nullptr);
    const cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platform_id]),  0, 0 };
    cl_context context = clCreateContext(contextProperties, 1, &deviceIds.data()[device_id], nullptr, nullptr, &error);
    clGetDeviceInfo(deviceIds.data()[device_id], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);

    for (int current_image = 0; current_image < img_i; current_image++)
    {
     int k = 6;
     int r = 7;

     fprintf(to_file,"\n%s - %s\n" ,buffer, folder_name[current_image].c_str());
     printf("\n%s\n", folder_name[current_image].c_str());
     fprintf(to_file, "id\tmedL_solo\tmedR_solo\tmed_full\tcross_h\tcross_v\tcross_full\taggregation\tintegral_h\taggr_h\tintegral_v\taggr_v\tinit_disp\tfinal_disp\tcross method total\t\t\taggr\tsupp_w\tv_aggr_mean\th_aggr_mean\ttotal aggregation\twta\tconsistency\tv_ref_mean_L\tv_ref_mean_R\th_ref_mean_L\th_ref_mean_R\twta_mean_LR\tconsistency_mean\ttotal refinement\tmedian\ttotal WTA method");

     Image imgL;
     lodepng::decode(imgL.pixel, imgL.width, imgL.height, left_path[current_image]);
     Image imgR;
     lodepng::decode(imgR.pixel, imgR.width, imgR.height, right_path[current_image]);

     //variables
     static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
     Image result[4] = { imgL, imgR, imgL, imgL };
     std::size_t local[3] = { 3, 3, 1 };
     std::size_t local_asw[3] = { 48, 1, 1 };
     size_t numLocalGroups[] = { ceil(result[0].width / local[0]), ceil(result[0].height / local[1]) };
     int sizei[3] = { result[0].width, result[0].height, 1 };
     //global work size
     std::size_t size[3] = { result[0].width, result[0].height, 1 };
     size_t globalSize[] = { local[0] * numLocalGroups[0], local[1] * numLocalGroups[1] };
     //defines offset, where should we read, z-buffer i zero for 2Ds
     std::size_t origin[3] = { 0,0,0 };
     //whats the size of rect we gonna read
     std::size_t region[3] = { result[0].width, result[0].height, 1 };
     std::size_t integralh_size[3] = { result[0].height, 61, 1 };
     std::size_t hcross_size[3] = { result[0].width, result[0].height, 61 };
     std::size_t integralv_size[3] = { result[0].width, 61, 1 };
     std::size_t vcross_size[3] = { result[0].width, result[0].height, 61 };



     // Create a program from source
     cl_program program = clCreateProgramWithSource(context, num_kernels, load_kernel, nullptr, nullptr);
     ErCheck(clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr));
     cl_command_queue queue = clCreateCommandQueue(context, deviceIds.data()[device_id], CL_QUEUE_PROFILING_ENABLE, &error);
    for (int run = 0; run < 10; run++)
    {
   

      fprintf(to_file, "\nRun %d \t", run + 1);
     
      //oii
      cl_kernel median = clCreateKernel(program, "Median", &error);
      cl_kernel cross = clCreateKernel(program, "Cross", &error);
      cl_kernel aggregation = clCreateKernel(program, "Aggregation", &error);
      cl_kernel integral_h = clCreateKernel(program, "Integral_h", &error);
      cl_kernel oii_hcross = clCreateKernel(program, "Oii_hcross", &error);
      cl_kernel integral_v = clCreateKernel(program, "Integral_v", &error);
      cl_kernel oii_vcross = clCreateKernel(program, "Oii_vcross", &error);
      cl_kernel init_disparity = clCreateKernel(program, "Init_disparity", &error);
      cl_kernel final_disparity = clCreateKernel(program, "Disparity", &error);
      //asw refinement
      cl_kernel asw_vsupport = clCreateKernel(program, "asw_vSupport", &error);
      cl_kernel asw_hsupport = clCreateKernel(program, "asw_hSupport", &error);
      cl_kernel asw_vcost = clCreateKernel(program, "asw_vCost", &error);
      cl_kernel asw_cost = clCreateKernel(program, "asw_Cost", &error);
      cl_kernel asw_disp = clCreateKernel(program, "asw_WTA", &error);
      cl_kernel asw_aggr = clCreateKernel(program, "asw_Aggr", &error);
      cl_kernel asw_vaggregation_kernel = clCreateKernel(program, "asw_vCostAggregation", &error);
      cl_kernel asw_haggregation_kernel = clCreateKernel(program, "asw_hCostAggregation", &error);
      cl_kernel asw_ref_v = clCreateKernel(program, "asw_ref_v", &error);
      cl_kernel asw_ref_h = clCreateKernel(program, "asw_ref_h", &error);
      cl_kernel asw_wta_ref = clCreateKernel(program, "asw_WTA_REF", &error);
      cl_kernel consist_kernel = clCreateKernel(program, "Constistency", &error);

      cl_mem inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
      cl_mem inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
      cl_mem median_l = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem median_r = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[1].width, result[1].height, 0, nullptr, &error);
      //width, height x 4 dimension for every pixel: H-, H+, V-, V+
      cl_mem cross_l = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[0].width * result[0].height * 4, nullptr, &error);
      cl_mem cross_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[1].width * result[1].height * 4, nullptr, &error);
      //width, height x disparity <0:60>
      cl_mem cost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      cl_mem im_size = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 3, sizei, &error);
      cl_mem temp_cost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      cl_mem disparity = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem f_disparity = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem cross_method = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);

      cl_event event_median[2];
      cl_event event_cross[2];
      cl_event event_aggro;
      cl_event event_integralh;
      cl_event event_oiih;
      cl_event event_integralv;
      cl_event event_oiiv;
      cl_event event_initd;
      cl_event event_disp;
      cl_event event_final_cross;

      //median filter
      printf("\nCross-Based Local Stereo Matching Using Orthogonal Integral Images ");
      printf("\n---Working...\nMedian filter..  ");
      clSetKernelArg(median, 0, sizeof(cl_mem), &inputImage_l);
      clSetKernelArg(median, 1, sizeof(cl_mem), &median_l);
      clEnqueueNDRangeKernel(queue, median, 2, nullptr, globalSize, nullptr, 0, nullptr, &event_median[0]);
      clFlush(queue);

      clSetKernelArg(median, 0, sizeof(cl_mem), &inputImage_r);
      clSetKernelArg(median, 1, sizeof(cl_mem), &median_r);
      clEnqueueNDRangeKernel(queue, median, 2, nullptr, globalSize, nullptr, 0, nullptr, &event_median[1]);
      clFlush(queue);

      //Cross construction
      printf("\nCross construction.. ");
      clSetKernelArg(cross, 0, sizeof(cl_mem), &median_l);
      clSetKernelArg(cross, 1, sizeof(cl_mem), &cross_l);
      ErCheck(clEnqueueNDRangeKernel(queue, cross, 2, nullptr, size, nullptr, 1, &event_median[0], &event_cross[0]));
      clFlush(queue);

      clSetKernelArg(cross, 0, sizeof(cl_mem), &median_r);
      clSetKernelArg(cross, 1, sizeof(cl_mem), &cross_r);
      ErCheck(clEnqueueNDRangeKernel(queue, cross, 2, nullptr, size, nullptr, 1, &event_median[1], &event_cross[1]));
      clFlush(queue);

      //aggregation cost
      printf("\nAggregation cost.. ");
      clSetKernelArg(aggregation, 0, sizeof(cl_mem), &median_l);
      clSetKernelArg(aggregation, 1, sizeof(cl_mem), &median_r);
      clSetKernelArg(aggregation, 2, sizeof(cl_mem), &cost);
      ErCheck(clEnqueueNDRangeKernel(queue, aggregation, 2, nullptr, size, nullptr, 2, event_median, &event_aggro));
      clFlush(queue);

      //horizontal integral image
      printf("\nOII - horizontal integral.. ");
      clSetKernelArg(integral_h, 0, sizeof(cl_mem), &cost);
      clSetKernelArg(integral_h, 1, sizeof(cl_mem), &im_size);
      ErCheck(clEnqueueNDRangeKernel(queue, integral_h, 2, nullptr, integralh_size, nullptr, 1, &event_aggro, &event_integralh));
      clFlush(queue);
      //horizontal image cross cost

      cl_event event_integralh_cross[3] = { event_integralh, event_cross[0], event_cross[1] };
      printf("\nOII - horizontal cross.. ");
      clSetKernelArg(oii_hcross, 0, sizeof(cl_mem), &cross_l);
      clSetKernelArg(oii_hcross, 1, sizeof(cl_mem), &cross_r);
      clSetKernelArg(oii_hcross, 2, sizeof(cl_mem), &cost);
      clSetKernelArg(oii_hcross, 3, sizeof(cl_mem), &temp_cost);
      clSetKernelArg(oii_hcross, 4, sizeof(cl_mem), &im_size);
      ErCheck(clEnqueueNDRangeKernel(queue, oii_hcross, 3, nullptr, hcross_size, nullptr, 3, event_integralh_cross, &event_oiih));
      clFlush(queue);

      //vertical integral image
      printf("\nOII - vertical integral.. ");
      clSetKernelArg(integral_v, 0, sizeof(cl_mem), &temp_cost);
      clSetKernelArg(integral_v, 1, sizeof(cl_mem), &im_size);
      ErCheck(clEnqueueNDRangeKernel(queue, integral_v, 2, nullptr, integralv_size, nullptr, 1, &event_oiih, &event_integralv));
      clFlush(queue);

      //vertical image cross cost
      printf("\nOII - vertical cross.. ");
      clSetKernelArg(oii_vcross, 0, sizeof(cl_mem), &cross_l);
      clSetKernelArg(oii_vcross, 1, sizeof(cl_mem), &cross_r);
      clSetKernelArg(oii_vcross, 2, sizeof(cl_mem), &temp_cost);
      clSetKernelArg(oii_vcross, 3, sizeof(cl_mem), &cost);
      clSetKernelArg(oii_vcross, 4, sizeof(cl_mem), &im_size);
      ErCheck(clEnqueueNDRangeKernel(queue, oii_vcross, 3, nullptr, vcross_size, nullptr, 1, &event_integralv, &event_oiiv));
      clFlush(queue);

      //initial disparity
      printf("\nInitial disparity.. ");
      clSetKernelArg(init_disparity, 0, sizeof(cl_mem), &cost);
      clSetKernelArg(init_disparity, 1, sizeof(cl_mem), &disparity);
      ErCheck(clEnqueueNDRangeKernel(queue, init_disparity, 2, nullptr, size, nullptr, 1, &event_oiiv, &event_initd));
      clFlush(queue);

      //final disparity
      printf("\nFinal disparity.. ");
      clSetKernelArg(final_disparity, 0, sizeof(cl_mem), &disparity);
      clSetKernelArg(final_disparity, 1, sizeof(cl_mem), &cross_l);
      clSetKernelArg(final_disparity, 2, sizeof(cl_mem), &f_disparity);
      ErCheck(clEnqueueNDRangeKernel(queue, final_disparity, 2, nullptr, size, nullptr, 1, &event_initd, &event_disp));
      clFlush(queue);

      clSetKernelArg(median, 0, sizeof(cl_mem), &f_disparity);
      clSetKernelArg(median, 1, sizeof(cl_mem), &cross_method);
      ErCheck(clEnqueueNDRangeKernel(queue, median, 2, nullptr, globalSize, nullptr, 1, &event_disp, &event_final_cross));
      clFlush(queue);
      clFinish(queue);
       clEnqueueReadImage(queue, disparity, CL_TRUE, origin, region, 0, 0, result[2].pixel.data(), 1, &event_final_cross, nullptr);
       s_path_out = folder_name[current_image] + "/cross_based_initial.png";
       lodepng::encode(s_path_out, result[2].pixel, result[1].width, result[1].height);

       clEnqueueReadImage(queue, cross_method, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 1, &event_final_cross, nullptr);
       s_path_out = folder_name[current_image] + "/cross_based_disparity.png";
       lodepng::encode(s_path_out, result[1].pixel, result[1].width, result[1].height);

       clEnqueueReadImage(queue, median_l, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 1, &event_final_cross, nullptr);
       s_path_out = folder_name[current_image] + "/median.png";
       lodepng::encode(s_path_out, result[0].pixel, result[1].width, result[1].height);
    


      //clReleaseMemObject(inputImage_l);
      //clReleaseMemObject(inputImage_r);
      //clReleaseMemObject(median_l);
      //clReleaseMemObject(median_r);
      //clReleaseMemObject(im_size);
      //clReleaseMemObject(f_disparity);
      clReleaseMemObject(cross_r);
      clReleaseMemObject(cost);
      clReleaseMemObject(temp_cost);
      clReleaseMemObject(cross_l);
      clReleaseMemObject(disparity);

      //clReleaseKernel(median);
      clReleaseKernel(cross);
      clReleaseKernel(aggregation);
      clReleaseKernel(integral_h);
      clReleaseKernel(oii_hcross);
      clReleaseKernel(integral_v);
      clReleaseKernel(oii_vcross);
      clReleaseKernel(init_disparity);
      clReleaseKernel(final_disparity);
    
      //Get measured time data
      fprintf(to_file,"%0.3f\t", compute_time(event_median[0], "\nSingle Median one: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_median[1], "\nSingle Median two: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_median, "\nMedian filter: "));

      fprintf(to_file, "%0.3f\t", compute_time(event_cross[0], "\nSingle Cross one: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_cross[1], "\nSingle Cross two: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_cross, "\nCross: "));

      fprintf(to_file, "%0.3f\t", compute_time(event_aggro, "\nAggregation: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_integralh, "\nHorizontal integral: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_oiih, "\nHorizontal cost matching: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_integralv, "\nVertical integral: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_oiiv, "\nVertical cost matching: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_initd, "\nInitial disparity: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_disp, "\nFinal disparity: "));
      cl_event total_event[2] = { event_median[0], event_disp };
      fprintf(to_file, "%0.3f\t", compute_time(total_event, "\n-- - Total time in milliseconds = "));
      fprintf(to_file, "\t\t");
      //asw refinement
      std::size_t asw_support_size[3] = { result[0].width, result[0].height, 33 };
      std::size_t asw_cost_size[3] = { result[0].width, result[0].height, 61 };

      cl_event event_vaggr[10];
      cl_event event_haggr[10];
      cl_event event_asw_cost;
      cl_event event_asw[2];
      cl_event event_asw_costh;
      cl_event event_asw_wta;
      cl_event event_image;
      cl_event event_aggr;
      cl_event event_consist;
      cl_event event_consist_refin[6];
      cl_event event_support[4];
      cl_event event_vreff_left[6];
      cl_event event_hreff_left[6];
      cl_event event_vreff_right[6];
      cl_event event_hreff_right[6];
      cl_event event_wta_ref_left[6];
      cl_event event_wta_ref_right[6];

      cl_mem asw_cost_buffer[2];
      cl_mem asw_initcost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      cl_mem asw_vsupp_left = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 33, nullptr, &error);
      cl_mem asw_hsupp_left = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 33, nullptr, &error);
      cl_mem asw_vsupp_right = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 33, nullptr, &error);
      cl_mem asw_hsupp_right = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 33, nullptr, &error);
      asw_cost_buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      asw_cost_buffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      cl_mem asw_left_wta = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem asw_right_wta = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem asw_d_est_reference = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 1, nullptr, &error);
      cl_mem asw_d_est_target = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 1, nullptr, &error);
      cl_mem asw_confidence_reference = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 1, nullptr, &error);
      cl_mem asw_confidence_target = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 1, nullptr, &error);
      cl_mem asw_vref_l = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_vref_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_href_l = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_href_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_wta_ref_l = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_wta_ref_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 2, nullptr, &error);
      cl_mem asw_denom = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * result[1].width * result[1].height * 61, nullptr, &error);
      cl_mem consistency_error = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem consistency_error_red = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
      cl_mem consistency_error_red_reff = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);


      printf("\n\n\nReal - time Stereo Matching on CUDA using an Iterative Refinement Method for Adaptive Support - Weight Correspondences");
      printf("\n---Working...\nRaw cost aggregation..  ");
      //per pixel 
      clSetKernelArg(asw_aggr, 0, sizeof(cl_mem), &inputImage_l);
      clSetKernelArg(asw_aggr, 1, sizeof(cl_mem), &inputImage_r);
      clSetKernelArg(asw_aggr, 2, sizeof(cl_mem), &asw_initcost);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_aggr, 2, nullptr, size, nullptr, 0, nullptr, &event_aggr));


      printf("\ngestalt principle - support area.. ");
      clSetKernelArg(asw_vsupport, 0, sizeof(cl_mem), &inputImage_l);
      clSetKernelArg(asw_vsupport, 1, sizeof(cl_mem), &asw_vsupp_left);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_vsupport, 3, nullptr, asw_support_size, nullptr, 1, &event_aggr, &event_support[0]));

      clSetKernelArg(asw_hsupport, 0, sizeof(cl_mem), &inputImage_l);
      clSetKernelArg(asw_hsupport, 1, sizeof(cl_mem), &asw_hsupp_left);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_hsupport, 3, nullptr, asw_support_size, nullptr, 1, &event_aggr, &event_support[1]));

      clSetKernelArg(asw_vsupport, 0, sizeof(cl_mem), &inputImage_r);
      clSetKernelArg(asw_vsupport, 1, sizeof(cl_mem), &asw_vsupp_right);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_vsupport, 3, nullptr, asw_support_size, nullptr, 1, &event_aggr, &event_support[2]));

      clSetKernelArg(asw_hsupport, 0, sizeof(cl_mem), &inputImage_r);
      clSetKernelArg(asw_hsupport, 1, sizeof(cl_mem), &asw_hsupp_right);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_hsupport, 3, nullptr, asw_support_size, nullptr, 1, &event_aggr, &event_support[3]));

      //cost - V
      printf("\nCost aggregation.. ");
      cl_mem * temp = &asw_initcost;
      cl_event* ptr = event_support;
      int event_number = 4;
      //loop r times
      for (int i = 0; i < r; i++)
      {
       clSetKernelArg(asw_vaggregation_kernel, 0, sizeof(cl_mem), &inputImage_l);
       clSetKernelArg(asw_vaggregation_kernel, 1, sizeof(cl_mem), &asw_vsupp_left);
       clSetKernelArg(asw_vaggregation_kernel, 2, sizeof(cl_mem), &asw_vsupp_right);
       clSetKernelArg(asw_vaggregation_kernel, 3, sizeof(cl_mem), temp);
       clSetKernelArg(asw_vaggregation_kernel, 4, sizeof(cl_mem), &asw_denom);
       clSetKernelArg(asw_vaggregation_kernel, 5, sizeof(cl_mem), &asw_cost_buffer[0]);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_vaggregation_kernel, 3, nullptr, asw_cost_size, nullptr, event_number, ptr, &event_vaggr[i]));


       clSetKernelArg(asw_haggregation_kernel, 0, sizeof(cl_mem), &inputImage_l);
       clSetKernelArg(asw_haggregation_kernel, 1, sizeof(cl_mem), &asw_hsupp_left);
       clSetKernelArg(asw_haggregation_kernel, 2, sizeof(cl_mem), &asw_hsupp_right);
       clSetKernelArg(asw_haggregation_kernel, 3, sizeof(cl_mem), &asw_cost_buffer[0]);
       clSetKernelArg(asw_haggregation_kernel, 4, sizeof(cl_mem), &asw_denom);
       clSetKernelArg(asw_haggregation_kernel, 5, sizeof(cl_mem), &asw_cost_buffer[1]);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_haggregation_kernel, 3, nullptr, asw_cost_size, nullptr, 1, &event_vaggr[i], &event_haggr[i]));


       temp = &asw_cost_buffer[1];
       ptr = &event_haggr[i];
       event_number = 1;
      }

      printf("\nWTA.. ");
      //WTA
      clSetKernelArg(asw_disp, 0, sizeof(cl_mem), &asw_cost_buffer[1]);
      clSetKernelArg(asw_disp, 1, sizeof(cl_mem), &asw_left_wta);
      clSetKernelArg(asw_disp, 2, sizeof(cl_mem), &asw_d_est_reference);
      clSetKernelArg(asw_disp, 3, sizeof(cl_mem), &asw_d_est_target);
      clSetKernelArg(asw_disp, 4, sizeof(cl_mem), &asw_right_wta);
      clSetKernelArg(asw_disp, 5, sizeof(cl_mem), &asw_confidence_reference);
      clSetKernelArg(asw_disp, 6, sizeof(cl_mem), &asw_confidence_target);
      ErCheck(clEnqueueNDRangeKernel(queue, asw_disp, 2, nullptr, size, nullptr, r, event_haggr, &event_asw_wta));


      printf("\nConsistency check.. ");
      //Consistency check
      clSetKernelArg(consist_kernel, 0, sizeof(cl_mem), &asw_left_wta);
      clSetKernelArg(consist_kernel, 1, sizeof(cl_mem), &asw_right_wta);
      clSetKernelArg(consist_kernel, 2, sizeof(cl_mem), &asw_confidence_reference);
      clSetKernelArg(consist_kernel, 3, sizeof(cl_mem), &asw_confidence_target);
      clSetKernelArg(consist_kernel, 4, sizeof(cl_mem), &consistency_error);
      clSetKernelArg(consist_kernel, 5, sizeof(cl_mem), &consistency_error_red);
      ErCheck(clEnqueueNDRangeKernel(queue, consist_kernel, 2, nullptr, size, nullptr, 1, &event_asw_wta, &event_consist));


      printf("\nRefinement..");
      temp = &consistency_error;
      cl_mem * temp_target = &asw_right_wta;
      ptr = &event_consist;
      cl_event ptr_doubled[2];
      for (int i = 0; i < k; i++)
      {
       //vertical refinement for left image
       clSetKernelArg(asw_ref_v, 0, sizeof(cl_mem), &inputImage_l);
       clSetKernelArg(asw_ref_v, 1, sizeof(cl_mem), temp);
       clSetKernelArg(asw_ref_v, 2, sizeof(cl_mem), &asw_confidence_reference);
       clSetKernelArg(asw_ref_v, 3, sizeof(cl_mem), &asw_vref_l);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_ref_v, 2, nullptr, size, nullptr, 1, ptr, &event_vreff_left[i]));


       //vertical refinement for right image
       clSetKernelArg(asw_ref_v, 0, sizeof(cl_mem), &inputImage_r);
       clSetKernelArg(asw_ref_v, 1, sizeof(cl_mem), temp_target);
       clSetKernelArg(asw_ref_v, 2, sizeof(cl_mem), &asw_confidence_target);
       clSetKernelArg(asw_ref_v, 3, sizeof(cl_mem), &asw_vref_r);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_ref_v, 2, nullptr, size, nullptr, 1, ptr, &event_vreff_right[i]));


       //horizontal refinement for left image
       clSetKernelArg(asw_ref_h, 0, sizeof(cl_mem), &inputImage_l);
       clSetKernelArg(asw_ref_h, 1, sizeof(cl_mem), &asw_confidence_reference);
       clSetKernelArg(asw_ref_h, 2, sizeof(cl_mem), &asw_vref_l);
       clSetKernelArg(asw_ref_h, 3, sizeof(cl_mem), &asw_href_l);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_ref_h, 2, nullptr, size, nullptr, 1, &event_vreff_left[i], &event_hreff_left[i]));


       //horizontal refinement for right image
       clSetKernelArg(asw_ref_h, 0, sizeof(cl_mem), &inputImage_r);
       clSetKernelArg(asw_ref_h, 1, sizeof(cl_mem), &asw_confidence_target);
       clSetKernelArg(asw_ref_h, 2, sizeof(cl_mem), &asw_vref_r);
       clSetKernelArg(asw_ref_h, 3, sizeof(cl_mem), &asw_href_r);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_ref_h, 2, nullptr, size, nullptr, 1, &event_vreff_right[i], &event_hreff_right[i]));


       //WTA left image
       clSetKernelArg(asw_wta_ref, 0, sizeof(cl_mem), &asw_cost_buffer[1]);
       clSetKernelArg(asw_wta_ref, 1, sizeof(cl_mem), &asw_href_l);
       clSetKernelArg(asw_wta_ref, 2, sizeof(cl_mem), &asw_href_r);
       clSetKernelArg(asw_wta_ref, 3, sizeof(cl_mem), &asw_left_wta);
       clSetKernelArg(asw_wta_ref, 4, sizeof(cl_mem), &asw_right_wta);
       clSetKernelArg(asw_wta_ref, 5, sizeof(cl_mem), &asw_wta_ref_l);
       clSetKernelArg(asw_wta_ref, 6, sizeof(cl_mem), &asw_wta_ref_r);
       clSetKernelArg(asw_wta_ref, 7, sizeof(cl_mem), &asw_confidence_reference);
       clSetKernelArg(asw_wta_ref, 8, sizeof(cl_mem), &asw_confidence_target);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_wta_ref, 2, nullptr, size, nullptr, 1, &event_hreff_left[i], &event_wta_ref_left[i]));
       /*
       //WTA right image
       clSetKernelArg(asw_wta_ref, 0, sizeof(cl_mem), &asw_cost_buffer[1]);
       clSetKernelArg(asw_wta_ref, 1, sizeof(cl_mem), &asw_href_r);
       clSetKernelArg(asw_wta_ref, 2, sizeof(cl_mem), &asw_right_wta);
       clSetKernelArg(asw_wta_ref, 3, sizeof(cl_mem), &asw_wta_ref_r);
       clSetKernelArg(asw_wta_ref, 4, sizeof(cl_mem), &asw_confidence_target);
       ErCheck(clEnqueueNDRangeKernel(queue, asw_wta_ref, 2, nullptr, size, nullptr, 1, &event_hreff_right[i], &event_wta_ref_right[i]));
       */
       //ptr_doubled[0] = event_wta_ref_left[i];
      // ptr_doubled[1] = event_wta_ref_right[i];
       //Consistency check
       clSetKernelArg(consist_kernel, 0, sizeof(cl_mem), &asw_left_wta);
       clSetKernelArg(consist_kernel, 1, sizeof(cl_mem), &asw_right_wta);
       clSetKernelArg(consist_kernel, 2, sizeof(cl_mem), &asw_confidence_reference);
       clSetKernelArg(consist_kernel, 3, sizeof(cl_mem), &asw_confidence_target);
       clSetKernelArg(consist_kernel, 4, sizeof(cl_mem), &consistency_error);
       clSetKernelArg(consist_kernel, 5, sizeof(cl_mem), &consistency_error_red_reff);
       ErCheck(clEnqueueNDRangeKernel(queue, consist_kernel, 2, nullptr, size, nullptr, 1, &event_wta_ref_left[i], &event_consist_refin[i]));


       ptr = &event_consist_refin[i];
       temp = &consistency_error;
       temp_target = &asw_right_wta;
      }
      
      printf("\nPost Processing..\n\n");
      clSetKernelArg(median, 0, sizeof(cl_mem), &consistency_error);
      clSetKernelArg(median, 1, sizeof(cl_mem), &f_disparity);
      clEnqueueNDRangeKernel(queue, median, 2, nullptr, size, nullptr, 1, ptr, &event_median[0]);

      clEnqueueReadImage(queue, f_disparity, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 1, &event_median[0], nullptr);
      s_path_out = folder_name[current_image] + "/asw_disparity.png";
      lodepng::encode(s_path_out, result[0].pixel, result[1].width, result[1].height);

      clEnqueueReadImage(queue, consistency_error_red, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 1, &event_median[0], nullptr);
      s_path_out = folder_name[current_image] + "/asw_consistency_pre-reff.png";
      lodepng::encode(s_path_out, result[1].pixel, result[1].width, result[1].height);

      clEnqueueReadImage(queue, consistency_error_red_reff, CL_TRUE, origin, region, 0, 0, result[2].pixel.data(), 1, &event_median[0], nullptr);
      s_path_out = folder_name[current_image] + "/asw_consistency_post-reff.png";
      lodepng::encode(s_path_out, result[2].pixel, result[1].width, result[1].height);
      
      //printf("\n\n---Results for %s: ", buffer);
      fprintf(to_file, "%0.3f\t", compute_time(event_aggr, "\nRaw Aggregation: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_support, "\nSupport weight: ", 3));

      float mean = 0;

      //printf("\nVertical - cost aggregation LOOP");
      for (int i = 0; i < r; i++)
      {
       mean += compute_time(event_vaggr[i], "\n---Vertical cost aggregation: ");
      }
      fprintf(to_file, "%0.3f\t", mean / r);
      //printf("\nMean vertical cost aggregation: %f\n", mean / r);

      mean = 0;
      //printf("\nHorizontal - cost aggregation LOOP");

      for (int i = 0; i < r; i++)
      {
      
       mean += compute_time(event_haggr[i], "\n---Horizontal cost aggregation: ");
      }

      fprintf(to_file, "%0.3f\t", mean / r);

      cl_event total_aggr[2] = { event_vaggr[0], event_haggr[r - 1] };
      fprintf(to_file, "%0.3f\t", compute_time(total_aggr, "\nTotal Aggregation time: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_asw_wta, "\nWTA: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_consist, "\nConsistency Check: "));

      mean = 0;
      float mean2 = 0;
      for (int i = 0; i < k; i++)
      {
       mean += compute_time(event_vreff_left[i], "\n---Vertical refinement left: ");
       mean2 += compute_time(event_vreff_right[i], "\n---Vertical refinement right: ");
      }
      
      fprintf(to_file, "%0.3f\t", mean / ( k));
      fprintf(to_file, "%0.3f\t", mean2 / ( k));

      mean = 0;
      mean2 = 0;
      for (int i = 0; i < k; i++)
      {
       mean += compute_time(event_hreff_left[0], "\n---Horizontal refinement left: ");
       mean2 += compute_time(event_hreff_right[0], "\n---Horizontal refinement right: ");
      }
      
      fprintf(to_file, "%0.3f\t", mean / ( k));
      fprintf(to_file, "%0.3f\t", mean2 / ( k));

      mean = 0;     
      mean2 = 0;
      for (int i = 0; i < k; i++)
      {       
       mean += compute_time(event_wta_ref_left[0], "\n---WTA refinement left: ");
      // mean2 += compute_time(event_wta_ref_right[0], "\n---WTA refinement right: ");
      }
     
      fprintf(to_file, "%0.3f\t", mean / ( k));
     // fprintf(to_file, "%0.3f\t", mean2 / ( k));

      mean = 0;
      for (int i = 0; i < k; i++)
      {
       mean += compute_time(event_consist_refin[i], "\n---Consistency check: ");
      }
      fprintf(to_file, "%0.3f\t", mean / (k));

      cl_event total_ref[2] = { event_vreff_left[0], event_consist_refin[k - 1] };
      fprintf(to_file, "%0.3f\t", compute_time(total_ref, "\nTotal Refinement time: "));
      fprintf(to_file, "%0.3f\t", compute_time(event_median[0], "\nMedian time: "));

      cl_event wta_total_event[2] = { event_aggr, event_median[0] };
      fprintf(to_file, "%0.3f\t", compute_time(wta_total_event, "\n--- Total time in milliseconds = "));
      //compute_time(event_asw, "\n--- testtime in milliseconds = ");


      clReleaseMemObject(inputImage_l);
      clReleaseMemObject(inputImage_r);
      clReleaseMemObject(asw_cost_buffer[0]);
      clReleaseMemObject(asw_cost_buffer[1]);
      clReleaseMemObject(asw_left_wta);
      clReleaseMemObject(asw_right_wta);
      clReleaseMemObject(asw_vsupp_left);
      clReleaseMemObject(asw_vsupp_right);
      clReleaseMemObject(asw_hsupp_right);
      clReleaseMemObject(asw_hsupp_left);
      clReleaseMemObject(im_size);
      clReleaseMemObject(asw_vref_l);
      clReleaseMemObject(asw_href_l);
      clReleaseMemObject(asw_vref_r);
      clReleaseMemObject(asw_href_r);
      clReleaseMemObject(asw_initcost);
      clReleaseMemObject(asw_wta_ref_l);
      clReleaseMemObject(asw_wta_ref_r);
      clReleaseMemObject(asw_d_est_target);
      clReleaseMemObject(consistency_error);
      clReleaseMemObject(consistency_error_red);
      clReleaseMemObject(consistency_error_red_reff);
      clReleaseMemObject(asw_d_est_reference);
      clReleaseMemObject(asw_confidence_reference);
      clReleaseMemObject(asw_confidence_target);
      clReleaseMemObject(asw_denom);
      clReleaseMemObject(f_disparity);

      clReleaseKernel(median);
      clReleaseKernel(asw_vsupport);
      clReleaseKernel(asw_hsupport);
      clReleaseKernel(asw_vcost);
      clReleaseKernel(asw_disp);
      clReleaseKernel(asw_aggr);
      clReleaseKernel(asw_vaggregation_kernel);
      clReleaseKernel(asw_haggregation_kernel);
      clReleaseKernel(consist_kernel);
      clReleaseKernel(asw_ref_v);
      clReleaseKernel(asw_ref_h);
      clReleaseKernel(asw_wta_ref);

      imgL.pixel.clear();
      imgR.pixel.clear();
      for (int i = 0; i < 4;i++)
      {
       result[i].pixel.clear();
      }
     }
     clReleaseCommandQueue(queue);
     clReleaseProgram(program);
    }
    clReleaseContext(context);
    fclose(to_file);
  }
 }
 delete [] folder_name;
  printf("\n\nAgain?...(y/n)  ");
  char y_or_n = 'n';  
  scanf_s(" %c", &y_or_n);
  if (y_or_n == 'y')
   again = true;
  else if (y_or_n == 'n')
   again = false;   
  else
  {
   printf("~I quit\n ");
   again = false;
  }
 }
 system("pause");

}
