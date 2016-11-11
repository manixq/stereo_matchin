#include "Header.h"
#include <chrono>

struct Image
{
 std::vector<unsigned char> pixel;
 unsigned width, height, cross;
};

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
int main()
{

 //CROSS-BASED METHOD
 bool again = true;
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
  size_t work_group_max = 0;
  cl_uint compute_units = 0;
  for (int i = 0; i < platformIdCount; i++)
  {
   printf("Platform Id: %d\n", i);
   clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount[i], deviceIds.data(), nullptr);
   for (int j = 0; j < deviceIdCount[i]; j++)
   {
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_max, nullptr);
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, nullptr);
    printf("\t- Device Id: %d\n", j);
    printf("\t- Device name: %s\n", buffer);
    printf("\t- Device max compute units: %d\n", compute_units);
    printf("\t- Device max work group size: %d\n\n", work_group_max);
    work_group_max = 0;
    compute_units = 0;
   }
  }

  //user_inputs
  int device_id = 0, platform_id = 0;
  printf("\nSelect platform: ");
  scanf("%d", &platform_id);
  printf("Select device: ");
  scanf("%d", &device_id);
  deviceIds.clear();

  clGetDeviceIDs(platformIds[platform_id], CL_DEVICE_TYPE_ALL, deviceIdCount[platform_id], deviceIds.data(), nullptr);
  //lets_go
  const cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platform_id]),  0, 0 };
  cl_int error = CL_SUCCESS;

  cl_context context = clCreateContext(contextProperties, 1, &deviceIds.data()[device_id], nullptr, nullptr, &error);
 
  Image imgL;
  lodepng::decode(imgL.pixel, imgL.width, imgL.height, "sukub/imP1.png");
  Image imgR;
  lodepng::decode(imgR.pixel, imgR.width, imgR.height, "sukub/imL1.png");

  //variables
  static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  Image result[3] = { imgL, imgR, imgL };
  std::size_t local[3] = { 3, 3, 1 };
  size_t numLocalGroups[] = { ceil(result[0].width / local[0]), ceil(result[0].height / local[1]) };
  //global work size
  std::size_t size[3] = { result[0].width, result[0].height, 1 };
  size_t globalSize[] = { local[0] * numLocalGroups[0], local[1] * numLocalGroups[1] };
  //defines offset, where should we read, z-buffer i zero for 2Ds
  std::size_t origin[3] = { 0,0,0 };
  //whats the size of rect we gonna read
  std::size_t region[3] = { result[0].width, result[0].height, 1 };

  //Median filter
  // Create a program from source
  cl_program program = CreateProgram(LoadKernel("kernels/median.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "Median", &error);
  cl_mem inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
  cl_mem outputImage_l = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[0].width, result[0].height, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage_l);
  //queue with profiling
  cl_command_queue queue = clCreateCommandQueue(context, deviceIds.data()[device_id], CL_QUEUE_PROFILING_ENABLE, &error);
  cl_event event_median_l;
  clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, local , 0, nullptr, &event_median_l);
  clWaitForEvents(1, &event_median_l);
  clEnqueueReadImage(queue, outputImage_l, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 0, nullptr,nullptr);
  lodepng::encode("sukub/median_L.png", result[0].pixel, result[0].width, result[0].height);
  clReleaseMemObject(inputImage_l);
  //second image
  cl_mem inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
  cl_mem outputImage_r = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[1].width, result[1].height, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage_r);
  cl_event event_median_r;
  clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, local, 0, nullptr, &event_median_r);
  clWaitForEvents(1, &event_median_r);
  clEnqueueReadImage(queue, outputImage_r, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 0, nullptr, nullptr);
  clReleaseMemObject(inputImage_r);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  lodepng::encode("sukub/median_R.png", result[1].pixel, result[1].width, result[1].height);

  //Cross construction
  //width, height x 4 dimension for every pixel: H-, H+, V-, V+
  //left image cross
  std::size_t cross_size[3] = { result[0].width, result[0].height, 1 };
  program = CreateProgram(LoadKernel("kernels/cross.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  kernel = clCreateKernel(program, "Cross", &error);
  inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
  cl_mem outputCross_l = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * result[0].width * result[0].height * 4, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputCross_l);
  cl_event event_cross_l;
  ErCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, cross_size, nullptr, 0, nullptr, &event_cross_l));
  clWaitForEvents(1, &event_cross_l);
  auto *cross_l = static_cast<int*>(malloc(sizeof(int) * result[0].width * result[0].height * 4));
  clEnqueueReadBuffer(queue, outputCross_l, CL_TRUE, 0, sizeof(int) * result[0].width * result[0].height * 4, cross_l, 0, nullptr, nullptr);
  //right image 
  inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
  cl_mem outputCross_r = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * result[1].width * result[1].height * 4, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputCross_r);
  cl_event event_cross_r;
  ErCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, cross_size, nullptr, 0, nullptr, &event_cross_r));
  clWaitForEvents(1, &event_cross_r);
  auto *cross_r = static_cast<int*>(malloc(sizeof(int) * result[1].width * result[1].height * 4));
  clEnqueueReadBuffer(queue, outputCross_r, CL_TRUE, 0, sizeof(int) * result[1].width * result[1].height * 4, cross_r, 0, nullptr, nullptr);
  clFinish(queue);
  clReleaseMemObject(outputCross_l);
  clReleaseMemObject(outputCross_r);
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  //aggregation cost
  program = CreateProgram(LoadKernel("kernels/aggregation.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  kernel = clCreateKernel(program, "Aggregation", &error);
  cl_mem inputCross_l = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * result[0].width * result[0].height * 4, cross_l, &error);
  cl_mem inputCross_r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * result[1].width * result[1].height * 4, cross_r, &error);
  cl_mem disparity = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[2].width, result[2].height, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputCross_l);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputCross_r);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &disparity);
  cl_event event_aggro;
  ErCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, size, nullptr, 0, nullptr, &event_aggro));
  ErCheck(clWaitForEvents(1, &event_aggro));
  clEnqueueReadImage(queue, disparity, CL_TRUE, origin, region, 0, 0, result[2].pixel.data(), 0, nullptr, nullptr);

  
  clReleaseMemObject(inputImage_l);
  clReleaseMemObject(inputImage_r);
  clReleaseMemObject(inputCross_r);
  clReleaseMemObject(disparity);
  lodepng::encode("sukub/initial_disparity.png", result[2].pixel, result[2].width, result[2].height);

  //final disparity
  program = CreateProgram(LoadKernel("kernels/disparity.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  kernel = clCreateKernel(program, "Disparity", &error);
  disparity = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[2].width, result[2].height, 0, const_cast<unsigned char*> (result[2].pixel.data()), &error);
  cl_mem final_disparity = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[0].width, result[0].height, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &disparity);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputCross_l);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &final_disparity);
  cl_event event_disp;
  ErCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, size, nullptr, 0, nullptr, &event_disp));
  ErCheck(clWaitForEvents(1, &event_disp));
  clEnqueueReadImage(queue, final_disparity, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 0, nullptr, nullptr);

  clReleaseMemObject(final_disparity);
  clReleaseMemObject(disparity);
  clReleaseMemObject(inputCross_l);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  lodepng::encode("sukub/final_disparity.png", result[0].pixel, result[0].width, result[0].height);

  //Get measured time data
  cl_ulong time_start, time_end;
  double total_time;
  clGetEventProfilingInfo(event_median_l, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_median_l, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nMedian(left) Execution time in milliseconds = %0.3f ms", (total_time / 1000000.0));
  clGetEventProfilingInfo(event_median_r, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_median_r, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nMedian(right) Execution time in milliseconds = %0.3f ms", (total_time / 1000000.0));
  clGetEventProfilingInfo(event_cross_l, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_cross_l, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nCross(left) Execution time in milliseconds = %0.3f ms", (total_time / 1000000.0));
  clGetEventProfilingInfo(event_cross_r, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_cross_r, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nCross(right) Execution time in milliseconds = %0.3f ms", (total_time / 1000000.0));
  clGetEventProfilingInfo(event_aggro, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_aggro, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nAggregation cost Execution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0));
  printf("Again?...(y/n)  ");
  char y_or_n = 'n';  
  scanf(" %c", &y_or_n);
  if (y_or_n == 'y')
   again = true;
  else if (y_or_n == 'n')
   again = false;   
  else
  {
   printf("~Whats wrong with you?\n ");
   again = false;
  }
  clReleaseContext(context);
 }
 system("pause");
}
