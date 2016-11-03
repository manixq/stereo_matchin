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
  lodepng::decode(imgL.pixel, imgL.width, imgL.height, "sukub/imP.png");
  Image imgR;
  lodepng::decode(imgR.pixel, imgR.width, imgR.height, "sukub/imL.png");

  //variables
  static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  Image result[2] = { imgL, imgR };
  std::size_t offset[3] = { 0 };
  //global work size
  std::size_t size[3] = { result[0].width, result[0].height, 1 };
  //defines offset, where should we read, z-buffer i zero for 2Ds
  std::size_t origin[3] = { 0,0,0 };
  //whats the size of rect we gonna read
  std::size_t region[3] = { result[0].width, result[0].height, 1 };

  //Median filter
  // Create a program from source
  cl_program program = CreateProgram(LoadKernel("kernels/median.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "Decimate", &error);
  cl_mem inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
  cl_mem inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
  cl_mem outputImage_l = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[0].width, result[0].height, 0, nullptr, &error);
  cl_mem outputImage_r = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[1].width, result[1].height, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImage_l);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputImage_r);
  //queue with profiling
  cl_command_queue queue = clCreateCommandQueue(context, deviceIds.data()[device_id], CL_QUEUE_PROFILING_ENABLE, &error);
  //ensure to have executed all queued tasks
  clFinish(queue);
  cl_event event_median;
  clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, size, nullptr, 0, nullptr, &event_median);
  clWaitForEvents(1, &event_median);
  clFinish(queue);
  clEnqueueReadImage(queue, outputImage_l, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 0, nullptr,nullptr);
  clEnqueueReadImage(queue, outputImage_r, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 0, nullptr, nullptr);
  clFinish(queue);

  clReleaseMemObject(inputImage_l);
  clReleaseMemObject(inputImage_r);
  clReleaseMemObject(outputImage_l);
  clReleaseMemObject(outputImage_r);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  lodepng::encode("sukub/decimated_depth_L.png", result[0].pixel, result[0].width, result[0].height);
  lodepng::encode("sukub/decimated_depth_R.png", result[1].pixel, result[1].width, result[1].height);

  //Cross construction
  //width, height x 4 dimension for every pixel: H-, H+, V-, V+
  std::size_t cross_size[3] = { result[0].width, result[0].height, 4 };

  program = CreateProgram(LoadKernel("kernels/cross.cl"), context);
  clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr);
  kernel = clCreateKernel(program, "Decimate", &error);
  inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
  inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
  cl_mem outputCross_l = clCreateImage3D(context, CL_MEM_WRITE_ONLY, &format, result[0].width, result[0].height, 4, 0, 0, nullptr, &error);
  cl_mem outputCross_r = clCreateImage3D(context, CL_MEM_WRITE_ONLY, &format, result[1].width, result[1].height, 4, 0, 0, nullptr, &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputCross_l);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputCross_r);
  //queue with profiling
  // queue = clCreateCommandQueue(context, deviceIds.data()[device_id], CL_QUEUE_PROFILING_ENABLE, &error);
  //ensure to have executed all queued tasks
  clFinish(queue);
  cl_event event_cross;
  clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, size, nullptr, 0, nullptr, &event_cross);
  clWaitForEvents(1, &event_cross);
  clFinish(queue);
  std::vector<std::vector<std::vector<size_t> > > cross_result(cross_size[0], std::vector<std::vector<size_t> >(cross_size[1], std::vector <size_t>(cross_size[2])));
  clEnqueueReadImage(queue, outputCross_l, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 0, nullptr, nullptr);
  clEnqueueReadImage(queue, outputCross_r, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 0, nullptr, nullptr);
  clFinish(queue);
  clReleaseMemObject(inputImage_l);
  clReleaseMemObject(inputImage_r);
  clReleaseMemObject(outputCross_l);
  clReleaseMemObject(outputCross_r);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  printf("%zu\n", cross_result.data()[0][0][0]);



  //Get measured time data
  cl_ulong time_start, time_end;
  double total_time;
  clGetEventProfilingInfo(event_median, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event_median, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
  total_time = time_end - time_start;
  printf("\nExecution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0));
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
 /*
 //DEPTH
 // Create a program from source
 program = CreateProgram(LoadKernel("kernels/depth.cl"), context);

 clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

 kernel = clCreateKernel(program, "Filter", &error);

 inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, imgL.width, imgL.height, 0,  // This is a bug in the spec
  const_cast<unsigned char*> (imgL.pixel.data()), &error);
 cl_mem inputImage2 = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, imgR.width, imgR.height, 0,  // This is a bug in the spec
  const_cast<unsigned char*> (imgR.pixel.data()), &error);

 int mywidth = imgL.width;
 outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, imgL.width, imgL.height, 0, nullptr, &error);
 cl_mem my_width = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &mywidth, &error);

 clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
 clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputImage2);
 clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImage);
 clSetKernelArg(kernel, 3, sizeof(cl_mem), &my_width);
 clSetKernelArg(kernel, 4, sizeof(int), nullptr);

 queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);


 clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);



 std::fill(result.pixel.begin(), result.pixel.end(), 0);

 // Get the result back to the host
 origin[3] = { 0 };
 region[0] = imgL.width / 4;// {imgL.width, imgL.height, 1};
 clEnqueueReadImage(queue, outputImage, CL_TRUE,  origin, region, 0, 0,  result.pixel.data(), 0, nullptr, nullptr);

 lodepng::encode("sukub/depth.png", result.pixel , result.width, result.height);

 clReleaseMemObject(outputImage);
 clReleaseMemObject(inputImage);
 clReleaseMemObject(inputImage2);
 clReleaseCommandQueue(queue);
 clReleaseKernel(kernel);
 clReleaseProgram(program);
 
 //DECYMACJA
 // Create a program from source
 program = CreateProgram(LoadKernel("kernels/decimate.cl"), context);

 clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

  kernel = clCreateKernel(program, "Filter", &error);

 inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result.width, result.height, 0,  // This is a bug in the spec
 const_cast<unsigned char*> (result.pixel.data()), &error);
 

 outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result.width, result.height, 0, nullptr, &error);
 
 clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
 clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);

 queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);

 clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);

 result = imgL;

 std::fill(result.pixel.begin(), result.pixel.end(), 0);

 region[0] = imgL.width;
 clEnqueueReadImage(queue, outputImage, CL_TRUE, origin, region, 0, 0, result.pixel.data(), 0, nullptr, nullptr);

 lodepng::encode("sukub/decimated_depth.png", result.pixel, result.width, result.height);



 clReleaseMemObject(outputImage);
 clReleaseMemObject(inputImage);
 clReleaseCommandQueue(queue);
 clReleaseKernel(kernel);
 clReleaseProgram(program);

 //INTERPOLACJA

 // Create a program from source
 program = CreateProgram(LoadKernel("kernels/interpolate.cl"), context);

 clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

 kernel = clCreateKernel(program, "Filter", &error);



 inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result.width, result.height, 0,  // This is a bug in the spec
  const_cast<unsigned char*> (result.pixel.data()), &error);


 outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result.width, result.height, 0, nullptr, &error);

 clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
 clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);

 queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);

 clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);

 result = imgL;

 std::fill(result.pixel.begin(), result.pixel.end(), 0);

 region[0] = imgL.width;
 clEnqueueReadImage(queue, outputImage, CL_TRUE, origin, region, 0, 0, result.pixel.data(), 0, nullptr, nullptr);

 lodepng::encode("sukub/interpolated_depth.png", result.pixel, result.width, result.height);


 clReleaseMemObject(outputImage);
 clReleaseMemObject(inputImage);
 clReleaseCommandQueue(queue);
 clReleaseKernel(kernel);
 clReleaseProgram(program);*/
 //clReleaseContext(context);
 system("pause");
}
