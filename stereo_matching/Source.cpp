#include "Header.h"

struct Image
{
 std::vector<unsigned char> pixel;
 unsigned width, height;
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
 cl_uint platformIdCount = 0;
 clGetPlatformIDs(0, nullptr, &platformIdCount);
 std::vector<cl_platform_id> platformIds(platformIdCount);
 clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

 cl_uint deviceIdCount = 0;
 clGetDeviceIDs(platformIds[1], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceIdCount);

 std::vector<cl_device_id> deviceIds(deviceIdCount);
 clGetDeviceIDs(platformIds[1], CL_DEVICE_TYPE_GPU, deviceIdCount, deviceIds.data(), nullptr);
 char buffer[10240];
 clGetDeviceInfo(deviceIds.data()[0], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
 printf("  DEVICE_NAME = %s\n", buffer);
 const cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[1]),  0, 0 };
 cl_int error = CL_SUCCESS;

 cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);

 Image imgL;
 lodepng::decode(imgL.pixel, imgL.width, imgL.height, "sukub/imP.png");
 Image imgR;
 lodepng::decode(imgR.pixel, imgR.width, imgR.height, "sukub/imL.png");

 //variables
 static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
 Image result = imgL;
 std::size_t offset[3] = { 0 };
 //global work size
 std::size_t size[3] = { result.width / 2, result.height / 2, 1 };
 //defines offset, where should we read, z-buffer i zero for 2Ds
 std::size_t origin[3] = { 0,0,0 };
 //whats the size of rect we gonna read
 std::size_t region[3] = { result.width / 2, result.height / 2, 1 };

 //DECIMATION
 // Create a program from source
 cl_program program = CreateProgram(LoadKernel("kernels/decimate.cl"), context);
 clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
 cl_kernel kernel = clCreateKernel(program, "Decimate", &error);
 cl_mem inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result.width, result.height, 0, const_cast<unsigned char*> (result.pixel.data()), &error);
 //Now our result is new result
 result.width /= 2;
 result.height /= 2;
// result.pixel.clear();
 cl_mem outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result.width, result.height, 0, nullptr, &error);
 clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
 clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
 cl_command_queue queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
 std::size_t local_group_size[2] = { 16, 16 };
 clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, size, local_group_size, 0, nullptr, nullptr);
 clEnqueueReadImage(queue, outputImage, CL_TRUE, origin, region, 0, 0, result.pixel.data(), 0, nullptr, nullptr);
 lodepng::encode("sukub/decimated_depth.png", result.pixel, result.width, result.height);
 clReleaseMemObject(outputImage);
 clReleaseMemObject(inputImage);
 clReleaseCommandQueue(queue);
 clReleaseKernel(kernel);
 clReleaseProgram(program);

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
 clReleaseContext(context);
 system("pause");
}
