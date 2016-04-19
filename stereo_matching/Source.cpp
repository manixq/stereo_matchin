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
 clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr,  &deviceIdCount);

 std::vector<cl_device_id> deviceIds(deviceIdCount);
 clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount,  deviceIds.data(), nullptr);

 const cl_context_properties contextProperties[] = {  CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]),  0, 0 };
 cl_int error = CL_SUCCESS;
 cl_context context = clCreateContext(contextProperties, deviceIdCount,  deviceIds.data(), nullptr, nullptr, &error);



 Image imgL;
 lodepng::decode(imgL.pixel, imgL.width, imgL.height, "sukub/imP.png");
 Image imgR;
 lodepng::decode(imgR.pixel, imgR.width, imgR.height, "sukub/imL.png");


 // Create a program from source
 cl_program program = CreateProgram(LoadKernel("kernels/image.cl"),  context);

 clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

 cl_kernel kernel = clCreateKernel(program, "Filter", &error);



 static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
 cl_mem inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,  imgL.width, imgL.height, 0,  // This is a bug in the spec
  const_cast<unsigned char*> (imgL.pixel.data()),  &error);
 cl_mem inputImage2 = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, imgR.width, imgR.height, 0,  // This is a bug in the spec
  const_cast<unsigned char*> (imgR.pixel.data()), &error);

 int mywidth = imgL.width;
 cl_mem outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format,  imgL.width, imgL.height, 0,  nullptr, &error);
 cl_mem my_width = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) , &mywidth, &error);

 clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
 clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputImage2);
 clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImage);
 clSetKernelArg(kernel, 3, sizeof(cl_mem), &my_width);
 clSetKernelArg(kernel, 4, sizeof(int), nullptr);

 cl_command_queue queue = clCreateCommandQueue(context, deviceIds[0],  0, &error);
 
 std::size_t offset[3] = { 0 };
 std::size_t size[3] = { imgL.width, imgL.height, 1 };
 clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);

 Image result = imgL;

 std::fill(result.pixel.begin(), result.pixel.end(), 0);

 // Get the result back to the host
 std::size_t origin[3] = { 0 };
 std::size_t region[3] = {imgL.width, imgL.height, 1 };
 clEnqueueReadImage(queue, outputImage, CL_TRUE,  origin, region, 0, 0,  result.pixel.data(), 0, nullptr, nullptr);

 lodepng::encode("sukub/niewiem.png", result.pixel , result.width, result.height);

 clReleaseMemObject(outputImage);
 clReleaseMemObject(inputImage);
 clReleaseMemObject(inputImage2);
 clReleaseCommandQueue(queue);
 clReleaseKernel(kernel);
 clReleaseProgram(program);
 clReleaseContext(context);
 system("pause");
}
