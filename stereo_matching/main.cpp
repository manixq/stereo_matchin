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

double compute_time(cl_event* clevent, char* name) 
{
 cl_ulong time_start[2], time_end[2];
 double total_time;

 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_START, sizeof(time_start[0]), &time_start[0], nullptr);
 clGetEventProfilingInfo(clevent[1], CL_PROFILING_COMMAND_START, sizeof(time_start[1]), &time_start[1], nullptr);
 clGetEventProfilingInfo(clevent[0], CL_PROFILING_COMMAND_END, sizeof(time_end[0]), &time_end[0], nullptr);
 clGetEventProfilingInfo(clevent[1], CL_PROFILING_COMMAND_END, sizeof(time_end[1]), &time_end[1], nullptr);

 total_time = (time_end[0] > time_end[1] ? time_end[0] : time_end[1]) - (time_start[0] < time_start[1] ? time_start[0] : time_start[1]);
 printf("%s %0.3f ms", name, (total_time / 1000000.0));
 return total_time / 1000000.0;
}

double compute_time(cl_event clevent, char* name)
{
 cl_ulong time_start, time_end;
 double total_time;

 clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
 clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);

 total_time = time_end - time_start;
 printf("%s %0.3f ms", name, (total_time / 1000000.0));
 return total_time / 1000000.0;
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
  cl_uint work_item_max[3];
  for (int i = 0; i < platformIdCount; i++)
  {
   printf("Platform Id: %d\n", i);
   clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount[i], deviceIds.data(), nullptr);
   for (int j = 0; j < deviceIdCount[i]; j++)
   {
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_max, nullptr);
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, nullptr);
    clGetDeviceInfo(deviceIds.data()[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(cl_uint) * 3, work_item_max, nullptr);
    printf("\t- Device Id: %d\n", j);
    printf("\t- Device name: %s\n", buffer);
    printf("\t- Device max compute units: %d\n", compute_units);
    printf("\t- Device max work group size: %d\n", work_group_max);
    printf("\t- Device max work item sizes: [%d, %d, %d]\n\n", work_item_max[0], work_item_max[1], work_item_max[2]);
    work_group_max = 0;
    compute_units = 0;
    work_item_max[0] = 0;
    work_item_max[1] = 0;
    work_item_max[2] = 0;
   }
  }

  //user_inputs
  int device_id = 0, platform_id = 0;
  char left_path[10240];
  char right_path[10240];
  printf("\n---Setup:\nSelect platform: ");
  scanf_s("%d", &platform_id);
  printf("Select device: ");
  scanf_s("%d", &device_id);
  printf("Select left image: ");
  scanf_s("%s", &left_path);
  printf("Select right image: ");
  scanf_s("%s", &right_path);
  deviceIds.clear();

  cl_int error = CL_SUCCESS;
  clGetDeviceIDs(platformIds[platform_id], CL_DEVICE_TYPE_ALL, deviceIdCount[platform_id], deviceIds.data(), nullptr);
  const cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platform_id]),  0, 0 };
  cl_context context = clCreateContext(contextProperties, 1, &deviceIds.data()[device_id], nullptr, nullptr, &error);
  clGetDeviceInfo(deviceIds.data()[device_id], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);

  Image imgL;
  lodepng::decode(imgL.pixel, imgL.width, imgL.height, left_path);
  Image imgR;
  lodepng::decode(imgR.pixel, imgR.width, imgR.height, right_path);

  //variables
  static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  Image result[3] = { imgL, imgR, imgL };
  std::size_t local[3] = { 3, 3, 1 };
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

  //Reading all kernels
  std::string kernels[9] = { 
   sampler + LoadKernel("kernels/median.cl"), 
   LoadKernel("kernels/cross.cl"),
   LoadKernel("kernels/aggregation.cl"),
   LoadKernel("kernels/integral_h.cl"),
   LoadKernel("kernels/oii_hcross.cl"),
   LoadKernel("kernels/integral_v.cl"),
   LoadKernel("kernels/oii_vcross.cl"),
   LoadKernel("kernels/init_disparity.cl"),
   LoadKernel("kernels/disparity.cl")
  };

  int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

  const char* load_kernel[9];
  for (int i = 0; i < num_kernels; i++)
  {
   load_kernel[i] = kernels[i].data();
  }

  // Create a program from source
  cl_program program = clCreateProgramWithSource(context, num_kernels, load_kernel, nullptr, nullptr);
  ErCheck(clBuildProgram(program, 1, &deviceIds.data()[device_id], nullptr, nullptr, nullptr));
  cl_command_queue queue = clCreateCommandQueue(context, deviceIds.data()[device_id], CL_QUEUE_PROFILING_ENABLE, &error);

  cl_kernel median = clCreateKernel(program, "Median", &error);
  cl_kernel cross = clCreateKernel(program, "Cross", &error);
  cl_kernel aggregation = clCreateKernel(program, "Aggregation", &error);
  cl_kernel integral_h = clCreateKernel(program, "Integral_h", &error);
  cl_kernel oii_hcross = clCreateKernel(program, "Oii_hcross", &error);
  cl_kernel integral_v = clCreateKernel(program, "Integral_v", &error);
  cl_kernel oii_vcross = clCreateKernel(program, "Oii_vcross", &error);
  cl_kernel init_disparity = clCreateKernel(program, "Init_disparity", &error);
  cl_kernel final_disparity = clCreateKernel(program, "Disparity", &error);

  cl_mem inputImage_l = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[0].width, result[0].height, 0, const_cast<unsigned char*> (result[0].pixel.data()), &error);
  cl_mem inputImage_r = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, result[1].width, result[1].height, 0, const_cast<unsigned char*> (result[1].pixel.data()), &error);
  cl_mem median_l = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
  cl_mem median_r = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[1].width, result[1].height, 0, nullptr, &error);
  //width, height x 4 dimension for every pixel: H-, H+, V-, V+
  cl_mem cross_l = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[0].width * result[0].height * 4, nullptr, &error);
  cl_mem cross_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[1].width * result[1].height * 4, nullptr, &error);
  //width, height x disparity <0:60>
  cl_mem cost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[1].width * result[1].height * 61, nullptr, &error);
  cl_mem im_size = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 3, sizei, &error);
  cl_mem temp_cost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * result[1].width * result[1].height * 61, nullptr, &error);
  cl_mem disparity = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, result[0].width, result[0].height, 0, nullptr, &error);
  cl_mem f_disparity = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, result[0].width, result[0].height, 0, nullptr, &error);

  cl_event event_median[2];
  cl_event event_cross[2];
  cl_event event_aggro;
  cl_event event_integralh;
  cl_event event_oiih;
  cl_event event_integralv;
  cl_event event_oiiv;
  cl_event event_initd;
  cl_event event_disp;

  //median filter
  printf("\n---Working...\nMedian filter..  ");
  clSetKernelArg(median, 0, sizeof(cl_mem), &inputImage_l);
  clSetKernelArg(median, 1, sizeof(cl_mem), &median_l); 
  clEnqueueNDRangeKernel(queue, median, 2, nullptr, globalSize, local, 0, nullptr, &event_median[0]);
  clSetKernelArg(median, 0, sizeof(cl_mem), &inputImage_r);
  clSetKernelArg(median, 1, sizeof(cl_mem), &median_r);
  clEnqueueNDRangeKernel(queue, median, 2, nullptr, globalSize, local, 0, nullptr, &event_median[1]);
  //Cross construction
  printf("\nCross construction.. ");
  clSetKernelArg(cross, 0, sizeof(cl_mem), &median_l);
  clSetKernelArg(cross, 1, sizeof(cl_mem), &cross_l);
  ErCheck(clEnqueueNDRangeKernel(queue, cross, 2, nullptr, size, nullptr, 2, event_median, &event_cross[0]));
  clSetKernelArg(cross, 0, sizeof(cl_mem), &median_r);
  clSetKernelArg(cross, 1, sizeof(cl_mem), &cross_r);
  ErCheck(clEnqueueNDRangeKernel(queue, cross, 2, nullptr, size, nullptr, 2, event_median, &event_cross[1]));
  //aggregation cost
  printf("\nAggregation cost.. ");
  clSetKernelArg(aggregation, 0, sizeof(cl_mem), &median_l);
  clSetKernelArg(aggregation, 1, sizeof(cl_mem), &median_r);
  clSetKernelArg(aggregation, 2, sizeof(cl_mem), &cost);
  ErCheck(clEnqueueNDRangeKernel(queue, aggregation, 2, nullptr, size, nullptr, 2, event_cross, &event_aggro));
  //horizontal integral image
  printf("\nOII - horizontal integral.. ");
  clSetKernelArg(integral_h, 0, sizeof(cl_mem), &cost);
  clSetKernelArg(integral_h, 1, sizeof(cl_mem), &im_size);
  ErCheck(clEnqueueNDRangeKernel(queue, integral_h, 2, nullptr, integralh_size, nullptr, 1, &event_aggro, &event_integralh));
  //horizontal image cross cost
  printf("\nOII - horizontal cross.. ");
  clSetKernelArg(oii_hcross, 0, sizeof(cl_mem), &cross_l);
  clSetKernelArg(oii_hcross, 1, sizeof(cl_mem), &cross_r);
  clSetKernelArg(oii_hcross, 2, sizeof(cl_mem), &cost);
  clSetKernelArg(oii_hcross, 3, sizeof(cl_mem), &temp_cost);
  clSetKernelArg(oii_hcross, 4, sizeof(cl_mem), &im_size);
  ErCheck(clEnqueueNDRangeKernel(queue, oii_hcross, 3, nullptr, hcross_size, nullptr, 1, &event_integralh, &event_oiih));
  //vertical integral image
  printf("\nOII - vertical integral.. ");
  clSetKernelArg(integral_v, 0, sizeof(cl_mem), &temp_cost);
  clSetKernelArg(integral_v, 1, sizeof(cl_mem), &im_size);
  ErCheck(clEnqueueNDRangeKernel(queue, integral_v, 2, nullptr, integralv_size, nullptr, 1, &event_oiih, &event_integralv));
  //vertical image cross cost
  printf("\nOII - vertical cross.. ");
  clSetKernelArg(oii_vcross, 0, sizeof(cl_mem), &cross_l);
  clSetKernelArg(oii_vcross, 1, sizeof(cl_mem), &cross_r);
  clSetKernelArg(oii_vcross, 2, sizeof(cl_mem), &temp_cost);
  clSetKernelArg(oii_vcross, 3, sizeof(cl_mem), &cost);
  clSetKernelArg(oii_vcross, 4, sizeof(cl_mem), &im_size);
  ErCheck(clEnqueueNDRangeKernel(queue, oii_vcross, 3, nullptr, vcross_size, nullptr, 1, &event_integralv, &event_oiiv));
  //initial disparity
  printf("\nInitial disparity.. ");
  clSetKernelArg(init_disparity, 0, sizeof(cl_mem), &cost);
  clSetKernelArg(init_disparity, 1, sizeof(cl_mem), &disparity);
  ErCheck(clEnqueueNDRangeKernel(queue, init_disparity, 2, nullptr, size, nullptr, 1, &event_oiiv, &event_initd));
  //final disparity
  printf("\nFinal disparity.. ");
  clSetKernelArg(final_disparity, 0, sizeof(cl_mem), &disparity);
  clSetKernelArg(final_disparity, 1, sizeof(cl_mem), &cross_l);
  clSetKernelArg(final_disparity, 2, sizeof(cl_mem), &f_disparity);
  std::cout << "set kernel";
  ErCheck(clEnqueueNDRangeKernel(queue, final_disparity, 2, nullptr, size, nullptr, 1, &event_initd, &event_disp));
  std::cout << "enqueuereakernel";

  clEnqueueReadImage(queue, disparity, CL_TRUE, origin, region, 0, 0, result[0].pixel.data(), 1, &event_initd, nullptr);
  clEnqueueReadImage(queue, f_disparity, CL_TRUE, origin, region, 0, 0, result[1].pixel.data(), 1, &event_disp, nullptr);

  clReleaseMemObject(inputImage_l);
  clReleaseMemObject(inputImage_r);
  clReleaseMemObject(median_l);
  clReleaseMemObject(median_r);
  clReleaseMemObject(temp_cost);
  clReleaseMemObject(cross_r);
  clReleaseMemObject(im_size);
  clReleaseMemObject(cost);
  clReleaseMemObject(cross_l);
  clReleaseMemObject(disparity);
  clReleaseMemObject(f_disparity);

  clReleaseKernel(median);
  clReleaseKernel(cross);
  clReleaseKernel(aggregation);
  clReleaseKernel(integral_h);
  clReleaseKernel(oii_hcross);
  clReleaseKernel(integral_v);
  clReleaseKernel(oii_vcross);
  clReleaseKernel(init_disparity);
  clReleaseKernel(final_disparity);

  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  lodepng::encode("sukub/initial_disparity.png", result[0].pixel, result[0].width, result[0].height);
  lodepng::encode("sukub/final_disparity.png", result[1].pixel, result[1].width, result[1].height);

  //Get measured time data
  printf("\n\n---Results for %s: ", buffer);
  compute_time(event_median, "\nMedian filter: ");
  compute_time(event_cross, "\nCross: ");
  compute_time(event_aggro, "\nAggregation: ");
  compute_time(event_integralh, "\nHorizontal integral: ");
  compute_time(event_oiih, "\nHorizontal cost matching: ");
  compute_time(event_integralv, "\nVertical integral: ");
  compute_time(event_oiiv, "\nVertical cost matching: ");
  compute_time(event_initd, "\nInitial disparity: ");
  compute_time(event_disp, "\nFinal disparity: ");
  printf("\n");
  cl_event total_event[2] = { event_median[0], event_disp };
  compute_time(total_event, "\n-- - Total time in milliseconds = ");
  printf("\n\nAgain?...(y/n)  ");
  char y_or_n = 'n';  
  scanf_s(" %c", &y_or_n);
  if (y_or_n == 'y')
   again = true;
  else if (y_or_n == 'n')
   again = false;   
  else
  {
   printf("~Whats wrong with you?\n ");
   again = false;
  }
 }
 system("pause");
}
