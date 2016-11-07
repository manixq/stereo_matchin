__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;

//T=60
//tau = 20
//L=17
int disparity(image2d_t left, image2d_t right, int2 pos, int d)
{
 
 float4 left_pixel = read_imagef(left, sampler, pos);
 float4 right_pixel = read_imagef(right, sampler, pos + (int2)(d, 0));
 
 int color_similarity_r = abs_diff((int)(100 * left_pixel.x), (int)(100 * right_pixel.x));
 int color_similarity_g = abs_diff((int)(100 * left_pixel.y), (int)(100 * right_pixel.y));
 int color_similarity_b = abs_diff((int)(100 * left_pixel.z), (int)(100 * right_pixel.z));

 int result = color_similarity_r + color_similarity_g + color_similarity_b;

 return result;
}

int row_check(image2d_t left, image2d_t right, int2 pos, int2 dim, int position, int* cross_l, int* cross_r)
{
 int result = 0;

 //left and right horizontal
 int h_minus_l = cross_l[pos.x + (pos.y + position) * dim.x];
 int h_plus_l = cross_l[pos.x + (pos.y + position) * dim.x + dim.x * dim.y];

 //find dispariity then calc new cross region
 int d = 1;
 int disp = disparity(left, right, pos, d);
 int d_temp = 2;
 int disp_temp = disparity(left, right, pos, d_temp);
 //2
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //3
 disp_temp = disparity(left, right, pos, d_temp); 
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //4
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //5
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //6
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //7
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //8
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //9
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //10
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //11
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //12
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //13
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //14
 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;

 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;

 disp_temp = disparity(left, right, pos, d_temp);
 disp = select(disp, disp_temp, islessequal(float)(disp_temp), (float)(disp));
 d = select(d, d_temp, islessequal(float)(disp_temp), (float)(disp));
 d_temp = d_temp + 1;
 //int h_minus_r = cross_r[pos.x + (pos.y + position) * dim.x];
 //int h_plus_r = cross_r[pos.x + (pos.y + position) * dim.x + dim.x * dim.y];
 
 return result;
}


__kernel void Aggregation (
	__read_only image2d_t input_l,
 __read_only image2d_t input_r,
	__global int* input_cross_l,
 __global int* input_cross_r,
 __write_only image2d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input_l);


    write_imagef(output, (int2)(pos.x, pos.y), (float4)(0.5f,0.0f,0.0f,1.0f));
}