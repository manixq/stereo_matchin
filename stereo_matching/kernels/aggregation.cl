__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

//T=60
//tau = 20
//L=25
int disparity(image2d_t left, image2d_t right, int2 pos, int d)
{
 float4 left_pixel = read_imagef(left, sampler, pos);
 float4 right_pixel = read_imagef(right, sampler, pos + (int2)(d, 0));
 
 int color_similarity_r = abs_diff((int)(10000 * left_pixel.x), (int)(10000 * right_pixel.x));
 int color_similarity_g = abs_diff((int)(10000 * left_pixel.y), (int)(10000 * right_pixel.y));
 int color_similarity_b = abs_diff((int)(10000 * left_pixel.z), (int)(10000 * right_pixel.z));
 //????WARTOSCI PIKSELI CZY ROZBIEZNOSC????
 int result = color_similarity_r + color_similarity_g + color_similarity_b;
 return result;
}

int matching_row(int2 pos, int h_minus, int h_plus, int d, image2d_t left, image2d_t right)
{
 int result = 0;
 for (int i = h_minus; i <= h_plus; i++)
 {
  result = result + disparity(left, right, pos, d);
 }
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
    
    //left and right horizontal
    int h_minus_l = 0;
    int h_plus_l = 0;
    //up and down vertical
    int v_minus_l = input_cross_l[pos.x + pos.y * dim.x + dim.x * dim.y * 2];
    int v_plus_l = input_cross_l[pos.x + pos.y * dim.x + dim.x * dim.y * 3];

    //find dispariity 
    int d = -1;
    int disp = disparity(input_l, input_r, pos, d);
    int disp_temp;
    for (int d_temp = -2; d_temp >= -60; d_temp--)
    {
     disp_temp = disparity(input_l, input_r, pos, d_temp);
     disp = select(disp, disp_temp, islessequal((float)(disp_temp), (float)(disp)));
     d = select(d, d_temp, islessequal((float)(disp_temp), (float)(disp)));
    }
    //now 'd' is offset for pixel from right image
    int h_minus_r = 0;
    int h_plus_r = 0;
    int v_minus_r = input_cross_l[pos.x + d + pos.y * dim.x + dim.x * dim.y * 2];
    int v_plus_r = input_cross_l[pos.x + d + pos.y * dim.x + dim.x * dim.y * 3];

    //combined local support
    int v_minus = select(v_minus_r, v_minus_l, islessequal((float)(v_minus_r), (float)(v_minus_l)));
    int v_plus = select(v_plus_l, v_plus_r, islessequal((float)(v_plus_r), (float)(v_plus_l)));

    int pix_num = 0;
    int disp_sum = 0;
    int h_minus = 0;
    int h_plus = 0;
    for (int i = v_minus; i <= v_plus; i++)
    {
     //get current row from both images
     h_minus_r = input_cross_r[pos.x + d + (pos.y + i) * dim.x];
     h_plus_r = input_cross_r[pos.x + d + (pos.y + i) * dim.x + dim.x * dim.y];
     h_minus_l = input_cross_l[pos.x + (pos.y + i) * dim.x];
     h_plus_l = input_cross_l[pos.x + (pos.y + i) * dim.x + dim.x * dim.y];
     
     //calc current combined row
     h_minus = select(h_minus_r, h_minus_l, islessequal((float)(h_minus_r), (float)(h_minus_l)));
     h_plus = select(h_plus_l, h_plus_r, islessequal((float)(h_plus_r), (float)(h_plus_l)));
     pix_num = pix_num + h_plus - h_minus;
     disp_sum = disp_sum + matching_row((int2)(pos.x,pos.y + i), h_minus, h_plus, d, input_l, input_r);
    }
    float result = (float)(disp_sum)/(float)(pix_num);
    result = result/10000;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
}