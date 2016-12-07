
float supp_v(float4 p, float4 q, int2 pos, int y)
{
 //SAD
 c_diff = (-1) * (fabs(p.x - q.x) + fabs(p.y - q.y) + fabs(p.z - q.z)) / 300.91f;

 g_dist = distance((float2)(pos.x, pos.y), (float2)(pos.x, y)) / 280.21f;
 w = exp(c_diff - g_dist);
 return w;
}

//for every pixel p(x,y)
__kernel void asw_ref_v (
 __read_only image2d_t input,
 __global float* input_cost,
 __global float* input_est,
 __global float* output_REF
 )
{
    //x_width, y_height
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input);

    int y = 0;
    float c_num_v = 0;
    float c_denom_v = 0; 
    float ww_v = 0;
    float F = 0;

    for (int i = 0; i < 33; i++)
    {
     //V
     y = clamp(pos.y + i - 16, 0, dim.y-1);
     ww_v = supp_v(read_imagef(input, sampler, (int2)(pos.x, pos.y)), read_imagef(input, sampler, (int2)(pos.x, y)), y);
     F = (input_est[pos.x + dim.x * y + dim.x * dim.y * 2] - input_est[pos.x + dim.x * y + dim.x * dim.y * 1]) / input_est[pos.x + dim.x * y + dim.x * dim.y * 2];

     c_num_v += ww_v * F * input_est[pos.x + dim.x * y];
     c_denom_v += ww_v * F;
    }

    float result = c_num_v / c_denom_v;
   // printf("ww %f \n",ww);
    output_REF[pos.x + dim.x * pos.y] = result;
    output_REF[pos.x + dim.x * pos.y + dim.x * dim.y] = c_denom_v;
}