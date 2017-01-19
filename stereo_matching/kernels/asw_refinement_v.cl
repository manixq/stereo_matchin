
float supp_v(float4 p, float4 q, int2 pos, int y)
{
 //SAD
 float c_diff = (-1) * (fabs(p.x - q.x) + fabs(p.y - q.y) + fabs(p.z - q.z)) / 10.94f;

 float g_dist = distance((float2)(pos.x, pos.y), (float2)(pos.x, y)) / 118.78f;
 float w = exp(c_diff - g_dist);
 return w;
}

//for every pixel p(x,y)
__kernel void asw_ref_v (
 __read_only image2d_t input,
 __read_only image2d_t input_est,
 __global float* confidence,
 __global float* output_REF
 )
{
    //x_width, y_height
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input);

    float x_mult = 255;
    int y = 0;
    float c_num_v = 0.00001;
    float c_denom_v = 0.00001; 
    float ww_v = 0;
    float F = 0;
    float4 p = read_imagef(input, sampler, (int2)(pos.x, pos.y)) * x_mult;
    float4 q;
    float4 D;
    for (int i = 0; i < 33; i++)
    {
     //V
     y = clamp(pos.y + i - 16, 0, dim.y - 1);
     q = read_imagef(input, sampler, (int2)(pos.x, y)) * x_mult;
     D = read_imagef(input_est, sampler, (int2)(pos.x, y)) * 60;
     ww_v = supp_v(p, q, (int2)(pos.x, pos.y), y);
     F = confidence[pos.x + dim.x * y];
     
     c_num_v += ww_v * F * D.x;
     c_denom_v += ww_v * F;
     
    }

    float result = c_num_v / c_denom_v;
 
    output_REF[pos.x + dim.x * pos.y] = result;
    output_REF[pos.x + dim.x * pos.y + dim.x * dim.y] = c_denom_v;
}