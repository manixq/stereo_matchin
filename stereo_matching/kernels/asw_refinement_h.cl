
float supp_h(float4 p, float4 q, int2 pos, int x)
{
 //SAD
 float c_diff = (-1) * (fabs(p.x - q.x) + fabs(p.y - q.y) + fabs(p.z - q.z)) / 10.94f;

 float g_dist = distance((float2)(pos.x, pos.y), (float2)(x, pos.y)) / 118.78f;
 float w = exp(c_diff - g_dist);
 return w;
}

//for every pixel p(x,y)
__kernel void asw_ref_h (
 __read_only image2d_t input,
 __global float* input_est,
 __global float* input_REF,
 __global float* output_REF
 )
{
   //x_width, y_height
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input);

    int x = 0;
    float c_num_h = 0;
    float c_denom_h = 0;
    float ww_h;
    float F;
    
    for (int i = 0; i < 33; i++)
    {
     //H     
     x = clamp(pos.x + i - 16, 0, dim.x - 1);
     ww_h = supp_h(read_imagef(input, sampler, (int2)(pos.x, pos.y))*1000, read_imagef(input, sampler, (int2)(x, pos.y)), pos, x)*1000;
     F = (input_est[x + dim.x * pos.y + dim.x * dim.y * 2] - input_est[x + dim.x * pos.y + dim.x * dim.y * 1]) / input_est[x + dim.x * pos.y + dim.x * dim.y * 2];
     //here input ref is vertical refinement result
     c_num_h += ww_h * F * input_REF[x + dim.x * pos.y] * input_REF[x + dim.x * pos.y + dim.x * dim.y];
     c_denom_h += ww_h * F * input_REF[x + dim.x * pos.y];   
     
    }

    float result = c_num_h / c_denom_h;
   // printf("ww %f \n",ww);
    output_REF[pos.x + dim.x * pos.y] = result;
    output_REF[pos.x + dim.x * pos.y + dim.x * dim.y] = c_denom_h;
}