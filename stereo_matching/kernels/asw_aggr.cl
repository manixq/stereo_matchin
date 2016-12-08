

__kernel void asw_Aggr (
	__read_only image2d_t input_l,
 __read_only image2d_t input_r,
 __global float* output_cost
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input_l);

    float4 left_pixel = read_imagef(input_l, sampler, pos);
    float4 right_pixel;

    float result;
    for (int d = 0; d > -61; d--)
    {  
     right_pixel = read_imagef(input_r, sampler, pos + (int2)(d, 0));
     result = min(fabs(left_pixel.x - right_pixel.x) + fabs(left_pixel.y - right_pixel.y) + fabs(left_pixel.z - right_pixel.z), 0.10f);

     output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * (-1) * d] = result;
    }
}