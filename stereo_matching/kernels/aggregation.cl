

__kernel void Aggregation (
	__read_only image2d_t input_l,
 __read_only image2d_t input_r,
 __global int* output_cost
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input_l);

    float4 left_pixel = read_imagef(input_l, sampler, pos);
    float4 right_pixel;

    int color_similarity_r;
    int color_similarity_g;
    int color_similarity_b;
    int result;
    for (int d = 0; d >= -60; d--)
    {  
     right_pixel = read_imagef(input_r, sampler, pos + (int2)(d, 0));
     color_similarity_r = abs_diff((int)(10000 * left_pixel.x), (int)(10000 * right_pixel.x));
     color_similarity_g = abs_diff((int)(10000 * left_pixel.y), (int)(10000 * right_pixel.y));
     color_similarity_b = abs_diff((int)(10000 * left_pixel.z), (int)(10000 * right_pixel.z));
     result = color_similarity_r + color_similarity_g + color_similarity_b;
     output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * (-1) * d] = result;
    }
}