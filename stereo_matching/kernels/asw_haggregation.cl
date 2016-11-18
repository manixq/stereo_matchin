

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

     float g_dist = fast_distance(pos, pos + (int2)(d, 0));
     float c_diff = fast_distance(left_pixel, right_pixel);
     float w = exp(-c_diff - g_dist);

     output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * (-1) * d] = (int)(result/3);
    }
}



