

__kernel void asw_hAggregation (
	__read_only image2d_t input,
 __global int* output_cost
 )
{
   //x_width, y_height, z_support_window( <-16,16>\{0} )
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    float4 p;
    float4 q;

    float g_dist;
    float c_diff;
    float w;
   
    int y = clamp(pos.y + pos.z - 16, 0, dim.y - 1);
    p = read_imagef(input, sampler, (int2)(pos.x, pos.y));
    q = read_imagef(input, sampler, (int2)(pos.x, y));
    c_diff = fast_distance(p, q) / 30.91f;

    g_dist = fast_distance(pos, (int2)(pos.x, y)) / 28.21f;
    w = exp(- c_diff - g_dist);

    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = w;
}