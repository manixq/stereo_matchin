

__kernel void asw_hSupport (
	__read_only image2d_t input,
 __global float* output_cost
 )
{
   //x_width, y_height, z_support_window( <-16,16>\{0} )
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input);

    float4 p;
    float4 q;

    float g_dist;
    float c_diff;
    float w;
   
   int x = clamp(pos.x + pos.z - 16, 0, dim.x - 1);
    p = read_imagef(input, sampler, (int2)(pos.x, pos.y));
    q = read_imagef(input, sampler, (int2)(x, pos.y));
    c_diff = (-1) * distance(p, q) / 30.91f;

    g_dist = distance((float2)(pos.x, pos.y), (float2)(x, pos.y)) / 28.21f;
    w = exp( c_diff - g_dist);

    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = w;
}