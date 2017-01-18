

__kernel void Init_disparity (
 __global float* cost,
 __write_only image2d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output);
    int min_d = 0;
    float min_result = cost[pos.x + dim.x * pos.y];

    for (int i = 0; i <= 60; i++)
    {
     min_d = select(min_d, i, isless((float)(cost[pos.x + dim.x * pos.y + dim.x * dim.y * i]), (float)(min_result)));
     min_result = select(min_result, cost[pos.x + dim.x * pos.y + dim.x * dim.y * i], isless((float)(cost[pos.x + dim.x * pos.y + dim.x * dim.y * i]), (float)(min_result)));
    }
    float d_result = (float)(min_d) / 60.0f;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(d_result, d_result, d_result, 1.0f));
}