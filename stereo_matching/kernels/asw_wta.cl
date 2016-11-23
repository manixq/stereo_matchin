

__kernel void asw_WTA (
 __global float* output_cost,
 __write_only image2d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output);

    float current_cost = output_cost[pos.x + dim.x * pos.y];
    float temp;
    int min_d = 0;
    for (int i = 0; i < 61; i++)
    {
     temp = output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * i];
     min_d = select(min_d, i, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
    }
    float result = (float)(min_d) / 60.0f;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
    
}