
__kernel void asw_WTA_REF (
 __global float* agg_d,
 __global float* ref,
 __write_only image2d_t output,
 __global float* disp_ref,
 __global float* confidence
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output);

    int min_d = 0;
    float penalty = 0.085f * ref[pos.x + dim.x * pos.y + dim.x * dim.y] * fabs(ref[pos.x + dim.x * pos.y] - 0);
    float current_cost = 100000;
    float last_current_cost = 100000;
    
    for (int i = 0; i < 61; i++)
    {
     //its actualy penalty + cost
     penalty = 0.085f * ref[pos.x + dim.x * pos.y + dim.x * dim.y] * fabs(ref[pos.x + dim.x * pos.y] - i) + agg_d[pos.x + dim.x * pos.y + dim.x * dim.y * i];

     //need little fix here, same in asw_wta
     last_current_cost = select(last_current_cost, penalty, isless(penalty, last_current_cost));
     min_d = select(min_d, i, isless(penalty, current_cost));
     last_current_cost = select(last_current_cost, current_cost, isless(penalty, current_cost));
     current_cost = select(current_cost, penalty, isless(penalty, current_cost));
    }
    float result = (float)(min_d) / 60.0f;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
    disp_ref[pos.x + dim.x * pos.y] = (float)(min_d);
    confidence[pos.x + dim.x * pos.y] = (last_current_cost - current_cost) / last_current_cost;
    
}