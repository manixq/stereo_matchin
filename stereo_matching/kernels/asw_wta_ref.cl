
__kernel void asw_WTA_REF (
 __global float* agg_d,
 __global float* ref,
 __global float* ref_target,
 __write_only image2d_t output,
 __write_only image2d_t output_target,
 __global float* disp_ref,
 __global float* disp_ref_target,
 __global float* confidence,
 __global float* confidence_target
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output);

    int min_d = 0;
    float penalty;
    float current_cost = 100000;
    float last_current_cost = 100000;
    int d_r = 0;

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

    int b = 0;
    d_r = min_d;
    int min_d_r = min_d;
    float current_cost_target = 100000;
    float last_current_cost_target = 100000;
    for (int i = 0; i < d_r; i++)
    {
     b = bresenham((int2)(0, pos.x - d_r), (int2)(min_d, pos.x), max(0, pos.x - i)); //d_xr instead of 0
                                                                                     //last_current_cost_target = output_cost[max(0, pos.x - i) + dim.x * pos.y + dim.x * dim.y * b];
                                                                                     //min_d_r = select(min_d_r, b, isless(last_current_cost_target, current_cost_target));
                                                                                     // current_cost_target = select(current_cost_target, last_current_cost_target, isless(last_current_cost_target, current_cost_target));
     penalty = 0.085f * ref_target[pos.x + dim.x * pos.y + dim.x * dim.y] * fabs(ref_target[pos.x + dim.x * pos.y] - i) + agg_d[max(0, pos.x - i) + dim.x * pos.y + dim.x * dim.y * b];
     last_current_cost_target = select(last_current_cost_target, penalty, isless(penalty, last_current_cost_target));
     min_d_r = select(min_d_r, b, isless(penalty, current_cost_target));
     last_current_cost_target = select(last_current_cost_target, current_cost_target, isless(penalty, current_cost_target));
     current_cost_target = select(current_cost_target, penalty, isless(penalty, current_cost_target));
    }





    float result = (float)(min_d) / 60.0f;
    float result_target = (float)(min_d_r) / 60.0f;

    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
    write_imagef(output_target, (int2)(pos.x, pos.y), (float4)(result_target, result_target, result_target, 1.0f));
    disp_ref[pos.x + dim.x * pos.y] = (float)(min_d);
    confidence[pos.x + dim.x * pos.y] = (last_current_cost - current_cost) / last_current_cost;

    disp_ref_target[pos.x + dim.x * pos.y] = (float)(min_d_r);
    confidence[pos.x + dim.x * pos.y] = (last_current_cost_target - current_cost_target) / last_current_cost_target;
    
}