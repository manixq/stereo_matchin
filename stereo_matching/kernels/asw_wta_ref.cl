
__kernel void asw_WTA_REF (
 __global float* raw_d,
 __global float* ref,
 __write_only image2d_t output,
 __global float* out_d,
 __global float* out_fractional
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output_l);

    int min_d = 0;
    float penalty = 0.085 * ref[pos.x + dim.x * pos.y + dim.x * dim.y] * fabs(ref[pos.x + dim.x * pos.y] - 0);
    float current_cost = raw_d[pos.x + dim.x * pos.y + dim.x * dim.y * 0] + penalty;
    float last_current_cost = raw_d[pos.x + dim.x * pos.y + dim.x * dim.y * 0] + penalty;

    for (int i = 1; i < 61; i++)
    {
     penalty = 0.085 * ref[pos.x + dim.x * pos.y + dim.x * dim.y] * fabs(ref[pos.x + dim.x * pos.y] - i) + raw_d[pos.x + dim.x * pos.y + dim.x * dim.y * i];

     last_current_cost = select(last_current_cost, current_cost, isless(penalty, current_cost));
     min_d = select(min_d, i, isless(penalty, current_cost));
     current_cost = select(current_cost, penalty, isless(penalty, current_cost));
    }

    float result = (float)(min_d) / 60.0f;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
    out_d[pos.x + dim.x * pos.y] = min_d;
    out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 1] = current_cost;
    out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 2] = last_current_cost;//min_d_r;
    //second min
    //out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 2] = last_min;
   // out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 3] = last_min_r;
}