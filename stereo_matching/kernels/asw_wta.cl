//p1 is [d1, x1] where d1 is depth

int bresenham(int2 p1, int2 p2, int x)
{
 int y = p1.x;
 if((p1.y - p2.y) != 0)
  y = (p1.x - p2.x) / (p1.y - p2.y) * (x - p2.y) + p2.x;
 return y;
}


__kernel void asw_WTA (
 __global float* output_cost,
 __write_only image2d_t output,
 __global float* d_est_reference,
 __global float* d_est_target,
 __write_only image2d_t output_target,
 __global float* confidence_reference,
 __global float* confidence_target
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output);

    float current_cost = 100000;
    float last_current_cost = 100000;
    int d_r = 0;    
    int min_d = 0;

    

    float temp;

    for (int i = 0; i < 61; i++)
    {
     
     temp = output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * i];
     /*
     last_current_cost = select(last_current_cost, current_cost, isless(temp, current_cost));
     min_d = select(min_d, i, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
     */
     last_current_cost = select(last_current_cost, temp, isless(temp, last_current_cost));
     min_d = select(min_d, i, isless(temp, current_cost));
     last_current_cost = select(last_current_cost, current_cost, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
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

     temp = output_cost[max(0, pos.x - i) + dim.x * pos.y + dim.x * dim.y * b];
     last_current_cost_target = select(last_current_cost_target, temp, isless(temp, last_current_cost_target));
     min_d_r = select(min_d_r, b, isless(temp, current_cost_target));
     last_current_cost_target = select(last_current_cost_target, current_cost_target, isless(temp, current_cost_target));
     current_cost_target = select(current_cost_target, temp, isless(temp, current_cost_target));
    }
    
    
    float result = (float)(min_d) / 60.0f;
    float result_target = (float)(min_d_r) / 60.0f;
  
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(result, result, result, 1.0f));
    write_imagef(output_target, (int2)(pos.x, pos.y), (float4)(result_target, result_target, result_target, 1.0f));

    d_est_reference[pos.x + dim.x * pos.y ] = min_d;
    confidence_reference[pos.x + dim.x * pos.y] = (last_current_cost - current_cost) / last_current_cost;
    
    d_est_target[pos.x + dim.x * pos.y] = min_d_r;
    confidence_target[pos.x + dim.x * pos.y ] = (last_current_cost_target - current_cost_target) / last_current_cost_target;
   
}