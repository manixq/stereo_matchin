//p1 is [d1, x1] where d1 is depth
int bresenham(int2 p1, int2 p2, int x)
{
 int y = p1.x;
 if((p1.y - p2.y) != 0)
  y = (p1.x - p2.x) / (p1.y - p2.y) * (x - p2.y) + p2.x;
 return y;
}


__kernel void asw_WTA_REF (
 __global float* raw_d,
 __global float* refinement,
 __write_only image2d_t output_l,
 __write_only image2d_t output_r,
 __global float* out_d,
 __global float* out_fractional
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(output_l);

    float current_cost = output_cost[pos.x + dim.x * pos.y];
    float temp;
    int d_r = 0;    
    int min_d = 0;

    int last_min = 0;
    int last_min_r = 0;

    for (int i = 0; i < 61; i++)
    {
     temp = output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * i];
     last_min = select(last_min, min_d, isless(temp, current_cost));
     min_d = select(min_d, i, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
    }

    //from d = zero or from  d = dminR??

    /*
    int d_xr = 0;
    d_r = min_d;
    current_cost = output_cost[max(0, pos.x - d_r) + dim.x * pos.y];
    for (int i = 0; i < 61; i++)
    {
     temp = output_cost[max(0, pos.x - d_r) + dim.x * pos.y + dim.x * dim.y * i];
     d_xr = select(d_xr, i, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
    }
    */

    int b = 0;
    d_r = min_d;
    int min_d_r = min_d;
    current_cost = output_cost[max(0, pos.x - d_r) + dim.x * pos.y];
    for (int i = 0; i < d_r; i++)
    {
     b = bresenham((int2)(0, pos.x - d_r), (int2)(min_d, pos.x), max(0, pos.x - i)); //d_xr instead of 0
     temp = output_cost[max(0, pos.x - i) + dim.x * pos.y + dim.x * dim.y * b];
     last_min_r = select(last_min_r, min_d_r, isless(temp, current_cost));
     min_d_r = select(min_d_r, b, isless(temp, current_cost));
     current_cost = select(current_cost, temp, isless(temp, current_cost));
    }
    

    float result_l = (float)(min_d) / 60.0f;
    float result_r = (float)(min_d_r) / 60.0f;
    write_imagef(output_r, (int2)(pos.x, pos.y), (float4)(result_r, result_r, result_r, 1.0f));
    write_imagef(output_l, (int2)(pos.x, pos.y), (float4)(result_l, result_l, result_l, 1.0f));
    out_d[pos.x + dim.x * pos.y ] = min_d;
    out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 1] = min_d_r;
    //second min
    out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 2] = last_min;
    out_d[pos.x + dim.x * pos.y + dim.x * dim.y * 3] = last_min_r;
}