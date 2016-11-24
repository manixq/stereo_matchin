

__kernel void asw_Cost (
	__global float* v_support_l,
 __global float* v_support_r,
 __global float* h_support_l,
 __global float* h_support_r,
 __read_only image2d_t input_l,
 __read_only image2d_t input_r,
 __global float* old_cost,
 __global float* output_cost
 )
{
   //x_width, y_height, z_support_window( <-16,16>\{0} )
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    float4 q;
    float4 q_;

    float c_num = 0;
    float c_denom = 0;
    float ww;
    for (int i = 0; i < 33; i++)
    {
     //V
     q = read_imagef(input_l, sampler, (int2)(pos.x, clamp(pos.y + i - 16, 0, dim.y - 1)));
     q_ = read_imagef(input_r, sampler, (int2)(max(0, pos.x - pos.z), clamp(pos.y + i - 16, 0, dim.y - 1)));
     ww = v_support_l[pos.x + dim.x * pos.y + dim.x * dim.y * i] * v_support_r[max(0, pos.x - pos.z) + dim.x * pos.y + dim.x * dim.y * i];

     c_num += ww * old_cost[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * pos.z];
     c_denom += ww;
    
     for (int j = 0; j < 33; j++)
     {
      //H per every V
      q = read_imagef(input_l, sampler, (int2)(clamp(pos.x + j - 16, 0, dim.x - 1), clamp(pos.y + i - 16, 0, dim.y - 1)));
      q_ = read_imagef(input_r, sampler, (int2)(clamp(pos.x + j - 16 - pos.z, 0, dim.x - 1), clamp(pos.y + i - 16, 0, dim.y - 1)));
      ww = h_support_l[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * j] * h_support_r[max(0, pos.x - pos.z) + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * j];

      c_num += ww * old_cost[clamp(pos.x + j - 16, 0, dim.x - 1) + dim.x * pos.y + dim.x * dim.y * pos.z];
      c_denom += ww;
     }
    }

    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = c_num / c_denom;
}