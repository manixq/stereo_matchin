__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;



__kernel void Integral_v (
 __global int* new_cost,
 __global int* size
 )
{
 //x=width, y=depth
 const int2 pos = {get_global_id(0), get_global_id(1)};

 int sum = 0;
 for (int i = 0; i < size[1]; i++)
 {
  sum = sum + new_cost[pos.x + size[0] * i + size[0] * size[1] * pos.y];
  new_cost[pos.x + size[0] * i + size[0] * size[1] * pos.y] = sum;
 }
}