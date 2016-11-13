__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;



__kernel void Integral_h (
 __global int* cost_ping,
 __global int* cost_pong,
 __global int* size
 )
{
 const int2 pos = {get_global_id(1), get_global_id(2)};

 int sum = cost_ping[size[0] * pos.x + size[0] * size[1] * pos.y];
 for (int i = 1; i < size[0]; i++)
 {
  cost_pong[i + size[0] * pos.x + size[0] * size[1] * pos.y] = sum + cost_ping[i + size[0] * pos.x + size[0] * size[1] * pos.y];
  sum = sum + cost_ping[i + size[0] * pos.x + size[0] * size[1] * pos.y];
 }
}