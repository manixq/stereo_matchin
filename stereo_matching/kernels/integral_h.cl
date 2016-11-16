


__kernel void Integral_h (
 __global int* cost,
 __global int* size
 )
{
 //x=height, y=depth
 const int2 pos = {get_global_id(0), get_global_id(1)};

 int sum = 0;
 for (int i = 0; i < size[0]; i++)
 {
  sum = sum + cost[i + size[0] * pos.x + size[0] * size[1] * pos.y];
  cost[i + size[0] * pos.x + size[0] * size[1] * pos.y] = sum;
 }
}