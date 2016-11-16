

__kernel void Oii_hcross (
	__global int* cross_l,
 __global int* cross_r,
 __global int* cost,
 __global int* temp_cost,
 __global int* size
 )
{
 const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

 int h_minus_r = cross_r[pos.x - pos.z + pos.y * size[0]];
 int h_plus_r = cross_r[pos.x - pos.z + pos.y * size[0] + size[0] * size[1]];
 int h_minus_l = cross_l[pos.x + pos.y * size[0]];
 int h_plus_l = cross_l[pos.x + pos.y * size[0] + size[0] * size[1]];

 //calc current combined row
 int h_minus = select(h_minus_r, h_minus_l, isless((float)(h_minus_r), (float)(h_minus_l)));
 int h_plus = select(h_plus_l, h_plus_r, isless((float)(h_plus_r), (float)(h_plus_l)));

 int delta = h_plus - h_minus;
 int new_one = (cost[pos.x + h_plus + size[0] * pos.y + size[0] * size[1] * pos.z] - cost[pos.x + h_minus - 1 + size[0] * pos.y + size[0] * size[1] * pos.z]) / delta;
// barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 temp_cost[pos.x + size[0] * pos.y + size[0] * size[1] * pos.z] = new_one;
}
