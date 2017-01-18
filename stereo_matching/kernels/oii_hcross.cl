

__kernel void Oii_hcross (
	__global int* cross_l,
 __global int* cross_r,
 __global float* cost,
 __global float* temp_cost,
 __global int* size
 )
{
 const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

 int temp_val = max(0, pos.x - pos.z) + pos.y * size[0];
 int area = size[0] * size[1];

 int h_minus_r = cross_r[temp_val];
 int h_plus_r = cross_r[temp_val + area];

 temp_val = pos.x + pos.y * size[0];

 int h_minus_l = cross_l[temp_val];
 int h_plus_l = cross_l[temp_val + area];

 //calc current combined row
 int h_minus = max(h_minus_r, h_minus_l);
 int h_plus = min(h_plus_r, h_plus_l);

 int delta = h_plus - h_minus;
 float new_one = (cost[min(size[0]-1, pos.x + h_plus) + size[0] * pos.y + area * pos.z] - cost[max(0, pos.x + h_minus - 1) + size[0] * pos.y + area * pos.z]) / delta;
// barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 temp_cost[pos.x + size[0] * pos.y + area * pos.z] = new_one;
}
