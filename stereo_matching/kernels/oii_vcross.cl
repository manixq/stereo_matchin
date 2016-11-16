

__kernel void Oii_vcross (
	__global int* cross_l,
 __global int* cross_r,
 __global int* temp_cost,
 __global int* cost,
 __global int* size
 )
{
 const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

 int v_minus_r = cross_r[pos.x - pos.z + pos.y * size[0] + size[0] * size[1] * 2];
 int v_plus_r = cross_r[pos.x - pos.z + pos.y * size[0] + size[0] * size[1] * 3];
 int v_minus_l = cross_l[pos.x + pos.y * size[0] + size[0] * size[1] * 2];
 int v_plus_l = cross_l[pos.x + pos.y * size[0] + size[0] * size[1] * 3];

 //calc current combined row
 int v_minus = select(v_minus_r, v_minus_l, isless((float)(v_minus_r), (float)(v_minus_l)));
 int v_plus = select(v_plus_l, v_plus_r, isless((float)(v_plus_r), (float)(v_plus_l)));

 int delta = v_plus - v_minus;
 int new_one = (temp_cost[pos.x + size[0] * (pos.y + v_plus) + size[0] * size[1] * pos.z] - temp_cost[pos.x + size[0] * (pos.y + v_minus - 1) + size[0] * size[1] * pos.z]) / delta;
// barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 cost[pos.x + size[0] * pos.y + size[0] * size[1] * pos.z] = new_one;
}
