

__kernel void Oii_vcross (
	__global int* cross_l,
 __global int* cross_r,
 __global int* temp_cost,
 __global int* cost,
 __global int* size
 )
{
 const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

 int area = size[0] * size[1];
 int temp_val = max(0, pos.x - pos.z) + pos.y * size[0];

 int v_minus_r = cross_r[temp_val + area * 2];
 int v_plus_r = cross_r[temp_val + area * 3];

 temp_val = pos.x + pos.y * size[0];
 int v_minus_l = cross_l[temp_val + area * 2];
 int v_plus_l = cross_l[temp_val + area * 3];

 //calc current combined row
 int v_minus = max(v_minus_l, v_minus_r);
 int v_plus = min(v_plus_l, v_plus_r);

 int delta = v_plus - v_minus;
 int new_one = (int)((temp_cost[pos.x + size[0] * min(size[1] - 1, pos.y + v_plus) + area * pos.z] - temp_cost[pos.x + size[0] * max(0, pos.y + v_minus) + area * pos.z])/delta);
 //printf("\n%d $d %d %d %d", pos.x ,size[0] , min(size[1] - 1, pos.y + v_plus) , area , pos.z);
 //printf("\n%d", temp_cost[pos.x + size[0] * min(size[1] - 1, pos.y + v_plus) + area * pos.z]);
 cost[pos.x + size[0] * pos.y + area * pos.z] = new_one;
}
