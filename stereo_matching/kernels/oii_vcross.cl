__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;



__kernel void Oii_vcross (
	__global int* cross_l,
 __global int* cross_r,
 __global int* cost_pong,
 __global int* cost_ping,
 __global int* size
 )
{
 const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

 int v_minus_r = cross_r[pos.x - pos.z + pos.y * size[0] + size[0] * size[1] * 2];
 int v_plus_r = cross_r[pos.x - pos.z + pos.y * size[0] + size[0] * size[1] * 3];
 int v_minus_l = cross_l[pos.x + pos.y * size[0] + size[0] * size[1] * 2];
 int v_plus_l = cross_l[pos.x + pos.y * size[0] + size[0] * size[1] * 3];

 //calc current combined row
 int v_minus = select(v_minus_r, v_minus_l, islessequal((float)(v_minus_r), (float)(v_minus_l)));
 int v_plus = select(v_plus_l, v_plus_r, islessequal((float)(v_plus_r), (float)(v_plus_l)));
 int delta = v_plus - v_minus;
 cost_ping[pos.x + size[0] * pos.y + size[0] * size[1] * pos.z] = (cost_pong[pos.x + size[0] * (pos.y + v_plus) + size[0] * size[1] * pos.z] - cost_pong[pos.x + size[0] * (pos.y + v_minus) + size[0] * size[1] * pos.z]) / delta;
}
