

/*
left image,
right image,
raw disparity per pixel || result cost from previous iteration,
raw disparity - added by me to improve quality,
result vertical cost,
vertical cost denominator,
output,
*/
__kernel void asw_hCostAggregation(
 __read_only image2d_t input_l,
 __global float* supp_left,
 __global float* supp_right,
 __global float* vertical_cost,
 __global float* denom_v,
 __global float* output_cost
 )
{
 const int3 pos = { get_global_id(0), get_global_id(1), get_global_id(2) };
 const int2 dim = get_image_dim(input_l);

 float x_mult = 500;
 int x = 0;
 float c_num_h = 0.00001;
 float c_denom_h = 0.00001;
 float ww = 0; 
 int index_d = max(pos.x - pos.z, 0) + pos.y * dim.x;
 int index = pos.x + pos.y * dim.x;
 int size = dim.x * dim.y;
 int size_ext = size * pos.z;

 for (int i = 0; i < 33; i++)
 {
  //H
  x = clamp(pos.x + i - 16, 0, dim.x - 1);
  ww = supp_left[index + size * i] * supp_right[index_d + size * i];
  c_num_h += ww * denom_v[x + dim.x * pos.y + size_ext] * vertical_cost[x + dim.x * pos.y + size_ext];
  c_denom_h += ww * denom_v[x + dim.x * pos.y + size_ext];
 }
 float result = c_num_h / c_denom_h;
 output_cost[pos.x + dim.x * pos.y + size_ext] = result;
}