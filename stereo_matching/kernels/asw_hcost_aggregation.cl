

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
 float ww_h = 0;
 float ww_h_ = 0;
 for (int i = 0; i < 33; i++)
 {
  //H
  x = clamp(pos.x + i - 16, 0, dim.x );
  ww_h = supp_left[pos.x + pos.y * dim.x + dim.x * dim.y * i];
  ww_h_ = supp_right[max(pos.x - pos.z, 0) + pos.y * dim.x + dim.x * dim.y * i];

  c_num_h += ww_h * ww_h_ * denom_v[x + dim.x * pos.y + dim.x * dim.y * pos.z] * vertical_cost[x + dim.x * pos.y + dim.x * dim.y * pos.z];
  c_denom_h += ww_h * ww_h_ * denom_v[x + dim.x * pos.y + dim.x * dim.y * pos.z];
 }
 float result = c_num_h / c_denom_h;
 output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = result;
}