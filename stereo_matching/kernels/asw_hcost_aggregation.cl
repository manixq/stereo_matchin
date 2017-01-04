float supp_hh(float4 p, float4 q, int2 pos, int x)
{
 //SAD
 float c_diff = (-1) * (fabs(p.x - q.x) + fabs(p.y - q.y) + fabs(p.z - q.z)) / 30.91f;

 float g_dist = distance((float2)(pos.x, pos.y), (float2)(x, pos.y)) / 28.21f;
 float w = exp(c_diff - g_dist);
 return w;
}

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
 __read_only image2d_t input_r,
 __global float* vertical_cost,
 __global float* denom_v,
 __global float* output_cost
 )
{
 const int3 pos = { get_global_id(0), get_global_id(1), get_global_id(2) };
 const int2 dim = get_image_dim(input_l);

 int x = 0;
 float c_num_h = 0;
 float c_denom_h = 0;
 float ww_h = 0;
 float ww_h_ = 0;
 float4 p = read_imagef(input_l, sampler, (int2)(pos.x, pos.y)) * 255;
 float4 p_ = read_imagef(input_r, sampler, (int2)(max(pos.x - pos.z, 0), pos.y)) * 255;
 float4 q;
 float4 q_;
 for (int i = 0; i < 33; i++)
 {
  //H
  x = clamp(pos.x + i - 16, 0, dim.x );
  q = read_imagef(input_l, sampler, (int2)(x, pos.y)) * 255;
  q_ = read_imagef(input_r, sampler, (int2)(max(x - pos.z, 0), pos.y)) * 255;
  ww_h = supp_hh(p, q, (int2)(pos.x, pos.y), x);
  ww_h_ = supp_hh(p_, q_, (int2)(max(pos.x - pos.z, 0), pos.y), max(x - pos.z, 0));

  c_num_h += ww_h * ww_h_ * denom_v[x + dim.x * pos.y + dim.x * dim.y * pos.z] * vertical_cost[x + dim.x * pos.y + dim.x * dim.y * pos.z];
  c_denom_h += ww_h * ww_h_ * denom_v[x + dim.x * pos.y + dim.x * dim.y * pos.z];
 }
 float result = c_num_h / c_denom_h;
 output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = result;
}