

/*
left image,
right image,
raw disparity per pixel || result cost from previous iteration,
vertical cost denominator - output,
raw disparity - not used,
output,
*/
__kernel void asw_vCostAggregation (
 __read_only image2d_t input_l,
 __global float* supp_left,
 __global float* supp_right,
 __global float* input_cost,
 __global float* output_denom,
 __global float* output_cost
 )
{
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    int y = 0;
    float c_num_v = 0.00001;
    float c_denom_v = 0.00001;
    int index_d = max(pos.x - pos.z, 0) + pos.y * dim.x;
    int index = pos.x + pos.y * dim.x;
    int size = dim.x * dim.y;
    int size_ext = size * pos.z;
    float ww = 0;
    float result = 0;

    for (int i = 0; i < 33; i++)
    {
     //V
     y = clamp(pos.y + i - 16, 0, dim.y-1);
     ww = supp_left[index + size * i] *supp_right[index_d + size * i];
     c_num_v += ww * input_cost[pos.x + dim.x * y + size_ext];
     c_denom_v += ww;
    }
    result = c_num_v / c_denom_v;
    output_cost[index + size_ext] = result;
    output_denom[index + size_ext] = c_denom_v;
}