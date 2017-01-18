

__kernel void Asw(
 __read_only image2d_t input_l,
 __global float* supp_left,
 __global float* supp_right,
 __global float* init_cost,
 __global float* output_cost
 )
{
 //dla kazdego cnum dnum i w vcost_aggr tylko juz sumuj
 
 const int3 pos = { get_global_id(0), get_global_id(1), get_global_id(2) };
 const int2 dim = get_image_dim(input_l);
 
 float ww = supp_left[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] * supp_right[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z];
 output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = ww;
}