

__kernel void asw_vCost (
	__global float* v_support_l,
 __global float* v_support_r,
 __global float* h_support_l,
 __global float* h_support_r,
 __read_only image2d_t input_l,
 __read_only image2d_t input_r,
 __global float* input_cost,
 __global float* output_cost
 )
{
 //NAJPIERW KOSZT PER PUNKTY, DOPIERO POTEM LACZYMY KOSZT + SUPPORT AREAS
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    float4 q;
    float4 q_;

    float c_num_v = 0;
    float c_denom_v = 0; 
    float c_num_h = 0;
    float c_denom_h = 0;
    float ww_v;
    float ww_h;
    for (int i = 0; i < 33; i++)
    {
     //V
    
     ww_v = v_support_l[pos.x + dim.x * pos.y + dim.x * dim.y * i] * v_support_r[max(0, pos.x - pos.z) + dim.x * pos.y + dim.x * dim.y * i];//- pos.z
     //verticals are just summed up coz it was explained like that in IEE article
     c_num_v +=  input_cost[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * pos.z];
     //c_denom_v += ww_v;
    // printf("ww: %f\n",ww);
    // printf("cost: %f\n", input_cost[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * pos.z]);
     for (int j = 0; j < 33; j++)
     {
      //H per every V
      ww_h = h_support_l[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * j] * h_support_r[max(0, pos.x - pos.z) + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * j];// - pos.z
      c_num_h += ww_h * ww_v * input_cost[clamp(pos.x + j - 16, 0, dim.x - 1) + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y * pos.z];
      c_denom_h += ww_h;
     }
    }
    float result = c_num_v/33  + c_num_h / c_denom_h;
   // printf("ww %f \n",ww);
    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = result;
}