

//SUPPORT FUNCTION - DO IT YOURSELF

//for every pixel p(x,y)
__kernel void asw_ref_h (
	__global float* v_support_l,
 __global float* v_support_r,
 __global float* h_support_l,
 __global float* h_support_r,
 __read_only image2d_t input_l,
 __read_only image2d_t input_r,
 __global float* input_est,
 __global float* output_REF
 )
{
 //NAJPIERW KOSZT PER PUNKTY, DOPIERO POTEM LACZYMY KOSZT + SUPPORT AREAS
   //x_width, y_height, z_support_window( <-16,16>\{0} )
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    float4 q;
    float4 q_;

    float c_num_h = 0;
    float c_denom_h = 0;
    float ww_h;
    float F;
    for (int i = 0; i < 33; i++)
    {
     //V
    
     ww_h = h_support_l[pos.x + dim.x * pos.y + dim.x * dim.y * i];
     F = input_est[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1) + dim.x * dim.y] / input_est[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1)];
     //here input est is vertical refinement result
     c_num_h += ww_h * F * input_est[pos.x + dim.x * clamp(pos.y + i - 16, 0, dim.y - 1)];
     c_denom_h += ww_h * F;
    
    }
    float result = c_num_h / c_denom_h;
   // printf("ww %f \n",ww);
    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = result;
}