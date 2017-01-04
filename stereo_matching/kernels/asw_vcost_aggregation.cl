
float supp_vv(float4 p, float4 q, int2 pos, int y)
{
 //SAD
 float c_diff = (-1) * (fabs(p.x - q.x) + fabs(p.y - q.y) + fabs(p.z - q.z)) / 30.91f;

 float g_dist = distance((float2)(pos.x, pos.y), (float2)(pos.x, y)) / 28.21f;

 float w = exp(c_diff - g_dist);
 return w;
}

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
 __read_only image2d_t input_r,
 __global float* input_cost,
 __global float* output_denom,
 __global float* output_cost
 )
{
    const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    const int2 dim = get_image_dim(input_l);

    int y = 0;
    float c_num_v = 0;
    float c_denom_v = 0;
    float ww_v = 0;
    float ww_v_ = 0;

    // * 255 okazuje sie byc kluczowe. Wartosc ta jest optymalna, 
    // dla nizszych otrzymujemy znaczna deformacje, dla wyzszych wyostrzamy szczegoly kosztem wysokich zaklocen
    float4 p = read_imagef(input_l, sampler, (int2)(pos.x, pos.y)) * 255; 
    float4 p_ = read_imagef(input_r, sampler, (int2)(max(pos.x - pos.z, 0), pos.y)) * 255;
    float4 q;
    float4 q_;
    
    for (int i = 0; i < 33; i++)
    {
     //V
     y = clamp(pos.y + i - 16, 0, dim.y);
     q = read_imagef(input_l, sampler, (int2)(pos.x, y)) * 255;
     q_ = read_imagef(input_r, sampler, (int2)(max(pos.x - pos.z, 0), y)) * 255;
     ww_v = supp_vv(p, q, (int2)(pos.x, pos.y), y);
     ww_v_ = supp_vv(p_, q_, (int2)(max(pos.x - pos.z, 0), pos.y), y);

     c_num_v += ww_v * ww_v_ * input_cost[pos.x + dim.x * y + dim.x * dim.y * pos.z];
     c_denom_v += ww_v * ww_v_;
    }
    float result = c_num_v / c_denom_v;
    output_cost[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = result; 
    output_denom[pos.x + dim.x * pos.y + dim.x * dim.y * pos.z] = c_denom_v;
}