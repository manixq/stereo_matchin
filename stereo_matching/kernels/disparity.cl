

__kernel void Disparity (
	__read_only image2d_t input,
	__global int* input_cross,
 __write_only image2d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input);

    int h_minus = input_cross[pos.x + pos.y * dim.x];
    int h_plus = input_cross[pos.x + pos.y * dim.x + dim.x * dim.y];
    int v_minus = input_cross[pos.x + pos.y * dim.x + dim.x * dim.y * 2];
    int v_plus = input_cross[pos.x + pos.y * dim.x + dim.x * dim.y * 3];

    int tab[61] = { 0 };
    float4 pixel ;
    int result = 0;
    int result_indx=0;
    for (int i = v_minus; i <= v_plus; i++)
    {
     h_minus = input_cross[pos.x + (clamp(pos.y + i, 0, dim.y - 1)) * dim.x];
     h_plus = input_cross[pos.x + (clamp(pos.y + i, 0, dim.y - 1)) * dim.x + dim.x * dim.y];
     for (int j = h_minus; j <= h_plus; j++)
     {

      pixel = read_imagef(input, sampler, pos+(int2)(j,i))*60;
      tab[(int)(pixel.x)]++;
     }
    }
    for (int i = 0; i <= 60; i++)
    {
     result_indx = select(i, result_indx, isless((float)(tab[i]), (float)(result)));
     result = select(tab[i], result, isless((float)(tab[i]), (float)(result)));
    }
    float d_result = (float)(result_indx) / 60.0;
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(d_result, d_result, d_result, 1.0f));
}