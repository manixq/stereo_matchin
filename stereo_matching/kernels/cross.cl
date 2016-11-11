__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

int check_similarity(image2d_t input, int2 pos, int old_one, int current_one, float4 color_data)
{
 float4 neighbour_data = read_imagef(input, sampler, pos);
 int color_similarity_r = abs_diff((int)(1000000 * color_data.x), (int)(1000000 * neighbour_data.x));
 int color_similarity_g = abs_diff((int)(1000000 * color_data.y), (int)(1000000 * neighbour_data.y));
 int color_similarity_b = abs_diff((int)(1000000 * color_data.z), (int)(1000000 * neighbour_data.z));
 //10% error - and let it go
 int check_r = select(0, 1, islessequal((float)(color_similarity_r)/10000, 20.0f));
 int check_g = select(0, 1, islessequal((float)(color_similarity_g)/10000, 20.0f));
 int check_b = select(0, 1, islessequal((float)(color_similarity_b)/10000, 20.0f));
 //if 1 then fail
 int flag = select(0, 1,isgreater((float)(current_one - old_one), 1.0f));
 current_one = select(old_one, current_one, islessequal(3.0f, (float)(check_r + check_b + check_g)));
 return select(current_one, old_one, flag);
}

int check_all(image2d_t input, int2 pos, int2 offset)
{
 int x = 1;
 float4 color_data = read_imagef(input, sampler, pos);
 int2 current = offset + offset;
 for (int i = 1; i <= 30; i++)
 {
  x = check_similarity(input, pos + current, x, i, color_data);
  current = current + offset;
 }
 return x;
}

__kernel void Cross (
	__read_only image2d_t input,
	__global int* output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 dim = get_image_dim(input);
    int h_minus = 1;
    int h_plus = 1;
    int v_minus = 1;
    int v_plus = 1;

    //start searchin H-, H+, V-, V+
    
    h_minus = check_all(input, pos, (int2)(-1, 0));
    h_plus = check_all(input, pos, (int2)( 1, 0));
    v_minus = check_all(input, pos, (int2)(0, -1));
    v_plus = check_all(input, pos, (int2)(0, 1));
    output[pos.x + dim.x * pos.y] = (-1)*h_minus;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 1] = h_plus;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 2] = (-1)*v_minus;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 3] = v_plus;
}