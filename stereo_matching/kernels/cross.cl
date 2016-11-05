__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;

int check_similarity(image2d_t input, int2 pos, int old_one, int current_one, float4 color_data)
{
 float4 neighbour_data = read_imagef(input, sampler, pos);

 uint color_similarity_r = abs_diff((int)(color_data.x), (int)(neighbour_data.x));
 uint color_similarity_g = abs_diff((int)(color_data.y), (int)(neighbour_data.y));
 uint color_similarity_b = abs_diff((int)(color_data.z), (int)(neighbour_data.z));

 int check_r = select(0, 1, islessequal(color_similarity_r, 5));
 int check_g = select(0, 1, islessequal(color_similarity_g, 5));
 int check_b = select(0, 1, islessequal(color_similarity_b, 5));
 //if 1 then fail
 int flag = select(0, 1,isgreater(current_one - old_one, 1));
 current_one = select(old_one, current_one, isequal(3, check_r + check_b + check_g));
 return select(current_one, old_one, flag);
}

int check_all(image2d_t input, int2 pos, int2 offset)
{
 int x = 1;
 float4 color_data = read_imagef(input, sampler, pos);
 int2 current = offset + offset;

 x= check_similarity(input, pos + current, x, 2, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 3, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 4, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 5, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 6, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 7, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 8, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 9, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 10, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 11, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 12, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 13, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 14, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 15, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 16, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 17, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 18, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 19, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 20, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 21, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 22, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 23, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 24, color_data);
 current = current + offset;
 x = check_similarity(input, pos + current, x, 25, color_data);
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

    //start searchin H-
    h_minus = check_all(input, pos,(int2)(-1, 0));
    h_plus = check_all(input, pos, (int2)( 1, 0));
    
    output[pos.x + dim.x * pos.y] = h_minus;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 1] = 2;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 2] = 3;
    output[pos.x + dim.x * pos.y + dim.x*dim.y * 3] = 4;
}