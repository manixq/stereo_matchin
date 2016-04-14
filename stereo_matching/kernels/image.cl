__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


int ispixeleq(float4 a,float4 b)
{
 if (a[0] <= b[0] + 0.01 && a[0] >= b[0] - 0.01 && a[1] <= b[1]+0.01 && a[1] >= b[1] - 0.01 && a[2] <= b[2] + 0.01 && a[2] >= b[2] - 0.01 )
 {
  return 1;
 }
 else
 return 0;
}
__kernel void Filter (
	__read_only image2d_t input,
 __read_only image2d_t input2,
	__constant float* filterWeights,
	__write_only image2d_t output,
 __constant int* my_width)
{

    const int2 pos = {get_global_id(0), get_global_id(1)};

    float4 sum = (float4)(0.0f);
    
     float4 a1 = read_imagef(input, sampler, pos);
     float4 a2 = read_imagef(input, sampler, pos + (int2)(1, 0));
     float4 a3 = read_imagef(input, sampler, pos + (int2)(2, 0));
     for (float x = 1; x <= *my_width; x ++) {
      float4 b1 = read_imagef(input2, sampler, pos + (int2)(x, 0));
      float4 b2 = read_imagef(input2, sampler, pos + (int2)(x + 1, 0));
      float4 b3 = read_imagef(input2, sampler, pos + (int2)(x + 2, 0));
      if (ispixeleq(a1, b1) || ispixeleq(a2, b2) || ispixeleq(a3, b3))//mogloby byc a1,b2 ; a1,b3 ->ale wtedy nie 'rozmyje'
      {
       sum = (float4)(0, x / (*my_width) * 30, 0, 1);
       break;
      }
     
    }
    write_imagef (output, (int2)(pos.x, pos.y), sum);
}