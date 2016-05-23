
__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


__kernel void Filter (
	__read_only image2d_t input,
 __read_only image2d_t input2,
	__write_only image2d_t output,
 __constant int* my_width,
 __local int* local_var)
{

    const int2 pos = {get_global_id(0), get_global_id(1)};

     float4 sum = (float4)(0.0f);
     float min = 0.0;
     float next = 0.0;
     float4 a1 = read_imagef(input, sampler, pos);
     float4 a2;
     float4 a3;

     if (pos.x < *my_width)
      a2 = read_imagef(input, sampler, pos + (int2)(1, 0));
     else
      a2 = a1;

     if (pos.x != 0)
      a3 = read_imagef(input, sampler, pos + (int2)(-1, 0));
     else
      a3 = a1;

     float4 b1 = (float4)(0.0f);
     float4 b2 = (float4)(0.0f);
     float4 b3 = (float4)(0.0f);

     for (float x = 0;; x ++) 
     {
      if (pos.x + x > *my_width)
       break;
      b1 = read_imagef(input2, sampler, pos + (int2)(x, 0));

      if (pos.x + x < *my_width)
       b2 = read_imagef(input2, sampler, pos + (int2)(x + 1, 0));
      else
       b2 = b1;

      if (pos.x != 0)
       b3 = read_imagef(input2, sampler, pos + (int2)(x - 1, 0));
      else
       b3 = b1;

      next = 0;

      if (!x)
       for (int i = 0; i < 3; i++)
       {
        min += fabs(b1[i] - a1[i]);
        min += fabs(b2[i] - a2[i]);
        min += fabs(b3[i] - a3[i]);
       }


      for (int i = 0; i < 3; i++)
      {
       next += fabs(b1[i] - a1[i]);
       next += fabs(b2[i] - a2[i]);
       next += fabs(b3[i] - a3[i]);
      }


     

      if (next < min)
      {
       min = next;
       sum = (float4)(x / 45, x / 45, x / 45, 1.0);
      }  
    }


     //barrier(CLK_LOCAL_MEM_FENCE);
    write_imagef (output, (int2)(pos.x, pos.y), sum);
}