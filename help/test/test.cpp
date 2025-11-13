#include <vector>
#include <iostream>

void PixelShuffle(float* input, float* output, int height, int width, int in_channels, int upscale_factor)
{
    int r = upscale_factor;
    int out_h = height * r;
    int out_w = width * r;
    int in_c = in_channels;
    int out_c = in_c / (r * r);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < out_c; ++c)
            {
                for (int i = 0; i < r; ++i)
                {
                    for (int j = 0; j < r; ++j)
                    {
                        int in_index  = ((h * width + w) * in_c) + c * (r * r) + i * r + j;
                        int out_h_idx = h * r + i;
                        int out_w_idx = w * r + j;
                        int out_index = ((out_h_idx * out_w + out_w_idx) * out_c) + c;

                        output[out_index] = input[in_index];
                    }
                }
            }
        }
    }
}

int main()
{
    int height2 = 3;
    int width2  = 3;
    int upscale_factor = 2;
    int in_channels = 4;   // 3x3x4 input
    int out_channels = in_channels / (upscale_factor * upscale_factor); // -> 1

    std::vector<float> output_chw(height2 * width2 * in_channels);
    float* mpDnnOutBuf = output_chw.data();

    std::vector<float> output((height2 * upscale_factor) * (width2 * upscale_factor) * out_channels);
    float* mpTileOutBuf = output.data();



    // Fill NHWC input with sequential values to visualize mapping
    int val = 0;
    for (int h = 0; h < height2; ++h)
    {
        for (int w = 0; w < width2; ++w)
        {
            int base = ((h * width2 + w) * in_channels);
            for (int c = 0; c < in_channels; ++c)
            {
                mpDnnOutBuf[base + c] = static_cast<float>(++val);
            }
        }
    }

    // Perform pixel shuffle
    PixelShuffle(mpDnnOutBuf, mpTileOutBuf, height2, width2, in_channels, upscale_factor);

     // Print output matrix
    std::cout << "Pixel shuffled matrix:\n";
    for (int i = 0; i < (height2 * upscale_factor) * (width2 * upscale_factor) * out_channels; ++i)
    {
        std::cout << output[i] << " ";
        if ((i + 1) % ((width2 * upscale_factor) * out_channels) == 0)
            std::cout << "\n";
    }
    std::cout << "\n";
    return 0;
}
