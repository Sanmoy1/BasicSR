#include <iostream>
#include <iomanip>
#include <cstring>

using namespace std;

void pixel_unshuffle_nhwc( float* in, float* out,
                          int h, int w, int c, int r = 2)
{
    int out_h = h / r;
    int out_w = w / r;
    int out_c = c * r * r;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {

            int out_base = (oh * out_w + ow) * out_c;

            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < r; ++j) {

                    int ih = oh * r + i;//input height
                    int iw = ow * r + j;//input width

                    int in_base = (ih * w + iw) * c;//input base
                    int out_channel_offset = (i * r + j) * c;//output channel offset

                    memcpy(out + out_base + out_channel_offset,
                           in + in_base,
                           sizeof(float) * c);//copy c values
                }
            }
        }
    }
}

int main()
{
    int H = 6, W = 6, C = 1;
    int r = 2;

    int OH = H / r;
    int OW = W / r;
    int OC = C * r * r;

    float* input  = new float[H * W * C];
    float* output = new float[OH * OW * OC];

    // Populate input with sequential values: 1, 2, 3, ...
    for (int i = 0; i < H * W * C; i++)
        input[i] = i + 1;

    // Print Input
    cout << "Input (6x6x1):" << endl;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            cout << setw(4) << input[h * W + w];
        }
        cout << endl;
    }
    cout << endl;

    // Pixel Unshuffle
    pixel_unshuffle_nhwc(input, output, H, W, C, r);

    // Print Output (3x3x4)
    cout << "Output (3x3x4):" << endl;

    for (int h = 0; h < OH; ++h) {
        for (int w = 0; w < OW; ++w) {

            cout << "Block (" << h << "," << w << "): ";
            cout << "[ ";

            for (int ch = 0; ch < OC; ++ch) {
                int idx = (h * OW + w) * OC + ch;
                cout << output[idx];
                if (ch < OC - 1) cout << ", ";
            }

            cout << " ]" << endl;
        }
        cout << endl;
    }
    return 0;
}
