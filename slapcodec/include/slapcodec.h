// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef slapcodec_h__
#define slapcodec_h__

#include <stdint.h>

#ifndef bool_t
#define bool_t uint8_t
#endif

enum slapResult
{
  slapSuccess,
  slapError
};

typedef struct slapEncoder
{
  size_t frameIndex;
  size_t iframeStep;
  size_t resX;
  size_t resY;
  bool_t stereo;
} slapEncoder;

struct slapEncoder *slapCreateEncoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
void slapDestroyEncoder(struct slapEncoder *ppEncoder);

#endif // slapcodec_h__