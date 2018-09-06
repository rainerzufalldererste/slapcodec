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
#include <malloc.h>
#include <memory.h>
#include <stdio.h>

#ifndef bool_t
#define bool_t uint8_t
#endif // !bool_t

#ifndef IN
#define IN
#endif // !IN

#ifndef OUT
#define OUT
#endif // !OUT

#ifndef IN_OUT
#define IN_OUT IN OUT
#endif // !IN_OUT

#define slapAlloc(Type, count) (Type *)malloc(sizeof(Type) * (count))
#define slapFreePtr(ptr)  do { if (ptr && *ptr) { free(*ptr); *ptr = 0; } } while (0)
#define slapSetZero(ptr, Type) memset(ptr, 0, sizeof(Type))

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum slapResult
  {
    slapSuccess,
    slapError_Generic,
    slapError_ArgumentNull,
    slapError_Compress_Internal,
    slapError_FileError
  } slapResult;

  typedef struct slapEncoder
  {
    size_t frameIndex;
    size_t iframeStep;
    size_t resX;
    size_t resY;
    bool_t stereo;
    int quality;
    void *pAdditionalData;
  } slapEncoder;

  slapEncoder * slapCreateEncoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
  void slapDestroyEncoder(IN_OUT slapEncoder **ppEncoder);

  slapResult slapFinalizeEncoder(IN slapEncoder *pEncoder);

  // @param ppCompressedData should be NULL on when the first frame is added.
  slapResult slapAddFrameYUV420(IN slapEncoder *pEncoder, IN void *pData, const size_t stride, OUT void **ppCompressedData, OUT size_t *pSize);

#define SLAP_HEADER_BLOCK_SIZE 256

  typedef struct slapFileWriter
  {
    FILE *pMainFile;
    FILE *pHeaderFile;
    uint64_t headerPosition;
    uint64_t frameCount;
    slapEncoder *pEncoder;
    void *pData;
    uint64_t frameSizeOffsets[SLAP_HEADER_BLOCK_SIZE];
    size_t frameSizeOffsetIndex;
  } slapFileWriter;

  slapFileWriter * slapInitFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
  void slapDestroyFileWriter(IN_OUT slapFileWriter **ppFileWriter);

  slapResult slapFinalizeFileWriter(IN slapFileWriter *pFileWriter);

  slapResult slapFileWriter_AddFrameYUV420(IN slapFileWriter *pFileWriter, IN void *pData, const size_t stride);

#ifdef __cplusplus
}
#endif

#endif // slapcodec_h__