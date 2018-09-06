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
#include <string.h>
#include <inttypes.h>

#ifndef bool_t
#define bool_t uint64_t
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
#define slapRealloc(ptr, Type, count) (*ptr = (Type *)realloc(*ptr, sizeof(Type) * (count)))
#define slapFreePtr(ptr)  do { if (ptr && *ptr) { free(*ptr); *ptr = NULL; } } while (0)
#define slapSetZero(ptr, Type) memset(ptr, 0, sizeof(Type))
#define slapStrCpy(target, source) do { size_t size = strlen(source) + 1; target = slapAlloc(char, size); if (target) { memcpy(target, source, size); } } while (0)

#define slapLog(str, ...) printf(str, __VA_ARGS__);

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum slapResult
  {
    slapSuccess,
    slapError_Generic,
    slapError_ArgumentNull,
    slapError_Compress_Internal,
    slapError_FileError,
    slapError_EndOfStream,
    slapError_MemoryAllocation
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
    unsigned long __data0;
  } slapEncoder;

  slapEncoder * slapCreateEncoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
  void slapDestroyEncoder(IN_OUT slapEncoder **ppEncoder);

  slapResult slapFinalizeEncoder(IN slapEncoder *pEncoder);

  // @param ppCompressedData should be NULL on when the first frame is added.
  slapResult slapAddFrameYUV420(IN slapEncoder *pEncoder, IN void *pData, const size_t stride, OUT void **ppCompressedData, OUT size_t *pSize);

#define SLAP_HEADER_BLOCK_SIZE 1024

#define SLAP_PRE_HEADER_SIZE 8
#define SLAP_PRE_HEADER_HEADER_SIZE_INDEX 0
#define SLAP_PRE_HEADER_FRAME_COUNT_INDEX 1
#define SLAP_PRE_HEADER_FRAME_SIZEX_INDEX 2
#define SLAP_PRE_HEADER_FRAME_SIZEY_INDEX 3
#define SLAP_PRE_HEADER_IFRAME_STEP_INDEX 4
#define SLAP_PRE_HEADER_CODEC_FLAGS_INDEX 5

#define SLAP_HEADER_PER_FRAME_SIZE 2

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
    char *filename;
  } slapFileWriter;

  slapFileWriter * slapCreateFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
  void slapDestroyFileWriter(IN_OUT slapFileWriter **ppFileWriter);

  slapResult slapFinalizeFileWriter(IN slapFileWriter *pFileWriter);

  slapResult slapFileWriter_AddFrameYUV420(IN slapFileWriter *pFileWriter, IN void *pData, const size_t stride);

  typedef struct slapDecoder
  {
    size_t frameIndex;
    size_t iframeStep;
    size_t resX;
    size_t resY;
    bool_t stereo;
    void *pAdditionalData;
  } slapDecoder;

  slapDecoder * slapCreateDecoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d);
  void slapDestroyDecoder(IN_OUT slapDecoder **ppDecoder);

  slapResult slapDecodeFrame(IN slapDecoder *pDecoder, IN void *pData, const size_t length, IN_OUT void *pYUVData);

  typedef struct slapFileReader
  {
    FILE *pFile;
    void *pCurrentFrame;
    size_t currentFrameAllocatedSize;

    void *pDecodedFrameYUV;

    uint64_t preHeaderBlock[SLAP_PRE_HEADER_SIZE];
    uint64_t *pHeader;
    size_t headerOffset;
    size_t frameIndex;

    slapDecoder *pDecoder;
  } slapFileReader;

  slapFileReader * slapCreateFileReader(const char *filename);
  void slapDestroyFileReader(IN_OUT slapFileReader **ppFileReader);

  slapResult _slapFileReader_ReadNextFrame(IN slapFileReader *pFileReader);

#ifdef __cplusplus
}
#endif

#endif // slapcodec_h__