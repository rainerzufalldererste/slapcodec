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

// enables prettier downscaling
#define SSSE3 1

#define slapAlloc(Type, count) (Type *)malloc(sizeof(Type) * (count))
#define slapRealloc(ptr, Type, count) (*ptr = (Type *)realloc(*ptr, sizeof(Type) * (count)))
#define slapFreePtr(ptr)  do { if (ptr && *ptr) { free(*ptr); *ptr = NULL; } } while (0)
#define slapSetZero(ptr, Type) memset(ptr, 0, sizeof(Type))
#define slapStrCpy(target, source) do { size_t size = strlen(source) + 1; target = slapAlloc(char, size); if (target) { memcpy(target, source, size); } } while (0)

#define slapLog(str, ...) printf(str, __VA_ARGS__);

#ifdef __cplusplus
extern "C" {
#endif

  void slapMemcpy(OUT void *pDest, IN void *pSrc, const size_t size);
  void slapMemmove(OUT void *pDest, IN_OUT void *pSrc, const size_t size);

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

  slapResult slapWriteJpegFromYUV(const char *filename, IN void *pData, const size_t resX, const size_t resY);

#define SLAP_SUB_BUFFER_COUNT 1
#define SLAP_LOW_RES_BUFFER_INDEX SLAP_SUB_BUFFER_COUNT

#define SLAP_FLAG_STEREO 1

  typedef union mode
  {
    uint64_t flagsPack;

    struct flags
    {
      unsigned int stereo : 1;
      unsigned int encoder : 4;
    } flags;

  } mode;

  typedef struct slapEncoder
  {
    size_t frameIndex;
    size_t iframeStep;
    size_t resX;
    size_t resY;
    uint8_t *pLowResData;
    uint8_t *pLastFrame;
    size_t lowResX;
    size_t lowResY;

    mode mode;

    int quality;
    int iframeQuality;
    int lowResQuality;
    void **ppEncoderInternal;
    void **ppLowResEncoderInternal;
    void **ppDecoderInternal;
    void **ppCompressedBuffers;
    size_t compressedSubBufferSizes[SLAP_SUB_BUFFER_COUNT + 1];
  } slapEncoder;

  slapEncoder * slapCreateEncoder(const size_t sizeX, const size_t sizeY, const uint64_t flags);
  void slapDestroyEncoder(IN_OUT slapEncoder **ppEncoder);

  slapResult slapFinalizeEncoder(IN slapEncoder *pEncoder);

  // After slapEncoder_BeginFrame has finished, the subFrame can be compressed and written.
  slapResult slapEncoder_BeginFrame(IN slapEncoder *pEncoder, IN void *pData);

  // After slapEncoder_BeginSubFrame has finished, the frame can be written to disk.
  slapResult slapEncoder_BeginSubFrame(IN slapEncoder *pEncoder, IN void *pData, OUT void **ppCompressedData, OUT size_t *pSize, const size_t subFrameIndex);
  slapResult slapEncoder_EndSubFrame(IN slapEncoder *pEncoder, IN void *pData, const size_t subFrameIndex);

  slapResult slapEncoder_EndFrame(IN slapEncoder *pEncoder, IN void *pData);

#define SLAP_HEADER_BLOCK_SIZE 1024

#define SLAP_PRE_HEADER_SIZE 8
#define SLAP_PRE_HEADER_HEADER_SIZE_INDEX 0
#define SLAP_PRE_HEADER_FRAME_COUNT_INDEX 1
#define SLAP_PRE_HEADER_FRAME_SIZEX_INDEX 2
#define SLAP_PRE_HEADER_FRAME_SIZEY_INDEX 3
#define SLAP_PRE_HEADER_IFRAME_STEP_INDEX 4
#define SLAP_PRE_HEADER_CODEC_FLAGS_INDEX 5

#define SLAP_HEADER_PER_FRAME_FULL_FRAME_OFFSET 4
#define SLAP_HEADER_PER_FRAME_SIZE (SLAP_HEADER_PER_FRAME_FULL_FRAME_OFFSET + SLAP_SUB_BUFFER_COUNT * 2)

#define SLAP_HEADER_FRAME_OFFSET_INDEX 0
#define SLAP_HEADER_FRAME_DATA_SIZE_INDEX 1

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
    void *pLowResBuffer;
    size_t lowResBufferSize;
  } slapFileWriter;

  slapFileWriter * slapCreateFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const uint64_t flags);
  void slapDestroyFileWriter(IN_OUT slapFileWriter **ppFileWriter);

  slapResult slapFinalizeFileWriter(IN slapFileWriter *pFileWriter);

  slapResult slapFileWriter_AddFrameYUV420(IN slapFileWriter *pFileWriter, IN void *pData);

  typedef struct slapDecoder
  {
    size_t frameIndex;
    size_t iframeStep;
    size_t resX;
    size_t resY;

    mode mode;

    void **ppDecoders;
    uint8_t *pLowResData;
    uint8_t *pLastFrame;
  } slapDecoder;

  slapDecoder * slapCreateDecoder(const size_t sizeX, const size_t sizeY, const uint64_t flags);
  void slapDestroyDecoder(IN_OUT slapDecoder **ppDecoder);

  slapResult slapDecoder_DecodeSubFrame(IN slapDecoder *pDecoder, const size_t decoderIndex, IN void **ppCompressedData, IN size_t *pLength, IN_OUT void *pYUVData);
  slapResult slapDecoder_FinalizeFrame(IN slapDecoder *pDecoder, IN void *pData, const size_t length, IN_OUT void *pYUVData);

  typedef struct slapFileReader
  {
    FILE *pFile;
    void *pCurrentFrame;
    size_t currentFrameAllocatedSize;
    size_t currentFrameSize;

    void *pDecodedFrameYUV;

    uint64_t preHeaderBlock[SLAP_PRE_HEADER_SIZE];
    uint64_t *pHeader;
    size_t headerOffset;
    size_t frameIndex;

    slapDecoder *pDecoder;
  } slapFileReader;

  slapFileReader * slapCreateFileReader(const char *filename);
  void slapDestroyFileReader(IN_OUT slapFileReader **ppFileReader);

  slapResult slapFileReader_GetResolution(IN slapFileReader *pFileReader, OUT size_t *pResolutionX, OUT size_t *pResolutionY);
  slapResult slapFileReader_GetLowResFrameResolution(IN slapFileReader *pFileReader, OUT size_t *pResolutionX, OUT size_t *pResolutionY);

  slapResult _slapFileReader_ReadNextFrameFull(IN slapFileReader *pFileReader);
  slapResult _slapFileReader_DecodeCurrentFrameFull(IN slapFileReader *pFileReader);

  slapResult _slapFileReader_ReadNextFrameLowRes(IN slapFileReader *pFileReader);
  slapResult _slapFileReader_DecodeCurrentFrameLowRes(IN slapFileReader *pFileReader);

#ifdef __cplusplus
}
#endif

#endif // slapcodec_h__