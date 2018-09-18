// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "slapcodec.h"
#include "turbojpeg.h"

#include "apex_memmove/apex_memmove.h"
#include "apex_memmove/apex_memmove.c"

#include "threadpool.h"

#include <intrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#ifdef SSSE3
#include <tmmintrin.h>

#define SLAP_HIGH_QUALITY_DOWNSCALE 1
#endif


slapResult _slapCompressChannel(IN void *pData, IN_OUT void **ppCompressedData, IN_OUT size_t *pCompressedDataSize, const size_t width, const size_t height, const int quality, IN void *pCompressor);
slapResult _slapCompressYUV420(IN void *pData, IN_OUT void **ppCompressedData, IN_OUT size_t *pCompressedDataSize, const size_t width, const size_t height, const int quality, IN void *pCompressor);
slapResult _slapDecompressChannel(IN void *pData, IN_OUT void *pCompressedData, const size_t compressedDataSize, const size_t width, const size_t height, IN void *pDecompressor);
slapResult _slapDecompressYUV420(IN void *pData, IN_OUT void *pCompressedData, const size_t compressedDataSize, const size_t width, const size_t height, IN void *pDecompressor);
void _slapLastFrameDiffAndStereoDiffAndSubBufferYUV420(IN_OUT void *pLastFrame, IN_OUT void *pData, IN_OUT void *pLowRes, const size_t resX, const size_t resY);
void _slapCopyToLastFrameAndGenSubBufferAndStereoDiffYUV420(IN_OUT void *pData, OUT void *pLowResData, OUT void *pLastFrame, const size_t resX, const size_t resY);
void _slapAddStereoDiffYUV420(IN_OUT void *pData, const size_t resX, const size_t resY);
void _slapAddStereoDiffYUV420AndCopyToLastFrame(IN_OUT void *pData, OUT void *pLastFrame, const size_t resX, const size_t resY);
void _slapAddStereoDiffYUV420AndAddLastFrameDiff(IN_OUT void *pData, OUT void *pLastFrame, const size_t resX, const size_t resY);

typedef struct _slapFrameEncoderBlock
{
  size_t frameSize;
  void *pFrameData;
} _slapFrameEncoderBlock;

#ifdef SLAP_MULTITHREADED
typedef struct _slapEncoderSubTaskData0
{
  slapEncoder *pEncoder;
  void *pData;
  _slapFrameEncoderBlock *pSubFrameEncoderData;
  size_t index;
} _slapEncoderSubTaskData0;

typedef struct _slapDecoderSubTaskData0
{
  slapDecoder *pDecoder;
  size_t index;
  void **pDataAddrs;
  size_t *pDataSizes;
  void *pYUVFrame;
} _slapDecoderSubTaskData0;
#endif

//////////////////////////////////////////////////////////////////////////

void slapMemcpy(OUT void *pDest, IN void *pSrc, const size_t size)
{
  apex_memcpy(pDest, pSrc, size);
}

void slapMemmove(OUT void *pDest, IN_OUT void *pSrc, const size_t size)
{
  apex_memmove(pDest, pSrc, size);
}

slapResult slapWriteJpegFromYUV(const char *filename, IN void *pData, const size_t resX, const size_t resY)
{
  slapResult result = slapSuccess;
  tjhandle jpegHandle = NULL;
  unsigned char *pBuffer = NULL;
  unsigned long bufferSize = 0;
  FILE *pFile = NULL;

  if (!filename || !pData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  jpegHandle = tjInitCompress();

  if (!jpegHandle)
  {
    result = slapError_Compress_Internal;
    goto epilogue;
  }

  if (tjCompressFromYUV(jpegHandle, (unsigned char *)pData, (int)resX, 32, (int)resY, TJSAMP_420, &pBuffer, &bufferSize, 75, 0))
  {
    slapLog(tjGetErrorStr2(jpegHandle));
    result = slapError_Compress_Internal;
    goto epilogue;
  }

  pFile = fopen(filename, "wb");

  if (!pFile)
  {
    result = slapError_FileError;
    goto epilogue;
  }

  if (bufferSize != fwrite(pBuffer, 1, bufferSize, pFile))
  {
    result = slapError_FileError;
    goto epilogue;
  }

epilogue:
  if (jpegHandle)
    tjDestroy(jpegHandle);

  if (pBuffer)
    tjFree(pBuffer);

  if (pFile)
  {
    fclose(pFile);

    if (result != slapSuccess)
      remove(filename);
  }

  return result;
}

slapEncoder * slapCreateEncoder(const size_t sizeX, const size_t sizeY, const uint64_t flags)
{
  if (sizeX & 31 || sizeY & 31) // must be multiple of 32.
    return NULL;

  slapEncoder *pEncoder = slapAlloc(slapEncoder, 1);

  if (!pEncoder)
    goto epilogue;

  slapSetZero(pEncoder, slapEncoder);

  pEncoder->resX = sizeX;
  pEncoder->resY = sizeY;
  pEncoder->iframeStep = 30;
  pEncoder->mode.flagsPack = flags;
  pEncoder->quality = 75;
  pEncoder->iframeQuality = 75;
  pEncoder->lowResQuality = 85;

  pEncoder->lowResX = pEncoder->resX >> 3;
  pEncoder->lowResY = pEncoder->resY >> 3;
  
  if (pEncoder->mode.flags.stereo)
    pEncoder->lowResY >>= 1;

  pEncoder->pLowResData = slapAlloc(uint8_t, pEncoder->lowResX * pEncoder->lowResY * 3 / 2);

  if (!pEncoder->pLowResData)
    goto epilogue;

  pEncoder->pLastFrame = slapAlloc(uint8_t, sizeX * sizeY * 3 / 2);

  if (!pEncoder->pLastFrame)
    goto epilogue;

  pEncoder->ppEncoderInternal = slapAlloc(void *, SLAP_SUB_BUFFER_COUNT + 1);

  if (!pEncoder->ppEncoderInternal)
    goto epilogue;

  memset(pEncoder->ppEncoderInternal, 0, sizeof(void *) * (SLAP_SUB_BUFFER_COUNT + 1));

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT + 1; i++)
  {
    pEncoder->ppEncoderInternal[i] = tjInitCompress();

    if (!pEncoder->ppEncoderInternal[i])
      goto epilogue;
  }

  pEncoder->ppLowResEncoderInternal = tjInitCompress();

  pEncoder->ppDecoderInternal = slapAlloc(void *, SLAP_SUB_BUFFER_COUNT);

  if (!pEncoder->ppDecoderInternal)
    goto epilogue;

  memset(pEncoder->ppDecoderInternal, 0, sizeof(void *) * SLAP_SUB_BUFFER_COUNT);

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    pEncoder->ppDecoderInternal[i] = tjInitDecompress();

    if (!pEncoder->ppDecoderInternal[i])
      goto epilogue;
  }

  pEncoder->ppCompressedBuffers = slapAlloc(void *, SLAP_SUB_BUFFER_COUNT + 1);

  if (!pEncoder->ppCompressedBuffers)
    goto epilogue;

  memset(pEncoder->ppCompressedBuffers, 0, sizeof(void *) * (SLAP_SUB_BUFFER_COUNT + 1));

  const size_t threadCount = ThreadPool_GetSystemThreadCount();

  pEncoder->pThreadPoolHandle = ThreadPool_Init(threadCount);

  if (!pEncoder->pThreadPoolHandle)
    goto epilogue;

  return pEncoder;

epilogue:
  if (pEncoder->pLowResData)
    slapFreePtr(&pEncoder->pLowResData);

  if (pEncoder->ppEncoderInternal)
  {
    for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT + 1; i++)
      if (pEncoder->ppEncoderInternal[i])
        tjDestroy(pEncoder->ppEncoderInternal[i]);

    slapFreePtr(&pEncoder->ppEncoderInternal);
  }

  if (pEncoder->ppLowResEncoderInternal)
    tjDestroy(pEncoder->ppLowResEncoderInternal);

  if (pEncoder->ppDecoderInternal)
  {
    for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
      if (pEncoder->ppDecoderInternal[i])
        tjDestroy(pEncoder->ppDecoderInternal[i]);

    slapFreePtr(&pEncoder->ppDecoderInternal);
  }

  if ((pEncoder)->pLastFrame)
    slapFreePtr(&(pEncoder)->pLastFrame);

  if ((pEncoder)->ppCompressedBuffers)
    slapFreePtr(&(pEncoder)->ppCompressedBuffers);

  if (pEncoder->pThreadPoolHandle)
    ThreadPool_Destroy(pEncoder->pThreadPoolHandle);

  slapFreePtr(&pEncoder);

  return NULL;
}

void slapDestroyEncoder(IN_OUT slapEncoder **ppEncoder)
{
  if (ppEncoder && *ppEncoder)
  {
    if ((*ppEncoder)->ppEncoderInternal)
    {
      for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT + 1; i++)
        if ((*ppEncoder)->ppEncoderInternal[i])
          tjDestroy((*ppEncoder)->ppEncoderInternal[i]);

      slapFreePtr(&(*ppEncoder)->ppEncoderInternal);
    }

    if ((*ppEncoder)->ppLowResEncoderInternal)
      tjDestroy((*ppEncoder)->ppLowResEncoderInternal);

    if ((*ppEncoder)->ppDecoderInternal)
    {
      for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
        if ((*ppEncoder)->ppDecoderInternal[i])
          tjDestroy((*ppEncoder)->ppDecoderInternal[i]);

      slapFreePtr(&(*ppEncoder)->ppDecoderInternal);
    }

    if ((*ppEncoder)->ppCompressedBuffers)
      slapFreePtr(&(*ppEncoder)->ppCompressedBuffers);

    if ((*ppEncoder)->pLowResData)
      slapFreePtr(&(*ppEncoder)->pLowResData);

    if ((*ppEncoder)->pLastFrame)
      slapFreePtr(&(*ppEncoder)->pLastFrame);

    if ((*ppEncoder)->pThreadPoolHandle)
      ThreadPool_Destroy((*ppEncoder)->pThreadPoolHandle);
  }

  slapFreePtr(ppEncoder);
}

slapResult slapFinalizeEncoder(IN slapEncoder *pEncoder)
{
  (void)pEncoder;

  return slapSuccess;
}

slapResult slapEncoder_BeginFrame(IN slapEncoder *pEncoder, IN void *pData)
{
  slapResult result = slapSuccess;

  if (!pEncoder || !pData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  if (pEncoder->mode.flags.encoder == 0)
  {
    if (pEncoder->frameIndex % pEncoder->iframeStep != 0)
      _slapLastFrameDiffAndStereoDiffAndSubBufferYUV420(pEncoder->pLastFrame, pData, pEncoder->pLowResData, pEncoder->resX, pEncoder->resY);
    else
      _slapCopyToLastFrameAndGenSubBufferAndStereoDiffYUV420(pData, pEncoder->pLowResData, pEncoder->pLastFrame, pEncoder->resX, pEncoder->resY);
  }

epilogue:
  return result;
}

slapResult slapEncoder_BeginSubFrame(IN slapEncoder *pEncoder, IN void *pData, OUT void **ppCompressedData, OUT size_t *pSize, const size_t subFrameIndex)
{
  slapResult result = slapSuccess;

  if (!pEncoder || !pData || !ppCompressedData || !pSize)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  const size_t subFrameHeight = pEncoder->resY * 3 / 2 / SLAP_SUB_BUFFER_COUNT;

  if (pEncoder->mode.flags.encoder == 0)
  {
    if (subFrameHeight * subFrameIndex * 2 / 3 < pEncoder->resY)
    {
      result = _slapCompressChannel(((uint8_t *)pData) + subFrameIndex * subFrameHeight * pEncoder->resX, &pEncoder->ppCompressedBuffers[subFrameIndex], &pEncoder->compressedSubBufferSizes[subFrameIndex], pEncoder->resX, subFrameHeight, (pEncoder->frameIndex % pEncoder->iframeStep == 0) ? pEncoder->quality : pEncoder->iframeQuality, pEncoder->ppEncoderInternal[subFrameIndex]);
    }
    else
    {
      result = _slapCompressChannel(((uint8_t *)pData) + subFrameIndex * subFrameHeight * pEncoder->resX, &pEncoder->ppCompressedBuffers[subFrameIndex], &pEncoder->compressedSubBufferSizes[subFrameIndex], pEncoder->resX >> 1, subFrameHeight << 1, (pEncoder->frameIndex % pEncoder->iframeStep == 0) ? pEncoder->quality : pEncoder->iframeQuality, pEncoder->ppEncoderInternal[subFrameIndex]);
    }

    if (result != slapSuccess)
      goto epilogue;

    *pSize = pEncoder->compressedSubBufferSizes[subFrameIndex];
    *ppCompressedData = pEncoder->ppCompressedBuffers[subFrameIndex];
  }

epilogue:
  return result;
}

slapResult slapEncoder_EndSubFrame(IN slapEncoder *pEncoder, IN void *pData, const size_t subFrameIndex)
{
  slapResult result = slapSuccess;

  const size_t subFrameHeight = pEncoder->resY * 3 / 2 / SLAP_SUB_BUFFER_COUNT;
  
  if (pEncoder->mode.flags.encoder == 0)
  {
    if (pEncoder->frameIndex % pEncoder->iframeStep != 0)
    {
      if (subFrameHeight * subFrameIndex * 2 / 3 < pEncoder->resY)
        result = _slapDecompressChannel(((uint8_t *)pData) + subFrameIndex * subFrameHeight * pEncoder->resX, pEncoder->ppCompressedBuffers[subFrameIndex], pEncoder->compressedSubBufferSizes[subFrameIndex], pEncoder->resX, subFrameHeight, pEncoder->ppDecoderInternal[subFrameIndex]);
      else
        result = _slapDecompressChannel(((uint8_t *)pData) + subFrameIndex * subFrameHeight * pEncoder->resX, pEncoder->ppCompressedBuffers[subFrameIndex], pEncoder->compressedSubBufferSizes[subFrameIndex], pEncoder->resX >> 1, subFrameHeight << 1, pEncoder->ppDecoderInternal[subFrameIndex]);

      if (result != slapSuccess)
        goto epilogue;
    }
    else
    {
      result = _slapDecompressChannel(pEncoder->pLastFrame + pEncoder->resX * subFrameHeight * subFrameIndex, pEncoder->ppCompressedBuffers[subFrameIndex], pEncoder->compressedSubBufferSizes[subFrameIndex], pEncoder->resX, subFrameHeight, pEncoder->ppDecoderInternal[subFrameIndex]);

      if (result != slapSuccess)
        goto epilogue;
    }
  }

epilogue:
  return result;
}

slapResult slapEncoder_EndFrame(IN slapEncoder *pEncoder, IN void *pData)
{
  slapResult result = slapSuccess;

  if (!pEncoder)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }
  
  if (pEncoder->mode.flags.encoder == 0)
  {
    if (pEncoder->frameIndex % pEncoder->iframeStep != 0)
      _slapAddStereoDiffYUV420AndAddLastFrameDiff(pData, pEncoder->pLastFrame, pEncoder->resX, pEncoder->resY);
    else
      _slapAddStereoDiffYUV420(pEncoder->pLastFrame, pEncoder->resX, pEncoder->resY);
  }

  pEncoder->frameIndex++;

epilogue:
  return result;
}

slapResult _slapWriteToHeader(IN slapFileWriter *pFileWriter, const uint64_t data)
{
  slapResult result = slapSuccess;

  pFileWriter->frameSizeOffsets[pFileWriter->frameSizeOffsetIndex++] = data;
  pFileWriter->headerPosition++;

  if (pFileWriter->frameSizeOffsetIndex >= (uint64_t)SLAP_HEADER_BLOCK_SIZE)
  {
    if ((size_t)SLAP_HEADER_BLOCK_SIZE != fwrite(pFileWriter->frameSizeOffsets, sizeof(uint64_t), SLAP_HEADER_BLOCK_SIZE, pFileWriter->pHeaderFile))
    {
      result = slapError_FileError;
      goto epilogue;
    }

    pFileWriter->frameSizeOffsetIndex = 0;
  }

epilogue:
  return result;
}

slapFileWriter * slapCreateFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const uint64_t flags)
{
  slapFileWriter *pFileWriter = slapAlloc(slapFileWriter, 1);
  char filenameBuffer[0xFF];
  char headerFilenameBuffer[0xFF];

  if (!pFileWriter)
    goto epilogue;

  slapSetZero(pFileWriter, slapFileWriter);
  slapStrCpy(pFileWriter->filename, filename);

  if (!pFileWriter->filename)
    goto epilogue;

  pFileWriter->pEncoder = slapCreateEncoder(sizeX, sizeY, flags);

  if (!pFileWriter->pEncoder)
    goto epilogue;

  sprintf_s(filenameBuffer, 0xFF, "%s.raw", filename);
  sprintf_s(headerFilenameBuffer, 0xFF, "%s.header", filename);

  pFileWriter->pMainFile = fopen(filenameBuffer, "wb");

  if (!pFileWriter->pMainFile)
    goto epilogue;

  pFileWriter->pHeaderFile = fopen(headerFilenameBuffer, "wb");

  if (!pFileWriter->pHeaderFile)
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, (uint64_t)pFileWriter->pEncoder->resX))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, (uint64_t)pFileWriter->pEncoder->resY))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, (uint64_t)pFileWriter->pEncoder->iframeStep))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, (uint64_t)pFileWriter->pEncoder->mode.flagsPack))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
    goto epilogue;

  if (pFileWriter->headerPosition != SLAP_PRE_HEADER_SIZE)
    goto epilogue;

  return pFileWriter;

epilogue:

  if (pFileWriter)
  {
    slapDestroyEncoder(&pFileWriter->pEncoder);

    if (pFileWriter->pMainFile)
      fclose(pFileWriter->pMainFile);

    if (pFileWriter->pHeaderFile)
      fclose(pFileWriter->pHeaderFile);
  }

  slapFreePtr(&pFileWriter);
  return NULL;
}

void slapDestroyFileWriter(IN_OUT slapFileWriter **ppFileWriter)
{
  if (ppFileWriter && *ppFileWriter)
  {
    slapDestroyEncoder(&(*ppFileWriter)->pEncoder);

    if ((*ppFileWriter)->pData)
      tjFree((*ppFileWriter)->pData);

    if ((*ppFileWriter)->pLowResBuffer)
      tjFree((*ppFileWriter)->pLowResBuffer);

    if ((*ppFileWriter)->filename)
      free((*ppFileWriter)->filename);
  }

  slapFreePtr(ppFileWriter);
}

slapResult slapFinalizeFileWriter(IN slapFileWriter *pFileWriter)
{
  slapResult result = slapError_Generic;
  FILE *pFile = NULL;
  FILE *pReadFile = NULL;
  char filenameBuffer[0xFF];
  void *pData = NULL;
  size_t fileSize = 0; 
  const size_t maxBlockSize = 1024 * 1024 * 64;
  size_t remainingSize = 0;

  if (!pFileWriter)
    goto epilogue;

  if (pFileWriter->pEncoder)
    slapFinalizeEncoder(pFileWriter->pEncoder);

  if (pFileWriter->pHeaderFile)
  {
    if (pFileWriter->frameSizeOffsetIndex != 0)
      fwrite(pFileWriter->frameSizeOffsets, 1, sizeof(uint64_t) * pFileWriter->frameSizeOffsetIndex, pFileWriter->pHeaderFile);

    fflush(pFileWriter->pHeaderFile);
    fclose(pFileWriter->pHeaderFile);
    pFileWriter->pHeaderFile = NULL;
  }

  if (pFileWriter->pMainFile)
  {
    fflush(pFileWriter->pMainFile);
    fclose(pFileWriter->pMainFile);
    pFileWriter->pMainFile = NULL;
  }

  pFile = fopen(pFileWriter->filename, "wb");

  if (!pFile)
    goto epilogue;

  sprintf_s(filenameBuffer, 0xFF, "%s.header", pFileWriter->filename);
  pReadFile = fopen(filenameBuffer, "rb");

  if (!pReadFile)
    goto epilogue;

  pData = slapAlloc(unsigned char, pFileWriter->headerPosition * sizeof(uint64_t));

  if (!pData)
    goto epilogue;

  if (pFileWriter->headerPosition != (fread(pData, sizeof(uint64_t), pFileWriter->headerPosition, pReadFile)))
    goto epilogue;

  ((uint64_t *)pData)[SLAP_PRE_HEADER_HEADER_SIZE_INDEX] = pFileWriter->headerPosition - SLAP_PRE_HEADER_SIZE;
  ((uint64_t *)pData)[SLAP_PRE_HEADER_FRAME_COUNT_INDEX] = pFileWriter->frameCount;

  if (pFileWriter->headerPosition != fwrite(pData, sizeof(uint64_t), pFileWriter->headerPosition, pFile))
    goto epilogue;

  fclose(pReadFile);
  remove(filenameBuffer);

  sprintf_s(filenameBuffer, 0xFF, "%s.raw", pFileWriter->filename);
  pReadFile = fopen(filenameBuffer, "rb");

  if (!pFile)
    goto epilogue;

  fseek(pReadFile, 0, SEEK_END);
  fileSize = ftell(pReadFile);
  fseek(pReadFile, 0, SEEK_SET);

  remainingSize = fileSize;

  slapRealloc(&pData, uint8_t, fileSize < maxBlockSize ? fileSize : maxBlockSize);

  while (remainingSize > maxBlockSize)
  {
    if (maxBlockSize != fread(pData, 1, maxBlockSize, pReadFile))
      goto epilogue;

    if (maxBlockSize != fwrite(pData, 1, maxBlockSize, pFile))
      goto epilogue;

    remainingSize -= maxBlockSize;
  }

  if (remainingSize != fread(pData, 1, remainingSize, pReadFile))
    goto epilogue;

  if (remainingSize != fwrite(pData, 1, remainingSize, pFile))
    goto epilogue;

  fclose(pReadFile);
  pReadFile = NULL;
  remove(filenameBuffer);

  result = slapSuccess;

epilogue:

  if (pFile)
    fclose(pFile);

  if (pReadFile)
    fclose(pReadFile);

  if (pData)
    slapFreePtr(&pData);

  return result;
}

#ifdef SLAP_MULTITHREADED

size_t _slapEncoderTask_CallBeginSubframe(void *pData)
{
  _slapEncoderSubTaskData0 *pUserData = (_slapEncoderSubTaskData0 *)pData;

  return (size_t)slapEncoder_BeginSubFrame(pUserData->pEncoder, pUserData->pData, &pUserData->pSubFrameEncoderData->pFrameData, &pUserData->pSubFrameEncoderData->frameSize, pUserData->index);
}

size_t _slapEncoderTask_CallEndSubframe(void *pData)
{
  _slapEncoderSubTaskData0 *pUserData = (_slapEncoderSubTaskData0 *)pData;

  return (size_t)slapEncoder_EndSubFrame(pUserData->pEncoder, pUserData->pData, pUserData->index);
}

#endif

slapResult slapFileWriter_AddFrameYUV420(IN slapFileWriter *pFileWriter, IN void *pData)
{
  slapResult result = slapSuccess;
  size_t filePosition = 0;
  _slapFrameEncoderBlock subFrames[SLAP_SUB_BUFFER_COUNT];
  size_t totalFullFrameSize = 0;
#ifdef SLAP_MULTITHREADED
  ThreadPool_TaskHandle tasks[SLAP_SUB_BUFFER_COUNT];
  _slapEncoderSubTaskData0 encoderData[SLAP_SUB_BUFFER_COUNT];
#endif

  if (!pFileWriter || !pData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  result = slapEncoder_BeginFrame(pFileWriter->pEncoder, pData);

  if (result != slapSuccess)
    goto epilogue;

  // compress sub frame
  result = _slapCompressYUV420(pFileWriter->pEncoder->pLowResData, &pFileWriter->pEncoder->ppCompressedBuffers[SLAP_LOW_RES_BUFFER_INDEX], &pFileWriter->pEncoder->compressedSubBufferSizes[SLAP_LOW_RES_BUFFER_INDEX], pFileWriter->pEncoder->lowResX, pFileWriter->pEncoder->lowResY, pFileWriter->pEncoder->lowResQuality, pFileWriter->pEncoder->ppEncoderInternal[SLAP_LOW_RES_BUFFER_INDEX]);

  if (result != slapSuccess)
    goto epilogue;

  // compress full frame
#ifdef SLAP_MULTITHREADED

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    encoderData[i].pEncoder = pFileWriter->pEncoder;
    encoderData[i].pData = pData;
    encoderData[i].pSubFrameEncoderData = &subFrames[i];
    encoderData[i].index = i;

    tasks[i] = ThreadPool_CreateTask(_slapEncoderTask_CallBeginSubframe, (void *)&encoderData[i]);
    ThreadPool_EnqueueTask(pFileWriter->pEncoder->pThreadPoolHandle, tasks[i]);
  }

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
    ThreadPool_JoinTask(tasks[i]);

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    tasks[i] = ThreadPool_CreateTask(_slapEncoderTask_CallEndSubframe, (void *)&encoderData[i]);
    ThreadPool_EnqueueTask(pFileWriter->pEncoder->pThreadPoolHandle, tasks[i]);
  }

#else

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    result = slapEncoder_BeginSubFrame(pFileWriter->pEncoder, pData, &subFrames[i].pFrameData, &subFrames[i].frameSize, i);

    if (result != slapSuccess)
      goto epilogue;
  }

#endif

  // save to disk
  filePosition = ftell(pFileWriter->pMainFile);

  if ((result = _slapWriteToHeader(pFileWriter, filePosition)) != slapSuccess) goto epilogue;
  if ((result = _slapWriteToHeader(pFileWriter, pFileWriter->pEncoder->compressedSubBufferSizes[SLAP_LOW_RES_BUFFER_INDEX])) != slapSuccess) goto epilogue;

  if (pFileWriter->pEncoder->compressedSubBufferSizes[SLAP_LOW_RES_BUFFER_INDEX] != fwrite(pFileWriter->pEncoder->ppCompressedBuffers[SLAP_LOW_RES_BUFFER_INDEX], 1, pFileWriter->pEncoder->compressedSubBufferSizes[SLAP_LOW_RES_BUFFER_INDEX], pFileWriter->pMainFile))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  filePosition = ftell(pFileWriter->pMainFile);
  if ((result = _slapWriteToHeader(pFileWriter, filePosition)) != slapSuccess) goto epilogue;

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
    totalFullFrameSize += subFrames[i].frameSize;

  if ((result = _slapWriteToHeader(pFileWriter, totalFullFrameSize)) != slapSuccess) goto epilogue;

  filePosition = 0;

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    if ((result = _slapWriteToHeader(pFileWriter, filePosition)) != slapSuccess) goto epilogue;
    if ((result = _slapWriteToHeader(pFileWriter, subFrames[i].frameSize)) != slapSuccess) goto epilogue;

    filePosition += subFrames[i].frameSize;

    if (subFrames[i].frameSize != fwrite(subFrames[i].pFrameData, 1, subFrames[i].frameSize, pFileWriter->pMainFile))
    {
      result = slapError_FileError;
      goto epilogue;
    }
  }

  // get ready for next frame
#ifdef SLAP_MULTITHREADED

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
    ThreadPool_JoinTask(tasks[i]);

#else

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    result = slapEncoder_EndSubFrame(pFileWriter->pEncoder, pData, i);

    if (result != slapSuccess)
      goto epilogue;
  }

#endif

  // finalize frame.
  result = slapEncoder_EndFrame(pFileWriter->pEncoder, pData);

  if (result != slapSuccess)
    goto epilogue;

  pFileWriter->frameCount++; 

epilogue:
  return result;
}

slapDecoder * slapCreateDecoder(const size_t sizeX, const size_t sizeY, const uint64_t flags)
{
  if (sizeX & 63 || sizeY & 63) // must be multiple of 64.
    return NULL;

  slapDecoder *pDecoder = slapAlloc(slapDecoder, 1);

  if (!pDecoder)
    goto epilogue;

  slapSetZero(pDecoder, slapDecoder);

  pDecoder->resX = sizeX;
  pDecoder->resY = sizeY;
  pDecoder->iframeStep = 30;
  pDecoder->mode.flagsPack = flags;

  pDecoder->ppDecoders = slapAlloc(void *, SLAP_SUB_BUFFER_COUNT);
  
  if (!pDecoder->ppDecoders)
    goto epilogue;

  memset(pDecoder->ppDecoders, 0, sizeof(void *) * SLAP_SUB_BUFFER_COUNT);

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    pDecoder->ppDecoders[i] = tjInitDecompress();

    if (!pDecoder->ppDecoders[i])
      goto epilogue;
  }

  size_t lowResSizeX = sizeX >> 3;
  size_t lowResSizeY = sizeY >> 3;

  if (pDecoder->mode.flags.stereo)
    lowResSizeY <<= 1;

  pDecoder->pLowResData = slapAlloc(uint8_t, lowResSizeX * lowResSizeY * 3 / 2);

  if (!pDecoder->pLowResData)
    goto epilogue;

  pDecoder->pLastFrame = slapAlloc(uint8_t, sizeX * sizeY * 3 / 2);

  if (!pDecoder->pLastFrame)
    goto epilogue;

  const size_t threadCount = ThreadPool_GetSystemThreadCount();

  pDecoder->pThreadPoolHandle = ThreadPool_Init(threadCount);

  if (!pDecoder->pThreadPoolHandle)
    goto epilogue;

  return pDecoder;

epilogue:
  if (pDecoder->ppDecoders)
  {
    for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
      if (pDecoder->ppDecoders[i])
        tjDestroy(pDecoder->ppDecoders[i]);

    slapFreePtr(&pDecoder->ppDecoders);
  }

  if (pDecoder->pLowResData)
    slapFreePtr(&pDecoder->pLowResData);

  if (pDecoder->pLastFrame)
    slapFreePtr(&pDecoder->pLastFrame);

  if (pDecoder->pThreadPoolHandle)
    ThreadPool_Destroy(pDecoder->pThreadPoolHandle);

  slapFreePtr(&pDecoder);

  return NULL;
}

void slapDestroyDecoder(IN_OUT slapDecoder **ppDecoder)
{
  if (ppDecoder && *ppDecoder)
  {
    if ((*ppDecoder)->ppDecoders)
    {
      for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
        if ((*ppDecoder)->ppDecoders[i])
          tjDestroy((*ppDecoder)->ppDecoders[i]);

      slapFreePtr(&(*ppDecoder)->ppDecoders);
    }

    if ((*ppDecoder)->pLowResData)
      slapFreePtr(&(*ppDecoder)->pLowResData);

    if ((*ppDecoder)->pLastFrame)
      slapFreePtr(&(*ppDecoder)->pLastFrame);

    if ((*ppDecoder)->pThreadPoolHandle)
      ThreadPool_Destroy((*ppDecoder)->pThreadPoolHandle);
  }

  slapFreePtr(ppDecoder);
}

slapResult slapDecoder_DecodeSubFrame(IN slapDecoder *pDecoder, const size_t decoderIndex, IN void **ppCompressedData, IN size_t *pLength, IN_OUT void *pYUVData)
{
  slapResult result = slapSuccess;

  const size_t subFrameHeight = pDecoder->resY * 3 / 2 / SLAP_SUB_BUFFER_COUNT;
  uint8_t *pOutData = ((uint8_t *)pYUVData) + decoderIndex * subFrameHeight * pDecoder->resX;

  if (pDecoder->mode.flags.encoder == 0)
  {
    if (subFrameHeight * decoderIndex * 2 / 3 < pDecoder->resY)
      result = _slapDecompressChannel(pOutData, ppCompressedData[decoderIndex], pLength[decoderIndex], pDecoder->resX, subFrameHeight, pDecoder->ppDecoders[decoderIndex]);
    else
      result = _slapDecompressChannel(pOutData, ppCompressedData[decoderIndex], pLength[decoderIndex], pDecoder->resX >> 1, subFrameHeight << 1, pDecoder->ppDecoders[decoderIndex]);

    if (result != slapSuccess)
      goto epilogue;
  }

epilogue:
  return result;
}

slapResult slapDecoder_FinalizeFrame(IN slapDecoder *pDecoder, IN void *pData, const size_t length, IN_OUT void *pYUVData)
{
  slapResult result = slapSuccess;

  if (!pDecoder || !pData || !length || !pYUVData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  if (pDecoder->mode.flags.encoder == 0)
  {
    if (pDecoder->frameIndex % pDecoder->iframeStep != 0)
      _slapAddStereoDiffYUV420AndAddLastFrameDiff(pYUVData, pDecoder->pLastFrame, pDecoder->resX, pDecoder->resY);
    else
      _slapAddStereoDiffYUV420AndCopyToLastFrame(pYUVData, pDecoder->pLastFrame, pDecoder->resX, pDecoder->resY);
  }

  pDecoder->frameIndex++;

epilogue:
  return result;
}

slapFileReader * slapCreateFileReader(const char *filename)
{
  slapFileReader *pFileReader = slapAlloc(slapFileReader, 1);
  size_t frameSize = 0;

  if (!pFileReader)
    goto epilogue;

  slapSetZero(pFileReader, slapFileReader);

  pFileReader->pFile = fopen(filename, "rb");

  if (!pFileReader->pFile)
    goto epilogue;

  if (SLAP_PRE_HEADER_SIZE != fread(pFileReader->preHeaderBlock, sizeof(uint64_t), SLAP_PRE_HEADER_SIZE, pFileReader->pFile))
    goto epilogue;

  pFileReader->pHeader = slapAlloc(uint64_t, pFileReader->preHeaderBlock[SLAP_PRE_HEADER_HEADER_SIZE_INDEX]);

  if (!pFileReader->pHeader)
    goto epilogue;

  if (pFileReader->preHeaderBlock[SLAP_PRE_HEADER_HEADER_SIZE_INDEX] != fread(pFileReader->pHeader, sizeof(uint64_t), pFileReader->preHeaderBlock[SLAP_PRE_HEADER_HEADER_SIZE_INDEX], pFileReader->pFile))
    goto epilogue;

  pFileReader->headerOffset = ftell(pFileReader->pFile);

  pFileReader->pDecoder = slapCreateDecoder(pFileReader->preHeaderBlock[SLAP_PRE_HEADER_FRAME_SIZEX_INDEX], pFileReader->preHeaderBlock[SLAP_PRE_HEADER_FRAME_SIZEY_INDEX], pFileReader->preHeaderBlock[SLAP_PRE_HEADER_CODEC_FLAGS_INDEX]);

  if (!pFileReader->pDecoder)
    goto epilogue;

  frameSize = pFileReader->pDecoder->resX * pFileReader->pDecoder->resX * 3 / 2;

  pFileReader->pDecodedFrameYUV = slapAlloc(uint8_t, frameSize);

  if (!pFileReader->pDecodedFrameYUV)
    goto epilogue;

  return pFileReader;

epilogue:

  slapFreePtr(&(pFileReader)->pHeader);
  slapFreePtr(&(pFileReader)->pCurrentFrame);
  slapFreePtr(&(pFileReader)->pDecodedFrameYUV);

  if (pFileReader->pFile)
    fclose(pFileReader->pFile);

  if (pFileReader->pDecoder)
    slapDestroyDecoder(&pFileReader->pDecoder);

  slapFreePtr(&pFileReader);

  return NULL;
}

void slapDestroyFileReader(IN_OUT slapFileReader **ppFileReader)
{
  if (ppFileReader && *ppFileReader)
  {
    slapFreePtr(&(*ppFileReader)->pHeader);
    slapFreePtr(&(*ppFileReader)->pCurrentFrame);
    slapFreePtr(&(*ppFileReader)->pDecodedFrameYUV);
    slapDestroyDecoder(&(*ppFileReader)->pDecoder);
    fclose((*ppFileReader)->pFile);
  }

  slapFreePtr(ppFileReader);
}

slapResult slapFileReader_GetResolution(IN slapFileReader *pFileReader, OUT size_t *pResolutionX, OUT size_t *pResolutionY)
{
  if (!pFileReader || !pResolutionX || !pResolutionY)
    return slapError_ArgumentNull;

  *pResolutionX = pFileReader->pDecoder->resX;
  *pResolutionY = pFileReader->pDecoder->resY;

  return slapSuccess;
}

slapResult slapFileReader_GetLowResFrameResolution(IN slapFileReader * pFileReader, OUT size_t * pResolutionX, OUT size_t * pResolutionY)
{
  if (!pFileReader || !pResolutionX || !pResolutionY)
    return slapError_ArgumentNull;

  *pResolutionX = pFileReader->pDecoder->resX >> 3;
  *pResolutionY = pFileReader->pDecoder->resY >> 3;

  if (pFileReader->pDecoder->mode.flags.stereo)
    *pResolutionY >>= 1;

  return slapSuccess;
}

slapResult _slapFileReader_ReadNextFrameFull(IN slapFileReader *pFileReader)
{
  slapResult result = slapSuccess;
  uint64_t position;

  if (!pFileReader)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  if (pFileReader->frameIndex >= pFileReader->preHeaderBlock[SLAP_PRE_HEADER_FRAME_COUNT_INDEX])
  {
    result = slapError_EndOfStream;
    goto epilogue;
  }

  position = pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * pFileReader->frameIndex + 2 + SLAP_HEADER_FRAME_OFFSET_INDEX] + pFileReader->headerOffset;
  pFileReader->currentFrameSize = pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * pFileReader->frameIndex + 2 + SLAP_HEADER_FRAME_DATA_SIZE_INDEX];

  if (pFileReader->currentFrameAllocatedSize < pFileReader->currentFrameSize)
  {
    slapRealloc(&pFileReader->pCurrentFrame, uint8_t, pFileReader->currentFrameSize);
    pFileReader->currentFrameAllocatedSize = pFileReader->currentFrameSize;

    if (!pFileReader->pCurrentFrame)
    {
      pFileReader->currentFrameAllocatedSize = 0;
      result = slapError_MemoryAllocation;
      goto epilogue;
    }
  }

  if (fseek(pFileReader->pFile, (long)position, SEEK_SET))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  if (pFileReader->currentFrameSize != fread(pFileReader->pCurrentFrame, 1, pFileReader->currentFrameSize, pFileReader->pFile))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  pFileReader->frameIndex++;

epilogue:
  return result;
}

#ifdef SLAP_MULTITHREADED

size_t _slapDecoderTask_DecodeSubframe(void *pData)
{
  _slapDecoderSubTaskData0 *pUserData = (_slapDecoderSubTaskData0 *)pData;

  return (size_t)slapDecoder_DecodeSubFrame(pUserData->pDecoder, pUserData->index, pUserData->pDataAddrs, pUserData->pDataSizes, pUserData->pYUVFrame);
}

#endif

#include "time.h"

slapResult _slapFileReader_DecodeCurrentFrameFull(IN slapFileReader *pFileReader)
{
  slapResult result = slapSuccess;
  void *dataAddrs[SLAP_SUB_BUFFER_COUNT];
  size_t dataSizes[SLAP_SUB_BUFFER_COUNT];
#ifdef SLAP_MULTITHREADED
  ThreadPool_TaskHandle taskHandle[SLAP_SUB_BUFFER_COUNT];
  _slapDecoderSubTaskData0 taskData[SLAP_SUB_BUFFER_COUNT];
#endif

  if (!pFileReader)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    dataAddrs[i] = ((uint8_t *)pFileReader->pCurrentFrame) + pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * (pFileReader->frameIndex - 1) + SLAP_HEADER_PER_FRAME_FULL_FRAME_OFFSET + i * 2 + SLAP_HEADER_FRAME_OFFSET_INDEX];
    dataSizes[i] = pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * (pFileReader->frameIndex - 1) + SLAP_HEADER_PER_FRAME_FULL_FRAME_OFFSET + i * 2 + SLAP_HEADER_FRAME_DATA_SIZE_INDEX];
  }

  clock_t t = clock();

#ifdef SLAP_MULTITHREADED

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
  {
    taskData[i].pDataSizes = dataSizes;
    taskData[i].index = i;
    taskData[i].pDataAddrs = dataAddrs;
    taskData[i].pDecoder = pFileReader->pDecoder;
    taskData[i].pYUVFrame = pFileReader->pDecodedFrameYUV;

    taskHandle[i] = ThreadPool_CreateTask(_slapDecoderTask_DecodeSubframe, (void *)&taskData[i]);
    ThreadPool_EnqueueTask(pFileReader->pDecoder->pThreadPoolHandle, taskHandle[i]);
  }

  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
    ThreadPool_JoinTask(taskHandle[i]);

#else
  for (size_t i = 0; i < SLAP_SUB_BUFFER_COUNT; i++)
    result = slapDecoder_DecodeSubFrame(pFileReader->pDecoder, i, dataAddrs, dataSizes, pFileReader->pDecodedFrameYUV);
#endif

  t = clock() - t;
  printf("Decoding part1: %" PRIi32 " ms | ", t);
  t = clock();

  result = slapDecoder_FinalizeFrame(pFileReader->pDecoder, pFileReader->pCurrentFrame, pFileReader->currentFrameSize, pFileReader->pDecodedFrameYUV);

  t = clock() - t;
  printf("Decoding part2: %" PRIi32 " ms \n", t);

  if (result != slapSuccess)
    goto epilogue;

epilogue:
  return result;
}

slapResult _slapFileReader_ReadNextFrameLowRes(IN slapFileReader *pFileReader)
{
  slapResult result = slapSuccess;
  uint64_t position;

  if (!pFileReader)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  if (pFileReader->frameIndex >= pFileReader->preHeaderBlock[SLAP_PRE_HEADER_FRAME_COUNT_INDEX])
  {
    result = slapError_EndOfStream;
    goto epilogue;
  }

  position = pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * pFileReader->frameIndex + SLAP_HEADER_FRAME_OFFSET_INDEX] + pFileReader->headerOffset;
  pFileReader->currentFrameSize = pFileReader->pHeader[SLAP_HEADER_PER_FRAME_SIZE * pFileReader->frameIndex + SLAP_HEADER_FRAME_DATA_SIZE_INDEX];

  if (pFileReader->currentFrameAllocatedSize < pFileReader->currentFrameSize)
  {
    slapRealloc(&pFileReader->pCurrentFrame, uint8_t, pFileReader->currentFrameSize);
    pFileReader->currentFrameAllocatedSize = pFileReader->currentFrameSize;

    if (!pFileReader->pCurrentFrame)
    {
      pFileReader->currentFrameAllocatedSize = 0;
      result = slapError_MemoryAllocation;
      goto epilogue;
    }
  }

  if (fseek(pFileReader->pFile, (long)position, SEEK_SET))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  if (pFileReader->currentFrameSize != fread(pFileReader->pCurrentFrame, 1, pFileReader->currentFrameSize, pFileReader->pFile))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  pFileReader->frameIndex++;

epilogue:
  return result;
}

slapResult _slapFileReader_DecodeCurrentFrameLowRes(IN slapFileReader *pFileReader)
{
  slapResult result = slapSuccess;

  if (!pFileReader)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  size_t resX, resY;
  slapFileReader_GetLowResFrameResolution(pFileReader, &resX, &resY);

  if (pFileReader->pDecoder->mode.flags.encoder == 0)
  {
    if (tjDecompressToYUV2(pFileReader->pDecoder->ppDecoders[0], (unsigned char *)pFileReader->pCurrentFrame, (unsigned long)pFileReader->currentFrameSize, (unsigned char *)pFileReader->pDecodedFrameYUV, (int)resX, 4, (int)resY, TJFLAG_FASTDCT))
    {
      slapLog(tjGetErrorStr2(pFileReader->pDecoder->ppDecoders[0]));
      result = slapError_Compress_Internal;
      goto epilogue;
    }
  }

  pFileReader->pDecoder->frameIndex++;

  if (result != slapSuccess)
    goto epilogue;

epilogue:
  return result;
}

//////////////////////////////////////////////////////////////////////////
// Core En- & Decoding Functions
//////////////////////////////////////////////////////////////////////////

slapResult _slapCompressChannel(IN void *pData, IN_OUT void **ppCompressedData, IN_OUT size_t *pCompressedDataSize, const size_t width, const size_t height, const int quality, IN void *pCompressor)
{
  unsigned long length = (unsigned long)*pCompressedDataSize;

  if (tjCompress2(pCompressor, (unsigned char *)pData, (int)width, (int)width, (int)height, TJPF_GRAY, (unsigned char **)ppCompressedData, &length, TJSAMP_GRAY, quality, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pCompressor));
    return slapError_Compress_Internal;
  }

  *pCompressedDataSize = length;

  return slapSuccess;
}

slapResult _slapCompressYUV420(IN void *pData, IN_OUT void **ppCompressedData, IN_OUT size_t *pCompressedDataSize, const size_t width, const size_t height, const int quality, IN void *pCompressor)
{
  unsigned long length = (unsigned long)*pCompressedDataSize;

  if (tjCompressFromYUV(pCompressor, (unsigned char *)pData, (int)width, 32, (int)height, TJSAMP_420, (unsigned char **)ppCompressedData, &length, quality, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pCompressor));
    return slapError_Compress_Internal;
  }

  *pCompressedDataSize = length;

  return slapSuccess;
}

slapResult _slapDecompressChannel(IN void *pData, IN_OUT void *pCompressedData, const size_t compressedDataSize, const size_t width, const size_t height, IN void *pDecompressor)
{
  if (tjDecompress2(pDecompressor, (unsigned char *)pCompressedData, (unsigned long)compressedDataSize, (unsigned char *)pData, (int)width, (int)width, (int)height, TJPF_GRAY, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pDecompressor));
    return slapError_Compress_Internal;
  }

  return slapSuccess;
}

slapResult _slapDecompressYUV420(IN void *pData, IN_OUT void *pCompressedData, const size_t compressedDataSize, const size_t width, const size_t height, IN void *pDecompressor)
{
  if (tjDecompressToYUV2(pDecompressor, (unsigned char *)pCompressedData, (unsigned long)compressedDataSize, (unsigned char *)pData, (int)width, 32, (int)height, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pDecompressor));
    return slapError_Compress_Internal;
  }

  return slapSuccess;
}

void _slapLastFrameDiffAndStereoDiffAndSubBufferYUV420(IN_OUT void *pLastFrame, IN_OUT void *pData, IN_OUT void *pLowRes, const size_t resX, const size_t resY)
{
  uint8_t *pMainFrameY = (uint8_t *)pData;
  uint16_t *pSubFrameYUV = (uint16_t *)pLowRes;
  __m128i *pLastFrameYUV = (__m128i *)pLastFrame;

  size_t resXdiv16 = resX >> 4;
  size_t halfFrameDiv16Quarter = resXdiv16 * resY >> 3;

  __m128i *pCB0 = (__m128i *)pMainFrameY;
  __m128i *pCB1 = (__m128i *)pCB0 + 1;

  __m128i *pCB0_ = (__m128i *)pMainFrameY + resXdiv16 * (resY >> 1);
  __m128i *pCB1_ = (__m128i *)pCB0_ + 1;

  __m128i *pLF0 = (__m128i *)pLastFrameYUV;
  __m128i *pLF1 = (__m128i *)pLF0 + 1;

  __m128i *pLF0_ = (__m128i *)pLastFrameYUV + resXdiv16 * (resY >> 1);
  __m128i *pLF1_ = (__m128i *)pLF0_ + 1;

  __m128i half = { 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };

#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
  __m128i shuffle = { 0, 7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };
#endif

  const size_t stepSize = 2;
  size_t itX = resX / (sizeof(__m128i) * stepSize);
  size_t itY = resY >> 1;

  for (size_t i = 0; i < 3; i++)
  {
    for (size_t y = 0; y < itY; y += 8)
    {
      for (size_t x = 0; x < itX; x++)
      {
        __m128i cb0 = _mm_load_si128(pCB0);
        __m128i cb1 = _mm_load_si128(pCB1);
        __m128i cb0_ = _mm_load_si128(pCB0_);
        __m128i cb1_ = _mm_load_si128(pCB1_);

        // SubBuffer
        {
#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          __m128i v = _mm_shuffle_epi8(cb0, shuffle);
#else
          __m128i v = _mm_srli_si128(cb0, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;

#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          v = _mm_shuffle_epi8(cb1, shuffle);
#else
          v = _mm_srli_si128(cb1, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;
        }

        __m128i lf0 = _mm_load_si128(pLF0);
        __m128i lf1 = _mm_load_si128(pLF1);
        __m128i lf0_ = _mm_load_si128(pLF0_);
        __m128i lf1_ = _mm_load_si128(pLF1_);

        // last frame diff
        cb0 = _mm_add_epi8(_mm_sub_epi8(lf0, cb0), half);
        cb0_ = _mm_add_epi8(_mm_sub_epi8(lf0_, cb0_), half);
        cb1 = _mm_add_epi8(_mm_sub_epi8(lf1, cb1), half);
        cb1_ = _mm_add_epi8(_mm_sub_epi8(lf1_, cb1_), half);

        _mm_store_si128(pCB0, cb0);
        _mm_store_si128(pCB1, cb1);

        // Stereo diff
        cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, cb0), half);
        cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, cb1), half);
        _mm_store_si128(pCB0_, cb0_);
        _mm_store_si128(pCB1_, cb1_);

        pCB0 += stepSize;
        pCB1 += stepSize;
        pCB0_ += stepSize;
        pCB1_ += stepSize;
        pLF0 += stepSize;
        pLF1 += stepSize;
        pLF0_ += stepSize;
        pLF1_ += stepSize;
      }


      for (size_t k = 0; k < 7; k++)
      {
        for (size_t x = 0; x < itX; x++)
        {
          __m128i cb0 = _mm_load_si128(pCB0);
          __m128i cb1 = _mm_load_si128(pCB1);
          __m128i cb0_ = _mm_load_si128(pCB0_);
          __m128i cb1_ = _mm_load_si128(pCB1_);
          __m128i lf0 = _mm_load_si128(pLF0);
          __m128i lf1 = _mm_load_si128(pLF1);
          __m128i lf0_ = _mm_load_si128(pLF0_);
          __m128i lf1_ = _mm_load_si128(pLF1_);

          // last frame diff
          cb0 = _mm_add_epi8(_mm_sub_epi8(lf0, cb0), half);
          cb0_ = _mm_add_epi8(_mm_sub_epi8(lf0_, cb0_), half);
          cb1 = _mm_add_epi8(_mm_sub_epi8(lf1, cb1), half);
          cb1_ = _mm_add_epi8(_mm_sub_epi8(lf1_, cb1_), half);

          _mm_store_si128(pCB0, cb0);
          _mm_store_si128(pCB1, cb1);

          // Stereo diff
          cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, cb0), half);
          cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, cb1), half);
          _mm_store_si128(pCB0_, cb0_);
          _mm_store_si128(pCB1_, cb1_);

          pCB0 += stepSize;
          pCB1 += stepSize;
          pCB0_ += stepSize;
          pCB1_ += stepSize;
          pLF0 += stepSize;
          pLF1 += stepSize;
          pLF0_ += stepSize;
          pLF1_ += stepSize;
        }
      }
    }

    if (i == 0)
    {
      itX >>= 1;
      itY >>= 1;
    }

    pCB0 = pCB0_;
    pCB1 = pCB1_;
    pCB0_ += halfFrameDiv16Quarter;
    pCB1_ = pCB0_ + 1;
    pLF0 = pLF0_;
    pLF1 = pLF1_;
    pLF0_ += halfFrameDiv16Quarter;
    pLF1_ = pLF0_ + 1;
  }
}

void _slapCopyToLastFrameAndGenSubBufferAndStereoDiffYUV420(IN_OUT void *pData, OUT void *pLowResData, OUT void *pLastFrame, const size_t resX, const size_t resY)
{
  uint8_t *pMainFrameY = (uint8_t *)pData;
  uint16_t *pSubFrameYUV = (uint16_t *)pLowResData;
  __m128i *pLastFrameYUV = (__m128i *)pLastFrame;

#define GREATER_OR_EQUAL_TO_6_BLOCKS
#define GREATER_OR_EQUAL_TO_8_BLOCKS

  size_t resXdiv16 = resX >> 4;
  size_t halfFrameDiv16Quarter = resXdiv16 * resY >> 3;

  __m128i *pCB0 = (__m128i *)pMainFrameY;
  __m128i *pCB1 = pCB0 + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pCB2 = pCB0 + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pCB3 = pCB0 + 3;
#endif

  __m128i *pCB0_ = (__m128i *)pMainFrameY + resXdiv16 * (resY >> 1);
  __m128i *pCB1_ = pCB0_ + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pCB2_ = pCB0_ + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pCB3_ = pCB0_ + 3;
#endif

  __m128i *pLF0 = (__m128i *)pLastFrameYUV;
  __m128i *pLF1 = pLF0 + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pLF2 = pLF0 + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pLF3 = pLF0 + 3;
#endif

  __m128i *pLF0_ = (__m128i *)pLastFrameYUV + resXdiv16 * (resY >> 1);
  __m128i *pLF1_ = pLF0_ + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pLF2_ = pLF0_ + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pLF3_ = pLF0_ + 3;
#endif

    __m128i half = { 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118 };

#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
  __m128i shuffle = { 0, 7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };
#endif

#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  const size_t stepSize = 4;
#else
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  const size_t stepSize = 3;
#else
  const size_t stepSize = 2;
#endif
#endif

  size_t itX = resX / (sizeof(__m128i) * stepSize);
  size_t itY = resY >> 1;

  for (size_t i = 0; i < 3; i++)
  {
    for (size_t y = 0; y < itY; y += 8)
    {
      for (size_t x = 0; x < itX; x++)
      {
        __m128i cb0 = _mm_load_si128(pCB0);
        __m128i cb1 = _mm_load_si128(pCB1);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        __m128i cb2 = _mm_load_si128(pCB2);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        __m128i cb3 = _mm_load_si128(pCB3);
#endif

        // SubBuffer
        {
#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          __m128i v = _mm_shuffle_epi8(cb0, shuffle);
#else
          __m128i v = _mm_srli_si128(cb0, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;

#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          v = _mm_shuffle_epi8(cb1, shuffle);
#else
          v = _mm_srli_si128(cb1, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;

#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          v = _mm_shuffle_epi8(cb2, shuffle);
#else
          v = _mm_srli_si128(cb2, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;
#endif

#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
#ifdef SLAP_HIGH_QUALITY_DOWNSCALE
          v = _mm_shuffle_epi8(cb3, shuffle);
#else
          v = _mm_srli_si128(cb3, 7);
#endif
          *pSubFrameYUV = *(uint16_t *)&v;
          pSubFrameYUV++;
#endif
        }

        __m128i cb0_ = _mm_load_si128(pCB0_);
        __m128i cb1_ = _mm_load_si128(pCB1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        __m128i cb2_ = _mm_load_si128(pCB2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        __m128i cb3_ = _mm_load_si128(pCB3_);
#endif

        // Copy to last frame
        _mm_store_si128(pLF0, cb0);
        _mm_store_si128(pLF1, cb1);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        _mm_store_si128(pLF2, cb2);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        _mm_store_si128(pLF3, cb3);
#endif
        _mm_store_si128(pLF0_, cb0_);
        _mm_store_si128(pLF1_, cb1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        _mm_store_si128(pLF2_, cb2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        _mm_store_si128(pLF3_, cb3_);
#endif

        // Stereo diff
        cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, cb0), half);
        cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, cb1), half);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        cb2_ = _mm_add_epi8(_mm_sub_epi8(cb2_, cb2), half);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        cb3_ = _mm_add_epi8(_mm_sub_epi8(cb3_, cb3), half);
#endif
        _mm_store_si128(pCB0_, cb0_);
        _mm_store_si128(pCB1_, cb1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        _mm_store_si128(pCB2_, cb2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        _mm_store_si128(pCB3_, cb3_);
#endif

        pCB0 += stepSize;
        pCB1 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        pCB2 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        pCB3 += stepSize;
#endif
        pCB0_ += stepSize;
        pCB1_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        pCB2_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        pCB3_ += stepSize;
#endif
        pLF0 += stepSize;
        pLF1 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        pLF2 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        pLF3 += stepSize;
#endif
        pLF0_ += stepSize;
        pLF1_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
        pLF2_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
        pLF3_ += stepSize;
#endif
      }


      for (size_t k = 0; k < 7; k++)
      {
        for (size_t x = 0; x < itX; x++)
        {
          __m128i cb0 = _mm_load_si128(pCB0);
          __m128i cb1 = _mm_load_si128(pCB1);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          __m128i cb2 = _mm_load_si128(pCB2);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          __m128i cb3 = _mm_load_si128(pCB3);
#endif
          __m128i cb0_ = _mm_load_si128(pCB0_);
          __m128i cb1_ = _mm_load_si128(pCB1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          __m128i cb2_ = _mm_load_si128(pCB2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          __m128i cb3_ = _mm_load_si128(pCB3_);
#endif

          // Copy to last frame
          _mm_store_si128(pLF0, cb0);
          _mm_store_si128(pLF1, cb1);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          _mm_store_si128(pLF2, cb2);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          _mm_store_si128(pLF3, cb3);
#endif
          _mm_store_si128(pLF0_, cb0_);
          _mm_store_si128(pLF1_, cb1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          _mm_store_si128(pLF2_, cb2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          _mm_store_si128(pLF3_, cb3_);
#endif

          // Stereo diff
          cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, cb0), half);
          cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, cb1), half);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          cb2_ = _mm_add_epi8(_mm_sub_epi8(cb2_, cb2), half);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          cb3_ = _mm_add_epi8(_mm_sub_epi8(cb3_, cb3), half);
#endif
          _mm_store_si128(pCB0_, cb0_);
          _mm_store_si128(pCB1_, cb1_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          _mm_store_si128(pCB2_, cb2_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          _mm_store_si128(pCB3_, cb3_);
#endif

          pCB0 += stepSize;
          pCB1 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          pCB2 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          pCB3 += stepSize;
#endif
          pCB0_ += stepSize;
          pCB1_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          pCB2_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          pCB3_ += stepSize;
#endif
          pLF0 += stepSize;
          pLF1 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          pLF2 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          pLF3 += stepSize;
#endif
          pLF0_ += stepSize;
          pLF1_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
          pLF2_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
          pLF3_ += stepSize;
#endif
        }
      }
    }

    if (i == 0)
    {
      itX >>= 1;
      itY >>= 1;

      __m128i tmp = { 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
      half = tmp;
    }

    pCB0 = pCB0_;
    pCB1 = pCB1_;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
    pCB2 = pCB2_;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
    pCB3 = pCB3_;
#endif
    pCB0_ += halfFrameDiv16Quarter;
    pCB1_ = pCB0_ + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
    pCB2_ = pCB0_ + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
    pCB3_ = pCB0_ + 3;
#endif
    pLF0 = pLF0_;
    pLF1 = pLF1_;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
    pLF2 = pLF2_;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
    pLF3 = pLF3_;
#endif
    pLF0_ += halfFrameDiv16Quarter;
    pLF1_ = pLF0_ + 1;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
    pLF2_ = pLF0_ + 2;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
    pLF3_ = pLF0_ + 3;
#endif
  }

#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
#undef GREATER_OR_EQUAL_TO_6_BLOCKS
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
#undef GREATER_OR_EQUAL_TO_8_BLOCKS
#endif
}

void _slapAddStereoDiffYUV420(IN_OUT void *pData, const size_t resX, const size_t resY)
{
  uint8_t *pMainFrameY = (uint8_t *)pData;

#define GREATER_OR_EQUAL_TO_6_BLOCKS
#define GREATER_OR_EQUAL_TO_8_BLOCKS

  size_t resXdiv16 = resX >> 4;

  __m128i *pCB0 = (__m128i *)pMainFrameY;
  __m128i *pCB1 = pCB0 + 1;
  __m128i *pCB2 = pCB0 + 2;
  __m128i *pCB3 = pCB0 + 3;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pCB4 = pCB0 + 4;
  __m128i *pCB5 = pCB0 + 5;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pCB6 = pCB0 + 6;
  __m128i *pCB7 = pCB0 + 7;
#endif

  __m128i *pCB0_ = (__m128i *)pMainFrameY + resXdiv16 * (resY >> 1);
  __m128i *pCB1_ = pCB0_ + 1;
  __m128i *pCB2_ = pCB0_ + 2;
  __m128i *pCB3_ = pCB0_ + 3;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  __m128i *pCB4_ = pCB0_ + 4;
  __m128i *pCB5_ = pCB0_ + 5;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  __m128i *pCB6_ = pCB0_ + 6;
  __m128i *pCB7_ = pCB0_ + 7;
#endif

  __m128i halfY = { 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118 };
  __m128i halfUV = { 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126 };

#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  const size_t stepSize = 8;
#else
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  const size_t stepSize = 6;
#else
  const size_t stepSize = 4;
#endif
#endif

  const size_t lumaX = resX / (sizeof(__m128i) * stepSize);
  const size_t chromaX = lumaX >> 1;

  for (size_t y = 0; y < (resY >> 1); y++)
  {
    for (size_t x = 0; x < lumaX; x++)
    {
      __m128i cb0 = _mm_load_si128(pCB0);
      __m128i cb1 = _mm_load_si128(pCB1);
      __m128i cb2 = _mm_load_si128(pCB2);
      __m128i cb3 = _mm_load_si128(pCB3);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      __m128i cb4 = _mm_load_si128(pCB4);
      __m128i cb5 = _mm_load_si128(pCB5);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      __m128i cb6 = _mm_load_si128(pCB6);
      __m128i cb7 = _mm_load_si128(pCB7);
#endif

      __m128i cb0_ = _mm_load_si128(pCB0_);
      __m128i cb1_ = _mm_load_si128(pCB1_);
      __m128i cb2_ = _mm_load_si128(pCB2_);
      __m128i cb3_ = _mm_load_si128(pCB3_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      __m128i cb4_ = _mm_load_si128(pCB4_);
      __m128i cb5_ = _mm_load_si128(pCB5_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      __m128i cb6_ = _mm_load_si128(pCB6_);
      __m128i cb7_ = _mm_load_si128(pCB7_);
#endif

      // Add stereo diff.
      cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfY), cb0);
      cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, halfY), cb1);
      cb2_ = _mm_add_epi8(_mm_sub_epi8(cb2_, halfY), cb2);
      cb3_ = _mm_add_epi8(_mm_sub_epi8(cb3_, halfY), cb3);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      cb4_ = _mm_add_epi8(_mm_sub_epi8(cb4_, halfY), cb4);
      cb5_ = _mm_add_epi8(_mm_sub_epi8(cb5_, halfY), cb5);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      cb6_ = _mm_add_epi8(_mm_sub_epi8(cb6_, halfY), cb6);
      cb7_ = _mm_add_epi8(_mm_sub_epi8(cb7_, halfY), cb7);
#endif

      _mm_store_si128(pCB0_, cb0_);
      _mm_store_si128(pCB1_, cb1_);
      _mm_store_si128(pCB2_, cb2_);
      _mm_store_si128(pCB3_, cb3_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      _mm_store_si128(pCB4_, cb4_);
      _mm_store_si128(pCB5_, cb5_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      _mm_store_si128(pCB6_, cb6_);
      _mm_store_si128(pCB7_, cb7_);
#endif

      pCB0 += stepSize;
      pCB1 += stepSize;
      pCB2 += stepSize;
      pCB3 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      pCB4 += stepSize;
      pCB5 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      pCB6 += stepSize;
      pCB7 += stepSize;
#endif
      pCB0_ += stepSize;
      pCB1_ += stepSize;
      pCB2_ += stepSize;
      pCB3_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      pCB4_ += stepSize;
      pCB5_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      pCB6_ += stepSize;
      pCB7_ += stepSize;
#endif
    }
  }

  size_t halfFrameDiv16Quarter = resXdiv16 * resY >> 3;

  int first = 1;
chromaSubSampleBuffer:

  pCB0 = pCB0_;
  pCB1 = pCB0 + 1;
  pCB2 = pCB0 + 2;
  pCB3 = pCB0 + 3;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  pCB4 = pCB0 + 4;
  pCB5 = pCB0 + 5;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  pCB6 = pCB0 + 6;
  pCB7 = pCB0 + 7;
#endif

  pCB0_ += halfFrameDiv16Quarter;
  pCB1_ = pCB0_ + 1;
  pCB2_ = pCB0_ + 2;
  pCB3_ = pCB0_ + 3;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
  pCB4_ = pCB0_ + 4;
  pCB5_ = pCB0_ + 5;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
  pCB6_ = pCB0_ + 6;
  pCB7_ = pCB0_ + 7;
#endif

  for (size_t y = 0; y < (resY >> 2); y++)
  {
    for (size_t x = 0; x < chromaX; x++)
    {
      __m128i cb0 = _mm_load_si128(pCB0);
      __m128i cb1 = _mm_load_si128(pCB1);
      __m128i cb2 = _mm_load_si128(pCB2);
      __m128i cb3 = _mm_load_si128(pCB3);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      __m128i cb4 = _mm_load_si128(pCB4);
      __m128i cb5 = _mm_load_si128(pCB5);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      __m128i cb6 = _mm_load_si128(pCB6);
      __m128i cb7 = _mm_load_si128(pCB7);
#endif

      __m128i cb0_ = _mm_load_si128(pCB0_);
      __m128i cb1_ = _mm_load_si128(pCB1_);
      __m128i cb2_ = _mm_load_si128(pCB2_);
      __m128i cb3_ = _mm_load_si128(pCB3_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      __m128i cb4_ = _mm_load_si128(pCB4_);
      __m128i cb5_ = _mm_load_si128(pCB5_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      __m128i cb6_ = _mm_load_si128(pCB6_);
      __m128i cb7_ = _mm_load_si128(pCB7_);
#endif

      // Add stereo diff.
      cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfUV), cb0);
      cb1_ = _mm_add_epi8(_mm_sub_epi8(cb1_, halfUV), cb1);
      cb2_ = _mm_add_epi8(_mm_sub_epi8(cb2_, halfUV), cb2);
      cb3_ = _mm_add_epi8(_mm_sub_epi8(cb3_, halfUV), cb3);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      cb4_ = _mm_add_epi8(_mm_sub_epi8(cb4_, halfUV), cb4);
      cb5_ = _mm_add_epi8(_mm_sub_epi8(cb5_, halfUV), cb5);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      cb6_ = _mm_add_epi8(_mm_sub_epi8(cb6_, halfUV), cb6);
      cb7_ = _mm_add_epi8(_mm_sub_epi8(cb7_, halfUV), cb7);
#endif

      _mm_store_si128(pCB0_, cb0_);
      _mm_store_si128(pCB1_, cb1_);
      _mm_store_si128(pCB2_, cb2_);
      _mm_store_si128(pCB3_, cb3_);
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      _mm_store_si128(pCB4_, cb4_);
      _mm_store_si128(pCB5_, cb5_);
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      _mm_store_si128(pCB6_, cb6_);
      _mm_store_si128(pCB7_, cb7_);
#endif

      pCB0 += stepSize;
      pCB1 += stepSize;
      pCB2 += stepSize;
      pCB3 += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      pCB4 += stepSize;
      pCB5 += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      pCB6 += stepSize;
      pCB7 += stepSize;
#endif
      pCB0_ += stepSize;
      pCB1_ += stepSize;
      pCB2_ += stepSize;
      pCB3_ += stepSize;
#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
      pCB4_ += stepSize;
      pCB5_ += stepSize;
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
      pCB6_ += stepSize;
      pCB7_ += stepSize;
#endif
    }
  }

  if (first)
  {
    first = 0;
    goto chromaSubSampleBuffer;
  }


#ifdef GREATER_OR_EQUAL_TO_6_BLOCKS
#undef GREATER_OR_EQUAL_TO_6_BLOCKS
#endif
#ifdef GREATER_OR_EQUAL_TO_8_BLOCKS
#undef GREATER_OR_EQUAL_TO_8_BLOCKS
#endif
}

void _slapAddStereoDiffYUV420AndCopyToLastFrame(IN_OUT void * pData, OUT void * pLastFrame, const size_t resX, const size_t resY)
{
  size_t max = (resY * resX) >> 5;

  __m128i *pCB0 = (__m128i *)pData;
  __m128i *pCB0_ = (__m128i *)pData + max;
  __m128i *pLF0 = (__m128i *)pLastFrame;
  __m128i *pLF0_ = (__m128i *)pLastFrame + max;

  __m128i halfYUV = { 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126 };

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    _mm_store_si128(pLF0, cb0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }

  max >>= 2;
  pCB0 = pCB0_;
  pLF0 = pLF0_;
  pCB0_ += max;
  pLF0_ += max;

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    _mm_store_si128(pLF0, cb0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }

  pCB0 = pCB0_;
  pLF0 = pLF0_;
  pCB0_ += max;
  pLF0_ += max;

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    _mm_store_si128(pLF0, cb0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }
}

void _slapAddStereoDiffYUV420AndAddLastFrameDiff(IN_OUT void * pData, OUT void * pLastFrame, const size_t resX, const size_t resY)
{
  size_t max = (resY * resX) >> 5;

  __m128i *pCB0 = (__m128i *)pData;
  __m128i *pCB0_ = (__m128i *)pData + max;
  __m128i *pLF0 = (__m128i *)pLastFrame;
  __m128i *pLF0_ = (__m128i *)pLastFrame + max;

  __m128i halfYUV = { 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126 };
  __m128i halfY = { 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129 };
  __m128i halfUV = { 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130 };

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    __m128i lf0 = _mm_load_si128(pLF0);

    lf0 = _mm_sub_epi8(lf0, _mm_add_epi8(cb0, halfY));
    _mm_store_si128(pCB0, lf0);
    _mm_store_si128(pLF0, lf0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    __m128i lf0_ = _mm_load_si128(pLF0_);

    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);
    cb0_ = _mm_sub_epi8(lf0_, _mm_add_epi8(cb0_, halfY));

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }

  max >>= 2;
  pCB0 = pCB0_;
  pLF0 = pLF0_;
  pCB0_ += max;
  pLF0_ += max;

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    __m128i lf0 = _mm_load_si128(pLF0);

    lf0 = _mm_sub_epi8(lf0, _mm_add_epi8(cb0, halfUV));
    _mm_store_si128(pCB0, lf0);
    _mm_store_si128(pLF0, lf0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    __m128i lf0_ = _mm_load_si128(pLF0_);

    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);
    cb0_ = _mm_sub_epi8(lf0_, _mm_add_epi8(cb0_, halfUV));

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }

  pCB0 = pCB0_;
  pLF0 = pLF0_;
  pCB0_ += max;
  pLF0_ += max;

  for (size_t i = 0; i < max; i++)
  {
    __m128i cb0 = _mm_load_si128(pCB0);
    __m128i lf0 = _mm_load_si128(pLF0);

    lf0 = _mm_sub_epi8(lf0, _mm_add_epi8(cb0, halfUV));
    _mm_store_si128(pCB0, lf0);
    _mm_store_si128(pLF0, lf0);

    __m128i cb0_ = _mm_load_si128(pCB0_);
    __m128i lf0_ = _mm_load_si128(pLF0_);

    cb0_ = _mm_add_epi8(_mm_sub_epi8(cb0_, halfYUV), cb0);
    cb0_ = _mm_sub_epi8(lf0_, _mm_add_epi8(cb0_, halfUV));

    _mm_store_si128(pCB0_, cb0_);
    _mm_store_si128(pLF0_, cb0_);

    pCB0++;
    pCB0_++;
    pLF0++;
    pLF0_++;
  }
}
