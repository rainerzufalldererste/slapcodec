// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "slapcodec.h"
#include "turbojpeg.h"

slapEncoder * slapCreateEncoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d)
{
  slapEncoder *pEncoder = slapAlloc(slapEncoder, 1);

  if (!pEncoder)
    goto epilogue;

  slapSetZero(pEncoder, slapEncoder);

  pEncoder->resX = sizeX;
  pEncoder->resY = sizeY;
  pEncoder->iframeStep = 30;
  pEncoder->stereo = isStereo3d;
  pEncoder->quality = 75;

  if (!isStereo3d)
    goto epilogue;

  pEncoder->pAdditionalData = tjInitCompress();

  if (!pEncoder->pAdditionalData)
    goto epilogue;

  return pEncoder;

epilogue:
  slapFreePtr(&pEncoder);
  return NULL;
}

void slapDestroyEncoder(IN_OUT slapEncoder **ppEncoder)
{
  if (ppEncoder && *ppEncoder)
    tjDestroy((*ppEncoder)->pAdditionalData);

  slapFreePtr(ppEncoder);
}

slapResult slapFinalizeEncoder(IN slapEncoder *pEncoder)
{
  (void)pEncoder;

  return slapSuccess;
}

slapResult slapAddFrameYUV420(IN slapEncoder *pEncoder, IN void *pData, const size_t stride, OUT void **ppCompressedData, OUT size_t *pSize)
{
  slapResult result = slapSuccess;

  if (!pEncoder || !pData || !ppCompressedData || !pSize)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  if (tjCompressFromYUV(pEncoder->pAdditionalData, (unsigned char *)pData, (int)pEncoder->resX, 4, (int)pEncoder->resY, TJSAMP_420, (unsigned char **)ppCompressedData, &pEncoder->__data0, pEncoder->quality, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pEncoder->pAdditionalData));
    result = slapError_Compress_Internal;
    goto epilogue;
  }

  *pSize = pEncoder->__data0;

  (void)stride;

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
    if ((size_t)SLAP_HEADER_BLOCK_SIZE != fwrite(pFileWriter->frameSizeOffsets, 1, SLAP_HEADER_BLOCK_SIZE, pFileWriter->pHeaderFile))
    {
      result = slapError_FileError;
      goto epilogue;
    }

    pFileWriter->frameSizeOffsetIndex = 0;
  }

epilogue:
  return result;
}

slapFileWriter * slapCreateFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const bool_t isStereo3d)
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

  pFileWriter->pEncoder = slapCreateEncoder(sizeX, sizeY, isStereo3d);

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

  if (slapSuccess != _slapWriteToHeader(pFileWriter, (uint64_t)pFileWriter->pEncoder->stereo))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
    goto epilogue;

  if (slapSuccess != _slapWriteToHeader(pFileWriter, 0))
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

  if (pFileWriter->headerPosition != fread(pData, sizeof(uint64_t), pFileWriter->headerPosition, pReadFile))
    goto epilogue;

  ((uint64_t *)pData)[0] = pFileWriter->headerPosition;
  ((uint64_t *)pData)[1] = pFileWriter->frameCount;

  if (pFileWriter->headerPosition != fwrite(pData, 1, pFileWriter->headerPosition, pFile))
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

  // TODO: do in multiple steps to not consume enormous amounts of memory.
  slapRealloc(&pData, uint8_t, fileSize);

  if (fileSize != fread(pData, 1, fileSize, pReadFile))
    goto epilogue;

  if (fileSize != fwrite(pData, 1, fileSize, pFile))
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

slapResult slapFileWriter_AddFrameYUV420(IN slapFileWriter *pFileWriter, IN void *pData, const size_t stride)
{
  slapResult result = slapSuccess;
  size_t dataSize = 0;

  if (!pFileWriter || !pData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  result = slapAddFrameYUV420(pFileWriter->pEncoder, pData, stride, &pFileWriter->pData, &dataSize);

  if (result != slapSuccess)
    goto epilogue;

  if (dataSize != fwrite(pFileWriter->pData, 1, dataSize, pFileWriter->pMainFile))
  {
    result = slapError_FileError;
    goto epilogue;
  }

  pFileWriter->frameCount++;
  result = _slapWriteToHeader(pFileWriter, dataSize);

  if (result != slapSuccess)
    goto epilogue;

epilogue:
  return result;
}

slapDecoder * slapCreateDecoder(const size_t sizeX, const size_t sizeY, const bool_t isStereo3d)
{
  slapDecoder *pDecoder = slapAlloc(slapDecoder, 1);

  if (!pDecoder)
    goto epilogue;

  slapSetZero(pDecoder, slapDecoder);

  pDecoder->resX = sizeX;
  pDecoder->resY = sizeY;
  pDecoder->iframeStep = 30;
  pDecoder->stereo = isStereo3d;

  if (!isStereo3d)
    goto epilogue;

  pDecoder->pAdditionalData = tjInitDecompress();

  if (!pDecoder->pAdditionalData)
    goto epilogue;

  return pDecoder;

epilogue:
  slapFreePtr(&pDecoder);
  return NULL;
}

void slapDestroyDecoder(IN_OUT slapDecoder **ppDecoder)
{
  if (ppDecoder && *ppDecoder)
    tjDestroy((*ppDecoder)->pAdditionalData);

  slapFreePtr(ppDecoder);
}

slapResult slapFinalizeDecoder(IN slapDecoder * pDecoder)
{
  (void)pDecoder;

  return slapSuccess;
}

slapResult slapDecodeFrame(IN slapDecoder * pDecoder, IN void * pData, const size_t length, IN_OUT void * pYUVData)
{
  slapResult result = slapSuccess;

  if (!pDecoder || !pData || !length || !pYUVData)
  {
    result = slapError_ArgumentNull;
    goto epilogue;
  }

  pDecoder->frameIndex++;

  if (tjDecompressToYUV2(pDecoder->pAdditionalData, (unsigned char *)pData, (unsigned long)length, (unsigned char *)pYUVData, (int)pDecoder->resX, 4, (int)pDecoder->resY, TJFLAG_FASTDCT))
  {
    slapLog(tjGetErrorStr2(pDecoder->pAdditionalData));
    result = slapError_Compress_Internal;
    goto epilogue;
  }

epilogue:
  return result;
}
