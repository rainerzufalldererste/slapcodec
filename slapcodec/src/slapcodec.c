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

  unsigned long jpegSize = 0;
  if (tjCompressFromYUV(pEncoder->pAdditionalData, (unsigned char *)pData, (int)pEncoder->resX, (int)stride, (int)pEncoder->resY, TJSAMP_420, (unsigned char **)ppCompressedData, &jpegSize, pEncoder->quality, 0))
  {
    result = slapError_Compress_Internal;
    goto epilogue;
  }

  *pSize = jpegSize;

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

slapFileWriter * slapInitFileWriter(const char *filename, const size_t sizeX, const size_t sizeY, const bool_t isStereo3d)
{
  slapFileWriter *pFileWriter = slapAlloc(slapFileWriter, 1);
  char filenameBuffer[0xFF];
  char headerFilenameBuffer[0xFF];

  if (!pFileWriter)
    goto epilogue;

  slapSetZero(pFileWriter, slapFileWriter);

  pFileWriter->pEncoder = slapCreateEncoder(sizeX, sizeY, isStereo3d);

  if (!pFileWriter->pEncoder)
    goto epilogue;

  sprintf_s(filenameBuffer, 0xFF, "%s.slap", filename);
  sprintf_s(headerFilenameBuffer, 0xFF, "%s.slap.header", filename);

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
  }

  slapFreePtr(ppFileWriter);
}

slapResult slapFinalizeFileWriter(IN slapFileWriter *pFileWriter)
{
  if (pFileWriter)
  {
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
  }

  return slapSuccess;
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

  result = slapAddFrameYUV420(pFileWriter->pEncoder, pData, stride, pFileWriter->pData, &dataSize);

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
